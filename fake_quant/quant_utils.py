# Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
import math
import transformers
import torch
import utils
import hadamard_utils
import fast_hadamard_transform


from signrot.functional.quantization import (
    get_minq_maxq,
    asym_quant,
    asym_dequant,
    asym_quant_dequant,
    sym_quant,
    sym_dequant,
    sym_quant_dequant,
    two_compl,
    pack_i4,
    unpack_i4,
)


class ActQuantizer(torch.nn.Module):
    """
    A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
    for the activations.
    """

    def __init__(self):
        super(ActQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0), persistent=False)
        self.register_buffer("scale", torch.zeros(1), persistent=False)
        self.register_buffer("zero", torch.zeros(1), persistent=False)
        self.bits = 16

    def free(self):
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        elif self.sym:
            return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
        return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(
        self, bits, groupsize=-1, sym=False, clip_ratio=1.0, extend_zero=False
    ):
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert (
            self.clip_ratio <= 1 and self.clip_ratio > 0
        ), "Clip ratio should be in (0, 1]"
        self.extend_zero = extend_zero

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(
            -1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize
        )

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            # xmin == xmax (e.g. a constant token after sign adjustment in bf16)
            # would give scale = 0 and zero = inf, producing NaN on dequant
            tmp = xmin == xmax
            xmin[tmp] -= 1
            xmax[tmp] += 1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        xmin = reshaped_x.min(1)[0]
        xmax = reshaped_x.max(1)[0]
        if self.extend_zero:
            tmp = torch.zeros(reshaped_x.shape[0], device=dev)
            xmin = torch.minimum(xmin, tmp)
            xmax = torch.maximum(xmax, tmp)
        xmin = xmin * self.clip_ratio
        xmax = xmax * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            # xmin == xmax (e.g. a constant token after sign adjustment in bf16)
            # would give scale = 0 and zero = inf, producing NaN on dequant
            tmp = xmin == xmax
            xmin[tmp] -= 1
            xmax[tmp] += 1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = (
                self.scale.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )
            self.zero = (
                self.zero.unsqueeze(1)
                .repeat(1, reshaped_x.shape[-1])
                .reshape(init_shape)
            )


class ActQuantWrapper(torch.nn.Module):
    """
    This class is a wrapper for the activation quantization.
    We extract the FP features in the forward pass and quantize the rest using
    the self.quantizer object.
    If a rotation Q is provided, the weight matrix will be rotated,
    a pre-forward hook will be registerd to rotate the activation before quantization.
    """

    weight: torch.Tensor
    bias: torch.Tensor
    had_K: torch.Tensor | None
    had_adj_sign: torch.Tensor | None

    def __init__(self, module: torch.nn.Linear):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer("had_K", torch.tensor(0), persistent=False)
        self._buffers["had_K"] = None
        self.K: int = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False
        self.had_adj_ready = False
        self.register_buffer("had_adj_sign", None)

        self.use_kernel = False
        self.kernel_quantizer = None
        self.kernel_module = None
        self.kernel_hadamard = None

        self.log_ma_position = False
        self.label = ""
        self.ma_logs_before_rotation = []  # per single batch
        self.ma_logs = []  # per single batch
        self.distribution_shift_logs = []  # per single batch
        self.distribution_shift_before_adjust_logs = []  # per single batch

    def __getattr__(self, name: str):
        # reflect module attributes
        if name in ["weight", "bias", "in_features", "out_features"]:
            return self.module.__getattr__(name)
        return super().__getattr__(name)

    def extra_repr(self) -> str:
        str_ = f"Input Quantizer Bits: {self.quantizer.bits}"
        if self.quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        str_ += f"\nOutput Quantizer Bits: {self.out_quantizer.bits}"
        if self.out_quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.out_quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        return str_

    def enable_kernel(self, weight_quantizer: "WeightQuantizer", free_fp16: bool = False):
        """
        Replace the fake quantize->linear path with the real signrot CUDA kernels.
        The weight is packed to int4 using the (already calibrated) weight quantizer
        scales; the activation is quantized/matmul-ed/dequantized by signrot kernels.
        If free_fp16 is set, the fp16/bf16 weight is dropped after packing so the
        model fits on small GPUs (the kernel path never reads it again).
        """
        import signrot.nn

        assert self.quantizer.bits == 4, "signrot kernel only supports 4-bit activations"
        assert self.quantizer.groupsize <= 0, "signrot kernel does not support groupwise activation quantization"
        assert not self.quantizer.extend_zero, "signrot kernel does not support extend_zero activation quantization"
        assert weight_quantizer.bits == 4, "signrot kernel only supports 4-bit weights"
        assert weight_quantizer.sym, "signrot kernel only supports symmetric weight quantization"
        assert weight_quantizer.scale.numel() == self.module.out_features, \
            "signrot kernel only supports per-channel weight scales"
        if self.had_adj_sign is not None and not self.had_adj_ready:
            # MoE experts that received no tokens during the (single-sample)
            # adjustment calibration keep their initial all-ones sign: identity
            # adjustment, nothing to fold into the weight
            assert bool((self.had_adj_sign == 1).all()), \
                "had_adj_sign was modified without being folded into the weight"
            self.had_adj_ready = True
        assert self.had_adj_sign is None or self.online_full_had, \
            "had_adj is only supported on online_full_had (down_proj) layers in the kernel path"
        assert self.out_quantizer.bits >= 16, \
            "output fake-quant must be off on the kernel path (V-cache is quantized by the kernel KV cache)"

        if self.online_full_had:
            self.kernel_hadamard = signrot.nn.OnlineHadamard(
                self.module.in_features,
                force_fp32=self.fp32_had,
                had_adj=self.had_adj_sign is not None,
            )
            if self.had_adj_sign is not None:
                self.kernel_hadamard.had_adj_sign = self.had_adj_sign.to(torch.int8)
        elif self.online_partial_had:
            self.kernel_hadamard = signrot.nn.OnlineHadamard(
                self.module.in_features // self.had_dim, force_fp32=self.fp32_had
            )

        if self.quantizer.sym:
            self.kernel_quantizer = signrot.nn.Quantizer(clip_ratio=self.quantizer.clip_ratio)
        else:
            self.kernel_quantizer = signrot.nn.AsymQuantizer(clip_ratio=self.quantizer.clip_ratio)

        # signrot kernels are fp16-only; bf16 models (e.g. Qwen3) need their
        # already-quantized weights cast before packing
        if self.module.weight.dtype != torch.float16:
            self.module.weight.data = self.module.weight.data.to(torch.float16)
            if self.module.bias is not None:
                self.module.bias.data = self.module.bias.data.to(torch.float16)

        weight_scales = weight_quantizer.scale.reshape(-1, 1).to(torch.float16)
        self.kernel_module = signrot.nn.Linear4bit.from_float(
            self.module, weight_scales=weight_scales
        )
        if free_fp16:
            self.module.weight.data = torch.empty(
                0, dtype=self.module.weight.dtype, device=self.module.weight.device
            )
        self.use_kernel = True

    def kernel_quantize(self, x):
        """Online rotation (+ had_adj sign) and activation quantization only.

        Split out from the matmul so sibling projections that consume the same
        activation (q/k/v, gate/up) can quantize it once instead of once each.
        """
        x = x.to(torch.float16)
        if self.online_full_had:
            x = self.kernel_hadamard(x)  # fused Hadamard + had_adj sign
        elif self.online_partial_had:
            init_shape = x.shape
            x = x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim).transpose(1, 2)
            x = self.kernel_hadamard(x)
            x = x.transpose(1, 2).reshape(init_shape)
        return self.kernel_quantizer(x)

    def kernel_matmul(self, packed_x, x_dtype):
        """int4 matmul + dequant on an already-quantized activation."""
        return self.kernel_module(packed_x).to(x_dtype)

    def shares_kernel_input_with(self, other):
        """Whether `other` can reuse this layer's quantized activation."""
        return (
            getattr(other, "use_kernel", False)
            and self.use_kernel
            and not (self.online_full_had or self.online_partial_had)
            and not (other.online_full_had or other.online_partial_had)
            and type(self.kernel_quantizer) is type(other.kernel_quantizer)
            and self.kernel_quantizer.clip_ratio == other.kernel_quantizer.clip_ratio
            and self.module.in_features == other.module.in_features
        )

    def _forward_kernel(self, x, x_dtype):
        # Deployment path: online rotation (+ had_adj sign), activation quant,
        # int4 matmul and dequant all run through the signrot kernel modules
        # (fp16-only, so bf16 activations are cast around them).
        return self.kernel_matmul(self.kernel_quantize(x), x_dtype)

    def forward(self, x):
        x_dtype = x.dtype

        if self.use_kernel and self.quantizer.bits < 16:
            return self._forward_kernel(x, x_dtype)

        if self.log_ma_position:
            self.ma_logs_before_rotation.extend(self.get_ma(x))

        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(
                    x_dtype
                )
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)
        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!
            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            x_reshaped = x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
            # x = hadamard_utils.matmul_hadU_cuda(x.mT, self.had_K, self.K).mT
            if self.K == 1:
                x_reshaped = fast_hadamard_transform.hadamard_transform(
                    x_reshaped.transpose(1, 2),
                    scale=1 / math.sqrt(init_shape[-1] // self.had_dim),
                ).transpose(1, 2)
            else:  # self.had_K is not None
                x_reshaped = (
                    self.had_K.to(x.dtype) @ x_reshaped
                ) / math.sqrt(init_shape[-1] // self.had_dim)
            x = x_reshaped.reshape(init_shape)

            if self.fp32_had:
                x = x.to(x_dtype)

        if self.log_ma_position:
            self.ma_logs.extend(self.get_ma(x))

        if self.had_adj_sign is not None:
            if self.log_ma_position:
                self.distribution_shift_before_adjust_logs.append(self.distribution_shift_check(x))

            if not self.had_adj_ready:  # not initialized
                self.had_adj_ready = True
                single_batch = x[0] if x.dim() == 3 else x
                ma_token_idx = single_batch.abs().max(dim=-1).values.argmax()
                self.had_adj_sign = ((single_batch[ma_token_idx] >= 0) * 2 - 1).char()
                self.module.weight.data = self.module.weight * self.had_adj_sign
            x = x * self.had_adj_sign

            if self.log_ma_position:
                # check distribution shift after hadamard adjustment
                self.distribution_shift_logs.append(self.distribution_shift_check(x))

        if self.quantizer.bits < 16:  # Quantize, if needed
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

        x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16:  # Quantize the output, if needed
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x

    def get_ma(self, x):
        # TODO: add quantization granularity(per-token)
        x_hidden_dim = x.shape[-1]
        x_flatten = x.flatten(-2, -1)
        ma_positions = x_flatten.abs().argmax(-1)
        ma_values = x_flatten.abs().amax(-1)
        mean_values = x_flatten.mean(-1)
        std_values = x_flatten.std(-1)
        var_values = x_flatten.var(-1)
        ma_tokens = ma_positions // x_hidden_dim
        ma_channels = ma_positions % x_hidden_dim
        return list(zip(ma_tokens.tolist(), ma_channels.tolist(), ma_values.tolist(), mean_values.tolist(), std_values.tolist(), var_values.tolist()))

    def distribution_shift_check(self, x):
        single_batch = x[0] if x.dim() == 3 else x
        ma_token_idx = single_batch.abs().max(dim=-1).values.argmax().item()
        ma_token_values = single_batch[ma_token_idx]
        mean_abs_value = ma_token_values.mean().item()
        return (ma_token_idx, mean_abs_value)

class WeightQuantizer(torch.nn.Module):
    """From GPTQ Repo"""

    scale: torch.Tensor
    zero: torch.Tensor

    def __init__(self, shape=(1,)):
        super(WeightQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))

        shape = [shape[0]] + [1] * (len(shape) - 1)
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        extend_zero=True,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)
        self.extend_zero = extend_zero
        # restrict sym quantization to the symmetric grid [-maxq, maxq] (drops the
        # -(maxq+1) level); used when a SignRot sign is refolded after quantization,
        # so the folded weight stays representable on the int4 grid
        self.sym_symmetric_grid = False

    def sym_minq(self):
        return -self.maxq if self.sym_symmetric_grid else None

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape  # (out_dim, in_dim, ...)
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        xmin = x.min(1)[0]
        xmax = x.max(1)[0]
        if self.extend_zero:
            tmp = torch.zeros(x.shape[0], device=dev)
            xmin = torch.minimum(xmin, tmp)
            xmax = torch.maximum(xmax, tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq, minq=self.sym_minq())
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(
                        x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                    )

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    def forward(self, x):
        x_dtype = x.dtype
        if not self.ready() or self.bits >= 16:
            return x

        if self.sym:
            return sym_quant_dequant(x, self.scale, self.maxq, minq=self.sym_minq()).to(x_dtype)
        return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

    def quantize(self, x):
        if not self.ready():
            raise RuntimeError(
                "Cannot quantize x because WeightQuantizer is not ready!"
            )

        if self.sym:
            return sym_quant(x, self.scale, self.maxq, minq=self.sym_minq())
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def dequantize(self, q):
        if not self.ready():
            raise RuntimeError(
                "Cannot dequantize q because WeightQuantizer is not ready!"
            )

        if self.sym:
            return sym_dequant(q, self.scale)
        else:
            return asym_dequant(q, self.scale, self.zero)

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def add_actquant(
    module,
    name="",
    layers=[
        torch.nn.Linear,
    ],
):
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) == torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + "." + name1 if name != "" else name1, layers)


def find_qlayers(module, layers=[torch.nn.Linear, ActQuantWrapper], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def disable_act_quant(module):
    bits_config = {}
    for name, m in module.named_modules():
        if isinstance(m, ActQuantWrapper):
            bits_config[name] = m.quantizer.bits
            m.quantizer.bits = 16

    return bits_config


def enable_act_quant(module, bits_config):
    for name, m in module.named_modules():
        if isinstance(m, ActQuantWrapper):
            m.quantizer.bits = bits_config[name]

