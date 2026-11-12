# Modified from QuaRot (https://github.com/spcl/QuaRot), Apache License 2.0.
import torch
import signrot.functional
import signrot.functional.quantization as quant_utils
import signrot.functional.kernel as kernel


class Linear4bit(torch.nn.Module):
    weight: torch.Tensor
    weight_scales: torch.Tensor
    weight_sum: torch.Tensor
    bias: torch.Tensor | None
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        '''
        Symmetric 4-bit Linear Layer.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scales',
                             torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', (torch.ones((self.out_features, self.in_features // 2),
                                                             # SubByte weight
                                                             dtype=torch.uint8, requires_grad=False)))
        if bias:                                                        
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None
        
    def forward(self, x):
        if not isinstance(x, kernel.PackedQuantizedTensor):
            # dequantize the weight at online
            w_deq = quant_utils.sym_dequant(quant_utils.unpack_i4(self.weight), self.weight_scales)
            return torch.nn.functional.linear(x, w_deq, self.bias)

        assert type(x) == kernel.PackedQuantizedTensor #Quantized input is given
        x, scales_x, zeros_x = x.quantized_x, x.scales_x, x.zeros_x
        if zeros_x is not None:
            x = kernel.matmul_ui4_si4(x, self.weight)
            if not hasattr(self, 'weight_sum'):
                self.register_buffer('weight_sum', kernel.sum_i4(self.weight), persistent=False)
            zeros = kernel.matmul_colrow_i8_i32(zeros_x, self.weight_sum.mT)
            x -= zeros
        else:
            x = kernel.matmul_si4_si4(x, self.weight)

        if self.bias is not None:
            return kernel.sym_dequant(x, scales_x, self.weight_scales) + self.bias
        else:
            return kernel.sym_dequant(x, scales_x, self.weight_scales)

    @staticmethod
    def from_float(module: torch.nn.Linear, weight_scales=None,):
        '''
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        weight_matrix = module.weight.data
        
        
        int_module = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, dtype=weight_matrix.dtype).to(weight_matrix.dtype)
        if weight_scales is not None:
            assert weight_scales.shape == (module.out_features, 1), 'weight_scales should have shape (out_features, 1)'
            weight_matrix = weight_matrix.cuda()
            int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
            int_rounded_weight = (weight_matrix/weight_scales.cuda()).round()
            int_module.weight.copy_(signrot.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())
        
            if module.bias is not None:
                int_module.bias.copy_(module.bias)
        
        return int_module

class AsymSymmLinear4bit(torch.nn.Module):
    weight: torch.Tensor
    weight_scales: torch.Tensor
    weight_sum: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        '''
        Asym-act Symm-weight 4-bit Linear Layer.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight', (torch.ones((self.out_features, self.in_features // 2),
                                                             # SubByte weight
                                                             dtype=torch.uint8, requires_grad=False)))
        self.register_buffer('weight_scales',
                             torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight_sum', (torch.ones((self.out_features, 1),
                                                             dtype=torch.uint8, requires_grad=False)))
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None
        
    def forward(self, x):
        assert type(x) == kernel.PackedQuantizedTensor #Quantized input is given
        x, scales_x, zeros_x = x.quantized_x, x.scales_x, x.zeros_x
        # TODO: Fuse kernel?
        x = kernel.matmul_si4_si4(x, self.weight)
        zeros = kernel.matmul_colrow_i8_i32(zeros_x, self.weight_sum.mT)
        out = kernel.sym_dequant(x-zeros, scales_x, self.weight_scales)
        if self.bias is not None:
            out += self.bias
        return out

    @staticmethod
    def from_float(module: torch.nn.Linear, weight_scales=None,):
        '''
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        weight_matrix = module.weight.data
        
        
        int_module = AsymSymmLinear4bit(module.in_features, module.out_features, bias=module.bias is not None, dtype=weight_matrix.dtype).to(weight_matrix.dtype)
        if weight_scales is not None:
            assert weight_scales.shape == (module.out_features, 1), 'weight_scales should have shape (out_features, 1)'
            weight_matrix = weight_matrix.cuda()
            int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
            int_rounded_weight = (weight_matrix/weight_scales.cuda()).round()
            int_weight_packed = signrot.functional.pack_i4(int_rounded_weight.to(torch.int8))
            int_weight_sum = kernel.sum_i4(int_weight_packed)
            int_module.weight.copy_(int_weight_packed.cpu())
            int_module.weight_sum.copy_(int_weight_sum.cpu())

            if module.bias is not None:
                int_module.bias.copy_(module.bias)

        return int_module
