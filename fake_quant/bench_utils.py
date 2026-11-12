import contextlib
import gc
import logging
import os
import subprocess
import time

import numpy as np
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_max_memory
from transformers.cache_utils import DynamicCache, StaticCache


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def _sync_all():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def _reset_peak_memory():
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def _peak_memory():
    return sum(
        torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
    )


def dispatch_for_bench(model, gpu_mem_scale=0.9):
    """Place the model on the available GPU(s), refusing CPU/disk offload.

    Unlike the eval path, the device map is inferred from the current (possibly
    int4-packed) tensor sizes, so it must run after enable_kernel_modules.
    """
    max_memory = get_max_memory()
    max_memory.pop("cpu", None)
    max_memory = {k: int(v * gpu_mem_scale) for k, v in max_memory.items()}
    device_map = infer_auto_device_map(
        model, max_memory, no_split_module_classes=model._no_split_modules
    )
    if any(device in ("cpu", "disk") for device in device_map.values()):
        raise RuntimeError(
            f"Model does not fit on the available GPUs for benchmarking: {device_map}"
        )
    logging.info(f"Benchmark device map: {device_map}")
    dispatch_model(model, device_map)
    _cleanup()


@contextlib.contextmanager
def _sample_gpu_telemetry(path, interval_ms=100):
    """Log SM clock / temperature / power to `path` for the duration of the block.

    Sampled by a background `nvidia-smi`, not from the timing loop: polling in
    the loop would idle the GPU between samples, which is exactly the drift this
    is meant to detect. `perf_counter` offsets recorded per sample line the trace
    up against `*_start_s` in the results.
    """
    if not path:
        yield
        return
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active",
        "--format=csv,noheader",
        f"--loop-ms={interval_ms}",
    ]
    proc = None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            # both clocks, so the nvidia-smi wall-clock timestamps can be mapped
            # exactly onto the perf_counter offsets recorded per sample (its
            # --loop-ms cadence is not even enough to interpolate from)
            f.write(
                f"# perf_counter at start: {time.perf_counter()}\n"
                f"# time at start: {time.time()}\n"
            )
            f.flush()
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL)
            yield
    except (OSError, subprocess.SubprocessError) as exc:
        logging.warning(f"GPU telemetry unavailable ({exc}); continuing without it")
        yield
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.kill()
            logging.info(f"GPU telemetry written to {path}")


def _time_fn(fn, repeats, warmup, soak_s=0.0):
    """Time `fn` `repeats` times, after warming the GPU up on `fn` itself.

    The clocks on a consumer card are not lockable, so the only defence against
    DVFS drift is to reach a steady thermal state before measuring and to stay
    there: `soak_s` extends the warm-up until the GPU has been busy that long,
    and the timed iterations run back to back. Nothing that idles the GPU may
    sit between the soak and the samples, or after it -- `_cleanup()` therefore
    runs *before* the soak. A gc pass over a 30B model plus an `empty_cache()`
    costs ~75ms, which is enough to cost the sample that follows it: with the
    cleanup after the soak, prefill sample 0 measured +2.4% while samples 1-9
    held within 0.05% of each other.
    """
    # before the soak, not after: this is the only point where the GPU may idle
    _cleanup()

    warm_start = time.perf_counter()
    done = 0
    while done < warmup or (time.perf_counter() - warm_start) < soak_s:
        fn()
        done += 1
        # a soak is only a soak if the GPU is actually kept busy, so sync per
        # iteration rather than letting the queue run dry at the end
        _sync_all()
    _sync_all()

    times_ms, peaks, starts = [], [], []
    for _ in range(repeats):
        _reset_peak_memory()
        _sync_all()
        start = time.perf_counter()
        fn()
        _sync_all()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)
        peaks.append(_peak_memory())
        # wall-clock offsets let a sample be lined up against the clock/temp
        # trace written by _sample_gpu_telemetry
        starts.append(start)
    return np.array(times_ms), np.array(peaks), starts


class StaticGraphedDecoder:
    """Replay a single decode step over a transformers StaticCache from a CUDA graph.

    The FP16 baseline decodes into a DynamicCache, which grows by concatenation
    and cannot be captured, and `kernel_utils.GraphedDecoder` only handles the
    paged int4 cache of the kernel path. Left eager, the FP16 decode is
    host-bound like the quantized one, so an FP16-vs-int4 decode comparison
    would measure Python dispatch rather than the GPU.

    A StaticCache is allocated by the prefill that fills it and written in place
    at `cache_position`, so the step is capture-safe afterwards. Rewinding only
    moves the write position: entries past it are masked out and overwritten by
    the following steps. Capture writes a throwaway token at the start position,
    which the first real step overwrites.
    """

    def __init__(self, model, cache, batch_size, position):
        self.model = model
        self.cache = cache
        self.position = position
        device = model.get_input_embeddings().weight.device
        self.input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        self.cache_position = torch.zeros(1, dtype=torch.long, device=device)
        self.graph = None
        self.logits = None

    def _forward(self):
        return self.model(
            self.input_ids,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=self.cache_position,
        ).logits

    def rewind(self, position):
        self.position = position

    def capture(self, warmup=3):
        self.cache_position.fill_(self.position)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                self._forward()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self._forward()
        return self

    def step(self, input_ids):
        """Run one decode step; returns the logits buffer (overwritten next step)."""
        self.input_ids.copy_(input_ids)
        self.cache_position.fill_(self.position)
        self.position += 1
        self.graph.replay()
        return self.logits


def _try_static_graph_decoder(model, cache, batch_size, position, probes=2, tol=1e-3):
    """Capture a decode graph over a filled StaticCache, checked against eager."""
    try:
        decoder = StaticGraphedDecoder(model, cache, batch_size, position).capture()
    except Exception as exc:  # capture is best-effort; the eager path still works
        logging.warning(f"CUDA graph capture over the static cache failed, falling back to eager decode: {exc}")
        torch.cuda.synchronize()
        return None
    device = decoder.input_ids.device
    vocab = model.get_input_embeddings().weight.shape[0]
    for _ in range(probes):
        # CPU sampling: the CUDA RNG may be unusable after a failed capture
        token = torch.randint(0, vocab, (batch_size, 1), dtype=torch.long).to(device)
        eager = model(
            token, past_key_values=cache, use_cache=True,
            cache_position=torch.tensor([position], device=device),
        ).logits.float()
        decoder.rewind(position)
        graphed = decoder.step(token).float()
        rel = (graphed - eager).abs().max() / eager.abs().max().clamp(min=1e-6)
        if not torch.isfinite(rel) or rel.item() > tol:
            logging.warning(
                f"CUDA graph output diverges from eager (relative {rel.item():.3e}); "
                f"falling back to eager decode"
            )
            return None
    decoder.rewind(position)
    logging.info("Captured a CUDA graph for the decode step (static KV cache)")
    return decoder


@torch.no_grad()
def runtime_bench(model, args):
    device = model.get_input_embeddings().weight.device
    results = []
    telemetry_path = None
    if getattr(args, "bench_log_clocks", False):
        telemetry_path = os.path.join(
            getattr(args, "save_log_dir", ".") or ".", "gpu_telemetry.csv"
        )
    with _sample_gpu_telemetry(telemetry_path):
        for bsz in args.bench_bsz:
            for prefill_len in args.bench_prefill_len:
                try:
                    results.append(
                        _bench_config(
                            model,
                            device,
                            bsz,
                            prefill_len,
                            args.bench_decode_steps,
                            args.bench_repeats,
                            args.bench_warmup,
                            use_cuda_graph=not getattr(
                                args, "bench_no_cuda_graph", False
                            ),
                            soak_s=getattr(args, "bench_soak_s", 0.0),
                        )
                    )
                except torch.OutOfMemoryError:
                    if not getattr(args, "bench_skip_oom", False):
                        raise
                    logging.warning(
                        f"Runtime bench: bsz={bsz} prefill_len={prefill_len} ran out of "
                        f"memory, skipping"
                    )
                _cleanup()
    _log_summary(results)
    return results


def _bench_config(model, device, bsz, prefill_len, decode_steps, repeats, warmup,
                  use_cuda_graph=True, soak_s=0.0):
    logging.info(
        f"Runtime bench: bsz={bsz} prefill_len={prefill_len} decode_steps={decode_steps} "
        f"(repeats={repeats}, warmup={warmup}, soak={soak_s:g}s)"
    )
    # sampled on the CPU: a failed CUDA-graph capture (e.g. a capture-unsafe
    # model falling back to eager) leaves the CUDA RNG unusable, and the bench
    # must still run after that fallback
    input_ids = torch.randint(
        1000, 8000, (bsz, prefill_len), dtype=torch.long
    ).to(device)
    next_ids = torch.randint(1000, 8000, (bsz, 1), dtype=torch.long).to(device)

    # kernel-patched models (kernel_utils) pre-size their paged KV cache from
    # this hint; without it the cache grows (realloc) during decode
    def hint_max_length():
        if hasattr(model, "_expected_max_length"):
            model._expected_max_length = prefill_len + decode_steps

    # logits_to_keep=1 mimics generation (only last-token logits), otherwise the
    # fp32 logits of the full prefill dominate memory on large-vocab models
    def prefill_fn():
        model(input_ids, use_cache=True, logits_to_keep=1)

    prefill_ms, prefill_peak, prefill_starts = _time_fn(
        prefill_fn, repeats, warmup, soak_s
    )
    logging.info(
        f"  prefill: {prefill_ms.mean():.2f} +- {1.96 * prefill_ms.std():.2f} ms"
    )

    hint_max_length()
    cache = model(input_ids, use_cache=True, logits_to_keep=1).past_key_values

    # Eager decode is host-bound on this stack (~1.3k kernel launches per token,
    # each shorter than its launch cost), so without a graph the measurement is
    # dominated by CPU dispatch rather than the kernels.
    decoder = None
    static_positions = None
    if use_cuda_graph and isinstance(cache, DynamicCache):
        # FP16 baseline: re-prefill into a preallocated StaticCache so that the
        # decode step can be captured like the kernel path's
        del cache
        _cleanup()
        cache = StaticCache(config=model.config, max_cache_len=prefill_len + decode_steps)
        static_positions = torch.arange(prefill_len, device=device)
        model(input_ids, past_key_values=cache, use_cache=True,
              cache_position=static_positions, logits_to_keep=1)
        decoder = _try_static_graph_decoder(model, cache, bsz, prefill_len)
        if decoder is None:
            del cache
            _cleanup()
            static_positions = None
            cache = model(input_ids, use_cache=True, logits_to_keep=1).past_key_values
    elif use_cuda_graph:
        import kernel_utils

        decoder = kernel_utils.try_graph_decoder(model, cache, bsz)

    if static_positions is not None:
        def decode_fn():
            decoder.rewind(prefill_len)
            for _ in range(decode_steps):
                decoder.step(next_ids)
    elif decoder is not None:
        def decode_fn():
            cache.crop(prefill_len)
            for _ in range(decode_steps):
                decoder.step(next_ids)
    else:
        def decode_fn():
            cache.crop(prefill_len)
            for _ in range(decode_steps):
                model(next_ids, past_key_values=cache, use_cache=True)

    decode_ms, decode_peak, decode_starts = _time_fn(
        decode_fn, repeats, warmup, soak_s
    )
    logging.info(
        f"  decode ({decode_steps} steps): {decode_ms.mean():.2f} +- {1.96 * decode_ms.std():.2f} ms "
        f"({decode_ms.mean() / decode_steps:.2f} ms/token, "
        f"{bsz * decode_steps / (decode_ms.mean() / 1000):.2f} tokens/s)"
    )

    if static_positions is not None:
        # the prefill rewrites the static cache in place, so the graph stays valid
        def e2e_fn():
            model(input_ids, past_key_values=cache, use_cache=True,
                  cache_position=static_positions, logits_to_keep=1)
            decoder.rewind(prefill_len)
            for _ in range(decode_steps):
                decoder.step(next_ids)
    elif decoder is not None and hasattr(cache, "reset"):
        # reuse the same cache so the captured graph stays valid; the pool and
        # index buffers do not move across a reset
        def e2e_fn():
            cache.reset()
            model(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            for _ in range(decode_steps):
                decoder.step(next_ids)
    else:
        def e2e_fn():
            hint_max_length()
            out = model(input_ids, use_cache=True, logits_to_keep=1)
            for _ in range(decode_steps):
                model(next_ids, past_key_values=out.past_key_values, use_cache=True)

    e2e_ms, e2e_peak, e2e_starts = _time_fn(e2e_fn, repeats, warmup, soak_s)
    del cache, decoder
    _cleanup()
    logging.info(f"  e2e: {e2e_ms.mean():.2f} +- {1.96 * e2e_ms.std():.2f} ms")

    peak_mem = max(prefill_peak.max(), decode_peak.max(), e2e_peak.max())
    logging.info(f"  peak memory: {peak_mem / 1024**3:.2f} GB")

    return {
        "bsz": bsz,
        "prefill_len": prefill_len,
        "decode_steps": decode_steps,
        "prefill_ms": prefill_ms.mean(),
        "prefill_ms_std": prefill_ms.std(),
        "prefill_ms_raw": prefill_ms.tolist(),
        "prefill_start_s": prefill_starts,
        "prefill_peak_mem_gb": prefill_peak.max() / 1024**3,
        "decode_ms": decode_ms.mean(),
        "decode_ms_std": decode_ms.std(),
        "decode_ms_per_token": decode_ms.mean() / decode_steps,
        "decode_ms_raw": decode_ms.tolist(),
        "decode_start_s": decode_starts,
        "decode_tokens_per_s": bsz * decode_steps / (decode_ms.mean() / 1000),
        "decode_peak_mem_gb": decode_peak.max() / 1024**3,
        "e2e_ms": e2e_ms.mean(),
        "e2e_ms_std": e2e_ms.std(),
        "e2e_ms_raw": e2e_ms.tolist(),
        "e2e_start_s": e2e_starts,
        "e2e_peak_mem_gb": e2e_peak.max() / 1024**3,
        "peak_mem_gb": peak_mem / 1024**3,
    }


def _log_summary(results):
    header = (
        "bsz | prefill_len | decode_steps | prefill_ms | decode_ms | ms/token | "
        "tokens/s | e2e_ms | peak_mem_gb"
    )
    logging.info("Runtime bench summary:")
    logging.info(header)
    for r in results:
        logging.info(
            f"{r['bsz']} | {r['prefill_len']} | {r['decode_steps']} | "
            f"{r['prefill_ms']:.2f} | {r['decode_ms']:.2f} | "
            f"{r['decode_ms_per_token']:.2f} | {r['decode_tokens_per_s']:.2f} | "
            f"{r['e2e_ms']:.2f} | {r['peak_mem_gb']:.2f}"
        )
