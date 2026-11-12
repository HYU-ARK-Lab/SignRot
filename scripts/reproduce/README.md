# Reproducing the paper results

Each script below reproduces tables or figures of the paper. Run them from the repository root after
installing the environment (`git submodule update --init --recursive && uv sync`), inside it
(e.g. `source .venv/bin/activate`).

```bash
scripts/reproduce/main_results.sh                       # all models
scripts/reproduce/main_results.sh meta-llama/Llama-2-7b-hf
DRY_RUN=1 scripts/reproduce/ablation_adjust_layers.sh   # print the commands only
```

The accuracy runs go through `scripts/fake_quant.sh`, whose `quarot` and `signrot` presets hold the
quantization settings. Each run writes its log, with the perplexities and the lm_eval results, to
`fake_quant/experiments/<model>/<save_name>/<datetime>/rank0.log`. The analysis and efficiency scripts
write their tables and figures to `fake_quant/experiments/reproduce/<topic>/`.

| Variable | Meaning |
|---|---|
| `CUDA_VISIBLE_DEVICES` | GPUs to use. The 70B models and Qwen3-30B-A3B need several GPUs or one large GPU. |
| `LM_EVAL_BS` | lm_eval batch size (default 32). Zero-shot accuracies can change slightly with the batch size. |
| `DRY_RUN=1` | Print the commands instead of running them. |

## Scripts

| Paper result | Script |
|---|---|
| Main results, QuaRot+SignRot rows (LLaMA2/3, RTN and GPTQ) | `main_results.sh` |
| Results on other models (Mistral-7B-v0.3, Qwen3-30B-A3B) | `other_models.sh` |
| End-to-end latency (main text and appendix) | `efficiency_e2e_latency.sh` |
| Ablation: symmetric vs asymmetric activation quantization | `ablation_sym_asym.sh` |
| Ablation: calibration data of the sign adjustment vector | `ablation_calibration_data.sh` |
| Ablation: sign adjustment at other projection inputs | `ablation_adjust_layers.sh` |
| Appendix: massive outlier statistics, early massive outlier layers, activation distribution statistics | `ma_statistics.sh` |
| Appendix: single-layer quantization KL (figure and table) | `single_layer_kl.sh` (run `ma_statistics.sh` first) |
| Appendix: activation clipping ratio of LLaMA3-70B | `appendix_clip_sweep.sh` |
| Appendix: single linear layer speed | `efficiency_speedup_single_linear.sh` |

Notes:
- LLaMA3-70B with GPTQ uses an activation clipping ratio of 1.0; the ablations run several seeds.
- The efficiency scripts use the CUDA kernels (`--use_kernel`); the paper measured them on one RTX 3090.
  Keep the GPU otherwise idle while measuring.

## Results not produced by this repository

**QuaRot rows of the main results.** They are taken from the QuaRot paper. In the ablations and on
Mistral-7B and Qwen3-30B-A3B, QuaRot is this repository's pipeline with the sign adjustment disabled
(a bare `--adj_layers`, or the `quarot` preset), and the scripts above include those rows.

**SpinQuant and DartQuant rows.** SpinQuant(+SignRot) and DartQuant(+SignRot) were run with the SpinQuant
and DartQuant code bases and their rotation matrices; they are not part of this repository.
