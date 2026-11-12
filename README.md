
# SignRot: LLM Quantization with Massive Outlier-Aware Sign-Adjusted Rotation
This repository contains the code for **SignRot**: LLM Quantization with Massive Outlier-Aware Sign-Adjusted Rotation



## Abstract
we propose SignRot (Sign-adjusted Rotation), which identifies massive outlier channels to construct a sign adjust vector, applying rotation prevents distribution fragmentation. SignRot significantly improves handling of outliers, applying SignRot to QuaRot reduces the Wikitext2 perplexity to 5.88 and increases the average zero-shot reasoning accuracy to 72.73, surpassing SpinQuant on the LLaMA3-70B model.

## Usage

To use SignRot, first clone the repository and install the required dependencies:

```bash
git submodule update --init --recursive
uv sync
```

To quantize a model, use the following command:
```bash
./scripts/fake_quant.sh --model meta-llama/Meta-Llama-3-8B --preset signrot --save_name signrot_gptq --w_quantizer gptq --save_qmodel
```

Simulated results can be evaluated using the following command:
```bash
./scripts/fake_quant.sh --model meta-llama/Meta-Llama-3-8B --preset signrot --save_name signrot_gptq --w_quantizer gptq --load_qmodel --ppl_eval --lm_eval
```

Or, to obtain all results needed for the main table, you can use the following command:
```bash
./scripts/reproduce/main_results.sh
```

Scripts for the other tables and figures of the paper are listed in [scripts/reproduce/README.md](scripts/reproduce/README.md).
