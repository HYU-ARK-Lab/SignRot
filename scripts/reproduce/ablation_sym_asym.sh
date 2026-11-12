#!/bin/bash
# Ablation table: symmetric vs asymmetric activation quantization (QuaRot and QuaRot+SignRot, GPTQ).
# Usage: scripts/reproduce/ablation_sym_asym.sh

source "$(dirname "$0")/common.sh"

# QuaRot = the same pipeline with the sign adjustment disabled (a bare --adj_layers)
declare -A METHOD_ARGS=(
  [quarot]="--adj_layers"
  [signrot]="--adj_cal_dataset wikitext2"
)

# LLaMA2-7B, LLaMA3-8B: seeds 1-10; quantize once, then evaluate asymmetric and symmetric activations
for MODEL_NAME in "meta-llama/Llama-2-7b-hf" "meta-llama/Meta-Llama-3-8B"; do
  for SEED in 1 2 3 4 5 6 7 8 9 10; do
    for METHOD in quarot signrot; do
      read -ra M <<<"${METHOD_ARGS[$METHOD]}"
      ARGS=(--model "$MODEL_NAME" --save_name "ablation_sym_asym/seed$SEED/$METHOD" --preset signrot "${M[@]}"
        --w_quantizer gptq --seed "$SEED")
      fake_quant "${ARGS[@]}" --save_qmodel
      fake_quant "${ARGS[@]}" --load_qmodel "${EVAL[@]}"
      fake_quant "${ARGS[@]}" --load_qmodel --no-a_asym "${EVAL[@]}"
    done
  done
done

# LLaMA3-70B: QuaRot (asym) and QuaRot+SignRot (asym), main-results setting
fake_quant --model meta-llama/Meta-Llama-3-70B --save_name ablation_sym_asym/quarot_asym \
  --preset signrot --adj_layers --w_quantizer gptq --a_clip_ratio 1.0 "${EVAL[@]}"
fake_quant --model meta-llama/Meta-Llama-3-70B --save_name ablation_sym_asym/signrot_asym \
  --preset signrot --w_quantizer gptq --a_clip_ratio 1.0 "${EVAL[@]}"
