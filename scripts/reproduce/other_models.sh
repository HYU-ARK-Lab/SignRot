#!/bin/bash
# Results on other models table: Mistral-7B-v0.3 and Qwen3-30B-A3B (FP16, QuaRot, QuaRot+SignRot).
# Usage: scripts/reproduce/other_models.sh

source "$(dirname "$0")/common.sh"

declare -A W_QUANTIZERS=(
  ["mistralai/Mistral-7B-v0.3"]="gptq"
  ["Qwen/Qwen3-30B-A3B"]="rtn gptq"
)

for MODEL_NAME in "mistralai/Mistral-7B-v0.3" "Qwen/Qwen3-30B-A3B"; do
  fake_quant --model "$MODEL_NAME" --save_name fp16 "${EVAL[@]}"
  for WQ in ${W_QUANTIZERS[$MODEL_NAME]}; do
    for METHOD in quarot signrot; do
      fake_quant --model "$MODEL_NAME" --save_name "$METHOD/$WQ" --preset "$METHOD" --w_quantizer "$WQ" "${EVAL[@]}"
    done
  done
done
