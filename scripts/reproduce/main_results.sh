#!/bin/bash
# Main results table: QuaRot+SignRot rows (LLaMA2/3, RTN and GPTQ).
# Usage: scripts/reproduce/main_results.sh [MODEL ...]

source "$(dirname "$0")/common.sh"

MODEL_LIST=(
  "meta-llama/Llama-2-7b-hf"
  "meta-llama/Llama-2-13b-hf"
  "meta-llama/Llama-2-70b-hf"
  "meta-llama/Meta-Llama-3-8B"
  "meta-llama/Meta-Llama-3-70B"
)
declare -A EXTRA_ARGS=(
  ["meta-llama/Meta-Llama-3-70B gptq"]="--a_clip_ratio 1.0"
)

if [[ $# -gt 0 ]]; then
  MODEL_LIST=("$@")
fi

for MODEL_NAME in "${MODEL_LIST[@]}"; do
  for WQ in rtn gptq; do
    # shellcheck disable=SC2086
    fake_quant --model "$MODEL_NAME" --save_name "signrot/$WQ" --preset signrot --w_quantizer "$WQ" \
      "${EVAL[@]}" ${EXTRA_ARGS["$MODEL_NAME $WQ"]:-}
  done
done
