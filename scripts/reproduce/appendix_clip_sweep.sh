#!/bin/bash
# Appendix, detailed quantization configuration: activation clipping ratio sweep (LLaMA3-70B, GPTQ).
# Usage: scripts/reproduce/appendix_clip_sweep.sh [MODEL]

source "$(dirname "$0")/common.sh"

MODEL_NAME="${1:-meta-llama/Meta-Llama-3-70B}"

fake_quant --model "$MODEL_NAME" --save_name clip_sweep --preset signrot --w_quantizer gptq --save_qmodel
for A_CLIP in 1.0 0.975 0.95 0.925; do
  fake_quant --model "$MODEL_NAME" --save_name clip_sweep --preset signrot --w_quantizer gptq --load_qmodel \
    --a_clip_ratio "$A_CLIP" "${EVAL[@]}"
done
