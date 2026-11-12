#!/bin/bash
# Ablation table: calibration data of the sign adjustment vector (LLaMA3-8B, GPTQ).
# Usage: scripts/reproduce/ablation_calibration_data.sh

source "$(dirname "$0")/common.sh"

for DATASET in wikitext2 c4; do
  for CAL_SEED in 0 1 2; do
    fake_quant --model meta-llama/Meta-Llama-3-8B --save_name "ablation_calibration_data/$DATASET/seed$CAL_SEED" \
      --preset signrot --w_quantizer gptq --adj_cal_dataset "$DATASET" --adj_cal_dataset_seed "$CAL_SEED" "${EVAL[@]}"
  done
done
