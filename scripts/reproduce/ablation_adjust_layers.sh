#!/bin/bash
# Ablation table: sign adjustment at other projection inputs (LLaMA3-8B, GPTQ, seeds 0-5).
# Usage: scripts/reproduce/ablation_adjust_layers.sh

source "$(dirname "$0")/common.sh"

declare -A LAYERS=(
  [qkv]="q_proj k_proj v_proj"
  [o]="o_proj"
  [ug]="up_proj gate_proj"
  [d]="down_proj"
  [qkvo]="q_proj k_proj v_proj o_proj"
  [ugd]="up_proj gate_proj down_proj"
  [od]="o_proj down_proj"
  [all]="q_proj k_proj v_proj o_proj up_proj gate_proj down_proj"
)

for SEED in 0 1 2 3 4 5; do
  for CFG in qkv o ug d qkvo ugd od all; do
    read -ra ADJ <<<"${LAYERS[$CFG]}"
    fake_quant --model meta-llama/Meta-Llama-3-8B --save_name "ablation_adjust_layers/$CFG/seed$SEED" \
      --preset signrot --w_quantizer gptq --seed "$SEED" \
      --adj_layers "${ADJ[@]}" --adj_cal_dataset wikitext2 "${EVAL[@]}"
  done
done
