#!/bin/bash
# Appendix single-layer quantization KL: Figure "single_layer_kl" and Table "single_layer_kl" (GPTQ W4A4, QuaRot vs QuaRot+SignRot).
# Usage: scripts/reproduce/single_layer_kl.sh [MODEL ...]   (run scripts/reproduce/ma_statistics.sh first: gray bands / table rows)

source "$(dirname "$0")/common.sh"

MODEL_LIST=(
  "meta-llama/Llama-2-7b-hf"
  "meta-llama/Llama-2-13b-hf"
  "meta-llama/Meta-Llama-3-8B"
  "meta-llama/Llama-2-70b-hf"
  "meta-llama/Meta-Llama-3-70B"
)
if [[ $# -gt 0 ]]; then
  MODEL_LIST=("$@")
fi
ANALYSIS=scripts/reproduce/analysis/single_layer_kl

run() {
  echo "+ $*"
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    (cd "$REPO_ROOT" && "$@")
  fi
}

for MODEL_NAME in "${MODEL_LIST[@]}"; do
  LAYERS=all
  [[ "$MODEL_NAME" == *70[bB]* ]] && LAYERS="0-7,8:80:4,79"
  for METHOD in quarot signrot; do
    if [[ "$METHOD" == signrot ]]; then
      ADJ=(--adj_layers down_proj)
      CAL=(--adj_cal_dataset wikitext2)
    else
      ADJ=(--adj_layers)
      CAL=()
    fi
    RUN_DIR="$REPO_ROOT/fake_quant/experiments/$MODEL_NAME/single_layer_kl/$METHOD"
    if [[ ! -f "$RUN_DIR/qmodel/config.json" ]]; then
      fake_quant --model "$MODEL_NAME" --save_name "single_layer_kl/$METHOD" --preset signrot "${ADJ[@]}" "${CAL[@]}" \
        --w_quantizer gptq --seed 0 --save_qmodel
    fi
    if [[ ! -f "$RUN_DIR/single_layer_kl.npz" ]]; then
      run python -u "$ANALYSIS/measure.py" --nseq 8 --layers "$LAYERS" \
        --model "$MODEL_NAME" --save_name "single_layer_kl/$METHOD" --load_qmodel --rotate \
        --a_bits 4 --a_asym --a_clip_ratio 0.9 "${ADJ[@]}"
    fi
  done
done

run python "$ANALYSIS/figure_table.py" --models "${MODEL_LIST[@]}"
