#!/bin/bash
# Appendix tables ma_statistics, ma_statistics_large, early_ma_layers and activation_distribution_statistics.
# Usage: scripts/reproduce/ma_statistics.sh [--out_dir DIR] [MODEL ...]   (default: all seven models)

source "$(dirname "$0")/common.sh"

PY_DIR="$REPO_ROOT/scripts/reproduce/analysis/ma_statistics"
OUT_DIR="$REPO_ROOT/fake_quant/experiments/reproduce/ma_statistics"
if [[ "${1:-}" == --out_dir ]]; then
  OUT_DIR="$2"
  shift 2
fi
MODELS=("$@")
[[ ${#MODELS[@]} -gt 0 ]] || MODELS=(
  meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf meta-llama/Llama-2-70b-hf
  meta-llama/Meta-Llama-3-8B meta-llama/Meta-Llama-3-70B
  mistralai/Mistral-7B-v0.3 Qwen/Qwen3-30B-A3B
)

run() {
  echo "+ $*"
  [[ "${DRY_RUN:-0}" == 1 ]] || "$@"
}

captures=() records=()
for model in "${MODELS[@]}"; do
  raw="$REPO_ROOT/fake_quant/experiments/$model/ma_statistics"
  topk="$raw/topk_wikitext2_n128.pt"
  bound="$raw/range_bound_wikitext2_n128.npz"
  extra=()
  [[ "$model" == *70b* || "$model" == *70B* ]] && extra=(--distribute)
  [[ -f "$topk" ]] || run python "$PY_DIR/capture_topk.py" --model "$model" --output "$topk" "${extra[@]}"
  [[ -f "$bound" ]] || run python "$PY_DIR/range_bound.py" --model "$model" --capture "$topk" --out "$bound" "${extra[@]}"
  captures+=(--input "$topk")
  records+=(--input "$bound")
done

run python "$PY_DIR/ma_statistics_table.py" "${captures[@]}" --out "$OUT_DIR/ma_statistics.csv"
run python "$PY_DIR/early_ma_layers_table.py" "${records[@]}" --out "$OUT_DIR/early_ma_layers.csv"
if [[ " ${MODELS[*]} " == *" meta-llama/Llama-2-7b-hf "* ]]; then
  run python "$PY_DIR/activation_distribution_statistics_table.py" --model meta-llama/Llama-2-7b-hf --layer 2 \
    --out "$OUT_DIR/activation_distribution_statistics.csv"
fi
