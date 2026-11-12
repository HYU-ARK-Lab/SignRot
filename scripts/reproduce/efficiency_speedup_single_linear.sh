#!/bin/bash
# Single linear layer latency, FP16 / Int4 / +QuaRot / +SignRot (table speedup_single_linear).
# Usage: [CUDA_VISIBLE_DEVICES=0] scripts/reproduce/efficiency_speedup_single_linear.sh [OUT_DIR] [speedup_single_linear.py args...]

source "$(dirname "$0")/common.sh"

OUT_DIR="${1:-$REPO_ROOT/fake_quant/experiments/reproduce/efficiency}"
shift || true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "+ speedup_single_linear.py --out_dir $OUT_DIR $*"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
python "$REPO_ROOT/scripts/reproduce/analysis/efficiency/speedup_single_linear.py" --out_dir "$OUT_DIR" "$@"
