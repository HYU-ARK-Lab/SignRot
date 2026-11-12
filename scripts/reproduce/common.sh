# Shared settings for scripts/reproduce. Source this file.
# CUDA_VISIBLE_DEVICES selects the GPUs, LM_EVAL_BS the lm_eval batch size, DRY_RUN=1 prints the commands only.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LM_EVAL_BS="${LM_EVAL_BS:-32}"
EVAL=(--ppl_eval --lm_eval --lm_eval_batch_size "$LM_EVAL_BS")

fake_quant() {
  echo "+ scripts/fake_quant.sh $*"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    return 0
  fi
  "$REPO_ROOT/scripts/fake_quant.sh" "$@"
}
