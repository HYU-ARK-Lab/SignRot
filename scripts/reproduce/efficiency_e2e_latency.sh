#!/bin/bash
# End-to-end latency and peak memory (tables e2e_latency and e2e_latency_full): bsz 1, 2048 prefill + 128 decode, one GPU.
# Usage: [CUDA_VISIBLE_DEVICES=0] [MODELS=...] [CONFIGS=...] [ROUNDS=2] scripts/reproduce/efficiency_e2e_latency.sh [OUT_DIR]

source "$(dirname "$0")/common.sh"

OUT_DIR="${1:-$REPO_ROOT/fake_quant/experiments/reproduce/efficiency}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
read -ra MODELS <<<"${MODELS:-meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf}"
read -ra CONFIGS <<<"${CONFIGS:-fp16 quarot_sym signrot_sym quarot_asym signrot_asym}"
ROUNDS="${ROUNDS:-2}"

BENCH=(--runtime_bench --bench_bsz 1 --bench_prefill_len 2048 --bench_decode_steps 128
  --bench_repeats "${BENCH_REPEATS:-5}" --bench_warmup 1 --bench_soak_s "${BENCH_SOAK_S:-90}"
  --bench_log_clocks --bench_skip_oom --seed 0)
# W4A4KV4 with activation/KV clip 0.9 for every quantized config; RTN weights.
QUANT=(--preset signrot --w_quantizer rtn --use_kernel)
declare -A CFG_ARGS=(
  [fp16]=""
  [quarot_sym]="${QUANT[*]} --no-a_asym --adj_layers"
  [signrot_sym]="${QUANT[*]} --no-a_asym --adj_cal_dataset wikitext2"
  [quarot_asym]="${QUANT[*]} --adj_layers"
  [signrot_asym]="${QUANT[*]} --adj_cal_dataset wikitext2"
)

for model in "${MODELS[@]}"; do
  for ((r = 1; r <= ROUNDS; r++)); do
    for cfg in "${CONFIGS[@]}"; do
      run="reproduce/efficiency/e2e/$cfg/r$r" stdout=/dev/null
      if [[ "${DRY_RUN:-0}" != 1 ]]; then
        stdout="$REPO_ROOT/fake_quant/experiments/$model/$run/stdout.txt"
        mkdir -p "$(dirname "$stdout")"
      fi
      # shellcheck disable=SC2086
      fake_quant --model "$model" --save_name "$run" ${CFG_ARGS[$cfg]} "${BENCH[@]}" 2>&1 | tee "$stdout" ||
        echo "[failed] $model $cfg r$r" >&2
    done
  done
done

echo "+ e2e_latency.py --out_dir $OUT_DIR"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
python "$REPO_ROOT/scripts/reproduce/analysis/efficiency/e2e_latency.py" --out_dir "$OUT_DIR"
