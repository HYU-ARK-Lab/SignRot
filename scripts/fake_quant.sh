ARGS=""
PRESET=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --preset)
      PRESET="$2"
      shift 2
      ;;
    *)
      ARGS="$ARGS $1"
      shift
      ;;
  esac
done

# preset can be quarot, signrot
if [[ -n "$PRESET" ]]; then
  echo "Using preset: $PRESET"
  if [[ "$PRESET" == "quarot" ]]; then
    ARGS="--w_bits 4 --w_clip --a_bits 4 --k_bits 4 --k_asym --v_bits 4 --v_asym --a_clip_ratio 0.9 --k_clip_ratio 0.95 --v_clip_ratio 0.95 --k_groupsize head --v_groupsize head --rotate $ARGS"
  elif [[ "$PRESET" == "signrot" ]]; then
    ARGS="--w_bits 4 --w_clip --a_bits 4 --a_asym --k_bits 4 --k_asym --v_bits 4 --v_asym --a_clip_ratio 0.9 --k_clip_ratio 0.9 --v_clip_ratio 0.9 --k_groupsize head --v_groupsize head --rotate --adj_layers down_proj $ARGS"
  else
    echo "Unknown preset: $PRESET"
    exit 1
  fi
else
  echo "No preset specified, using default args"
fi


cd "$(dirname "$0")/../fake_quant"

python -u main.py $ARGS
