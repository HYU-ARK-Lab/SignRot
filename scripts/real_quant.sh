ARGS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    *)
      ARGS="$ARGS $1"
      shift
      ;;
  esac
done

cd "$(dirname "$0")/.."

python e2e/benchmark_realquant.py $ARGS
