#!/bin/bash
# Live progress watcher for the slamio sweep (24 combos):
#   SplaTAM + Gaussian-SLAM : 3 scenes x {baseline, optimization}            = 12
#   MonoGS                  : 3 scenes x {baseline, optimization} x {single, multi} = 12
# Start the sweep with:
#   nohup bash launch.sh --all --only-opt slamio > logs/launch_all_slamio_$(date +%Y%m%d_%H%M%S).out 2>&1 &
#
# For every expected combo it shows done (row in the results CSV) / RUNNING / FAILED / pending,
# the metrics of the finished ones, and the progress of the run that is active right now.
#
# Usage: bash watch_slamio.sh [--once] [csv_path] [refresh_seconds]
#   --once           print one snapshot and exit
#   csv_path         default: same auto-detection as launch.sh (results_thor.csv on Thor)
#   refresh_seconds  default: 30
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MLSYS_ROOT="$PWD"

ONCE=0
[[ "${1:-}" == "--once" ]] && { ONCE=1; shift; }

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU_NAME" in
  *H100*) GPU="H100" ;;
  *Thor*) GPU="Thor" ;;
  *5090*) GPU="RTX5090" ;;
  *) GPU=$(echo "${GPU_NAME#NVIDIA }" | tr ' ' '_') ;;
esac
if [[ "$GPU" == "H100" ]]; then
  DEFAULT_CSV="$MLSYS_ROOT/experiments/results.csv"
else
  DEFAULT_CSV="$MLSYS_ROOT/experiments/results_$(echo "$GPU" | tr '[:upper:]' '[:lower:]').csv"
fi
CSV="${1:-$DEFAULT_CSV}"
REFRESH="${2:-30}"

fmt_secs() { printf '%dh%02dm' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )); }

# model-key | CSV model name | scenes' dataset
COMBOS=()
for m in splatam gaussianslam; do
  for s in fr1_desk room0 scene0059_00; do
    for md in baseline optimization; do COMBOS+=("$m|$s|$md|"); done
  done
done
for s in fr1_desk room0 scene0059_00; do
  for md in baseline optimization; do
    for th in single multi; do COMBOS+=("monogs|$s|$md|$th"); done
  done
done
TOTAL=${#COMBOS[@]}

csv_model() { case "$1" in splatam) echo SplaTAM;; gaussianslam) echo Gaussian-SLAM;; monogs) echo MonoGS;; esac; }
csv_dataset() { case "$1" in fr1_desk) echo TUM;; room0) echo Replica;; scene0059_00) echo ScanNet;; esac; }

# Echo "<total_s>,<track_ms_f>,<map_ms_f>,<ate_cm>,<psnr>" for a CSV row, or nothing.
csv_row() {
  local model="$1" scene="$2" opt="$3"
  [[ -f "$CSV" ]] || return 0
  awk -F, -v m="$model" -v s="$scene" -v o="$opt" '
    NR>1 && $1==m && $3==s && $4==o { r=$7","$8","$10","$12","$13 } END { if (r!="") print r }' "$CSV"
}

snapshot() {
  echo "=== slamio sweep status on $(hostname) [$GPU] ($(date)) ==="
  echo

  # --- sweep process / console log ---
  local OUT ALIVE="NOT running"
  OUT=$(ls -t "$MLSYS_ROOT"/logs/launch_all_slamio_*.out 2>/dev/null | head -1)
  pgrep -f "launch.sh --all --only-opt slamio" >/dev/null && ALIVE="running"
  echo "--- sweep: $ALIVE${OUT:+ -- $OUT} ---"
  if [[ -n "$OUT" ]]; then
    grep -m1 "starting at" "$OUT"
    grep -E "^=== \[[0-9]+/[0-9]+\]" "$OUT" | tail -1 | sed 's/^/current: /'
    grep -E "finished at .*succeeded" "$OUT" | tail -1
  fi
  echo

  # --- per-combo table ---
  local done_n=0 run_n=0 fail_n=0 line m s md th model dataset opt suffix log status row
  local rows=""
  for line in "${COMBOS[@]}"; do
    IFS='|' read -r m s md th <<< "$line"
    model=$(csv_model "$m"); dataset=$(csv_dataset "$s")
    opt="slamio_${md}"; suffix="$md"
    if [[ "$m" == "monogs" ]]; then opt="${opt}_${th}thread"; suffix="${md}_${th}thread"; fi
    log="$MLSYS_ROOT/logs/slamio-${m}/${s}_${suffix}.log"
    row=$(csv_row "$model" "$s" "$opt")
    if [[ -n "$row" ]]; then
      status="done"; done_n=$((done_n+1))
    elif [[ -f "$log" ]] && grep -q "MLSYS_START_EPOCH" "$log" && ! grep -q "MLSYS_END_EPOCH" "$log"; then
      status="RUNNING"; run_n=$((run_n+1)); row=",,,,"
    elif [[ -f "$log" ]] && grep -q "MLSYS_END_EPOCH" "$log"; then
      status="FAILED"; fail_n=$((fail_n+1)); row=",,,,"
    else
      status="pending"; row=",,,,"
    fi
    IFS=, read -r tt tr mp ate psnr <<< "$row"
    rows+="$status"$'\t'"$model"$'\t'"$dataset"$'\t'"$s"$'\t'"$opt"$'\t'"${tt:--}"$'\t'"${tr:--}"$'\t'"${mp:--}"$'\t'"${ate:--}"$'\t'"${psnr:--}"$'\n'
  done
  echo "=== completed: $done_n/$TOTAL  (running $run_n, failed $fail_n, pending $((TOTAL-done_n-run_n-fail_n))) -- $CSV ==="
  {
    printf 'status\tmodel\tdataset\tscene\topt_type\ttotal_s\ttrack_ms/f\tmap_ms/f\tate_cm\tpsnr_db\n'
    printf '%s' "$rows" | awk -F'\t' '{ for (i=6;i<=10;i++) if ($i ~ /^[0-9.eE+-]+$/ && $i+0 == $i) $i = sprintf("%.2f", $i); print }' OFS='\t'
  } | column -t -s "$(printf '\t')"
  echo

  # --- current / most recent run ---
  local LOG
  LOG=$(find "$MLSYS_ROOT"/logs/slamio-* -maxdepth 1 -name "*.log" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d" " -f2-)
  if [[ -z "$LOG" ]]; then
    echo "(no slamio per-combo log yet)"
  else
    echo "--- currently running / most recently active ---"
    echo "Active log: $LOG"
    grep -m1 "cmd:" "$LOG" 2>/dev/null | cut -c1-200
    local START END
    START=$(grep -m1 -oE "MLSYS_START_EPOCH=[0-9]+" "$LOG" | cut -d= -f2)
    END=$(grep -m1 -oE "MLSYS_END_EPOCH=[0-9]+" "$LOG" | cut -d= -f2)
    if [[ -n "$START" ]]; then
      if [[ -n "$END" ]]; then echo "Finished after $(fmt_secs $(( END - START )))"
      else echo "Elapsed: $(fmt_secs $(( $(date +%s) - START )))"; fi
    fi
    local TAIL STEP FRAME ATE
    TAIL=$(tail -c 12000 "$LOG" | tr '\r' '\n')
    # SplaTAM "Tracking Time Step: N"; Gaussian-SLAM "Tracking frame N" / "Mapping frame N";
    # MonoGS prints tqdm-style progress and "Keyframes" lines.
    STEP=$(grep -oE "(Tracking|Mapping) (Time Step:|frame) [0-9]+" <<< "$TAIL" | tail -1)
    FRAME=$(grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, *[0-9.]+(s/it|it/s)\]" <<< "$TAIL" | tail -1)
    [[ -n "$STEP" ]] && echo "Frame: $STEP"
    echo "Progress: ${FRAME:-(not available)}"
    grep -E "Iterations per frame|Tracking iterations/frame" <<< "$TAIL" | tail -1 | cut -c1-200
    ATE=$(grep -oE "(ATE RMSE=[0-9.eE+-]+|Eval: RMSE ATE \[m\][0-9. eE+-]+|ATE-RMSE: [0-9.]+ cm)" "$LOG" | tail -1)
    echo "ATE (last eval): ${ATE:-(not yet available)}"
  fi
  echo

  # --- machine ---
  echo "--- machine ---"
  nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null \
    | awk -F', ' '{ printf "GPU util %s, temp %sC, power %s\n", $1, $2, $3 }'
  free -h | awk '/^Mem:/ { printf "RAM used %s / %s\n", $3, $2 }'
}

if [[ $ONCE -eq 1 ]]; then
  snapshot
  exit 0
fi
while true; do
  clear
  echo "(refresh every ${REFRESH}s, Ctrl+C to stop)"
  snapshot
  sleep "$REFRESH"
done
