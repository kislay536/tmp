#!/bin/bash
# Run (or resume) the 24-combo slamio sweep, skipping every combo that already has a row in the results CSV.
# Safe to run again after a crash / reboot / session end -- just start it again.
#
#   nohup setsid bash resume_slamio.sh > logs/resume_slamio_$(date +%Y%m%d_%H%M%S).out 2>&1 < /dev/null &
#   TOTAL_COMBOS=24 bash watch_slamio.sh        # progress
#
# A combo that was interrupted mid-run has no CSV row (rows are appended only when a run finishes),
# so it is simply run again from the start.
# Optional: CSV=/path/to/results.csv
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MLSYS_ROOT="$PWD"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU_NAME" in
  *H100*) GPU="H100" ;;
  *Thor*) GPU="Thor" ;;
  *5090*) GPU="RTX5090" ;;
  *) GPU=$(echo "${GPU_NAME#NVIDIA }" | tr ' ' '_') ;;
esac
if [[ "$GPU" == "H100" ]]; then DEFAULT_CSV="$MLSYS_ROOT/experiments/results.csv"
else DEFAULT_CSV="$MLSYS_ROOT/experiments/results_$(echo "$GPU" | tr '[:upper:]' '[:lower:]').csv"; fi
CSV="${CSV:-$DEFAULT_CSV}"

# one sweep at a time (same lock file launch.sh --all uses)
mkdir -p logs
exec 9>"$MLSYS_ROOT/logs/.launch_all.lock"
flock -n 9 || { echo "ERROR: another sweep is already running on this machine" >&2; exit 1; }
if pgrep -af 'run_slam\.py|scripts/splatam\.py|slam\.py --config' | grep -v pgrep >/dev/null 2>&1; then
  echo "ERROR: a SLAM run is already active:" >&2; pgrep -af 'run_slam\.py|scripts/splatam\.py|slam\.py --config' >&2; exit 1
fi

csv_model() { case "$1" in splatam) echo SplaTAM;; gaussianslam) echo Gaussian-SLAM;; monogs) echo MonoGS;; esac; }
csv_dataset() { case "$1" in fr1_desk) echo TUM;; room0) echo Replica;; scene0059_00) echo ScanNet;; esac; }
have_row() {  # model scene opt_type
  [[ -f "$CSV" ]] && awk -F, -v m="$1" -v s="$2" -v o="$3" 'NR>1 && $1==m && $3==s && $4==o { f=1 } END { exit !f }' "$CSV"
}

COMBOS=()
for m in splatam gaussianslam; do for s in fr1_desk room0 scene0059_00; do for md in baseline optimization; do COMBOS+=("$m|$s|$md|"); done; done; done
for s in fr1_desk room0 scene0059_00; do for md in baseline optimization; do for th in single multi; do COMBOS+=("monogs|$s|$md|$th"); done; done; done
TOTAL=${#COMBOS[@]}

echo "=== resume_slamio.sh starting at $(date) on $(hostname), $TOTAL combos, csv=$CSV ==="
n=0; ran=0; skipped=0; FAILED=()
for line in "${COMBOS[@]}"; do
  IFS='|' read -r m s md th <<< "$line"
  n=$((n+1))
  opt="slamio_${md}"; extra=()
  if [[ "$m" == "monogs" ]]; then opt="${opt}_${th}thread"; extra=(--thread "$th"); fi
  if have_row "$(csv_model "$m")" "$s" "$opt"; then
    echo "=== [$n/$TOTAL] slamio/$m/$s/$md${th:+/${th}thread}: already in CSV -- skipped ==="
    skipped=$((skipped+1)); continue
  fi
  echo "=== [$n/$TOTAL] slamio/$m/$s/$md${th:+/${th}thread} ==="
  # launch.sh takes its own --all lock only for --all, so running single combos under our lock is fine
  if ! bash launch.sh --model "$m" --optimization slamio --scene "$s" --mode "$md" "${extra[@]}" --csv "$CSV"; then
    FAILED+=("$m/$s/$md${th:+/$th}")
  fi
  ran=$((ran+1))
done
echo "=== resume_slamio.sh finished at $(date): ran $ran, skipped $skipped, failed ${#FAILED[@]} ==="
if [[ ${#FAILED[@]} -gt 0 ]]; then printf '  FAILED: %s\n' "${FAILED[@]}"; exit 1; fi
exit 0
