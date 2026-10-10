#!/bin/bash
# Single unified launcher for all 48 combos: 3 models (splatam, gaussianslam,
# monogs) x 2 optimizations (splatonic, rtgs) x 3 scenes (fr1_desk, room0,
# scene0059_00) x 2 modes (baseline, optimization), with monogs additionally
# split by thread mode (single, multi) -- splatam/gaussianslam have no
# threading concept, so they stay at 2 modes x 3 scenes x 2 opts = 12 combos
# each; monogs (splatonic + rtgs) is 2 modes x 3 scenes x 2 opts x 2 threads
# = 24. Total: 12 + 12 + 24 = 48.
#
# Usage (one combo):
#   bash launch.sh --model {splatam|gaussianslam|monogs} \
#                   --optimization {splatonic|rtgs} \
#                   --scene {fr1_desk|room0|scene0059_00} \
#                   --mode {baseline|optimization} \
#                   [--thread {single|multi}] \
#                   [--csv PATH] [--gpu-label LABEL] [--compute-location LOC]
#
# --thread only matters for --model monogs (ignored otherwise). Default:
# multi, if omitted.
#
# Usage (all 48, sequential on one GPU):
#   bash launch.sh --all [--csv PATH] [--gpu-label LABEL] [--compute-location LOC]
#
# No idempotency/skip logic -- every invocation runs and appends a fresh
# row. Baseline is intentionally run once per (model, optimization) repo
# pair (e.g. splatonic-splatam's baseline AND rtgs-splatam's baseline both
# run, both get their own row) rather than deduplicated across repos, so
# the two baselines can be cross-checked against each other.
#
# Machine defaults are auto-detected (each overridable by its flag):
#   --gpu-label         from nvidia-smi: H100, Thor, RTX5090, else the raw name
#   --compute-location  "cluster" under SLURM, "local" otherwise
#   --csv               experiments/results.csv on H100 (the original
#                       cluster results), experiments/results_<gpu>.csv on
#                       any other GPU (e.g. results_thor.csv on Jetson Thor),
#                       so different machines never mix rows in one file.
#
# Each slamio repo runs from its own venv at slamio-<model>/<Fork>/.venv, built by
# setup_env.sh. This branch only contains the slamio variant:
#   bash launch.sh --all --only-opt slamio        # the 24 slamio combos
# (or use resume_slamio.sh, which skips combos already in the results CSV).
# FAISS_GPU (default 1) selects Gaussian-SLAM's neighbour search: 1 = faiss-gpu,
# 0 = pure-torch fallback (what the Jetson Thor runs used, since it has no faiss-gpu wheel).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MLSYS_ROOT="$PWD"

MODEL=""; OPT=""; SCENE=""; MODE=""; THREAD="multi"; RUN_ALL=0; RUN_TAG=""; ONLY_OPT=""
CSV=""; GPU=""; COMPUTE_LOCATION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) RUN_ALL=1; shift;;
    --model) MODEL="$2"; shift 2;;
    --optimization) OPT="$2"; shift 2;;
    --scene) SCENE="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --thread) THREAD="$2"; shift 2;;
    --run-tag) RUN_TAG="$2"; shift 2;;
    --only-opt) ONLY_OPT="$2"; shift 2;;
    --csv) CSV="$2"; shift 2;;
    --gpu-label) GPU="$2"; shift 2;;
    --compute-location) COMPUTE_LOCATION="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

usage() {
  echo "Usage: $0 --model {splatam|gaussianslam|monogs} --optimization {splatonic|rtgs|slamio} --scene {fr1_desk|room0|scene0059_00} --mode {baseline|optimization} [--thread {single|multi}] [--csv PATH] [--gpu-label LABEL] [--compute-location LOC]" >&2
  echo "   or: $0 --all [--only-opt {splatonic|rtgs|slamio}] [--csv PATH] [--gpu-label LABEL] [--compute-location LOC]" >&2
  exit 2
}

if [[ -z "$GPU" ]]; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  case "$GPU_NAME" in
    *H100*) GPU="H100" ;;
    *Thor*) GPU="Thor" ;;
    *5090*) GPU="RTX5090" ;;
    "") echo "ERROR: no GPU found via nvidia-smi; pass --gpu-label explicitly" >&2; exit 2 ;;
    *) GPU=$(echo "${GPU_NAME#NVIDIA }" | tr ' ' '_') ;;
  esac
fi
[[ -n "$COMPUTE_LOCATION" ]] || COMPUTE_LOCATION=$([[ -n "${SLURM_JOB_ID:-}" ]] && echo cluster || echo local)
if [[ -z "$CSV" ]]; then
  if [[ "$GPU" == "H100" ]]; then
    CSV="$MLSYS_ROOT/experiments/results.csv"
  else
    CSV="$MLSYS_ROOT/experiments/results_$(echo "$GPU" | tr '[:upper:]' '[:lower:]').csv"
  fi
fi

export WANDB_MODE=offline
# Must be unique per concurrent invocation, not shared -- matplotlib's TeX
# text-rendering cache (used by splatonic-gaussianslam's logger) writes
# and reads PNG glyph files under $MPLCONFIGDIR/tex.cache/ with no locking
# of its own. Under --all/array-job concurrency, two processes racing on
# the same shared cache dir produced a real FileNotFoundError (one process
# read a glyph file mid-write/mid-eviction by another). Scoped per SLURM
# array task (or per-PID outside SLURM) so no two concurrent invocations
# ever share a cache directory.
export MPLCONFIGDIR="$MLSYS_ROOT/.mplcache/${SLURM_JOB_ID:-$$}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR" "$MLSYS_ROOT/logs" "$(dirname "$CSV")"

CSV_HEADER="model	dataset	scene	optimization_type	gpu	compute_location	total_time_s	tracking_ms_per_frame	tracking_ms_per_iter	mapping_ms_per_frame	mapping_ms_per_iter	ate_cm	psnr_db	tracking_iters_per_frame	mapping_iters_per_frame"

append_row() {
  # $1=model_csv $2=dataset $3=scene $4=opt_type $5..=metric assoc array name
  # NOTE: the nameref below must NOT be named the same as the caller's array
  # (run_one's "M") -- a same-named nameref is a *circular* reference, which
  # bash only resolves reliably one call-frame deep. Through the real
  # run_one() -> append_row() chain (two frames) it silently fails to
  # dereference and every "${M[...]}" throws "unbound variable" under
  # set -u, producing a blank appended row. Distinct name avoids this
  # entirely.
  local model_csv="$1" dataset="$2" scene="$3" opt_type="$4"
  local -n metrics_ref="$5"
  local row
  row=$(printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' \
    "$model_csv" "$dataset" "$scene" "$opt_type" "$GPU" "$COMPUTE_LOCATION" \
    "${metrics_ref[total_time_s]:-}" "${metrics_ref[tracking_ms_per_frame]:-}" "${metrics_ref[tracking_ms_per_iter]:-}" \
    "${metrics_ref[mapping_ms_per_frame]:-}" "${metrics_ref[mapping_ms_per_iter]:-}" \
    "${metrics_ref[ate_cm]:-}" "${metrics_ref[psnr_db]:-}" \
    "${metrics_ref[tracking_iters_per_frame]:-}" "${metrics_ref[mapping_iters_per_frame]:-}")
  (
    flock -x 200
    if [[ ! -f "$CSV" ]]; then
      echo "model,dataset,scene,optimization_type,gpu,compute_location,total_time_s,tracking_ms_per_frame,tracking_ms_per_iter,mapping_ms_per_frame,mapping_ms_per_iter,ate_cm,psnr_db,tracking_iters_per_frame,mapping_iters_per_frame" > "$CSV"
    fi
    echo "$row" >> "$CSV"
  ) 200>"${CSV}.lock"
  echo "=== row appended to $CSV ==="
}

# ---------------------------------------------------------------------
# Per-combo config resolution: sets REPO, MODEL_CSV, DATASET, SCRIPT,
# CONFIG, EXTRA (extra CLI args), EXTRA_ENV (extra env KEY=VAL),
# EXTRACTOR_KIND (splatam|gaussianslam|monogs), and OPT_TYPE_LABEL (the
# CSV optimization_type value before any thread-mode suffix -- "baseline"
# everywhere, "optimization" for splatam/gaussianslam, "sparse_tracking"
# for splatonic-monogs, "downsample" for rtgs-monogs).
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# slamio: the user's own optimization stack (3DGS-framework @ slamio_final),
# copied into slamio-<model>/<Fork>/ (+ sibling utils/ and submodules/, which
# the forks import via sys.path). Unlike splatonic/rtgs it is configured
# mostly through environment variables:
#   optimization -> slamio-<model>/cells/<scene>_optimization.{env,cfg}, captured
#                   from the framework's own best_configs.sh / presets
#                   (splatam fr1_desk + scene0059_00 use configs/*/splatam_final.py,
#                   which sets its env inside the config)
#   baseline     -> the stock scene config of the same fork with NO slamio
#                   switch set (Adam, fixed iteration budget, no graph / reuse /
#                   early stop / adaptive mapping), i.e. the slamio code with every
#                   optimization off
# monogs single-thread = the <config>_singlethread_on.yaml variant
# (Dataset.single_thread + Training.single_thread, as in the other monogs repos).
# ---------------------------------------------------------------------
resolve_slamio() {
  local model="$1" scene="$2" mode="$3" thread="$4"
  local fork cells line
  case "$scene" in
    fr1_desk) DATASET="TUM" ;;
    room0) DATASET="Replica" ;;
    scene0059_00) DATASET="ScanNet" ;;
  esac
  case "$thread" in single|multi) : ;; *) echo "bad --thread: $thread" >&2; usage ;; esac
  EXTRA_ENV=(WANDB_MODE=disabled DISABLE_WANDB=true FAISS_GPU="${FAISS_GPU:-1}")
  case "$model" in
    splatam)      MODEL_CSV="SplaTAM";       EXTRACTOR_KIND="splatam";      fork="SplaTAM";       SCRIPT="scripts/splatam.py" ;;
    gaussianslam) MODEL_CSV="Gaussian-SLAM"; EXTRACTOR_KIND="gaussianslam"; fork="Gaussian-SLAM"; SCRIPT="run_slam.py"
                  # reconstruction / global-map eval are never read by the extractor (rendering + ATE still run)
                  EXTRA_ENV+=(GSLAM_SKIP_RECON_EVAL=1 GSLAM_SKIP_GLOBAL_MAP_EVAL=1) ;;
    monogs)       MODEL_CSV="MonoGS";        EXTRACTOR_KIND="monogs";       fork="MonoGS";        SCRIPT="slam.py" ;;
    *) echo "bad --model: $model" >&2; usage ;;
  esac
  REPO="$MLSYS_ROOT/slamio-${model}/${fork}"
  cells="$MLSYS_ROOT/slamio-${model}/cells"

  if [[ "$mode" == "optimization" ]]; then
    if [[ "$model" == "splatam" && "$scene" != "room0" ]]; then
      # the preset carries its own environment (_FINAL_ENV inside the file)
      case "$scene" in
        fr1_desk) CONFIG="configs/tum/splatam_final.py" ;;
        scene0059_00) CONFIG="configs/scannet/splatam_final.py" ;;
      esac
    else
      [[ -f "$cells/${scene}_optimization.cfg" ]] || { echo "missing $cells/${scene}_optimization.cfg" >&2; exit 1; }
      CONFIG="$(<"$cells/${scene}_optimization.cfg")"
      while IFS= read -r line; do [[ -n "$line" ]] && EXTRA_ENV+=("$line"); done < "$cells/${scene}_optimization.env"
    fi
  else
    case "$model:$scene" in
      splatam:fr1_desk)          CONFIG="configs/tum/splatam.py" ;;
      splatam:room0)             CONFIG="configs/replica/splatam_baseline_room0.py" ;;
      splatam:scene0059_00)      CONFIG="configs/scannet/splatam.py"; EXTRA_ENV+=(SCENE_NUM=1) ;;
      gaussianslam:fr1_desk)     CONFIG="configs/TUM_RGBD/rgbd_dataset_freiburg1_desk.yaml";  EXTRA_ENV+=(GSLAM_SKIP_VIS=1) ;;
      gaussianslam:room0)        CONFIG="configs/Replica/room0.yaml";                         EXTRA_ENV+=(GSLAM_SKIP_VIS=1) ;;
      gaussianslam:scene0059_00) CONFIG="configs/ScanNet/scene0059_00.yaml";                  EXTRA_ENV+=(GSLAM_SKIP_VIS=1) ;;
      monogs:fr1_desk)           CONFIG="configs/rgbd/tum/fr1_desk.yaml" ;;
      monogs:room0)              CONFIG="configs/rgbd/replica/room0.yaml" ;;
      monogs:scene0059_00)       CONFIG="configs/rgbd/scannet/scene0059_00_async_full.yaml" ;;
    esac
  fi
  # monogs: single-thread variant of whichever config was chosen
  if [[ "$model" == "monogs" && "$thread" == "single" ]]; then
    CONFIG="${CONFIG%.yaml}_singlethread_on.yaml"
  fi
  OPT_TYPE_LABEL="$mode"
  # Optional last-word overrides, e.g. SLAMIO_EXTRA_ENV="FRAMES=30 MONOGS_FRAMES=30 SPLATAM_FINAL_FRAMES=30"
  # for a smoke test (never set for a real sweep).
  if [[ -n "${SLAMIO_EXTRA_ENV:-}" ]]; then
    local _x; read -r -a _x <<< "$SLAMIO_EXTRA_ENV"; EXTRA_ENV+=("${_x[@]}")
  fi
}

resolve() {
  local model="$1" opt="$2" scene="$3" mode="$4" thread="${5:-multi}"
  EXTRA=(); EXTRA_ENV=()
  OPT_TYPE_LABEL="$mode"
  case "$scene" in
    fr1_desk) : ;;
    room0) : ;;
    scene0059_00) : ;;
    *) echo "bad --scene: $scene" >&2; usage ;;
  esac

  if [[ "$opt" == "slamio" ]]; then
    resolve_slamio "$model" "$scene" "$mode" "$thread"
    [[ -n "${CONFIG:-}" ]] || { echo "internal error: no config resolved for $model/$opt/$scene/$mode" >&2; exit 1; }
    return 0
  fi

  case "$model" in
    splatam)
      MODEL_CSV="SplaTAM"; EXTRACTOR_KIND="splatam"
      REPO="$MLSYS_ROOT/${opt}-splatam"
      case "$scene" in
        fr1_desk) DATASET="TUM" ;;
        room0) DATASET="Replica" ;;
        scene0059_00) DATASET="ScanNet" ;;
      esac
      SCRIPT="scripts/splatam.py"
      if [[ "$opt" == "splatonic" ]]; then
        [[ "$mode" == "optimization" ]] && SCRIPT="scripts/splatam_sparse.py"
        case "$scene:$mode" in
          fr1_desk:baseline)        CONFIG="configs/tum/splatam_baseline_freiburg1_desk.py" ;;
          fr1_desk:optimization)    CONFIG="configs/tum/splatam_sparse_final.py" ;;
          room0:baseline)           CONFIG="configs/replica/splatam_baseline_room0.py" ;;
          room0:optimization)       CONFIG="configs/replica/splatam_sparse_room0.py" ;;
          scene0059_00:baseline)     CONFIG="configs/scannet/splatam_baseline_scene0059_00.py" ;;
          scene0059_00:optimization) CONFIG="configs/scannet/splatam_sparse_scene0059_00.py" ;;
        esac
      elif [[ "$opt" == "rtgs" ]]; then
        case "$scene:$mode" in
          fr1_desk:baseline)        CONFIG="configs/tum/splatam_baseline_full.py" ;;
          fr1_desk:optimization)    CONFIG="configs/tum/splatam_rtgs_full.py" ;;
          room0:baseline)           CONFIG="configs/replica/splatam_baseline_room0_500.py" ;;
          room0:optimization)       CONFIG="configs/replica/splatam_rtgs.py" ;;
          scene0059_00:baseline)     CONFIG="configs/scannet/splatam_baseline_scene0059_00_500.py" ;;
          scene0059_00:optimization) CONFIG="configs/scannet/splatam_rtgs_scene0059_00.py" ;;
        esac
      else
        echo "bad --optimization: $opt" >&2; usage
      fi
      ;;

    gaussianslam)
      MODEL_CSV="Gaussian-SLAM"; EXTRACTOR_KIND="gaussianslam"
      REPO="$MLSYS_ROOT/${opt}-gaussianslam"
      SCRIPT="run_slam.py"
      case "$scene" in
        fr1_desk) DATASET="TUM"; BASE_CONFIG="configs/TUM_RGBD/rgbd_dataset_freiburg1_desk.yaml" ;;
        room0) DATASET="Replica"; BASE_CONFIG="configs/Replica/room0.yaml" ;;
        scene0059_00) DATASET="ScanNet"; BASE_CONFIG="configs/ScanNet/scene0059_00.yaml" ;;
      esac
      if [[ "$mode" == "baseline" ]]; then
        CONFIG="$BASE_CONFIG"
      elif [[ "$opt" == "splatonic" ]]; then
        CONFIG="$BASE_CONFIG"
        EXTRA=(--track_use_sparse_sampling --track_use_sparse_rasterizer --map_use_sparse_sampling --map_use_sparse_rasterizer)
        EXTRA_ENV=(SPLATONIC_MEASURE_TRACK_ITER_TIME=1)
      else  # rtgs optimization
        case "$scene" in
          fr1_desk) CONFIG="configs/TUM_RGBD/rgbd_dataset_freiburg1_desk_rtgs_fixed.yaml" ;;
          room0) CONFIG="configs/Replica/room0_rtgs.yaml" ;;
          scene0059_00) CONFIG="configs/ScanNet/scene0059_00_rtgs.yaml" ;;
        esac
      fi
      ;;

    monogs)
      MODEL_CSV="MonoGS"; EXTRACTOR_KIND="monogs"
      REPO="$MLSYS_ROOT/${opt}-monogs"
      SCRIPT="slam.py"
      case "$scene" in
        fr1_desk) DATASET="TUM" ;;
        room0) DATASET="Replica" ;;
        scene0059_00) DATASET="ScanNet" ;;
      esac
      case "$thread" in
        single|multi) : ;;
        *) echo "bad --thread: $thread" >&2; usage ;;
      esac
      if [[ "$opt" == "splatonic" ]]; then
        # "optimization" = tracking-only sparse kernel (NOT the combined
        # tracking+mapping kernel) -- the single representative for this
        # repo per the current experiment design.
        OPT_TYPE_LABEL="baseline"; [[ "$mode" == "optimization" ]] && OPT_TYPE_LABEL="sparse_tracking"
        case "$scene:$mode:$thread" in
          fr1_desk:baseline:multi)        CONFIG="configs/rgbd/tum/fr1_desk.yaml" ;;
          fr1_desk:baseline:single)       CONFIG="configs/rgbd/tum/fr1_desk_singlethread_on.yaml" ;;
          fr1_desk:optimization:multi)    CONFIG="configs/rgbd/tum/fr1_desk_sparse_gaussian_kernel.yaml" ;;
          fr1_desk:optimization:single)   CONFIG="configs/rgbd/tum/fr1_desk_sparse_gaussian_kernel_singlethread_on.yaml" ;;
          room0:baseline:multi)           CONFIG="configs/rgbd/replica/room0_async.yaml" ;;
          room0:baseline:single)          CONFIG="configs/rgbd/replica/room0_sp.yaml" ;;
          room0:optimization:multi)       CONFIG="configs/rgbd/replica/room0_async_sparse_gaussian_kernel.yaml" ;;
          room0:optimization:single)      CONFIG="configs/rgbd/replica/room0_sparse_gaussian_kernel_singlethread_on.yaml" ;;
          scene0059_00:baseline:multi)      CONFIG="configs/rgbd/scannet/scene0059_00.yaml" ;;
          scene0059_00:baseline:single)     CONFIG="configs/rgbd/scannet/scene0059_00_singlethread_on.yaml" ;;
          scene0059_00:optimization:multi)  CONFIG="configs/rgbd/scannet/scene0059_00_sparse_gaussian_kernel.yaml" ;;
          scene0059_00:optimization:single) CONFIG="configs/rgbd/scannet/scene0059_00_sparse_gaussian_kernel_singlethread_on.yaml" ;;
        esac
      elif [[ "$opt" == "rtgs" ]]; then
        OPT_TYPE_LABEL="baseline"; [[ "$mode" == "optimization" ]] && OPT_TYPE_LABEL="downsample"
        case "$scene:$mode:$thread" in
          fr1_desk:baseline:multi)        CONFIG="configs/rgbd/tum/fr1_desk_baseline_singlethread_off.yaml" ;;
          fr1_desk:baseline:single)       CONFIG="configs/rgbd/tum/fr1_desk_baseline.yaml" ;;
          fr1_desk:optimization:multi)    CONFIG="configs/rgbd/tum/fr1_desk_downsample_singlethread_off.yaml" ;;
          fr1_desk:optimization:single)   CONFIG="configs/rgbd/tum/fr1_desk_downsample_only.yaml" ;;
          room0:baseline:multi)           CONFIG="configs/rgbd/replica/room0_baseline_singlethread_off.yaml" ;;
          room0:baseline:single)          CONFIG="configs/rgbd/replica/room0_baseline.yaml" ;;
          room0:optimization:multi)       CONFIG="configs/rgbd/replica/room0_downsample_singlethread_off.yaml" ;;
          room0:optimization:single)      CONFIG="configs/rgbd/replica/room0_downsample_only.yaml" ;;
          scene0059_00:baseline:multi)      CONFIG="configs/rgbd/scannet/scene0059_00_baseline_singlethread_off.yaml" ;;
          scene0059_00:baseline:single)     CONFIG="configs/rgbd/scannet/scene0059_00_baseline.yaml" ;;
          scene0059_00:optimization:multi)  CONFIG="configs/rgbd/scannet/scene0059_00_downsample_singlethread_off.yaml" ;;
          scene0059_00:optimization:single) CONFIG="configs/rgbd/scannet/scene0059_00_downsample_only.yaml" ;;
        esac
      else
        echo "bad --optimization: $opt" >&2; usage
      fi
      ;;
    *) echo "bad --model: $model" >&2; usage ;;
  esac
  [[ -n "${CONFIG:-}" ]] || { echo "internal error: no config resolved for $model/$opt/$scene/$mode" >&2; exit 1; }
}

run_one() {
  local model="$1" opt="$2" scene="$3" mode="$4" thread="${5:-multi}"
  resolve "$model" "$opt" "$scene" "$mode" "$thread"
  [[ -f "$REPO/$CONFIG" ]] || { echo "ERROR: config not found: $REPO/$CONFIG" >&2; return 1; }

  # Thread mode only distinguishes monogs runs (splatam/gaussianslam have
  # no threading concept) -- keep their log naming otherwise unchanged.
  # OPT_TYPE_CSV always leads with the opt family (splatonic/rtgs) --
  # plain "baseline"/"optimization" alone can't tell a splatonic-splatam
  # row from an rtgs-splatam row, which are two different repos/configs
  # entirely.
  local NAME_SUFFIX="$mode"
  local OPT_TYPE_CSV="${opt}_${OPT_TYPE_LABEL}"
  if [[ "$model" == "monogs" ]]; then
    NAME_SUFFIX="${mode}_${thread}thread"
    OPT_TYPE_CSV="${opt}_${OPT_TYPE_LABEL}_${thread}thread"
  fi
  # RUN_TAG (e.g. "run1"/"run2"/"run3" for repeated trials) only affects
  # log/output file naming, never the CSV row -- the CSV schema stays
  # exactly 15 columns with no repeat/seed column; repeats are
  # distinguished by row order / timestamp if needed.
  [[ -n "$RUN_TAG" ]] && NAME_SUFFIX="${NAME_SUFFIX}_${RUN_TAG}"

  local TAG="${opt}-${model}_${scene}_${NAME_SUFFIX}"
  local LOG_DIR="$MLSYS_ROOT/logs/${opt}-${model}"
  local LOG_FILE="$LOG_DIR/${scene}_${NAME_SUFFIX}.log"
  local RUN_DIR="$MLSYS_ROOT/logs/${opt}-${model}/runs/${scene}_${NAME_SUFFIX}"
  mkdir -p "$LOG_DIR" "$RUN_DIR"
  local PY="$REPO/.venv/bin/python"
  [[ -x "$PY" ]] || { echo "ERROR: $PY not found -- build $REPO/.venv first (see header)" >&2; return 1; }

  local CMD
  if [[ "$EXTRACTOR_KIND" == "gaussianslam" ]]; then
    CMD=(env "${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}" "$PY" "$SCRIPT" "$CONFIG" --output_path "$RUN_DIR" "${EXTRA[@]+"${EXTRA[@]}"}")
  elif [[ "$EXTRACTOR_KIND" == "monogs" ]]; then
    CMD=(env "${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}" "$PY" "$SCRIPT" --config "$CONFIG" --eval)
  else  # splatam
    CMD=(env "${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}" "$PY" "$SCRIPT" "$CONFIG")
  fi

  echo "=== [$TAG] starting at $(date) on $(hostname) ==="
  {
    echo "=== cmd: ${CMD[*]} (cwd=$REPO) ==="
    echo "MLSYS_START_EPOCH=$(date +%s)"
  } | tee "$LOG_FILE"
  ( cd "$REPO" && "${CMD[@]}" ) >> "$LOG_FILE" 2>&1
  local STATUS=$?
  {
    echo "MLSYS_END_EPOCH=$(date +%s)"
    echo "=== [$TAG] finished at $(date), exit=$STATUS ==="
  } | tee -a "$LOG_FILE"

  local METRICS_OUT
  if [[ "$EXTRACTOR_KIND" == "splatam" ]]; then
    METRICS_OUT=$("$PY" "$MLSYS_ROOT/extract_splatam_metrics.py" "$LOG_FILE" 2>&1)
  elif [[ "$EXTRACTOR_KIND" == "gaussianslam" ]]; then
    METRICS_OUT=$("$PY" "$MLSYS_ROOT/extract_gaussianslam_metrics.py" "$RUN_DIR" "$LOG_FILE" 2>&1)
  else
    METRICS_OUT=$("$PY" "$MLSYS_ROOT/extract_monogs_metrics.py" "$LOG_FILE" 2>&1)
  fi
  if [[ $? -ne 0 ]]; then
    echo "WARNING: $TAG exited $STATUS -- metric extraction also failed, see below" >&2
    echo "$METRICS_OUT" >&2
    return 1
  fi
  declare -A M
  while IFS='=' read -r k v; do M["$k"]="$v"; done <<< "$METRICS_OUT"
  append_row "$MODEL_CSV" "$DATASET" "$scene" "$OPT_TYPE_CSV" M

  [[ $STATUS -ne 0 ]] && return 1
  return 0
}

if [[ $RUN_ALL -eq 1 ]]; then
  # One sweep per machine at a time: two concurrent --all sweeps on a
  # single-GPU box (Thor, a workstation) would contend for the GPU and
  # corrupt every timing column. Not taken under SLURM, where array
  # tasks are each scheduled onto their own GPU.
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec 9>"$MLSYS_ROOT/logs/.launch_all.lock"
    flock -n 9 || { echo "ERROR: another launch.sh --all is already running on this machine" >&2; exit 1; }
  fi
  NON_THREAD_MODELS=(splatam gaussianslam)
  OPTS=(splatonic rtgs)
  [[ -n "$ONLY_OPT" ]] && OPTS=("$ONLY_OPT")
  SCENES=(fr1_desk room0 scene0059_00)
  MODES=(baseline optimization)
  THREADS=(single multi)
  TOTAL=$(( ${#NON_THREAD_MODELS[@]} * ${#OPTS[@]} * ${#SCENES[@]} * ${#MODES[@]} \
            + ${#OPTS[@]} * ${#SCENES[@]} * ${#MODES[@]} * ${#THREADS[@]} ))
  COUNT=0
  FAILED=()
  echo "=== launch.sh --all starting at $(date) on $(hostname), $TOTAL total combos ==="
  echo "=== gpu=$GPU compute_location=$COMPUTE_LOCATION csv=$CSV ==="
  for m in "${NON_THREAD_MODELS[@]}"; do
    for o in "${OPTS[@]}"; do
      for s in "${SCENES[@]}"; do
        for md in "${MODES[@]}"; do
          COUNT=$((COUNT+1))
          echo "=== [$COUNT/$TOTAL] $o/$m/$s/$md ==="
          if ! run_one "$m" "$o" "$s" "$md"; then
            FAILED+=("$o/$m/$s/$md")
          fi
        done
      done
    done
  done
  for o in "${OPTS[@]}"; do
    for s in "${SCENES[@]}"; do
      for md in "${MODES[@]}"; do
        for th in "${THREADS[@]}"; do
          COUNT=$((COUNT+1))
          echo "=== [$COUNT/$TOTAL] $o-monogs/$s/$md/${th}thread ==="
          if ! run_one monogs "$o" "$s" "$md" "$th"; then
            FAILED+=("$o-monogs/$s/$md/${th}thread")
          fi
        done
      done
    done
  done
  echo "=== launch.sh --all finished at $(date): $((COUNT - ${#FAILED[@]}))/$TOTAL succeeded ==="
  if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "=== FAILED combos (${#FAILED[@]}): ==="
    printf '  %s\n' "${FAILED[@]}"
    exit 1
  fi
  exit 0
else
  [[ -n "$MODEL" && -n "$OPT" && -n "$SCENE" && -n "$MODE" ]] || usage
  run_one "$MODEL" "$OPT" "$SCENE" "$MODE" "$THREAD"
  exit $?
fi
