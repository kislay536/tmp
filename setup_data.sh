#!/bin/bash
# Link the three datasets into the places each slamio fork expects them.
#
#   DATA_ROOT=/path/to/data bash setup_data.sh
#
# DATA_ROOT must contain (download them with scripts/download_*.sh, which write here by default):
#   $DATA_ROOT/TUM_RGBD/rgbd_dataset_freiburg1_desk/{rgb,depth,rgb.txt,depth.txt,groundtruth.txt}
#   $DATA_ROOT/Replica/room0/{results/,traj.txt}
#   $DATA_ROOT/ScanNet/scene0059_00/{color,depth,pose,intrinsic}
# Only symlinks are created; nothing is copied. Safe to re-run.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$PWD"
DATA_ROOT="$(cd "${DATA_ROOT:-$ROOT/data}" 2>/dev/null && pwd)" || { echo "DATA_ROOT does not exist" >&2; exit 1; }

tum="$DATA_ROOT/TUM_RGBD"; rep="$DATA_ROOT/Replica"; scn="$DATA_ROOT/ScanNet"
ok=1
[[ -f "$tum/rgbd_dataset_freiburg1_desk/rgb.txt" ]] || { echo "missing: $tum/rgbd_dataset_freiburg1_desk (bash scripts/download_tum.sh fr1_desk)" >&2; ok=0; }
[[ -f "$rep/room0/traj.txt" ]]                      || { echo "missing: $rep/room0 (bash scripts/download_replica.sh room0)" >&2; ok=0; }
[[ -d "$scn/scene0059_00/color" ]]                  || { echo "missing: $scn/scene0059_00 (bash scripts/download_scannet.sh scene0059_00; needs ScanNet access)" >&2; ok=0; }
[[ $ok -eq 1 ]] || exit 1

link() { mkdir -p "$(dirname "$2")"; ln -sfn "$1" "$2"; echo "  $2 -> $1"; }

# SplaTAM
link "$tum" "$ROOT/slamio-splatam/SplaTAM/data/TUM_RGBD"
link "$rep" "$ROOT/slamio-splatam/SplaTAM/data/Replica"
link "$scn" "$ROOT/slamio-splatam/SplaTAM/data/scannet"
# Gaussian-SLAM
link "$tum" "$ROOT/slamio-gaussianslam/Gaussian-SLAM/data/TUM_RGBD-SLAM"
link "$rep" "$ROOT/slamio-gaussianslam/Gaussian-SLAM/data/Replica-SLAM/Replica"
link "$scn" "$ROOT/slamio-gaussianslam/Gaussian-SLAM/data/scannet/scans"
link "$scn" "$ROOT/slamio-gaussianslam/datasets/ScanNet"      # used by the slamio ScanNet config (../datasets/ScanNet)
# MonoGS
link "$tum" "$ROOT/slamio-monogs/MonoGS/datasets/tum"
link "$rep" "$ROOT/slamio-monogs/MonoGS/datasets/replica"
link "$scn" "$ROOT/slamio-monogs/MonoGS/datasets/scannet"
echo "data linked from $DATA_ROOT"
