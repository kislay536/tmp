#!/usr/bin/env bash
# Downloads TUM RGB-D sequences used across SplaTAM, MonoGS, and Gaussian-SLAM.
# Data goes to $DATA_ROOT (default: ../data relative to this script).
# All sequences land in datasets/TUM_RGBD/ — the junctions/symlinks take care of the rest.
#
# Usage:
#   bash download_tum.sh              # download all sequences
#   bash download_tum.sh fr1_desk     # download a specific sequence
#
# Available sequences:
#   fr1_desk   fr1_desk2   fr1_room   fr2_xyz   fr3_long_office
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${DATA_ROOT:-$SCRIPT_DIR/../data}/TUM_RGBD"

mkdir -p "$OUT"

declare -A URLS
URLS[fr1_desk]="https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
URLS[fr1_desk2]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz"
URLS[fr1_room]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz"
URLS[fr2_xyz]="https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz"
URLS[fr3_long_office]="https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz"

# Which models use each sequence
# fr1_desk, fr2_xyz, fr3_long_office -> all 3 models
# fr1_desk2, fr1_room                -> SplaTAM only

download_seq() {
    local url="$1"
    local tgz="$(basename $url)"
    local dir="$OUT/${tgz%.tgz}"

    if [ -d "$dir" ]; then
        echo "  already exists, skipping: $(basename $dir)"
        return
    fi

    echo "  downloading: $tgz"
    wget -q --show-progress -P "$OUT" "$url"
    echo "  extracting:  $tgz"
    tar -xzf "$OUT/$tgz" -C "$OUT"
    rm "$OUT/$tgz"
    echo "  done: $(basename $dir)"
}

if [ -n "$1" ]; then
    # Single sequence mode
    key="$1"
    if [ -z "${URLS[$key]}" ]; then
        echo "Unknown sequence '$key'. Available: ${!URLS[@]}"
        exit 1
    fi
    echo "Downloading TUM sequence '$key' to $OUT ..."
    download_seq "${URLS[$key]}"
else
    # All sequences
    echo "Downloading all TUM RGB-D sequences to $OUT ..."
    echo ""
    for key in fr1_desk fr2_xyz fr3_long_office fr1_desk2 fr1_room; do
        download_seq "${URLS[$key]}"
    done
fi

echo ""
echo "Sequences in $OUT:"
ls "$OUT"
