#!/usr/bin/env bash
# Downloads and extracts a ScanNet v2 scene into the per-frame layout used by
# SplaTAM and MonoGS in this repository. Data goes to $DATA_ROOT (default: ../data relative to this script).
#
# ScanNet ships each scan as a single compressed .sens blob. Both loaders glob
# <scene>/color/*.jpg, <scene>/depth/*.png and <scene>/pose/*.txt, so the .sens
# has to be unpacked before either model can run it.
# This script does both halves.
#
# Usage:
#   bash download_scannet.sh scene0000_00      # download + extract one scene
#   bash download_scannet.sh                   # same, defaults to scene0000_00
#
# The six scenes SplaTAM benchmarks on (indices match SCENE_NUM in
# SplaTAM/configs/scannet/splatam.py):
#   0 scene0000_00   1 scene0059_00   2 scene0106_00
#   3 scene0169_00   4 scene0181_00   5 scene0207_00
#
# Assumes you have already been granted ScanNet access and accepted their
# Terms of Use. Nothing here needs the download-scannet.py they email you:
# that script fetches with urllib.urlretrieve, which cannot resume a dropped
# multi-GB transfer, so the blob is pulled with wget -c instead.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${DATA_ROOT:-$SCRIPT_DIR/../data}/ScanNet"
RAW="${DATA_ROOT:-$SCRIPT_DIR/../data}/ScanNet_raw"

SCENE="${1:-scene0000_00}"

mkdir -p "$OUT"

if [ -d "$OUT/$SCENE/color" ]; then
    echo "Already extracted, nothing to do: $OUT/$SCENE"
    echo "  color: $(ls "$OUT/$SCENE/color" | wc -l) frames"
    exit 0
fi

mkdir -p "$RAW"

# ---------- 1. the .sens blob ----------
# ScanNet's own download-scannet.py fetches with urllib.urlretrieve, which
# cannot resume. A dropped connection partway through a multi-GB transfer
# discards everything already received and raises ContentTooShortError, so on
# a 3.7 GB scan it is largely a matter of luck. wget -c resumes instead.
#
# The blob sits under the v1 path even for a v2 scan: upstream never
# re-released the raw sensor streams for v2, which is why their script passes
# use_v1_sens. It is the same file.
#
# ScanNet's Terms of Use still apply - this assumes you have already been
# granted access and accepted them.
SENS_URL="https://kaldir.vc.cit.tum.de/scannet/v1/scans/$SCENE/$SCENE.sens"
SENS="$RAW/scans/$SCENE/$SCENE.sens"

mkdir -p "$(dirname "$SENS")"
rm -f "$SENS.tmp"   # leftover from an interrupted urlretrieve run, unusable

echo "Downloading $SCENE.sens (~GBs; resumable, safe to re-run) ..."
wget -c --tries=20 --waitretry=10 --timeout=30 --read-timeout=60 --show-progress -O "$SENS" "$SENS_URL"

if [ ! -s "$SENS" ]; then
    echo "Download produced no file at $SENS."
    exit 1
fi

# A 404 or a portal page arrives as HTML with a 200, which would otherwise
# reach the parser as garbage.
if head -c 512 "$SENS" | grep -qi "<!doctype\|<html"; then
    echo "$SENS is an HTML page, not sensor data. The URL was:"
    echo "  $SENS_URL"
    echo "Check the scene id is one ScanNet actually publishes."
    exit 1
fi

echo "Have $SCENE.sens ($(du -h "$SENS" | cut -f1))"

# ---------- 2. extract ----------
# Writes color/<i>.jpg, depth/<i>.png, pose/<i>.txt, intrinsic/*.txt.
# Filenames are bare integers; the dataloader natsorts them, so 10 lands after 9.
#
# Uses scannet_sens_to_frames.py, not ScanNet's own SensReader: theirs is
# Python 2 and cannot even be imported on py3, and it calls np.fromstring,
# which numpy 2.0 removed. See that file's header for the rest.
echo "Extracting $SCENE -> $OUT/$SCENE ..."
mkdir -p "$OUT/$SCENE"
python "$SCRIPT_DIR/scannet_sens_to_frames.py"     --filename "$SENS"     --output_path "$OUT/$SCENE"

# ---------- 3. sanity check ----------
NC=$(ls "$OUT/$SCENE/color" 2>/dev/null | wc -l)
ND=$(ls "$OUT/$SCENE/depth" 2>/dev/null | wc -l)
NP=$(ls "$OUT/$SCENE/pose"  2>/dev/null | wc -l)
echo ""
echo "$SCENE: $NC color, $ND depth, $NP pose"
if [ "$NC" -eq 0 ] || [ "$NC" -ne "$ND" ] || [ "$NC" -ne "$NP" ]; then
    echo "WARNING: counts are zero or disagree. The dataloader zips these by index,"
    echo "so a mismatch will silently misalign frames. Investigate before running."
    exit 1
fi

echo ""
echo "Intrinsics this scene actually has:"
cat "$OUT/$SCENE/intrinsic/intrinsic_color.txt" 2>/dev/null || echo "  (intrinsic_color.txt not found)"
echo ""
echo "SplaTAM ignores the above and uses the fixed values in"
echo "SplaTAM/configs/data/scannet.yaml (fx 1169.621094, fy 1167.105103,"
echo "cx 646.295044, cy 489.927032). That is upstream SplaTAM's own behaviour."
echo "MonoGS/configs/rgbd/scannet/base_config.yaml uses those same color"
echo "intrinsics scaled to its aligned 640x480 RGB-D working resolution."
echo ""
echo "The .sens blob is still in $RAW ($(du -sh "$RAW" | cut -f1)). Safe to delete once happy."
