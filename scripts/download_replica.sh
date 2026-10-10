#!/usr/bin/env bash
# Downloads Replica RGB-D sequences and cull meshes for SplaTAM, MonoGS, and Gaussian-SLAM.
# Data goes to $DATA_ROOT (default: ../data relative to this script).
#
# Usage:
#   bash download_replica.sh              # all 8 scenes + cull meshes
#   bash download_replica.sh room0        # single scene + cull meshes
#   bash download_replica.sh --no-cull    # skip cull_replica (no gslam reconstruction eval)
#   bash download_replica.sh room0 --no-cull
#
# Available scenes:
#   room0  room1  room2  office0  office1  office2  office3  office4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS="${DATA_ROOT:-$SCRIPT_DIR/../data}/Replica"
CULL_DIR="$SCRIPT_DIR/Gaussian-SLAM/data/Replica-SLAM/cull_replica"

NICESLAM_URL="https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
HF_REPO="https://huggingface.co/datasets/voviktyl/Replica-SLAM"

ALL_SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)

mkdir -p "$DS"

# --- parse args ---
TARGET=""
DOWNLOAD_CULL=1
for arg in "$@"; do
    case "$arg" in
        --no-cull) DOWNLOAD_CULL=0 ;;
        *) TARGET="$arg" ;;
    esac
done

green()  { echo -e "\033[1;32m$*\033[0m"; }
yellow() { echo -e "\033[1;33m$*\033[0m"; }

# ── RGB-D sequences (NICE-SLAM zip, all 8 scenes in one archive) ──────────────
download_sequences() {
    # check what's already present
    local missing=()
    if [ -n "$TARGET" ]; then
        [ -d "$DS/$TARGET" ] || missing=("$TARGET")
    else
        for s in "${ALL_SCENES[@]}"; do
            [ -d "$DS/$s" ] || missing+=("$s")
        done
    fi

    if [ ${#missing[@]} -eq 0 ]; then
        green "  RGB-D sequences already present, skipping download."
        return
    fi

    local zip="$DS/_Replica.zip"
    if [ ! -f "$zip" ]; then
        green "Downloading Replica RGB-D sequences (~1.5 GB)..."
        wget -q --show-progress -O "$zip" "$NICESLAM_URL"
    else
        yellow "  zip already downloaded, extracting..."
    fi

    echo "Extracting..."
    local tmp="$DS/_extract_tmp"
    mkdir -p "$tmp"
    unzip -q "$zip" -d "$tmp"
    # zip extracts to _extract_tmp/Replica/<scene>/
    for scene_dir in "$tmp/Replica"/*/; do
        local scene
        scene="$(basename "$scene_dir")"
        if [ -d "$DS/$scene" ]; then
            echo "  skipping (exists): $scene"
        else
            mv "$scene_dir" "$DS/$scene"
            echo "  extracted: $scene"
        fi
    done
    rm -rf "$tmp" "$zip"
    green "  done."
}

# ── cull_replica meshes (needed for gslam reconstruction eval) ────────────────
download_cull() {
    if [ -d "$CULL_DIR" ] && [ "$(ls -A "$CULL_DIR" 2>/dev/null)" ]; then
        green "  cull_replica already present, skipping."
        return
    fi

    if ! git lfs version &>/dev/null 2>&1; then
        yellow "  git-lfs not found — install with: sudo apt-get install -y git-lfs && git lfs install"
        yellow "  Skipping cull_replica (gslam reconstruction eval will not work)."
        return
    fi

    green "Downloading cull_replica meshes via huggingface_hub (~300 MB)..."
    mkdir -p "$CULL_DIR"

    local scenes_arg=""
    if [ -n "$TARGET" ]; then
        scenes_arg="$TARGET"
    else
        scenes_arg="${ALL_SCENES[*]}"
    fi

    python3 - "$CULL_DIR" "$scenes_arg" <<'PYEOF'
import sys, pathlib, re

try:
    from huggingface_hub import snapshot_download, list_repo_tree
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import snapshot_download, list_repo_tree

dst    = pathlib.Path(sys.argv[1])
scenes = sys.argv[2].split()
repo   = "voviktyl/Replica-SLAM"

# discover actual paths for .ply and _pc_unseen.npy in the repo
print("  listing repo tree...")
all_paths = [f.path for f in list_repo_tree(repo, repo_type="dataset", recursive=True)
             if hasattr(f, "path")]

needed = []
for scene in scenes:
    for p in all_paths:
        if re.search(rf"(^|/){re.escape(scene)}\.ply$", p) or \
           re.search(rf"(^|/){re.escape(scene)}_pc_unseen\.npy$", p):
            needed.append(p)

if not needed:
    print(f"  WARNING: no cull_replica files found for scenes {scenes}")
    print(f"  Repo contents: {all_paths[:20]}")
    sys.exit(1)

print(f"  found {len(needed)} files, downloading...")
tmp = snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    allow_patterns=needed,
    local_dir=str(dst / "_hf_tmp"),
)

# flatten into dst/
for p in needed:
    src = pathlib.Path(tmp) / p
    out = dst / src.name
    if out.exists():
        print(f"  already present: {out.name}")
    elif src.exists():
        src.rename(out)
        print(f"  downloaded: {out.name}")
    else:
        print(f"  WARNING: expected file not found after download: {src}")

import shutil
shutil.rmtree(dst / "_hf_tmp", ignore_errors=True)
print("done")
PYEOF
    green "  done: $CULL_DIR"
}

# ── main ──────────────────────────────────────────────────────────────────────
green "=== Replica dataset ==="
echo "Canonical location : $DS"
[ -n "$TARGET" ] && echo "Scene filter       : $TARGET" || echo "Scenes             : all"
[ $DOWNLOAD_CULL -eq 1 ] && echo "cull_replica       : yes" || echo "cull_replica       : skipped (--no-cull)"
echo ""

download_sequences

if [ $DOWNLOAD_CULL -eq 1 ]; then
    download_cull
fi

echo ""
green "=== Done ==="
echo "Scenes in $DS:"
ls "$DS"
