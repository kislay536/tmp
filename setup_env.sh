#!/bin/bash
# Build the three slamio environments: one venv per fork, each with the pinned stack the slamio
# framework was validated on (Python 3.10, torch 2.3.1+cu121) and the CUDA extensions compiled for
# the local GPU.
#
#   bash setup_env.sh                       # all three
#   bash setup_env.sh splatam monogs        # just these (splatam | gaussianslam | monogs)
#
# Overrides (defaults for an H100 box with CUDA 12.x installed):
#   PYTHON=/usr/bin/python3.10        interpreter used to create the venvs (must be 3.10)
#   CUDA_HOME=/usr/local/cuda-12.1    toolkit used to compile the extensions (nvcc must exist)
#   TORCH_CUDA_ARCH_LIST=9.0          9.0 = H100; 8.0 = A100; 8.9 = L40S / RTX 4090
#   MAX_JOBS=8                        parallel compile jobs
# Idempotent: an existing venv is reused, extensions are rebuilt only if they do not import.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$PWD"

PYTHON="${PYTHON:-python3.10}"
CUDA_HOME="${CUDA_HOME:-$(ls -d /usr/local/cuda 2>/dev/null)}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${MAX_JOBS:-8}"
TARGETS=("${@:-splatam gaussianslam monogs}")
read -r -a TARGETS <<< "${TARGETS[*]}"

red() { printf '\033[31m%s\033[0m\n' "$*"; }
grn() { printf '\033[32m%s\033[0m\n' "$*"; }

command -v "$PYTHON" >/dev/null || { red "python 3.10 not found; set PYTHON=/path/to/python3.10"; exit 1; }
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,10) else 1)' \
  || { red "$PYTHON is not Python 3.10 (the pinned torch 2.3.1 / opencv 4.8.1.78 stack targets 3.10)"; exit 1; }
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { red "nvcc not found under CUDA_HOME='$CUDA_HOME'; set CUDA_HOME=/usr/local/cuda-12.x"; exit 1; }
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
grn "python: $($PYTHON -V)   nvcc: $(nvcc --version | tail -1)   arch: $TORCH_CUDA_ARCH_LIST"

TORCH_PKGS=(--extra-index-url https://download.pytorch.org/whl/cu121 torch==2.3.1+cu121 torchvision==0.18.1+cu121)
COMMON=(tqdm Pillow imageio matplotlib scipy pyyaml wandb lpips torchmetrics pytorch-msssim "numpy<2" setuptools
        opencv-python-headless==4.8.1.78 open3d)

setup_one() {
  local name="$1" dir fork pkgs=() exts=()
  case "$name" in
    splatam)      fork=SplaTAM;       pkgs=(kornia natsort faiss-gpu-cu12)
                  exts=(slamio-splatam/submodules/diff-gaussian-rasterization) ;;
    gaussianslam) fork=Gaussian-SLAM; pkgs=(trimesh plyfile faiss-gpu-cu12)
                  exts=(slamio-gaussianslam/submodules/diff-gaussian-rasterization slamio-gaussianslam/submodules/simple-knn) ;;
    monogs)       fork=MonoGS;        pkgs=(plyfile trimesh imgviz PyOpenGL glfw PyGLM rich fvcore munch "evo==1.11.0" kornia)
                  exts=(slamio-monogs/submodules/diff-gaussian-rasterization slamio-monogs/MonoGS/submodules/simple-knn) ;;
    *) red "unknown target: $name"; return 1 ;;
  esac
  dir="$ROOT/slamio-$name/$fork"
  grn "=== $name: venv at $dir/.venv ==="
  [[ -x "$dir/.venv/bin/python" ]] || "$PYTHON" -m venv "$dir/.venv" || return 1
  local py="$dir/.venv/bin/python"
  "$py" -m pip install -q --upgrade pip setuptools wheel || return 1
  "$py" -m pip install -q "${TORCH_PKGS[@]}" || return 1
  "$py" -m pip install -q "${COMMON[@]}" "${pkgs[@]}" || return 1
  local e
  for e in "${exts[@]}"; do
    grn "  building $(basename "$e")"
    rm -rf "$ROOT/$e/build" "$ROOT/$e"/*.egg-info
    "$py" -m pip install --no-build-isolation --no-deps --force-reinstall "$ROOT/$e" || { red "  build failed: $e"; return 1; }
  done
  # smoke test: CUDA visible, arch present in torch, extensions import (slamio rasterizer exposes tracking_only etc.)
  ( cd "$dir" && "$py" - <<'EOF'
import torch, inspect
assert torch.cuda.is_available(), "CUDA not available in torch"
major, minor = torch.cuda.get_device_capability(0)
print(f"  device {torch.cuda.get_device_name(0)} sm_{major}{minor}; torch {torch.__version__}")
assert f"sm_{major}{minor}" in torch.cuda.get_arch_list(), "torch has no kernels for this GPU"
import diff_gaussian_rasterization as d
p = inspect.signature(d.rasterize_gaussians).parameters
assert "tracking_only" in p and "binning_capacity" in p, "not the slamio rasterizer"
try:
    import simple_knn._C
except ImportError:
    print("  (simple_knn not installed -- only needed by monogs / gaussianslam)")
import open3d.core as c
print("  open3d CUDA:", c.cuda.is_available(), "(needed by Gaussian-SLAM on Replica room0)")
EOF
  ) || { red "  smoke test failed for $name"; return 1; }
  grn "=== $name ready ==="
}

rc=0
for t in "${TARGETS[@]}"; do setup_one "$t" || rc=1; done
exit $rc
