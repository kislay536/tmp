#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# rebuild.sh — rebuild the rasterizer with a chosen variant, reliably.
#
#   bash rebuild.sh                    # defaults: precise exp, no warp reduce
#   bash rebuild.sh warp               # DGR_WARP_REDUCE=1
#   bash rebuild.sh fastexp            # DGR_FAST_EXP=1
#   bash rebuild.sh warp fastexp       # both
#   bash rebuild.sh minblocks 8        # DGR_MIN_BLOCKS_PER_SM=8
#   bash rebuild.sh block 8 8          # DGR_BLOCK_X=8 DGR_BLOCK_Y=8
#   bash rebuild.sh stubgrads          # CEILING PROBE - GRADIENTS ARE GARBAGE
#
# stubgrads replaces every per-Gaussian scatter write in
# renderCUDABackward with a thread-local accumulation, keeping the gradient
# MATH and removing only the accumulation STRUCTURE. The kernel duration is
# the floor no reduction rewrite could beat. TIMING ONLY - rebuild before any
# real run, and never read an ATE from it.
#
# block sets the TILE SIZE, 16x16 by default. That is 3DGS's original and
# never-ablated choice; the trade it makes (atomic contention and barrier depth
# down, tile-Gaussian intersections up) is written out in config.h. Like
# minblocks it changes no arithmetic - but it DOES change summation order, so
# expect the usual ~2e-06 atomicAdd nondeterminism rather than bit-identity.
#
# Anything building a per-tile structure must read the built value back via
# diff_gaussian_rasterization.tile_size(); utils/pixel_sample.py does. A
# hardcoded 16 against an 8x8 build makes a wrong-length mask and NOTHING
# RAISES.
#
# minblocks forces __launch_bounds__ occupancy on the two tile kernels. It
# changes NO ARITHMETIC, so it needs no gradient verification - but it is
# priced from the REGISTER SUMMARY this script prints, not from a run. A value
# that spills heavily in renderCUDABackward is discarded without ever being
# launched. See TILE_LAUNCH_BOUNDS in cuda_rasterizer/config.h.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS RATHER THAN A DOCUMENTED pip COMMAND.
#
# Three separate things have to be right or the build silently produces the
# WRONG BINARY, and each of them has already caused a wasted measurement:
#
#   --no-cache-dir        pip caches built wheels keyed on the SOURCE CONTENTS.
#                         An environment variable is not part of that key, so
#                         `DGR_WARP_REDUCE=1 pip install .` on unchanged source
#                         happily serves a cached wheel built WITHOUT the flag.
#                         This produced a run that reported warp_reduce=False
#                         after an explicit warp build, and nearly invalidated
#                         an entire A/B.
#
#   --no-build-isolation  without it pip builds in a fresh venv where CUDA_HOME
#                         is not set on this VM, so nvcc is not found.
#
#   rm -rf build/         stale objects survive a flag change otherwise.
#
# The run itself prints which variant it got:
#
#     [diff_gaussian_rasterization] build: warp_reduce=True  fast_exp=False
#
# ALWAYS CHECK THAT LINE before attributing a measurement to a build. It is the
# only thing that cannot be misremembered.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

WARP=0
FASTEXP=0
MINBLOCKS=""
BLOCKX=""
BLOCKY=""
STUBGRADS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        warp)      WARP=1 ;;
        fastexp)   FASTEXP=1 ;;
        minblocks)
            shift
            if [[ $# -eq 0 || ! "$1" =~ ^[0-9]+$ || "$1" -lt 1 ]]; then
                echo "minblocks needs a positive integer, e.g. 'minblocks 8'"; exit 1
            fi
            MINBLOCKS="$1"
            ;;
        stubgrads) STUBGRADS=1 ;;
        block)
            shift
            if [[ $# -lt 2 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^[0-9]+$ ]]; then
                echo "block needs two positive integers, e.g. 'block 8 8'"; exit 1
            fi
            BLOCKX="$1"; BLOCKY="$2"
            shift
            ;;
        *) echo "Unknown option '$1'. Valid: warp fastexp stubgrads 'minblocks N' 'block BX BY'"
           exit 1 ;;
    esac
    shift
done

ENVARGS=()
if [[ $WARP == 1 ]]; then ENVARGS+=("DGR_WARP_REDUCE=1"); fi
if [[ $FASTEXP == 1 ]]; then ENVARGS+=("DGR_FAST_EXP=1"); fi
if [[ -n "$MINBLOCKS" ]]; then ENVARGS+=("DGR_MIN_BLOCKS_PER_SM=$MINBLOCKS"); fi
if [[ -n "$BLOCKX" ]]; then ENVARGS+=("DGR_BLOCK_X=$BLOCKX" "DGR_BLOCK_Y=$BLOCKY"); fi
if [[ $STUBGRADS == 1 ]]; then ENVARGS+=("DGR_STUB_GRADS=1"); fi

echo "==> rebuilding with: ${ENVARGS[*]:-<defaults: precise exp, no warp reduce>}"

rm -rf build/ *.egg-info

BUILD_LOG="build_last.log"

# -v IS LOAD-BEARING, NOT NOISE. pip runs the PEP 517 build backend in a
# subprocess and CAPTURES its output, showing only "Building wheel ... done"
# unless the build fails or -v is passed. So nvcc's -Xptxas -v report never
# reached the terminal, and the first version of this script concluded from its
# absence that "the compile did not run" - on a build that had just compiled
# for thirty seconds and installed correctly. The diagnostic was wrong about
# the diagnosis, which is worse than not having it.
#
# The full log goes to a file rather than the terminal because -v is genuinely
# enormous; only the summary is printed.
echo "==> building (full log: $(pwd)/$BUILD_LOG)"
# ${ENVARGS[@]+...} guards the empty-array case under `set -u`.
if ! env ${ENVARGS[@]+"${ENVARGS[@]}"} \
        pip install -v --no-cache-dir --no-build-isolation . > "$BUILD_LOG" 2>&1; then
    echo "!! build FAILED - last 40 lines:"
    tail -40 "$BUILD_LOG"
    echo ""
    echo "   full log: $(pwd)/$BUILD_LOG"
    exit 1
fi
grep -E "^Successfully installed|^Successfully built" "$BUILD_LOG" | sed 's/^/    /' || true

echo ""

# ---------------------------------------------------------------------------
# THE REGISTER SUMMARY. This is what prices a minblocks value, and it costs no
# GPU time at all: -Xptxas -v (set unconditionally in setup.py) makes ptxas
# print, per kernel, the registers used and the bytes spilled.
#
# What to look for on the two tile kernels:
#
#   renderCUDA / renderCUDABackward at 48 registers, 0 spill  = the default
#   the same kernels at ~32 registers, 0 spill                = free occupancy,
#                                                               worth an A/B
#   the same kernels at ~32 registers with HUNDREDS of bytes  = the compiler
#     spilled the inner-loop accumulators to comply. Discard this value
#     WITHOUT RUNNING IT.
#
# A value that spills is not automatically worse - spills go through L1 and the
# kernel is not DRAM-bound (0.27% of peak) - but it is no longer a free change,
# and it stops being worth trying before the cheaper values have been.
# ---------------------------------------------------------------------------
echo "==> ptxas register summary (tile kernels)"
# NOT a grep. ptxas splits one kernel's usage over four lines and the spill
# bytes are on a different line from the register count, so a grep for "Used N
# registers" reports the registers and silently drops the spills - which is
# backwards, since a register target is only interesting if it was met WITHOUT
# spilling. See ptxas_summary.py.
# --threads-per-block MUST track the build. It defaults to 256, and the
# blocks/SM column is derived from it - so after an 8x8 build the untouched
# default reported occupancy for 256-thread blocks that were never compiled.
# It printed "4 blocks/SM" for a 64-thread build whose real figure is 18.
_TPB=$(( ${BLOCKX:-16} * ${BLOCKY:-16} ))
python "$(pwd)/ptxas_summary.py" --threads-per-block "$_TPB" < "$BUILD_LOG" || true
echo ""

# A stale in-place extension in the SOURCE package directory shadows the
# installed one for anything whose sys.path includes this directory. It
# survives `rm -rf build/`, so it can silently outlive many rebuilds.
STALE=$(ls diff_gaussian_rasterization/_C*.so 2>/dev/null || true)
if [[ -n "$STALE" ]]; then
    echo "!! WARNING: in-place extension present in the source tree:"
    echo "     $STALE"
    echo "   It shadows the installed package whenever this directory is on"
    echo "   sys.path. Remove it unless you deliberately built in place:"
    echo "     rm diff_gaussian_rasterization/_C*.so"
    echo ""
fi

echo "==> verifying what actually got installed"
# MUST run from somewhere else. `python - <<PY` puts the CURRENT DIRECTORY on
# sys.path, and this directory contains the source package - so verifying from
# here imports the local source (plus any stale in-place _C) rather than what
# pip just installed. That produced a false "UNKNOWN" on a build that was
# actually fine.
( cd / && python - <<'PY'
try:
    from diff_gaussian_rasterization import build_variant, __file__ as f
    v = build_variant()
    print(f"    module: {f}")
    if v["warp_reduce"] is None:
        print("    UNKNOWN - this .so predates the introspection binding.")
        print("    Something served a stale wheel or a stale in-place build.")
        raise SystemExit(1)
    mb = v.get("min_blocks_per_sm", 0)
    bx, by = v.get("block_x", 16), v.get("block_y", 16)
    print(f"    warp_reduce={v['warp_reduce']}  fast_exp={v['fast_exp']}  "
          f"min_blocks_per_sm={'default' if not mb else mb}  "
          f"tile={bx}x{by} ({bx * by} threads/block)")
    if v.get("stub_grads"):
        bar = "    " + "!" * 66
        print(bar)
        print("    !! CEILING PROBE BUILD - GRADIENTS ARE GARBAGE.")
        print("    !! renderCUDABackward does not accumulate them. Kernel")
        print("    !! TIMING only; rebuild before any run you read an ATE from.")
        print(bar)
except ImportError as e:
    print(f"    import failed: {e}")
    raise SystemExit(1)
PY
)
