"""Parse ptxas -v output into a register/spill table for the tile kernels.

WHY THIS IS A PARSER AND NOT A grep. ptxas splits one kernel's resource usage
across FOUR lines, and the number that decides a DGR_MIN_BLOCKS_PER_SM value -
the spill bytes - is NOT on the line that reports the register count:

    ptxas info    : Compiling entry function '_Z10renderCUDA...' for 'sm_80'
    ptxas info    : Function properties for _Z10renderCUDA...
        0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads   <- here
    ptxas info    : Used 48 registers, 380 bytes cmem[0]                 <- and here

A grep for "Used N registers" therefore reports the registers and silently
drops the spills, which is exactly backwards: a value that hits 32 registers is
only interesting IF it did so without spilling. The first version of this
extraction did that, and would have made every forced-occupancy build look free.

Reads a build log on stdin.

    bash rebuild.sh minblocks 8          (calls this itself)
    python ptxas_summary.py < build.log --all
"""

import argparse
import re
import shutil
import subprocess
import sys

RE_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '([^']+)'")
RE_PROPS = re.compile(r"Function properties for (\S+)")
RE_SPILL = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
RE_REGS = re.compile(r"Used (\d+) registers")

# The two kernels DGR_MIN_BLOCKS_PER_SM applies to. Matched against the MANGLED
# name, which still contains the identifier.
TILE_KERNELS = ("renderCUDA", "renderCUDABackward")


def demangle(names):
    """Best effort. c++filt is not always present and is not worth requiring."""
    exe = shutil.which("c++filt")
    if not exe or not names:
        return {n: n for n in names}
    try:
        out = subprocess.run([exe], input="\n".join(names), capture_output=True,
                             text=True, timeout=10)
        lines = out.stdout.splitlines()
        if len(lines) == len(names):
            return dict(zip(names, lines))
    except Exception:
        pass
    return {n: n for n in names}


def parse(text):
    """Return [{name, arch, regs, stack, spill_st, spill_ld}] in file order."""
    kernels = []
    by_name = {}
    current = None
    for line in text.splitlines():
        m = RE_ENTRY.search(line)
        if m:
            current = {"name": m.group(1), "arch": m.group(2), "regs": None,
                       "stack": 0, "spill_st": 0, "spill_ld": 0}
            kernels.append(current)
            by_name[m.group(1)] = current
            continue
        m = RE_PROPS.search(line)
        if m:
            # Properties can appear for a function other than the last entry
            # (inlined device functions get their own blocks).
            current = by_name.get(m.group(1), current)
            continue
        m = RE_SPILL.search(line)
        if m and current is not None:
            current["stack"] = int(m.group(1))
            current["spill_st"] = int(m.group(2))
            current["spill_ld"] = int(m.group(3))
            continue
        m = RE_REGS.search(line)
        if m and current is not None:
            current["regs"] = int(m.group(1))
    return kernels


# sm_80 (A100) hardware limits. Only binding once block size became a variant.
MAX_BLOCKS_PER_SM = 32
MAX_WARPS_PER_SM = 64


def blocks_per_sm(regs, threads_per_block, regs_per_sm):
    """Resident blocks/SM for a given per-thread register count.

    REGISTERS ARE NOT ALLOCATED AT THE GRANULARITY THEY ARE REPORTED. On sm_70+
    the allocation unit is 256 registers per WARP, i.e. the per-thread count is
    effectively rounded UP to a multiple of 8. A naive 65536/(regs*threads)
    happens to be right for 32, 40, 48 and 63 - every value observed so far -
    and is wrong for anything between, e.g. 41 registers really allocates 48 and
    gives 5 blocks, not 6.

    That is the kind of error that only shows up once a build lands on an
    unlucky number, and it would have overstated the occupancy of exactly the
    marginal values this tool exists to judge.
    """
    if not regs:
        return 0
    per_warp = -(-regs * 32 // 256) * 256      # ceil to the 256-register unit
    warps = regs_per_sm // per_warp
    blocks = warps // (threads_per_block // 32)
    # Hardware caps, which registers alone can leave unmentioned. Small blocks
    # hit these long before they exhaust the register file: at 64 threads a
    # low-register kernel computes 30+ blocks from registers while sm_80 allows
    # at most 32 resident blocks and 64 warps per SM. Without this the tool
    # reports occupancy the hardware will never deliver - and it only bites
    # once BLOCK_X/BLOCK_Y stopped being 16x16.
    blocks = min(blocks, MAX_BLOCKS_PER_SM,
                 MAX_WARPS_PER_SM // max(1, threads_per_block // 32))
    return blocks


def short(name):
    """Trim a demangled template signature to something readable."""
    base = name.split("(")[0]
    return base[-72:] if len(base) > 72 else base


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="every kernel, not just the tile kernels")
    ap.add_argument("--threads-per-block", type=int, default=256,
                    help="for the blocks/SM estimate (default 256)")
    ap.add_argument("--regs-per-sm", type=int, default=65536,
                    help="register file size per SM (default 65536, A100/sm_80)")
    args = ap.parse_args()

    kernels = parse(sys.stdin.read())
    if not kernels:
        print("    no 'ptxas info' lines in the build output.")
        print("    pip HIDES the compiler output unless it is run with -v, so")
        print("    this usually means the -v flag was lost rather than that the")
        print("    compile was skipped. Check that rebuild.sh still passes it.")
        return 2

    selected = kernels if args.all else [
        k for k in kernels if any(t in k["name"] for t in TILE_KERNELS)]
    if not selected:
        print(f"    {len(kernels)} kernels compiled, none matching "
              f"{TILE_KERNELS}. Use --all to see them.")
        return 2

    names = demangle([k["name"] for k in selected])

    # WARPS/SM, not just blocks/SM. Blocks are not comparable across builds once
    # the block SIZE is a variant: 5 blocks of 256 threads and 18 blocks of 64
    # threads are 40 and 36 resident warps respectively, and it is the warps
    # that hide latency. Comparing the blocks column across a tile-size sweep
    # reads backwards.
    wpb = max(1, args.threads_per_block // 32)
    print(f"    (threads/block={args.threads_per_block}, {wpb} warps/block)")
    print(f"    {'regs':>5} {'blocks/SM':>10} {'warps/SM':>9} "
          f"{'spill st':>9} {'spill ld':>9}  kernel")
    print(f"    {'-'*5} {'-'*10} {'-'*9} {'-'*9} {'-'*9}  {'-'*40}")
    any_spill = False
    for k in selected:
        regs = k["regs"]
        blocks = blocks_per_sm(regs, args.threads_per_block, args.regs_per_sm)
        spill = k["spill_st"] or k["spill_ld"]
        any_spill = any_spill or bool(spill)
        flag = "  <-- SPILLS" if spill else ""
        print(f"    {regs if regs is not None else '?':>5} {blocks:>10} "
              f"{blocks * wpb:>9} "
              f"{k['spill_st']:>9} {k['spill_ld']:>9}  {short(names[k['name']])}{flag}")

    print()
    # A BLANKET VERDICT IS WRONG HERE, and the first version gave one. The two
    # renderCUDABackward instantiations are launched by DIFFERENT MODELS, and
    # each model builds its own .so in its own conda env - so a spill in an
    # instantiation this env never launches costs exactly nothing.
    #
    # The build that prompted this fix read 40 regs / 0 spills on <3,false> and
    # 40 regs / 36 bytes on <3,true>, and the tool said "SPILLS PRESENT, try
    # lower values" - on a build that was perfectly free for the SplaTAM env it
    # had just been run in.
    print("    WHICH KERNEL MATTERS DEPENDS ON THE ENV THIS WAS BUILT IN:")
    print("      renderCUDABackward<C,false>   SplaTAM, Gaussian-SLAM")
    print("      renderCUDABackward<C,true>    MonoGS (pose-grad path)")
    print("      renderCUDA<C>                 forward, all models")
    print("    A spill in an instantiation this env never launches is free.")
    print()
    spillers = [k for k in selected if k["spill_st"] or k["spill_ld"]]
    clean = [k for k in selected if not (k["spill_st"] or k["spill_ld"])]
    if not spillers:
        print("    No spills anywhere. If a register count dropped, that occupancy")
        print("    is free and this value is worth a timing A/B.")
    else:
        for k in clean:
            if k["regs"]:
                print(f"    FREE for {short(names[k['name']])}: "
                      f"{k['regs']} regs, "
                      f"{blocks_per_sm(k['regs'], args.threads_per_block, args.regs_per_sm)} "
                      f"blocks/SM, no spills.")
        print("    Spilled: " + ", ".join(short(names[k["name"]]) for k in spillers))
        print("    A spill is not automatically worse - local memory is backed by L1")
        print("    and this kernel is not DRAM-bound (0.27% of peak) - but it is a")
        print("    trade rather than a free change, so measure it separately from")
        print("    any spill-free value.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
