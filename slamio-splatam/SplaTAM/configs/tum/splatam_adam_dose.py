# ADAM ITERATION DOSE-RESPONSE, CLOSED LOOP. The denominator of every
# tracking-speedup claim on this branch.
#
#   ITERS=40 python scripts/splatam.py configs/tum/splatam_adam_dose.py
#
# WHY THIS EXISTS. Every acceleration result here is of the form "equal ATE at
# fewer tracking iterations". That sentence is unmeasurable until we know what
# ATE Adam ALONE reaches at each iteration count. We have never measured it.
#
# SECOND_ORDER_TRACKING_RECORD.md §4.2 puts Adam-20 at ~64 cm, but that is an
# INFERENCE from a run whose GN step was inert (truncate tau=0.1 froze 4 of 6
# DOF, dGT +0.005, |step| ~1e-11), not a measurement. The direct run was
# proposed twice and declined twice. §7 names it as the single most important
# unknown for anything that follows. This is that run.
#
# WHAT THE CURVE DECIDES.
#
#   If ATE is FLAT from 200 down to ~40, then ~150 of the 200 iterations buy
#   nothing, the headroom is in truncation alone, and no new optimiser is
#   needed - just a better stopping rule. That would close the preconditioning
#   line before a line of it is written.
#
#   If ATE degrades SHARPLY below ~90, the iterations are load-bearing, and the
#   only way to spend fewer of them is to make each one worth more. That is the
#   case a preconditioner has to win.
#
#   Either answer is worth the hour it costs.
#
# ONE VARIABLE: the iteration cap. Early stopping is DISABLED because it makes
# the realised count vary per frame, which is the variable under test - with it
# on, "num_iters=40" means "at most 40", not "40". Everything else - graphs,
# pixel masking, binning capacity, the loss, the map pipeline - is inherited
# unchanged from the same base the entire GN campaign used, so this curve is
# directly comparable to every number in the record.
#
# THE ANCHOR. Run ITERS=200 as part of the sweep. It should land near the
# 3.32 / 3.64 / 3.69 cm Adam reference. If it does not, the chain has drifted
# and NOTHING in the curve is comparable to the record - stop and find out why
# before reading any other point. This self-anchoring is deliberate: it makes
# the curve valid regardless of what the reference runs had configured.
#
# READ ONLY ATE. Per-frame tracking error must not be used to rank these arms.
# It anti-correlated with ATE three separate times in the GN campaign (record
# §5.3), including once on this exact axis: open-loop Adam-20 (1.357 cm) beat
# Adam-89 (1.616 cm) per-frame while being ~20x worse in trajectory. The map
# co-adapts to whatever the tracker commits, so trajectory error is the only
# quantity that carries the mechanism.
#
# NOISE FLOOR. Two identical hybrid runs differed by 0.62 cm (record §4.1), on
# a 3.3 cm baseline - ~18%. Any difference smaller than ~1 cm needs repeats
# before it means anything. Sweep the shape first at one rep, then repeat only
# the points the shape makes interesting.
from configs.tum.splatam_es_graph_eps_pixel075_maskmul_bincap import config as _base
import copy
import os

_iters = int(os.environ.get("ITERS", "200"))
# FRAMES=-1 runs the whole sequence (573 for freiburg1_desk). Default stays 250
# so every result measured so far remains reproducible from this file.
#
# ATE IS NOT COMPARABLE ACROSS SEQUENCE LENGTHS. It is an RMSE over the whole
# trajectory after alignment, and drift compounds, so a longer run has a larger
# ATE for the same tracker quality. The record's 3.32/3.64/3.69 reference and
# the 5 cm "held" threshold are both 250-frame numbers - switching to full
# length needs its own anchor run before any arm can be read against anything.
_frames = int(os.environ.get("FRAMES", "250"))

config = copy.deepcopy(_base)
# RUN_REP keeps each replicate's params.npz. Without it every rep writes to
# one directory and overwrites the last, so only the final run's trajectory
# survives - which is why the 5/5 sweep could not be re-examined with
# ate_vs_length.py when a later full-length run disagreed with it. The
# trajectories were already paid for and had been thrown away.
_rep = os.environ.get("RUN_REP", "")
config["run_name"] = f"freiburg1_desk_adam_dose_i{_iters}_f{_frames}"
if _rep:
    config["run_name"] += f"_r{_rep}"
config["data"]["num_frames"] = _frames

config["tracking"]["num_iters"] = _iters
# Not "at most _iters" - exactly _iters, on every frame. See header.
config["tracking"].setdefault("early_stop", {})["enabled"] = False
# BELT AND BRACES. enabled=False alone was not enough: EarlyStop.check() tested
# _probe_mode before it tested enabled, so the retune probe opened at frame 100
# regardless and truncated 15 frames to 75% of the budget - in a config whose
# entire purpose is a hard, exact iteration cap. Fixed in early_stop.py, and
# pinned here too so this config cannot be broken again by a change over there.
config["tracking"]["early_stop"]["retune_every"] = 0
