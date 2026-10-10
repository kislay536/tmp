# GRADIENT STALENESS PROBE ON PLAIN ADAM - DENSE VARIANT.
#
# splatam_gradstale_adam.py is the sparse go/no-go probe (starts=[5,15],
# offsets=[1,2,4], 6 points/frame) - unchanged, still the source of the
# headline cos numbers. This traces nearly every consecutive-iteration pair
# instead, for a per-frame trend plot (feeds
# profiling/plot_grad_staleness_dense.py) rather than a go/no-go table. This
# is the config that reconstructs results/grad_staleness_adam/adam_consecutive_cos.png -
# that plot's own run was never committed, so this is a fresh equivalent, not
# a recovery of the original data.
#
# STARTS ARE ODD, NOT range(1, 200), BECAUSE GradStalenessProbe.wants() gives
# `iter_idx in self.starts` PRIORITY over closing the current window: if an
# even iteration were both "the offset-1 close of the previous start" and "the
# next start", observe() takes the start branch unconditionally, prints an
# "overlaps" warning, and drops the pending measurement - so back-to-back
# integer starts with offsets=[1] silently lose most rows. Spacing starts by 2
# (1,3,5,...) makes every window (k, k+1) close cleanly before the next opens,
# at the cost of tracing every OTHER consecutive pair rather than literally
# every one - still dense enough for a trend line, and it needs no change to
# grad_staleness.py's contract.
#
# Same optimizer/base as splatam_gradstale_adam.py: configs.tum.splatam
# (stock Adam, tracking_iters=200, early_stop min_iters=70), not
# splatam_precond. No preconditioner means no 6x6 curvature matrix, so
# cos_extrap/cos_secant read n/a - expected, only cos_stale/mag_ratio matter.
#
# Not a timed run - every probe point is an extra forward+backward, and this
# variant asks for ~100 of them per traced frame instead of 6.
from configs.tum.splatam import config as _base
import copy

config = copy.deepcopy(_base)
config["run_name"] = "freiburg1_desk_gradstale_adam_dense"
config["data"]["num_frames"] = 300

config["tracking"]["cuda_graph"]["enabled"] = False
config["tracking"].setdefault("iteration_graph", {})["enabled"] = False

config["tracking"]["grad_staleness"] = {
    "enabled": True,
    "frames": [50, 150, 250],
    "starts": list(range(1, 200, 2)),
    "offsets": [1],
    "usable_cos": 0.95,
    "out_path": "grad_staleness_adam_dense.jsonl",
}
