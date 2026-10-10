# CONSECUTIVE-ITERATION GRADIENT SIMILARITY, DENSE, PRECONDITIONER.
#
# splatam_gradstale.py measures cos(stale) at only 2 starts per frame
# (iterations 5 and 15), offsets 1/2/4 - enough for the operating-point table
# but not enough to see how cos(g_k, g_k+1) moves ACROSS a frame's
# optimization. This samples offset=1 only, at every other iteration, so it
# traces a near-continuous curve instead of 2 points.
#
# WHY EVERY OTHER ITERATION, NOT EVERY ONE. GradStalenessProbe.observe()
# checks `iter_idx in starts` before checking whether iter_idx completes a
# pending window - so consecutive integer starts (2,3,4,5,...) would open a
# new window on the very iteration that was supposed to close the previous
# one, and silently drop every measurement. Starts spaced by 2 with
# offsets=[1] sidesteps this: window opens at k, closes at k+1, next window
# opens at k+2 (already closed, no overlap).
#
# Frames run ~27 iterations at this operating point (grad_reuse.py docstring),
# so starts cover 2..24; later starts simply never fire on a frame that stops
# earlier under early_stop.
from configs.tum.splatam_precond import config as _base
import copy

config = copy.deepcopy(_base)
config["run_name"] = "freiburg1_desk_gradstale_consec_precond"
config["data"]["num_frames"] = 300

config["tracking"]["cuda_graph"]["enabled"] = False
config["tracking"].setdefault("iteration_graph", {})["enabled"] = False

config["tracking"]["grad_staleness"] = {
    "enabled": True,
    "frames": [50, 150, 250],
    "starts": list(range(2, 26, 2)),
    "offsets": [1],
    "usable_cos": 0.95,
    "out_path": "grad_staleness_precond_dense.jsonl",
}
