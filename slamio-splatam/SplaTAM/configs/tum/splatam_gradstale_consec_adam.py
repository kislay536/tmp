# CONSECUTIVE-ITERATION GRADIENT SIMILARITY, DENSE, PLAIN ADAM.
#
# Same idea as splatam_gradstale_consec_precond.py - offset=1 only, sampled
# at every other iteration so it traces a curve across the frame instead of
# 2 points - but based on stock configs.tum.splatam (plain Adam,
# tracking_iters=200, early_stop min_iters=70) instead of splatam_precond.
#
# Frames here run much longer (up to 200 iters, early-stop floor 70), so
# starts cover 2..98 - wide enough to see the pre-stop region and a bit past
# it. See splatam_gradstale_consec_precond.py for why starts are spaced by 2.
from configs.tum.splatam import config as _base
import copy

config = copy.deepcopy(_base)
config["run_name"] = "freiburg1_desk_gradstale_consec_adam"
config["data"]["num_frames"] = 300

config["tracking"]["cuda_graph"]["enabled"] = False
config["tracking"].setdefault("iteration_graph", {})["enabled"] = False

config["tracking"]["grad_staleness"] = {
    "enabled": True,
    "frames": [50, 150, 250],
    "starts": list(range(2, 100, 2)),
    "offsets": [1],
    "usable_cos": 0.95,
    "out_path": "grad_staleness_adam_dense.jsonl",
}
