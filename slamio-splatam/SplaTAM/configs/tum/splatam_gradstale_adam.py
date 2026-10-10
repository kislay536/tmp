# GRADIENT STALENESS PROBE ON PLAIN ADAM - not the preconditioner.
#
# Same probe, same frames/starts/offsets as splatam_gradstale.py, so the two
# jsonl outputs are directly comparable point-for-point. Only the optimizer
# differs: base is configs.tum.splatam (stock Adam, tracking_iters=200,
# early_stop min_iters=70), not splatam_precond. No preconditioner means no
# 6x6 curvature matrix, so cos_extrap/cos_secant will read n/a - that's
# expected, only cos_stale/mag_ratio are meaningful here.
#
# Not a timed run - every probe point is an extra forward+backward.
from configs.tum.splatam import config as _base
import copy

config = copy.deepcopy(_base)
config["run_name"] = "freiburg1_desk_gradstale_adam"
config["data"]["num_frames"] = 300

config["tracking"]["cuda_graph"]["enabled"] = False
config["tracking"].setdefault("iteration_graph", {})["enabled"] = False

config["tracking"]["grad_staleness"] = {
    "enabled": True,
    "frames": [50, 150, 250],
    "starts": [5, 15],
    "offsets": [1, 2, 4],
    "usable_cos": 0.95,
    "out_path": "grad_staleness_adam.jsonl",
}
