# Full sequence with ONLY early stopping + tracking CUDA-graph capture.
# Adaptive mapping, tile-mask pixel sampling and adaptive pruning are off.
#
# Sits between the two measured points on my-a100-vm-2:
#   main branch                          2739s   ATE 3.22cm
#   splatam_isolated_full(_graph)       ~2130s   ATE 3.56cm   (all opts off)
#   this config                            ?              <- early stop added
#
# Early stopping cuts iteration count without adding any per-iteration cost
# (pose_delta_norm and loss.item() are computed unconditionally in the
# tracking loop regardless of whether it is enabled), so this should land
# clearly below the isolated number - roughly 1700s if it fires at the ~50%
# / avg-iteration-100 rate seen on the other machine. That is an
# extrapolation, not a result.
#
# Early-stop thresholds are inherited as-is from splatam.py: 1e-4/1e-4 with
# periodic retuning every 100 frames. Those fire 0% of the time until the
# first retune at frame ~114 loosens loss_eps, after which they fire around
# half the time - so the first ~19% of the sequence runs the full 200
# iterations by design.
#
# Caveat: the ATE from a single run of this config is not trustworthy. With
# early stopping and retuning active, two runs of equivalent code produced
# 3.64cm and 6.31cm - retuning fits its thresholds from live loss data, so
# runs can diverge. The isolated configs (early stopping off) have been
# stable by comparison. Read the wall time here; treat the ATE as
# provisional until that instability is characterised.
from configs.tum.splatam import config as _base
import copy

config = copy.deepcopy(_base)
config["run_name"] = "freiburg1_desk_es_graph"

config["tracking"]["early_stop"]["enabled"] = True   # kept (already on in the base)
config["tracking"]["cuda_graph"]["enabled"] = True   # kept

config["adaptive_mapping"]["enabled"] = False
config["tracking"]["pixel_sample"]["enabled"] = False
config["mapping"]["adaptive_pruning"]["enabled"] = False
