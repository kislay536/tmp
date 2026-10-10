# GRADIENT STALENESS PROBE ON PLAIN ADAM - DENSE, EARLY STOP DISABLED.
#
# splatam_gradstale_adam_dense.py's first real run showed frames stopping at
# n=49/35/35 instead of the ~100 the dense starts (1,3,...,199) were meant to
# cover - early_stop's min_iters=70 (inherited from the base Adam config) cut
# every traced frame short before most of the configured starts ever fired.
# This variant disables early_stop entirely so every frame runs the full
# tracking_iters=200 budget, giving the dense trace its intended coverage.
#
# THIS CHANGES WHAT IS MEASURED, NOT JUST HOW MUCH OF IT. Consecutive-gradient
# behaviour late in a frame (iterations ~100-200) was never observed in the
# early-stop runs at all, since those frames never got there. Comparing this
# run's tail against the early-stop run's tail is not apples to apples - one
# has real optimizer behaviour there, the other has none.
from configs.tum.splatam_gradstale_adam_dense import config as _base
import copy

config = copy.deepcopy(_base)
config["run_name"] = "freiburg1_desk_gradstale_adam_dense_full"
config["tracking"]["early_stop"]["enabled"] = False
config["tracking"]["grad_staleness"]["out_path"] = \
    "grad_staleness_adam_dense_full.jsonl"

# early_stop is disabled for EVERY frame, not just the 3 traced ones (there is
# no per-frame override), so all 300 inherited frames would now run the full
# 200-iteration budget instead of stopping around 70 - a large, pointless cost
# increase since the probe only ever reads frames 50/150/250. Nothing past
# frame 250 is measured, so stop the sequence right after it.
config["data"]["num_frames"] = 260
