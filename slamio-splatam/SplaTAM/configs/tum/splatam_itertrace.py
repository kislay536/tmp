# PER-ITERATION TRACE for the two convergence figures (plot_iter_trace.py).
#
#   python scripts/splatam.py configs/tum/splatam_itertrace.py
#   python ../profiling/plot_iter_trace.py experiments/TUM/freiburg1_desk_itertrace/iter_trace.jsonl
#
# Stock SplaTAM (Adam, 200 tracking iterations) with EARLY STOPPING OFF, so
# every frame runs to the cap. The "pose settled before the cap" figure is
# only meaningful if the cap is actually reached - with the stopper on, the
# trace would end where the stopper decided, which is the conclusion, not the
# evidence.
#
# NOT A TIMED RUN: the trace syncs once per iteration.
#
# Env overrides: FRAMES (default -1 = all 573), ITERS (default 200),
# ITER_TRACE_EVERY (default 5: every 5th frame is traced, ~115 frames spread
# over the whole sequence; all frames are still tracked).
from configs.tum.splatam import config as _base
import copy
import os

config = copy.deepcopy(_base)

_frames = int(os.environ.get("FRAMES", "-1"))
_iters = int(os.environ.get("ITERS", "200"))

config["run_name"] = (f"freiburg1_desk_itertrace_it{_iters}"
                      + (f"_f{_frames}" if _frames > 0 else ""))
config["data"]["num_frames"] = _frames
config["tracking"]["num_iters"] = _iters

config["tracking"]["early_stop"] = dict(config["tracking"]["early_stop"],
                                        enabled=False, log_signals=None)
config["tracking"]["cuda_graph"]["enabled"] = False
config["tracking"].setdefault("iteration_graph", {})["enabled"] = False

config["tracking"]["iter_trace"] = dict(
    out_path=os.path.join(config["workdir"], config["run_name"],
                          "iter_trace.jsonl"),
    every=5,
)
