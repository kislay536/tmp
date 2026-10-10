# GRADIENT STALENESS PROBE ON PLAIN ADAM - EVERY CONSECUTIVE PAIR, MANY FRAMES.
#
# The dense configs (splatam_gradstale_adam_dense.py) only ever traced every
# OTHER consecutive pair (g1-g2, g3-g4, ...), skipping g2-g3 etc., because of
# a real bug in GradStalenessProbe.observe() - it checked "is this a new
# start" before "does this close the active window", so an iteration that was
# both silently dropped the close. Fixed in utils/grad_staleness.py (see
# utils/test_grad_staleness.py); starts can now be literally every integer.
#
# THIS ALSO SAMPLES 25 FRAMES (10, 20, ..., 250) INSTEAD OF 3 (50, 150, 250) -
# a much bigger n for the pooled histogram
# (profiling/plot_grad_staleness_histogram.py), at the cost of a
# proportionally longer run. Frame 0 is deliberately excluded: it is
# map-initialisation only on this model, with no tracking optimisation loop
# to trace.
#
# early_stop is disabled, same reasoning as splatam_gradstale_adam_dense_full.py:
# without it every one of these 25 frames would stop around iteration 70,
# throwing away most of starts=range(1,199)'s coverage.
#
# NOT A TIMED RUN, and the most expensive of the three Adam probes so far:
# 25 traced frames x up to ~199 extra forward+backward passes each, on top of
# num_frames=260 all running to the full 200-iteration cap (early_stop is a
# global switch, not a per-frame one - see splatam_gradstale_adam_dense_full.py's
# header for why num_frames is capped at 260 rather than left at 300).
from configs.tum.splatam_gradstale_adam_dense_full import config as _base
import copy

config = copy.deepcopy(_base)
config["run_name"] = "freiburg1_desk_gradstale_adam_allpairs"
config["tracking"]["grad_staleness"]["frames"] = list(range(10, 251, 10))
config["tracking"]["grad_staleness"]["starts"] = list(range(1, 200))
config["tracking"]["grad_staleness"]["out_path"] = \
    "grad_staleness_adam_allpairs.jsonl"
