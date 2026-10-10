# eps + pixel sampling + gradfix, with sample_ratio relaxed 0.6 -> 0.75.
#
# Reference: splatam_es_graph_eps_pixel_gradfix.py measured 1391s (1.97x vs
# main), ATE 3.82cm, tracking 13.948 ms/iter, 66.3% fire rate at avg
# iteration 99, 74,059 iterations.
#
# The problem being addressed. At sample_ratio 0.6 tile masking cut
# per-iteration tracking cost 10.8% (15.642 -> 13.948 ms/iter, against a
# -0.3% mapping control, so a real effect) but suppressed the early-stopping
# fire rate 71.2% -> 66.3%, adding 7.2% more iterations. That ate two thirds
# of the win: -10.8% per iteration became -3.5% on wall time. Masked
# rendering makes the loss noisier, so it plateaus later - and the config
# retuned 4x and still fired less, so this is inherent rather than a stale
# threshold.
#
# The bet: masking 25% of tiles instead of 40% gives a less noisy loss, so
# early stopping should hold closer to its unmasked 71.2% rate. Less saved
# per iteration (~6-7% rather than 10.8%) but without the iteration penalty,
# which could net out better than 0.6 did.
#
# Also worth watching for ATE: 0.6 measured 3.82cm against eps's 3.52cm
# (+8.5%). Lighter masking should sit between them.
from configs.tum.splatam_es_graph_eps_pixel_gradfix import config as _base
import copy

config = copy.deepcopy(_base)
config["run_name"] = "freiburg1_desk_es_graph_eps_pixel075"
config["tracking"]["pixel_sample"]["sample_ratio"] = 0.75
