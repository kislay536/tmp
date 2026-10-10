# maskmul + fixed-capacity binning. CORRECTNESS GATE for the CUDA-graph work.
#
# Reference to match: splatam_es_graph_eps_pixel075_maskmul.py over four runs
# measured 13.105 / 13.149 / 13.153 / 13.160 ms/iter (0.4% spread) and ATE
# 3.45 / 3.58 / 3.71 / 3.64.
#
# What this exercises. The rasterizer normally reads num_rendered back to the
# host to size its binning buffer, the sort and one launch grid - a
# synchronous memcpy, and the one thing that makes the rasterizer impossible
# to record into a CUDA graph. With this on, the first warmup_iters
# iterations of each frame run the original readback path to MEASURE
# num_rendered, then the rest of the frame uses capacity = max * margin, with
# unused sort keys padded to sort to the end and an overflow flag checked
# once per frame instead of once per iteration.
#
# This is NOT expected to be faster. It sorts capacity rather than the exact
# count, so a small slowdown (~2% at margin 1.25) is the correct outcome. The
# point is that results must be UNCHANGED - if padding or sentinel handling is
# wrong, it shows up here rather than tangled together with graph capture.
#
# Pass condition: ATE ~3.5 and ms/iter within a few percent of 13.15.
#
# Watch the summary line. "0 overflow frames" means the margin is safe and
# could come down toward 1.1 to reclaim sort cost. Any overflow means that
# frame's render dropped instances and its result is invalid - the margin
# self-raises, but the run is then not a clean comparison.
from configs.tum.splatam_es_graph_eps_pixel075_maskmul import config as _base
import copy

config = copy.deepcopy(_base)
config["run_name"] = "freiburg1_desk_es_graph_eps_pixel075_maskmul_bincap"
config["tracking"]["binning_capacity"] = dict(
    enabled=True,
    margin=1.25,
    warmup_iters=3,
)
