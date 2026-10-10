# pixel075 + mask_multiply_loss (the loss-mask sync removal, on its own).
#
# Reference: splatam_es_graph_eps_pixel075.py measured 1347s (2.03x vs main),
# ATE 3.46cm, PSNR 22.20, tracking 14.508 ms/iter, 72.9% fire rate.
#
# The tracking loss did `torch.abs(...)[mask].sum()` twice per iteration.
# Boolean-mask indexing is masked_select, whose output size depends on the
# data, so PyTorch reads the mask's element count back to the host to size the
# result - a device-to-host sync, mid-forward-pass, twice per iteration. This
# replaces it with (x * mask).sum(): static shape, no sync, no nonzero, no
# gather. It also drops the torch.tile that built the (3,H,W) colour mask,
# since (3,H,W) * (1,H,W) broadcasts.
#
# Verified locally that value and gradient match the indexed form exactly,
# including the broadcast, with an identical gradient nonzero pattern.
#
# This removes 2 of the loop's 5 syncs, versus 1 for sync_free_candidates - so
# it is the larger half of the sync experiment. Extra FLOPs (all pixels rather
# than the selected subset) are the right trade at ~40% GPU utilisation.
#
# Not bit-identical on GPU: summing over zero-padded values uses a different
# reduction order than summing a compacted array, so expect float-level
# differences and the usual atomicAdd run-to-run spread on top.
from configs.tum.splatam_es_graph_eps_pixel075 import config as _base
import copy

config = copy.deepcopy(_base)
config["run_name"] = "freiburg1_desk_es_graph_eps_pixel075_maskmul"
config["tracking"]["mask_multiply_loss"] = True
