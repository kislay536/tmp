# splatam_es_graph_eps.py + tile-mask pixel sampling + clear_gaussian_grads.
#
# Both changes on top of the stable eps config (1442s, 1.90x vs main, ATE
# 3.38/3.52 across two runs). See splatam_es_graph_eps_pixel.py for the tile
# masking rationale and splatam_es_graph_eps_gradfix.py for the leak.
#
# Note this bundles two changes, so a bad result will not say which caused it.
# Run splatam_es_graph_eps_pixel.py too if you need to attribute.
#
# On the leak fix: it did NOT rescue the marginal oneshot config - four runs
# at ~92% fire rate gave ATE 3.81 / 8.61 unfixed and 3.73 / 6.49 fixed, so
# the instability there is structural rather than caused by the leaked
# gradients. It is included here because it restores main's semantics and is
# correct on its own terms, not because it is expected to change ATE.
from configs.tum.splatam_es_graph_eps import config as _base
import copy

config = copy.deepcopy(_base)
config["run_name"] = "freiburg1_desk_es_graph_eps_pixel_gradfix"
config["tracking"]["pixel_sample"]["enabled"] = True
config["tracking"]["clear_gaussian_grads"] = True
