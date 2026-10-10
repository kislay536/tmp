# ALL OPTIMISATIONS OFF, for the "why two phases" traces
# (profiling/coupling/fig_phases.py). Same optimiser switch as
# configs/tum/splatam_precond.py (PRECOND=0 -> Adam, PRECOND=1 -> full matrix,
# PRE_DIAG_AFTER=N -> diagonal tail after N iterations), but every speed
# feature that file may turn on is forced OFF here, so the arms differ in the
# optimiser and nothing else:
#
#   tile masking / pixel sampling, early stopping, both CUDA graphs, binning
#   capacity, mask-multiply loss (tracking and mapping), adaptive mapping.
#
#   cd SplaTAM
#   ITERS=100 FRAMES=50 PRE_LR=0.004 ITER_TRACE_EVERY=1 \
#   PRECOND=0 ITER_TRACE=../results/coupling/traces/tum_f50_it100_adam.jsonl \
#       python scripts/splatam.py configs/tum/splatam_phases.py
#
# Not touched on purpose: clear_gaussian_grads (restores main's semantics, per
# the comment at its use in scripts/splatam.py) and map pruning (stock SplaTAM).
from configs.tum.splatam_precond import config as _base
import copy

config = copy.deepcopy(_base)
_t = config["tracking"]
_t["pixel_sample"]["enabled"] = False
_t["early_stop"]["enabled"] = False
_t["cuda_graph"]["enabled"] = False
_t.setdefault("iteration_graph", {})["enabled"] = False
_t["binning_capacity"]["enabled"] = False
_t["mask_multiply_loss"] = False
_t["sync_free_candidates"] = False
config["mapping"]["mask_multiply_loss"] = False
config.setdefault("adaptive_mapping", {})["enabled"] = False
if config["mapping"].get("adaptive_pruning"):
    config["mapping"]["adaptive_pruning"]["enabled"] = False
config["run_name"] += "_alloff"
