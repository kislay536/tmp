# splatam_es_graph.py with startup early-stop thresholds that can actually
# fire, so the frames before the first retune are not wasted.
#
# The problem this fixes. With retune_start defaulting to retune_every, the
# first retune lands at frame 114, so frames 0-113 (19% of the sequence) run
# on the static thresholds. Those were pose_eps=1e-4, and the criterion is an
# AND of loss-plateau and pose-plateau, so it could essentially never
# complete: a real run's own diagnostics measured a mean pose delta of
# 6.20e-04 (min 2.09e-05, max 5.29e-03) and tail_mass=1.00 against a 0.60
# threshold - i.e. the pose never settles on this scene. Early stopping was
# nominally enabled for those 114 frames and fired 0 times.
#
# What changed and why these values:
#   pose_eps 1e-4 -> 0.0   Disables the pose half of the AND. Not an
#                          arbitrary loosening - tail_mass=1.00 says the
#                          condition is unsatisfiable here, and every retune
#                          on this scene independently sets pose_eps to 0.
#   loss_eps 1e-4 -> 4e-3  Close to what retuning consistently fits from real
#                          probe data on this scene (~4.41e-3), rather than a
#                          value measurably ~40x too strict.
#
# Scope: retuning overwrites both values at frame 114 regardless of their
# starting point, so this changes behaviour for frames 0-113 only. Everything
# from frame 114 on is identical to splatam_es_graph.py.
#
# Expected effect. splatam_es_graph fired 343/591 frames overall, but with
# frames 0-113 firing 0%, the post-retune rate is really ~72%. If the early
# frames fire at a similar rate they save roughly 8-9k of the 77.7k tracking
# iterations, so ~1390s against splatam_es_graph's 1521s. An extrapolation.
#
# What to watch. min_iters=70 still applies, but these frames are exactly the
# thin-map window where stopping early is riskiest - a pose can look
# plateaued before it has genuinely converged. Compare ATE against
# splatam_es_graph's 3.54cm, and remember this config's family diverges in
# roughly 1 run of 3 (3.64 / 3.55 / 6.31), so one clean ATE is not proof.
from configs.tum.splatam_es_graph import config as _base
import copy

config = copy.deepcopy(_base)
config["run_name"] = "freiburg1_desk_es_graph_eps"

config["tracking"]["early_stop"]["pose_eps"] = 0.0
config["tracking"]["early_stop"]["loss_eps"] = 4e-3
