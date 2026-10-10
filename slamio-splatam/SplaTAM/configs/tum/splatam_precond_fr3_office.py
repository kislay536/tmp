# TUM fr3/long_office_household transfer of the frozen SplaTAM pose
# preconditioner.
#
# All optimizer, stopping, mapping and rendering settings come from
# splatam_precond.py. This wrapper changes only the dataset, frame count and
# output identity so fr3 cannot overwrite an fr1 or fr2 experiment - exactly
# the shape of splatam_precond_fr2_xyz.py.
#
# PRE_LR IS NOT ESTABLISHED ON THIS SEQUENCE. It has never been run. The
# inherited 0.004 is TUM's value, and the only evidence that it carries within
# the TUM family is fr2_xyz, which used the same 0.004. That is one transfer,
# not a rule - the ladder records PRE_LR failing to transfer ACROSS datasets
# in both directions (0.004 is catastrophic on Replica, Replica's 0.001 is far
# under-stepped on TUM). Treat the first fr3 number as a calibration point,
# not a result, and check `step |d| mean` against the achieved ATE before
# quoting it.
from configs.tum.splatam_precond import config as _base
import copy
import os

config = copy.deepcopy(_base)
# 500 by default: fr3/long_office_household is ~2500 frames, so a full-length
# run is not comparable in cost to fr1_desk's 591 and is not what the suite
# asks for. FRAMES=-1 still gives the whole sequence.
_frames = int(os.environ.get("FRAMES", "500"))

config["data"]["gradslam_data_cfg"] = \
    "./configs/data/TUM/freiburg3_long_office_household.yaml"
config["data"]["sequence"] = "rgbd_dataset_freiburg3_long_office_household"
config["data"]["end"] = _frames
config["data"]["num_frames"] = _frames
# THE FRAME COUNT GOES IN run_name, unlike fr2's wrapper. run_name keys the
# output directory and params.npz, and this sequence is long enough that a
# 500-frame and a full-length run are different experiments; without the tag
# the second would overwrite the first and look like a replicate of it.
config["run_name"] = config["run_name"].replace(
    "freiburg1_desk", f"freiburg3_office_f{_frames}", 1
)
