# TUM fr2_xyz transfer of the frozen SplaTAM pose preconditioner.
#
# All optimizer, stopping, mapping and rendering settings come from
# splatam_precond.py. This wrapper changes only the dataset, frame count and
# output identity so fr2 cannot overwrite an fr1 experiment.
from configs.tum.splatam_precond import config as _base
import copy
import os

config = copy.deepcopy(_base)
_frames = int(os.environ.get("FRAMES", "100"))

config["data"]["gradslam_data_cfg"] = "./configs/data/TUM/freiburg2_xyz.yaml"
config["data"]["sequence"] = "rgbd_dataset_freiburg2_xyz"
config["data"]["end"] = _frames
config["data"]["num_frames"] = _frames
config["run_name"] = config["run_name"].replace(
    "freiburg1_desk", "freiburg2_xyz", 1
)
