# First 500 frames of a ScanNet scene - a usable reference point without the
# ~1h40 of a full 1807-frame run. Inherits everything from
# configs/scannet/splatam.py; only the frame count and run_name change, so
# tracking/mapping budgets stay identical to the full-length baseline.
#
# eval_every is deliberately left at the inherited 500. That keeps this run on
# the same evaluation protocol as the full one, but note what it means: the
# rendering metrics (PSNR, depth RMSE/L1, MS-SSIM, LPIPS) are computed only on
# frames where time_idx == 0 or (time_idx+1) % 500 == 0, i.e. just frames 0 and
# 499 here. Two frames is far too few to compare optimization arms on. ATE is
# unaffected - it is computed over every frame regardless of eval_every - so
# treat ATE as the metric this config produces and the rendering numbers as
# indicative only.
#
# Scene is chosen with SCENE_NUM as usual. Note the -inf pose block in
# scene0059_00 sits at frames 1460-1479, so a 500-frame run never reaches it.
from configs.scannet.splatam import config as _base
import copy

config = copy.deepcopy(_base)
config["run_name"] = config["run_name"] + "_500"
config["data"]["end"] = 500
config["data"]["num_frames"] = 500
