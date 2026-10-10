#!/usr/bin/env python3
"""Parse a tee'd stdout log from splatam.py / splatam_sparse.py (SplaTAM family:
rtgs-splatam and splatonic-splatam) into the mlsys CSV row fields.

All numbers here come directly from lines splatam.py itself prints at the end of
a run (see utils/eval_helpers.py and the final block of scripts/splatam.py):

    Average Tracking/Iteration Time: <X> ms
    Average Tracking/Frame Time: <X> s
    Average Mapping/Iteration Time: <X> ms
    Average Mapping/Frame Time: <X> s
    Overall SLAM FPS (track+map): <X>
    Overall SLAM Wall Time: <X> s
    Final Average ATE RMSE: <X> cm
    Average PSNR: <X>

splatonic-splatam's scripts/splatam.py and scripts/splatam_sparse.py never got the
"Overall SLAM Wall Time"/"Overall SLAM FPS" summary block ported over from
rtgs-splatam (confirmed absent in both files) -- when that line is missing,
total_time_s falls back to the MLSYS_START_EPOCH/MLSYS_END_EPOCH markers that
mlsys/run_experiment.sh writes into the log immediately before/after invoking
the model script, i.e. wall-clock duration of the whole process (includes
Python/CUDA-context startup overhead, unlike the in-process timer used
elsewhere, which is a minor, acceptable overestimate for this fallback).

tracking/mapping_iters_per_frame are not printed directly -- they're derived
arithmetically from the frame-time/iter-time pair already printed
(iters_per_frame = frame_time_ms / iter_time_ms), which is exact as long as
every frame ran the same fixed iteration count (true for baseline splatam;
RTGS's early-stop means this is an average, not a per-frame constant, which is
the expected/correct interpretation of an "iters_per_frame" column anyway).
"""
import argparse
import re
import sys


PATTERNS = {
    "tracking_ms_per_iter": re.compile(r"Average Tracking/Iteration Time:\s*([\d.eE+-]+)\s*ms"),
    "tracking_s_per_frame": re.compile(r"Average Tracking/Frame Time:\s*([\d.eE+-]+)\s*s"),
    "mapping_ms_per_iter": re.compile(r"Average Mapping/Iteration Time:\s*([\d.eE+-]+)\s*ms"),
    "mapping_s_per_frame": re.compile(r"Average Mapping/Frame Time:\s*([\d.eE+-]+)\s*s"),
    "total_time_s": re.compile(r"Overall SLAM Wall Time:\s*([\d.eE+-]+)\s*s"),
    "ate_cm": re.compile(r"Final Average ATE RMSE:\s*([\d.eE+-]+)\s*cm"),
    "psnr_db": re.compile(r"^Average PSNR:\s*([\d.eE+-]+)\s*$"),
}


EPOCH_START = re.compile(r"MLSYS_START_EPOCH=(\d+)")
EPOCH_END = re.compile(r"MLSYS_END_EPOCH=(\d+)")


def extract(log_path: str) -> dict:
    found = {}
    start_epoch = end_epoch = None
    with open(log_path) as f:
        for line in f:
            for key, pat in PATTERNS.items():
                if key not in found:
                    m = pat.search(line)
                    if m:
                        found[key] = float(m.group(1))
            if start_epoch is None:
                m = EPOCH_START.search(line)
                if m:
                    start_epoch = int(m.group(1))
            if end_epoch is None:
                m = EPOCH_END.search(line)
                if m:
                    end_epoch = int(m.group(1))

    if "total_time_s" not in found and start_epoch is not None and end_epoch is not None:
        found["total_time_s"] = float(end_epoch - start_epoch)

    missing = [k for k in PATTERNS if k not in found]
    if missing:
        raise ValueError(f"{log_path}: could not find lines for: {missing}")

    tracking_ms_per_frame = found["tracking_s_per_frame"] * 1000.0
    mapping_ms_per_frame = found["mapping_s_per_frame"] * 1000.0
    tracking_iters_per_frame = tracking_ms_per_frame / found["tracking_ms_per_iter"]
    mapping_iters_per_frame = mapping_ms_per_frame / found["mapping_ms_per_iter"]

    return {
        "total_time_s": found["total_time_s"],
        "tracking_ms_per_frame": tracking_ms_per_frame,
        "tracking_ms_per_iter": found["tracking_ms_per_iter"],
        "mapping_ms_per_frame": mapping_ms_per_frame,
        "mapping_ms_per_iter": found["mapping_ms_per_iter"],
        "ate_cm": found["ate_cm"],
        "psnr_db": found["psnr_db"],
        "tracking_iters_per_frame": tracking_iters_per_frame,
        "mapping_iters_per_frame": mapping_iters_per_frame,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_path")
    args = ap.parse_args()
    try:
        metrics = extract(args.log_path)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    for k, v in metrics.items():
        print(f"{k}={v}")
