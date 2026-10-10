#!/usr/bin/env python3
"""Parse a Gaussian-SLAM family run's output directory + tee'd log (rtgs-gaussianslam
or splatonic-gaussianslam) into the mlsys CSV row fields.

Sources, in order of preference (exact over approximate):

  <output_path>/fps_metrics.json     -- {tracking_time_s, mapping_time_s, total_time_s,
                                          num_frames, num_mapping_frames} (from the
                                          instrumentation patch added to both repos'
                                          src/entities/gaussian_slam.py::run())
  <output_path>/ate_aligned.json     -- {"rmse": <meters>, ...} (Umeyama-aligned ATE)
  <output_path>/rendering_metrics.json -- {"psnr": <db>, ...}
  <output_path>/config.yaml          -- base tracking/mapping iteration counts
                                          (dumped by GaussianSLAM.__init__ at the
                                          START of the run, so always present even
                                          if the run crashed later)

tracking_ms_per_iter / tracking_iters_per_frame:
  - tracking_ms_per_iter is ALWAYS derived as tracking_ms_per_frame /
    tracking_iters_per_frame -- i.e. the real per-frame wall-clock time
    (Python harness + optimizer step + loss + rendering, everything)
    divided by the real iteration count. This is deliberate: it's the
    only apples-to-apples way to compare ms/iter between baseline and
    optimization runs.
  - splatonic-gaussianslam, if launched with SPLATONIC_MEASURE_TRACK_ITER_TIME=1,
    ALSO prints an exact "Average Tracking/Iteration Time: X ms (over N
    iterations, ...)" line -- but that X is deliberately NOT full
    wall-clock time (see src/entities/tracker.py's own comment: "pure
    rasterizer forward+backward GPU time only, no Python harness
    overhead"). Using it directly for tracking_ms_per_iter previously
    produced a false ~2.5-14x "speedup" vs baseline for every scene
    (baseline's ms/iter is wall-clock-derived, this X wasn't) --
    confirmed by tracking_ms_per_frame barely moving (5-15%) while this
    field implied 2.5-14x. Only N (the real iteration count) from that
    log line is still used now, for tracking_iters_per_frame -- more
    precise than the early-stop-line approximation below when available.
  - Otherwise (rtgs-gaussianslam always; splatonic-gaussianslam without the env
    var), iteration count approximated from config's base tracking.iterations,
    corrected using exact per-frame values recovered from RTGS's own
    "RTGS tracking early-stop at iter Y/X (frame F)" log lines where present
    (rtgs-gaussianslam only -- splatonic-gaussianslam's tracker has no early-stop
    mechanism). Frames without an early-stop line are assumed to have run the
    config's base iteration count; this ignores the rarer "Higher initial loss,
    increasing num_iters" doubling case, which is a known, documented
    approximation for this secondary/diagnostic column (the primary
    tracking_ms_per_frame figure is exact either way).

mapping_ms_per_iter / mapping_iters_per_frame:
  - Neither repo has a precise per-mapping-iteration timer, so both are always
    approximated from config's base mapping.iterations (the steady-state value;
    new-submap calls that use mapping.new_submap_iterations instead are rare and
    not separately accounted for).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import yaml


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract(output_path: str, log_path: str) -> dict:
    out = Path(output_path)
    fps = _load_json(out / "fps_metrics.json")
    ate = _load_json(out / "ate_aligned.json")
    render = _load_json(out / "rendering_metrics.json")
    with open(out / "config.yaml") as f:
        config = yaml.safe_load(f)

    num_frames = fps["num_frames"]
    num_tracked_frames = max(num_frames - 2, 1)  # frames 0,1 skip tracking
    num_mapping_frames = max(fps["num_mapping_frames"], 1)

    tracking_ms_per_frame = fps["tracking_time_s"] / num_tracked_frames * 1000.0
    mapping_ms_per_frame = fps["mapping_time_s"] / num_mapping_frames * 1000.0

    base_track_iters = config["tracking"]["iterations"]
    base_map_iters = config["mapping"]["iterations"]

    log_text = ""
    log_p = Path(log_path)
    if log_p.exists():
        log_text = log_p.read_text(errors="replace")

    # slamio-gaussianslam writes exact iteration totals straight into fps_metrics.json.
    exact_fps_iters = fps.get("tracking_iters_total")

    exact_iter_time = re.search(
        r"Average Tracking/Iteration Time:\s*([\d.eE+-]+)\s*ms\s*\(over\s*(\d+)\s*iterations",
        log_text,
    )
    if exact_fps_iters is not None:
        tracking_iters_per_frame = exact_fps_iters / num_tracked_frames
    elif exact_iter_time:
        total_track_iters = int(exact_iter_time.group(2))
        tracking_iters_per_frame = total_track_iters / num_tracked_frames
    else:
        early_stops = re.findall(
            r"RTGS tracking early-stop at iter (\d+)/(\d+) \(frame \d+\)", log_text
        )
        n_early = len(early_stops)
        sum_early = sum(int(y) for y, _x in early_stops)
        n_remaining = max(num_tracked_frames - n_early, 0)
        total_track_iters = sum_early + n_remaining * base_track_iters
        tracking_iters_per_frame = total_track_iters / num_tracked_frames
    # Always derived from wall-clock ms/frame, never from the GPU-kernel-only
    # "exact" print above -- see module docstring.
    tracking_ms_per_iter = tracking_ms_per_frame / tracking_iters_per_frame

    if fps.get("mapping_iters_total") is not None:
        mapping_iters_per_frame = fps["mapping_iters_total"] / num_mapping_frames
    else:
        mapping_iters_per_frame = float(base_map_iters)
    mapping_ms_per_iter = mapping_ms_per_frame / mapping_iters_per_frame

    return {
        "total_time_s": fps["total_time_s"],
        "tracking_ms_per_frame": tracking_ms_per_frame,
        "tracking_ms_per_iter": tracking_ms_per_iter,
        "mapping_ms_per_frame": mapping_ms_per_frame,
        "mapping_ms_per_iter": mapping_ms_per_iter,
        "ate_cm": ate["rmse"] * 100.0,
        "psnr_db": render["psnr"],
        "tracking_iters_per_frame": tracking_iters_per_frame,
        "mapping_iters_per_frame": mapping_iters_per_frame,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_path")
    ap.add_argument("log_path")
    args = ap.parse_args()
    try:
        metrics = extract(args.output_path, args.log_path)
    except (FileNotFoundError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    for k, v in metrics.items():
        print(f"{k}={v}")
