#!/usr/bin/env python3
"""Parse a tee'd stdout log from MonoGS-family slam.py (rtgs-monogs,
splatonic-monogs) into the mlsys CSV row fields.

Core numbers come directly from lines slam.py's own eval helpers print at
the end of a run:

    Eval: Total time <X> s
    Eval: RMSE ATE [m] <X>          (printed multiple times mid-run; last
                                      occurrence is the final, authoritative
                                      one)
    Eval: mean psnr: <X>, ssim: ..., lpips: ...   (also printed twice --
                                      tracking-time and post-color-refinement;
                                      last occurrence is post-refinement)

tracking_ms_per_frame comes from one of two conventions depending on which
repo's fork produced the log (both are handled here, whichever is
present):
  - rtgs-monogs: exact per-frame instrumentation prints
    "Eval: Tracking ms/frame: X" directly, plus ms/iter and iters/frame.
  - splatonic-monogs (sparsekernel branch): prints
    "Eval: Tracking Total time X" + "Eval: Tracking FPS X" instead --
    tracking_ms_per_frame is derived as 1000/FPS; ms/iter and
    iters/frame are not available from this repo's instrumentation and
    stay blank.

mapping_ms_per_frame/mapping_ms_per_iter/mapping_iters_per_frame: only
rtgs-monogs prints these ("Eval: Mapping ms/frame: X" etc.), and only
when single_thread=True (in async mode, BackEnd.map() isn't synchronized
1:1 with frame processing, so the source deliberately never prints it).
Always blank for splatonic-monogs, which has no mapping-side
instrumentation at all.

Only total_time_s/ate_m/psnr_db are required -- everything else is
best-effort per repo, left blank when the source log doesn't have it
rather than fabricated.
"""
import argparse
import re
import sys


PATTERNS = {
    "total_time_s": re.compile(r"Eval: Total time\s+([\d.eE+-]+)"),
    "ate_m": re.compile(r"Eval: RMSE ATE \[m\]\s+([\d.eE+-]+)"),
    "psnr_db": re.compile(r"Eval: mean psnr:\s*([\d.eE+-]+)"),
    "tracking_ms_per_frame": re.compile(r"Eval: Tracking ms/frame:\s*([\d.eE+-]+)"),
    "tracking_ms_per_iter": re.compile(r"Eval: Tracking ms/iter:\s*([\d.eE+-]+)"),
    "tracking_iters_per_frame": re.compile(r"Eval: Tracking iters/frame:\s*([\d.eE+-]+)"),
    "mapping_ms_per_frame": re.compile(r"Eval: Mapping ms/frame:\s*([\d.eE+-]+)"),
    "mapping_ms_per_iter": re.compile(r"Eval: Mapping ms/iter:\s*([\d.eE+-]+)"),
    "mapping_iters_per_frame": re.compile(r"Eval: Mapping iters/frame:\s*([\d.eE+-]+)"),
    "tracking_total_time_s": re.compile(r"Eval: Tracking Total time\s+([\d.eE+-]+)"),
    "tracking_fps": re.compile(r"Eval: Tracking FPS\s+([\d.eE+-]+)"),
}

# These are printed multiple times over a run (each mid-run eval pass) --
# always take the LAST match, not the first.
LAST_MATCH_WINS = {"ate_m", "psnr_db"}

REQUIRED = {"total_time_s", "ate_m", "psnr_db"}
OPTIONAL = {
    "tracking_ms_per_frame", "tracking_ms_per_iter", "tracking_iters_per_frame",
    "mapping_ms_per_frame", "mapping_ms_per_iter", "mapping_iters_per_frame",
}


def extract(log_path: str) -> dict:
    found = {}
    with open(log_path) as f:
        for line in f:
            for key, pat in PATTERNS.items():
                m = pat.search(line)
                if not m:
                    continue
                if key in found and key not in LAST_MATCH_WINS:
                    continue
                found[key] = float(m.group(1))

    missing = REQUIRED - found.keys()
    if missing:
        raise ValueError(f"{log_path}: could not find lines for: {sorted(missing)}")

    if "tracking_ms_per_frame" not in found and found.get("tracking_fps", 0) > 0:
        found["tracking_ms_per_frame"] = 1000.0 / found["tracking_fps"]

    result = {
        "total_time_s": found["total_time_s"],
        "ate_cm": found["ate_m"] * 100.0,
        "psnr_db": found["psnr_db"],
    }
    for key in OPTIONAL:
        result[key] = found.get(key, "")
    return result


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
