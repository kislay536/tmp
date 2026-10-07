#!/usr/bin/env python3
"""Replacement for Plot 1: x=frame number, y=error (cm), with the per-frame
*distribution* of error shown as a shaded percentile band (since a literal
per-frame scatter/boxplot is unreadable at ~2000 frames x ~90 iters x 5
conditions), plus a bold mean trend line on top. One combined axes per scene
with all 5 conditions (baseline + N=1..4) overlaid together -- each
condition keeps its own fixed palette color rather than literally all-red,
since 5 overlaid red lines would be indistinguishable; the user's "red line"
for a single-condition version becomes "this condition's own color" here.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_DIR = "/scratch/sws0/user/karya/3dgs/tmp/csv"
OUT_DIR = "/scratch/sws0/user/karya/3dgs/tmp/plots"
os.makedirs(OUT_DIR, exist_ok=True)

SCENES = ["room0", "room1", "fr1_desk"]
CONDITIONS = ["baseline", "n1", "n2", "n3", "n4"]
FILES = {
    ("room0", "baseline"): "room0_trace_baseline.csv",
    ("room0", "n1"): "room0_reuse_n1.csv",
    ("room0", "n2"): "room0_reuse_n2.csv",
    ("room0", "n3"): "room0_reuse_n3.csv",
    ("room0", "n4"): "room0_reuse_n4.csv",
    ("room1", "baseline"): "room1_trace_baseline.csv",
    ("room1", "n1"): "room1_reuse_n1.csv",
    ("room1", "n2"): "room1_reuse_n2.csv",
    ("room1", "n3"): "room1_reuse_n3.csv",
    ("room1", "n4"): "room1_reuse_n4.csv",
    ("fr1_desk", "baseline"): "fr1_desk_trace_baseline.csv",
    ("fr1_desk", "n1"): "fr1_desk_reuse_n1.csv",
    ("fr1_desk", "n2"): "fr1_desk_reuse_n2.csv",
    ("fr1_desk", "n3"): "fr1_desk_reuse_n3.csv",
    ("fr1_desk", "n4"): "fr1_desk_reuse_n4.csv",
}
COLORS = {
    "baseline": "#2a78d6",
    "n1": "#eb6834",
    "n2": "#1baf7a",
    "n3": "#eda100",
    "n4": "#e87ba4",
}
LABELS = {"baseline": "baseline", "n1": "N=1", "n2": "N=2", "n3": "N=3", "n4": "N=4"}
SCENE_TITLES = {"room0": "Replica room0", "room1": "Replica room1", "fr1_desk": "TUM fr1_desk"}
NBINS = 60

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c9c8c0",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e6e5de",
    "grid.linewidth": 0.6,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

print("Loading CSVs...")
RAW = {}
for key, fname in FILES.items():
    path = os.path.join(CSV_DIR, fname)
    df = pd.read_csv(path, usecols=["frame_idx", "tracking_itr", "t_err_cm"])
    RAW[key] = df
    print(f"  {key}: {len(df)} rows")

fig, axes = plt.subplots(3, 1, figsize=(11, 13), sharex=False)
for ax, scene in zip(axes, SCENES):
    fmin = min(RAW[(scene, c)]["frame_idx"].min() for c in CONDITIONS)
    fmax = max(RAW[(scene, c)]["frame_idx"].max() for c in CONDITIONS)
    bin_edges = np.linspace(fmin, fmax, NBINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for cond in CONDITIONS:
        df = RAW[(scene, cond)]
        bin_idx = np.digitize(df["frame_idx"].values, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, NBINS - 1)
        p10 = np.full(NBINS, np.nan)
        p90 = np.full(NBINS, np.nan)
        mean = np.full(NBINS, np.nan)
        err = df["t_err_cm"].values
        for b in range(NBINS):
            vals = err[bin_idx == b]
            if len(vals) == 0:
                continue
            p10[b] = np.percentile(vals, 10)
            p90[b] = np.percentile(vals, 90)
            mean[b] = vals.mean()

        ax.fill_between(bin_centers, p10, p90, color=COLORS[cond], alpha=0.15, linewidth=0)
        ax.plot(bin_centers, mean, color=COLORS[cond], linewidth=2.0, label=LABELS[cond])

    ax.set_title(SCENE_TITLES[scene], fontsize=11, loc="left")
    ax.set_ylabel("Translation error (cm)")
    ax.set_xlabel("Frame number")

axes[0].legend(loc="upper left", frameon=False, fontsize=9, ncol=5)
fig.suptitle(
    "Tracking error vs. frame number -- per-frame-bin distribution (shaded 10th-90th pct) "
    "+ mean trend, baseline vs. gradient-reuse N=1..4",
    fontsize=12, y=1.0,
)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "1_error_vs_frame.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved", out_path)
