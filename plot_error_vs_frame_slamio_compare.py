#!/usr/bin/env python3
"""Same frame-vs-error distribution+mean-line plot as plot_error_vs_frame.py,
but comparing the splatonic-monogs N-sweep (this session's own alternate-
iteration gradient reuse) against slamio-final's own GradReuse mechanism,
for whichever (scene, N) pairs have completed so far. Conditions with no
data yet are skipped, not faked.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLATONIC_DIR = "/scratch/sws0/user/karya/3dgs/tmp/csv"
SLAMIO_DIR = "/scratch/sws0/user/karya/3dgs/tmp/csv_slamio"
OUT_DIR = "/scratch/sws0/user/karya/3dgs/tmp/plots"
os.makedirs(OUT_DIR, exist_ok=True)

SCENES = ["room0", "room1", "fr1_desk"]
SCENE_TITLES = {"room0": "Replica room0", "room1": "Replica room1", "fr1_desk": "TUM fr1_desk"}
NS = ["baseline", "n1", "n2", "n3", "n4"]
NLABEL = {"baseline": "baseline", "n1": "N=1", "n2": "N=2", "n3": "N=3", "n4": "N=4"}

# source -> (dir, filename-builder, display color/style)
SPLATONIC_FILES = {
    ("room0", "baseline"): "room0_trace_baseline.csv", ("room0", "n1"): "room0_reuse_n1.csv",
    ("room0", "n2"): "room0_reuse_n2.csv", ("room0", "n3"): "room0_reuse_n3.csv", ("room0", "n4"): "room0_reuse_n4.csv",
    ("room1", "baseline"): "room1_trace_baseline.csv", ("room1", "n1"): "room1_reuse_n1.csv",
    ("room1", "n2"): "room1_reuse_n2.csv", ("room1", "n3"): "room1_reuse_n3.csv", ("room1", "n4"): "room1_reuse_n4.csv",
    ("fr1_desk", "baseline"): "fr1_desk_trace_baseline.csv", ("fr1_desk", "n1"): "fr1_desk_reuse_n1.csv",
    ("fr1_desk", "n2"): "fr1_desk_reuse_n2.csv", ("fr1_desk", "n3"): "fr1_desk_reuse_n3.csv", ("fr1_desk", "n4"): "fr1_desk_reuse_n4.csv",
}
SLAMIO_FILES = {
    ("room0", "baseline"): "room0_slamio_baseline.csv", ("room0", "n1"): "room0_slamio_reuse_n1.csv",
    ("room0", "n2"): "room0_slamio_reuse_n2.csv", ("room0", "n3"): "room0_slamio_reuse_n3.csv", ("room0", "n4"): "room0_slamio_reuse_n4.csv",
    ("room1", "baseline"): "room1_slamio_baseline.csv", ("room1", "n1"): "room1_slamio_reuse_n1.csv",
    ("room1", "n2"): "room1_slamio_reuse_n2.csv", ("room1", "n3"): "room1_slamio_reuse_n3.csv", ("room1", "n4"): "room1_slamio_reuse_n4.csv",
    ("fr1_desk", "baseline"): "fr1_desk_slamio_baseline.csv", ("fr1_desk", "n1"): "fr1_desk_slamio_reuse_n1.csv",
    ("fr1_desk", "n2"): "fr1_desk_slamio_reuse_n2.csv", ("fr1_desk", "n3"): "fr1_desk_slamio_reuse_n3.csv", ("fr1_desk", "n4"): "fr1_desk_slamio_reuse_n4.csv",
}

COLORS = {"baseline": "#2a78d6", "n1": "#eb6834", "n2": "#1baf7a", "n3": "#eda100", "n4": "#e87ba4"}
NBINS = 60

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#c9c8c0",
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5de", "grid.linewidth": 0.6,
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})

def load(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, usecols=["frame_idx", "tracking_itr", "t_err_cm"])

def plot_band(ax, df, color, label, linestyle="-"):
    fmin, fmax = df["frame_idx"].min(), df["frame_idx"].max()
    bin_edges = np.linspace(fmin, fmax, NBINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.clip(np.digitize(df["frame_idx"].values, bin_edges) - 1, 0, NBINS - 1)
    err = df["t_err_cm"].values
    p10 = np.full(NBINS, np.nan); p90 = np.full(NBINS, np.nan); mean = np.full(NBINS, np.nan)
    for b in range(NBINS):
        vals = err[bin_idx == b]
        if len(vals) == 0:
            continue
        p10[b] = np.percentile(vals, 10); p90[b] = np.percentile(vals, 90); mean[b] = vals.mean()
    ax.fill_between(bin_centers, p10, p90, color=color, alpha=0.15, linewidth=0)
    ax.plot(bin_centers, mean, color=color, linewidth=2.0, linestyle=linestyle, label=label)

available = []
for scene in SCENES:
    for n in NS:
        sp = os.path.join(SPLATONIC_DIR, SPLATONIC_FILES[(scene, n)])
        sl = os.path.join(SLAMIO_DIR, SLAMIO_FILES[(scene, n)])
        available.append((scene, n, os.path.exists(sp), os.path.exists(sl)))

scenes_with_any_slamio = sorted(set(s for s, n, hassp, hassl in available if hassl))
print("Scenes with at least one completed slamio run:", scenes_with_any_slamio)
for s, n, hassp, hassl in available:
    print(f"  {s:10s} {n:8s} splatonic={'Y' if hassp else '.'}  slamio={'Y' if hassl else '.'}")

if not scenes_with_any_slamio:
    raise SystemExit("No slamio data available yet -- nothing to plot.")

fig, axes = plt.subplots(len(scenes_with_any_slamio), 1, figsize=(11, 4.6 * len(scenes_with_any_slamio)), squeeze=False)
axes = axes[:, 0]

for ax, scene in zip(axes, scenes_with_any_slamio):
    any_line = False
    for n in NS:
        sp_path = os.path.join(SPLATONIC_DIR, SPLATONIC_FILES[(scene, n)])
        sl_path = os.path.join(SLAMIO_DIR, SLAMIO_FILES[(scene, n)])
        sp_df = load(sp_path)
        sl_df = load(sl_path)
        if sp_df is not None:
            plot_band(ax, sp_df, COLORS[n], f"splatonic {NLABEL[n]}", linestyle="-")
            any_line = True
        if sl_df is not None:
            plot_band(ax, sl_df, COLORS[n], f"slamio {NLABEL[n]}", linestyle="--")
            any_line = True
    ax.set_title(SCENE_TITLES[scene], fontsize=11, loc="left")
    ax.set_ylabel("Translation error (cm)")
    ax.set_xlabel("Frame number")
    if any_line:
        ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=2)

fig.suptitle(
    "Tracking error vs. frame number: splatonic-monogs reuse (solid) vs. slamio-final GradReuse (dashed)\n"
    "Full N=1..4 sweep, baseline + N=1..4, all three scenes",
    fontsize=11, y=1.0,
)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "6_error_vs_frame_slamio_compare.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved", out_path)
