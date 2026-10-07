#!/usr/bin/env python3
"""Generates the 5-plot set for the gradient-reuse N-sweep (baseline + N=1..4,
across room0/room1/fr1_desk), matching the original Sep 19 plot spec,
generalized from a binary baseline-vs-reuse comparison to the full 5-way sweep.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

# Fixed categorical order (dataviz skill default palette, slots 1-5), never
# reassigned across plots.
COLORS = {
    "baseline": "#2a78d6",  # blue
    "n1": "#eb6834",        # orange
    "n2": "#1baf7a",        # aqua
    "n3": "#eda100",        # yellow
    "n4": "#e87ba4",        # magenta
}
SCENE_TITLES = {"room0": "Replica room0", "room1": "Replica room1", "fr1_desk": "TUM fr1_desk"}

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
    df = pd.read_csv(path, usecols=[
        "frame_idx", "tracking_itr", "grad_computed", "grad_cosine_sim",
        "tau_norm", "ang_err_deg", "t_err_cm", "iter_wall_ms",
    ])
    RAW[key] = df
    print(f"  {key}: {len(df)} rows")

# ---------------------------------------------------------------------------
# Precompute per-frame final error and per-frame iteration counts.
# ---------------------------------------------------------------------------
FINAL = {}   # key -> DataFrame indexed by frame_idx, last row's t_err_cm/ang_err_deg
ITERS = {}   # key -> Series of iterations-per-frame
for key, df in RAW.items():
    g = df.groupby("frame_idx")
    FINAL[key] = g.tail(1).set_index("frame_idx")[["t_err_cm", "ang_err_deg"]]
    ITERS[key] = g.size()

# ===========================================================================
# Plot 1: headline comparison -- final per-frame error, 5 conditions x 3 scenes
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, metric, ylabel in zip(axes, ["t_err_cm", "ang_err_deg"], ["Translation error (cm)", "Rotation error (deg)"]):
    positions = []
    data = []
    colors = []
    xticks = []
    xticklabels = []
    pos = 0
    for scene in SCENES:
        group_start = pos
        for cond in CONDITIONS:
            vals = FINAL[(scene, cond)][metric].values
            vals = vals[np.isfinite(vals)]
            data.append(vals)
            positions.append(pos)
            colors.append(COLORS[cond])
            pos += 1
        xticks.append((group_start + pos - 1) / 2)
        xticklabels.append(SCENE_TITLES[scene])
        pos += 1.5  # gap between scene groups

    bp = ax.boxplot(
        data, positions=positions, widths=0.8, patch_artist=True,
        showfliers=False, medianprops={"color": "#0b0b0b", "linewidth": 1.4},
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
        patch.set_edgecolor("#0b0b0b")
        patch.set_linewidth(0.6)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.grid(axis="x", visible=False)

handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[c], edgecolor="#0b0b0b", linewidth=0.6) for c in CONDITIONS]
fig.legend(handles, ["baseline (real grad every iter)"] + [f"reuse N={n[1:]}" for n in CONDITIONS[1:]],
           loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04), frameon=False)
fig.suptitle("Final per-frame tracking error: baseline vs. gradient-reuse N=1..4", y=1.12, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "1_headline_final_error.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 1_headline_final_error.png")

# ===========================================================================
# Plot 2: convergence behavior -- CDF of iterations-needed-per-frame
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, scene in zip(axes, SCENES):
    for cond in CONDITIONS:
        vals = np.sort(ITERS[(scene, cond)].values)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        label = "baseline" if cond == "baseline" else f"N={cond[1:]}"
        ax.plot(vals, cdf, color=COLORS[cond], linewidth=1.8, label=label)
    ax.axvline(100, color="#8a8980", linewidth=1, linestyle=(0, (4, 3)))
    ax.text(100, 0.03, " 100-iter cap", color="#8a8980", fontsize=8, va="bottom")
    ax.set_title(SCENE_TITLES[scene], fontsize=11)
    ax.set_xlabel("Iterations to converge (or cap)")
axes[0].set_ylabel("Cumulative fraction of frames")
axes[0].legend(loc="upper left", frameon=False, fontsize=9)
fig.suptitle("Convergence behavior: iterations needed per frame", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "2_convergence_cdf.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 2_convergence_cdf.png")

# ===========================================================================
# Plot 3: gradient direction stability (baseline only) -- cosine similarity
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, scene in zip(axes, SCENES):
    df = RAW[(scene, "baseline")]
    vals = df["grad_cosine_sim"].dropna().values
    ax.hist(vals, bins=60, range=(-1, 1), color=COLORS["baseline"], alpha=0.85, edgecolor="white", linewidth=0.3)
    med = np.median(vals)
    ax.axvline(med, color="#0b0b0b", linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(med, ax.get_ylim()[1] * 0.92, f" median={med:.2f}", fontsize=8, color="#0b0b0b")
    ax.set_title(SCENE_TITLES[scene], fontsize=11)
    ax.set_xlabel("Cosine similarity, consecutive real gradients")
axes[0].set_ylabel("Iteration-pairs")
fig.suptitle("Gradient direction stability (baseline trace) -- the mechanism behind N-sweep degradation", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "3_gradient_cosine_similarity.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 3_gradient_cosine_similarity.png")

# ===========================================================================
# Plot 4: example trajectory -- one room0 frame, tau_norm & t_err_cm per-iter
# ===========================================================================
EXAMPLE_SCENE = "room0"
EXAMPLE_FRAME = 1000
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for cond in CONDITIONS:
    df = RAW[(EXAMPLE_SCENE, cond)]
    sub = df[df["frame_idx"] == EXAMPLE_FRAME].sort_values("tracking_itr")
    label = "baseline" if cond == "baseline" else f"N={cond[1:]}"
    axes[0].plot(sub["tracking_itr"], sub["tau_norm"], color=COLORS[cond], linewidth=1.6, label=label)
    axes[1].plot(sub["tracking_itr"], sub["t_err_cm"], color=COLORS[cond], linewidth=1.6, label=label)
axes[0].set_xlabel("Tracking iteration")
axes[0].set_ylabel("‖Adam step‖ (tau_norm)")
axes[1].set_xlabel("Tracking iteration")
axes[1].set_ylabel("Translation error (cm)")
axes[0].legend(loc="upper right", frameon=False, fontsize=9)
fig.suptitle(f"Example trajectory: {SCENE_TITLES[EXAMPLE_SCENE]}, frame {EXAMPLE_FRAME}", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "4_example_trajectory.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 4_example_trajectory.png")

# ===========================================================================
# Plot 5: cost-benefit -- speedup% vs accuracy-degradation multiplier
# ===========================================================================
SCENE_COLORS = {"room0": "#2a78d6", "room1": "#eb6834", "fr1_desk": "#1baf7a"}
fig, ax = plt.subplots(figsize=(7.5, 6))
for scene in SCENES:
    base_time = RAW[(scene, "baseline")]["iter_wall_ms"].sum()
    base_ate = FINAL[(scene, "baseline")]["t_err_cm"].pow(2).mean() ** 0.5  # frame-final RMSE proxy
    xs, ys = [], []
    for cond in CONDITIONS[1:]:
        t = RAW[(scene, cond)]["iter_wall_ms"].sum()
        a = FINAL[(scene, cond)]["t_err_cm"].pow(2).mean() ** 0.5
        speedup_pct = 100.0 * (1 - t / base_time)
        degrade_x = a / base_ate
        xs.append(speedup_pct)
        ys.append(degrade_x)
    ax.plot(xs, ys, color=SCENE_COLORS[scene], linewidth=1.6, marker="o", markersize=7, label=SCENE_TITLES[scene])
    for n_label, x, y in zip(["N=1", "N=2", "N=3", "N=4"], xs, ys):
        ax.annotate(n_label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8, color="#52514e")
ax.axhline(1.0, color="#8a8980", linewidth=1, linestyle=(0, (4, 3)))
ax.set_xlabel("Wall-clock speedup vs. baseline (%)")
ax.set_ylabel("Accuracy-degradation multiplier (RMSE ratio vs. baseline)")
ax.set_yscale("log")
ax.legend(loc="upper left", frameon=False)
ax.set_title("Cost-benefit: is any N worth it?", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "5_cost_benefit.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved 5_cost_benefit.png")

print("\nAll 5 plots written to", OUT_DIR)
