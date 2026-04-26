"""
FleetWatch — Before vs After Training Plots
Generates comparison plots from training_results.json (before) and
enhanced_training_results.json (after the grader/reward fixes).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Load data ────────────────────────────────────────────────────────────────

with open("training_results.json") as f:
    before = json.load(f)

with open("enhanced_training_results.json") as f:
    after = json.load(f)

before_rewards = before["episode_rewards"]
after_rewards  = after["episode_rewards"]

WINDOW = 5

def rolling_avg(rewards, w=WINDOW):
    out = []
    for i in range(len(rewards)):
        start = max(0, i - w + 1)
        out.append(np.mean(rewards[start:i+1]))
    return out

before_roll = rolling_avg(before_rewards)
after_roll  = rolling_avg(after_rewards)

TASK_NAMES = ["T1: Obvious", "T2: Pattern", "T3: Adversarial", "T4: Cascade", "T5: Collusion"]
COLORS     = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0"]

def task_avgs(data):
    tp = data["task_performance"]
    return [np.mean(tp[str(i)]) if tp[str(i)] else 0.0 for i in range(1, 6)]

before_task = task_avgs(before)
after_task  = task_avgs(after)

# ── Figure ───────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0f1117")

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

DARK_AX   = "#1a1d27"
GRID_COL  = "#2a2d3a"
TEXT_COL  = "#e0e0e0"
BEFORE_C  = "#ef5350"   # red
AFTER_C   = "#66bb6a"   # green
ACCENT    = "#42a5f5"   # blue

def style_ax(ax, title):
    ax.set_facecolor(DARK_AX)
    ax.set_title(title, color=TEXT_COL, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.6, alpha=0.8)

# ── 1. Training curves side-by-side ──────────────────────────────────────────

ax1 = fig.add_subplot(gs[0, 0])
ep_b = range(1, len(before_rewards) + 1)
ax1.plot(ep_b, before_rewards, alpha=0.25, color=BEFORE_C, linewidth=1)
ax1.plot(ep_b, before_roll,    color=BEFORE_C, linewidth=2.2, label=f"Rolling avg (w={WINDOW})")
ax1.axhline(np.mean(before_rewards), color=BEFORE_C, linestyle=":", linewidth=1.2,
            label=f"Mean: {np.mean(before_rewards):.3f}")
ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward")
ax1.set_ylim(-0.05, 1.05)
ax1.legend(fontsize=8, facecolor=DARK_AX, labelcolor=TEXT_COL, framealpha=0.8)
style_ax(ax1, "Before — Baseline Training (50 eps)")

ax2 = fig.add_subplot(gs[0, 1])
ep_a = range(1, len(after_rewards) + 1)
ax2.plot(ep_a, after_rewards, alpha=0.25, color=AFTER_C, linewidth=1)
ax2.plot(ep_a, after_roll,    color=AFTER_C, linewidth=2.2, label=f"Rolling avg (w={WINDOW})")
ax2.axhline(np.mean(after_rewards), color=AFTER_C, linestyle=":", linewidth=1.2,
            label=f"Mean: {np.mean(after_rewards):.3f}")
ax2.set_xlabel("Episode"); ax2.set_ylabel("Reward")
ax2.set_ylim(-0.05, 1.05)
ax2.legend(fontsize=8, facecolor=DARK_AX, labelcolor=TEXT_COL, framealpha=0.8)
style_ax(ax2, "After — Enhanced Training (75 eps)")

# ── 2. Per-task performance comparison ───────────────────────────────────────

ax3 = fig.add_subplot(gs[1, 0])
x = np.arange(len(TASK_NAMES))
w = 0.35
bars_b = ax3.bar(x - w/2, before_task, w, label="Before", color=BEFORE_C, alpha=0.85, edgecolor="white", linewidth=0.5)
bars_a = ax3.bar(x + w/2, after_task,  w, label="After",  color=AFTER_C,  alpha=0.85, edgecolor="white", linewidth=0.5)
for bar in bars_b:
    h = bar.get_height()
    if h > 0.01:
        ax3.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=7.5, color=TEXT_COL)
for bar in bars_a:
    h = bar.get_height()
    if h > 0.01:
        ax3.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=7.5, color=TEXT_COL)
ax3.set_xticks(x)
ax3.set_xticklabels(TASK_NAMES, rotation=30, ha="right", fontsize=8.5)
ax3.set_ylabel("Avg Reward"); ax3.set_ylim(0, 1.1)
ax3.legend(fontsize=9, facecolor=DARK_AX, labelcolor=TEXT_COL, framealpha=0.8)
style_ax(ax3, "Per-Task Performance: Before vs After")

# ── 3. Reward distribution comparison ────────────────────────────────────────

ax4 = fig.add_subplot(gs[1, 1])
bins = np.linspace(0, 1, 25)
ax4.hist(before_rewards, bins=bins, alpha=0.65, color=BEFORE_C, label="Before", edgecolor="white", linewidth=0.4)
ax4.hist(after_rewards,  bins=bins, alpha=0.65, color=AFTER_C,  label="After",  edgecolor="white", linewidth=0.4)
ax4.axvline(np.mean(before_rewards), color=BEFORE_C, linestyle="--", linewidth=1.5)
ax4.axvline(np.mean(after_rewards),  color=AFTER_C,  linestyle="--", linewidth=1.5)
ax4.set_xlabel("Reward"); ax4.set_ylabel("Frequency")
ax4.legend(fontsize=9, facecolor=DARK_AX, labelcolor=TEXT_COL, framealpha=0.8)
style_ax(ax4, "Reward Distribution Comparison")

# ── 4. Summary metrics bar chart ─────────────────────────────────────────────

ax5 = fig.add_subplot(gs[2, :])

metrics_labels = ["Mean Reward", "Best Reward", "Final Reward",
                  "Task4 Avg", "Task1 Avg", "Task2 Avg"]

def safe_mean(lst): return np.mean(lst) if lst else 0.0

before_vals = [
    np.mean(before_rewards),
    max(before_rewards),
    np.mean(before_rewards[-5:]),
    safe_mean(before["task_performance"]["4"]),
    safe_mean(before["task_performance"]["1"]),
    safe_mean(before["task_performance"]["2"]),
]
after_vals = [
    np.mean(after_rewards),
    max(after_rewards),
    np.mean(after_rewards[-5:]),
    safe_mean(after["task_performance"]["4"]),
    safe_mean(after["task_performance"]["1"]),
    safe_mean(after["task_performance"]["2"]),
]

x5 = np.arange(len(metrics_labels))
w5 = 0.38
b5 = ax5.bar(x5 - w5/2, before_vals, w5, label="Before", color=BEFORE_C, alpha=0.85, edgecolor="white", linewidth=0.5)
a5 = ax5.bar(x5 + w5/2, after_vals,  w5, label="After",  color=AFTER_C,  alpha=0.85, edgecolor="white", linewidth=0.5)

for bar, val in zip(b5, before_vals):
    ax5.text(bar.get_x() + bar.get_width()/2, val + 0.015, f"{val:.3f}",
             ha="center", va="bottom", fontsize=8.5, color=TEXT_COL, fontweight="bold")
for bar, val in zip(a5, after_vals):
    ax5.text(bar.get_x() + bar.get_width()/2, val + 0.015, f"{val:.3f}",
             ha="center", va="bottom", fontsize=8.5, color=TEXT_COL, fontweight="bold")

# Delta annotations
for i, (bv, av) in enumerate(zip(before_vals, after_vals)):
    delta = av - bv
    sign  = "+" if delta >= 0 else ""
    col   = AFTER_C if delta >= 0 else BEFORE_C
    ax5.text(i, max(bv, av) + 0.07, f"Δ {sign}{delta:.3f}",
             ha="center", fontsize=8, color=col, fontweight="bold")

ax5.set_xticks(x5)
ax5.set_xticklabels(metrics_labels, fontsize=10)
ax5.set_ylabel("Score"); ax5.set_ylim(0, 1.15)
ax5.legend(fontsize=10, facecolor=DARK_AX, labelcolor=TEXT_COL, framealpha=0.8)
style_ax(ax5, "Key Metrics Summary — Before vs After Improvements")

# ── Title ─────────────────────────────────────────────────────────────────────

fig.suptitle("FleetWatch AI — Training Analysis: Before vs After Grader Fixes",
             fontsize=15, fontweight="bold", color=TEXT_COL, y=0.98)

out_path = Path("before_after_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"✅  Saved → {out_path.resolve()}")
