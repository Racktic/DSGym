"""Regenerate fig4_step5_ace_delta_caveat_dual.png.

Fix: HARD subplot x-axis ends at step 4 (no empty step 5 tick).
EASY subplot still shows step 0-5.
"""

import matplotlib.pyplot as plt

# ---------- EASY split (Claude Sonnet 4.6 — public avg percentile) ----------
easy_steps_shared = [0, 1]
easy_baseline_step01 = [46.4, 48.8]  # step 0 baseline + step 1 v2 (shared)

# ACE has its own step 1
easy_ace_x = [0, 1, 2, 3, 4, 5, 6]
easy_ace_y = [46.4, 46.9, 54.7, 57.1, 58.2, 57.8, 58.8]

easy_delta_x = [0, 1, 2, 3, 4, 5, 6]
easy_delta_y = [46.4, 48.8, 58.9, 54.5, 58.1, 57.5, 55.9]

easy_caveat_x = [0, 1, 2, 3, 4, 5, 6]
easy_caveat_y = [46.4, 48.8, 60.2, 62.4, 56.6, 56.9, 57.4]

easy_dual_x = [0, 1, 2, 3, 4, 5, 6]
easy_dual_y = [46.4, 48.8, 61.8, 62.1, 59.2, 63.4, 64.7]

# ---------- HARD split (Claude Sonnet 4.6 — public avg percentile) ----------
hard_ace_x = [0, 1, 2, 3, 4]
hard_ace_y = [35.3, 37.8, 45.3, 47.9, 49.5]

hard_delta_x = [0, 1, 2, 3, 4]
hard_delta_y = [35.3, 42.2, 41.1, 42.4, 46.6]

hard_caveat_x = [0, 1, 2, 3, 4]
hard_caveat_y = [35.3, 42.2, 46.7, 47.3, 48.9]

hard_dual_x = [0, 1, 2, 3, 4]
hard_dual_y = [35.3, 42.2, 47.4, 49.4, 52.6]


# ---------- Plot ----------
fig, (ax_easy, ax_hard) = plt.subplots(1, 2, figsize=(14, 5))

# Styles
ace_style   = dict(color="grey",   linestyle=":",  marker="x", linewidth=1.8, markersize=7, label="ACE")
delta_style = dict(color="#ff7f0e", linestyle="--", marker="s", linewidth=1.8, markersize=7, label="grow-only")
caveat_style= dict(color="#1f77b4", linestyle="--", marker="^", linewidth=1.8, markersize=7, label="refine-only")
dual_style  = dict(color="#2ca02c", linestyle="-",  marker="D", linewidth=2.6, markersize=9, label="dual update (ours)")


def draw(ax, ace_x, ace_y, d_x, d_y, c_x, c_y, du_x, du_y, title, x_ticks):
    ax.plot(ace_x, ace_y, **ace_style)
    ax.plot(d_x, d_y, **delta_style)
    ax.plot(c_x, c_y, **caveat_style)
    ax.plot(du_x, du_y, **dual_style)

    # Numeric labels ONLY on dual update line (every point)
    for x, y in zip(du_x, du_y):
        ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color="#2ca02c", fontweight="bold")

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("public avg percentile")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"step {s}" for s in x_ticks])
    ax.set_xlim(x_ticks[0] - 0.3, x_ticks[-1] + 0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)


draw(ax_easy,
     easy_ace_x, easy_ace_y,
     easy_delta_x, easy_delta_y,
     easy_caveat_x, easy_caveat_y,
     easy_dual_x, easy_dual_y,
     title="EASY split",
     x_ticks=[0, 1, 2, 3, 4, 5, 6])

draw(ax_hard,
     hard_ace_x, hard_ace_y,
     hard_delta_x, hard_delta_y,
     hard_caveat_x, hard_caveat_y,
     hard_dual_x, hard_dual_y,
     title="HARD split",
     x_ticks=[0, 1, 2, 3, 4])

# Give HARD subplot extra headroom so the "52.6" dual-update label at step 4
# sits comfortably inside the plot area instead of touching the top frame.
ax_easy.set_ylim(top=68)
ax_hard.set_ylim(top=56)

fig.suptitle("Claude Sonnet 4.6 — public avg percentile across memory-update strategies",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])

out = "/home/bohanlyu/qixin/DSGym/dev_logs/0523_figures/fig4_step5_ace_delta_caveat_dual.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
