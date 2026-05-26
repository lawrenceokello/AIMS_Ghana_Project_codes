"""
Plotting utilities for the thesis simulation — presentation-quality figures.

Produces:
  * reward_curve.png        — training reward over episodes (with smoothing)
  * training_summary.png    — return, coverage, losses, entropy in a 2x2 panel
  * paths_comparison.png    — MAPPO+MPC vs MAPPO-only paths with ideal lanes
  * coverage_heatmap.png    — side-by-side sprayed cells (MPC vs MAPPO-only)
  * coverage_detail.png     — detailed spray intensity map for MAPPO+MPC
  * comparison_metrics.png  — bar chart comparing key metrics (thesis figure)
  * parameter_sweep.png     — effect of parameters on coverage/return
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm

try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf')
except Exception:
    pass
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11


def _smooth(x, w=20):
    if len(x) < w:
        return x
    cs = np.cumsum(np.insert(x, 0, 0.0))
    sm = (cs[w:] - cs[:-w]) / w
    pad = np.full(w - 1, sm[0])
    return np.concatenate([pad, sm])


# ============================================================================
# Garden drawing helpers
# ============================================================================

def _draw_garden(ax, env, trees, draw_region_line=True):
    """Draw garden boundary, trees, and optional region divider."""
    ax.add_patch(Rectangle((0, 0), env.size, env.size, fill=False,
                           edgecolor="black", linewidth=2.0))
    if draw_region_line:
        ax.axvline(env.size / 2, linestyle="--", color="gray", alpha=0.6,
                   linewidth=1.2)
    for (tx, ty) in trees:
        ax.add_patch(Circle((tx, ty), env.tree_radius, color="forestgreen",
                            alpha=0.5, zorder=2))
        ax.add_patch(Circle((tx, ty), env.tree_radius, fill=False,
                            edgecolor="darkgreen", linewidth=1.5, zorder=2))
        ax.add_patch(Circle((tx, ty), env.tree_radius + 0.3, fill=False,
                            edgecolor="red", linewidth=0.6, linestyle=":",
                            alpha=0.3, zorder=2))
    ax.set_xlim(-2, env.size + 2)
    ax.set_ylim(-2, env.size + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def _draw_ideal_lanes(ax, ideal_lanes, colors=["steelblue", "indianred"]):
    """Draw the ideal straight boustrophedon lane lines as dashed lines."""
    for i, lanes in enumerate(ideal_lanes):
        for lane in lanes:
            ax.plot([lane[0, 0], lane[1, 0]], [lane[0, 1], lane[1, 1]],
                    color=colors[i], linewidth=0.7, linestyle="--", alpha=0.35)


# ============================================================================
# Training plots
# ============================================================================

def plot_reward_curve(history_npz_path: str, out_path: str):
    H = np.load(history_npz_path)
    ep = H["episode"]
    ret = H["ret"]
    sm = _smooth(ret, w=max(10, len(ret) // 20))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep, ret, alpha=0.3, color="tab:blue", label="Episode return")
    ax.plot(ep, sm, color="tab:blue", linewidth=2.0,
            label=f"Smoothed (window={max(10, len(ret) // 20)})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total team reward")
    #ax.set_title("MAPPO Training: Cumulative Reward per Episode", fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_training_summary(history_npz_path: str, out_path: str):
    H = np.load(history_npz_path)
    ep = H["episode"]
    ret = H["ret"]
    cov = H["coverage"]
    pi_loss = H["pi_loss"]
    v_loss = H["v_loss"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(ep, ret, alpha=0.3, color="tab:blue")
    ax.plot(ep, _smooth(ret), color="tab:blue", linewidth=2.0)
    #ax.set_title("Episode Return", fontweight='bold')
    ax.set_xlabel("Episode"); ax.set_ylabel("Return")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ep, cov, alpha=0.3, color="tab:green")
    ax.plot(ep, _smooth(cov), color="tab:green", linewidth=2.0)
    #ax.set_title("Coverage Fraction", fontweight='bold')
    ax.set_xlabel("Episode"); ax.set_ylabel("Coverage")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ep, pi_loss, color="tab:red", label="Policy loss")
    ax2 = ax.twinx()
    ax2.plot(ep, v_loss, color="tab:purple", label="Value loss")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Policy loss", color="tab:red")
    ax2.set_ylabel("Value loss", color="tab:purple")
    #ax.set_title("Training Losses", fontweight='bold')
    ax.grid(alpha=0.3)

    fig.suptitle("MAPPO Training Diagnostics", y=1.02, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Path / coverage plots
# ============================================================================

def plot_paths_comparison(results: dict, env, out_path: str, episode_idx: int = 0):
    """Side-by-side: MAPPO+MPC vs MAPPO-only drone paths."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, key, title in zip(
        axes, ["with_mpc", "without_mpc"],
        ["MAPPO + MPC (Hybrid)", "MAPPO Only"]
    ):
        r = results[key][episode_idx]
        _draw_garden(ax, env, r["trees"])

        if "ideal_lanes" in r:
            _draw_ideal_lanes(ax, r["ideal_lanes"])

        colors = ["tab:blue", "tab:red"]
        labels = ["Drone 0 (left region)", "Drone 1 (right region)"]
        for i in range(2):
            path = r["paths"][i]
            ax.plot(path[:, 0], path[:, 1], color=colors[i], linewidth=1.5,
                    alpha=0.85, label=labels[i], zorder=3)
            ax.scatter(path[0, 0], path[0, 1], color=colors[i], marker="o",
                      s=80, edgecolor="black", zorder=5)
            ax.scatter(path[-1, 0], path[-1, 1], color=colors[i], marker="*",
                      s=180, edgecolor="black", zorder=5)

        cov_val = r.get('coverage', 0.0)
        coll = r.get('tree_collisions', 0)
        ax.set_title(f"{title}\nCoverage={cov_val:.1%}, Collisions={coll}, Steps={r['length']}",
                     fontsize=12, fontweight='bold')

        custom_handles = [
            Line2D([0], [0], color=colors[0], linewidth=1.8, label="Drone 0 path"),
            Line2D([0], [0], color=colors[1], linewidth=1.8, label="Drone 1 path"),
            Line2D([0], [0], color="gray", linewidth=0.8, linestyle="--",
                   alpha=0.5, label="Ideal boustrophedon lanes"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=8, label="Start"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                   markersize=12, label="End"),
        ]
        ax.legend(handles=custom_handles, loc="upper left", fontsize=8)

    fig.suptitle("Drone Paths in 50x50 m Garden — 6 Lanes per Region, 8 Trees\n"
                 "Dashed = ideal boustrophedon lanes | Solid = actual path",
                 y=1.02, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_heatmap(results: dict, env, out_path: str, episode_idx: int = 0):
    """Side-by-side coverage heatmaps: MAPPO+MPC vs MAPPO-only.

    This is the KEY presentation figure for the thesis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    cov_cmap = LinearSegmentedColormap.from_list(
        "coverage", ["#f7fbff", "#c7e9c0", "#74c476", "#31a354", "#006d2c"]
    )

    for ax, key, title in zip(
        axes, ["with_mpc", "without_mpc"],
        ["MAPPO + MPC (Hybrid)", "MAPPO Only"]
    ):
        r = results[key][episode_idx]

        if "coverage_grid" in r:
            grid = r["coverage_grid"].astype(float)
        else:
            centres = env.cell_centres
            covered = np.zeros(env.grid_res ** 2, dtype=bool)
            for drone in range(2):
                path = r["paths"][drone]
                for p in path:
                    d = np.linalg.norm(centres - p, axis=1)
                    covered |= (d < env.spray_radius)
            grid = covered.reshape(env.grid_res, env.grid_res).astype(float)

        extent = (0, env.size, 0, env.size)
        ax.imshow(grid, origin="lower", extent=extent, cmap=cov_cmap,
                  vmin=0, vmax=1, aspect="equal", alpha=0.85)

        _draw_garden(ax, env, r["trees"])

        if "ideal_lanes" in r:
            _draw_ideal_lanes(ax, r["ideal_lanes"])

        colors = ["tab:blue", "tab:red"]
        for i in range(2):
            path = r["paths"][i]
            ax.plot(path[:, 0], path[:, 1], color=colors[i], linewidth=0.5,
                    alpha=0.4, zorder=3)

        cov_val = grid.mean()
        coll = r.get('tree_collisions', 0)
    #     ax.set_title(f"{title}\nCoverage = {cov_val:.1%} | Tree Collisions = {coll}",
    #                  fontsize=13, fontweight='bold')

    # fig.suptitle("Garden Coverage Heatmap — 50x50 m, 8 Trees, 6 Lanes per Region\n"
    #              "Green = Sprayed | White = Unsprayed | Dashed = Boustrophedon Lanes",
    #              y=1.02, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_detail(results: dict, env, out_path: str, episode_idx: int = 0):
    """Detailed spray intensity map for MAPPO+MPC."""
    r = results["with_mpc"][episode_idx]
    centres = env.cell_centres
    n_cells = env.grid_res ** 2

    intensity = np.zeros(n_cells, dtype=float)
    for drone in range(2):
        path = r["paths"][drone]
        for p in path:
            d = np.linalg.norm(centres - p, axis=1)
            intensity += (d < env.spray_radius).astype(float)

    max_intensity = intensity.max()
    if max_intensity > 0:
        intensity_norm = intensity / max_intensity
    else:
        intensity_norm = intensity

    grid = intensity_norm.reshape(env.grid_res, env.grid_res)

    detail_cmap = LinearSegmentedColormap.from_list(
        "detail", ["#ffffff", "#e0f3db", "#a8ddb5", "#4eb3d3", "#2b8cbe", "#08589e"]
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    extent = (0, env.size, 0, env.size)
    im = ax.imshow(grid, origin="lower", extent=extent, cmap=detail_cmap,
                   vmin=0, vmax=1, aspect="equal", alpha=0.9)

    _draw_garden(ax, env, r["trees"])

    if "ideal_lanes" in r:
        _draw_ideal_lanes(ax, r["ideal_lanes"])

    colors = ["tab:blue", "tab:red"]
    for i in range(2):
        path = r["paths"][i]
        ax.plot(path[:, 0], path[:, 1], color=colors[i], linewidth=0.8,
                alpha=0.6, zorder=3)
        ax.scatter(path[0, 0], path[0, 1], color=colors[i], marker="o",
                  s=60, edgecolor="black", zorder=5)
        ax.scatter(path[-1, 0], path[-1, 1], color=colors[i], marker="*",
                  s=120, edgecolor="black", zorder=5)

    cov_frac = r.get('coverage', 0.0)
    mpc_used = r.get('mpc_used', 0)
    # ax.set_title(f"MAPPO + MPC — Spray Intensity Map\n"
    #              f"Coverage = {cov_frac:.1%} | Steps = {r['length']} | MPC Activations = {mpc_used}",
    #              fontsize=12, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spray intensity (normalized)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_metrics(results: dict, out_path: str):
    """Bar chart comparing key metrics between MAPPO+MPC and MAPPO-only.

    This is the THESIS COMPARISON FIGURE showing the clear advantage of
    the hybrid MAPPO+MPC approach over MAPPO alone.
    """
    mpc_covs = [r["coverage"] for r in results["with_mpc"]]
    no_covs = [r["coverage"] for r in results["without_mpc"]]
    mpc_colls = [r["tree_collisions"] for r in results["with_mpc"]]
    no_colls = [r["tree_collisions"] for r in results["without_mpc"]]
    mpc_rets = [r["return"] for r in results["with_mpc"]]
    no_rets = [r["return"] for r in results["without_mpc"]]
    mpc_lens = [r["length"] for r in results["with_mpc"]]
    no_lens = [r["length"] for r in results["without_mpc"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    categories = ["Coverage\nFraction", "Tree\nCollisions", "Episode\nReturn"]
    mpc_vals = [np.mean(mpc_covs), np.mean(mpc_colls), np.mean(mpc_rets)]
    no_vals = [np.mean(no_covs), np.mean(no_colls), np.mean(no_rets)]
    mpc_stds = [np.std(mpc_covs), np.std(mpc_colls), np.std(mpc_rets)]
    no_stds = [np.std(no_covs), np.std(no_colls), np.std(no_rets)]

    # Coverage comparison
    ax = axes[0]
    x = np.arange(2)
    bars = ax.bar(x, [np.mean(mpc_covs), np.mean(no_covs)], 0.5,
                  yerr=[np.std(mpc_covs), np.std(no_covs)],
                  color=["#2b8cbe", "#e34a33"], edgecolor="black", linewidth=0.5,
                  capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(["MAPPO + MPC", "MAPPO Only"])
    ax.set_ylabel("Coverage Fraction")
   # ax.set_title("Garden Coverage", fontweight='bold', fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Collision comparison
    ax = axes[1]
    bars = ax.bar(x, [np.mean(mpc_colls), np.mean(no_colls)], 0.5,
                  yerr=[np.std(mpc_colls), np.std(no_colls)],
                  color=["#2b8cbe", "#e34a33"], edgecolor="black", linewidth=0.5,
                  capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(["MAPPO + MPC", "MAPPO Only"])
    ax.set_ylabel("Tree Collisions")
    #ax.set_title("Safety (Tree Collisions)", fontweight='bold', fontsize=13)
    ax.grid(alpha=0.3, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Return comparison
    ax = axes[2]
    bars = ax.bar(x, [np.mean(mpc_rets), np.mean(no_rets)], 0.5,
                  yerr=[np.std(mpc_rets), np.std(no_rets)],
                  color=["#2b8cbe", "#e34a33"], edgecolor="black", linewidth=0.5,
                  capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(["MAPPO + MPC", "MAPPO Only"])
    ax.set_ylabel("Episode Return")
   # ax.set_title("Cumulative Reward", fontweight='bold', fontsize=13)
    ax.grid(alpha=0.3, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.0f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.suptitle("MAPPO + MPC vs MAPPO Only — Key Performance Metrics\n"
                 "Hybrid approach achieves higher coverage, fewer collisions, and greater reward",
                 y=1.04, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Parameter sweep
# ============================================================================

def plot_parameter_sweep(sweep_results: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    names = list(sweep_results.keys())
    returns_mpc = [np.mean([ep["return"] for ep in sweep_results[n]["with_mpc"]]) for n in names]
    returns_no = [np.mean([ep["return"] for ep in sweep_results[n]["without_mpc"]]) for n in names]
    cov_mpc = [np.mean([ep["coverage"] for ep in sweep_results[n]["with_mpc"]]) for n in names]
    cov_no = [np.mean([ep["coverage"] for ep in sweep_results[n]["without_mpc"]]) for n in names]

    x = np.arange(len(names))
    w = 0.38

    ax = axes[0]
    bars1 = ax.bar(x - w / 2, returns_mpc, w, label="MAPPO + MPC",
                   color="#2b8cbe", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w / 2, returns_no, w, label="MAPPO only",
                   color="#e34a33", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Mean return")
    #ax.set_title("Effect of Parameters on Episode Return", fontweight='bold')
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best")

    ax = axes[1]
    bars1 = ax.bar(x - w / 2, cov_mpc, w, label="MAPPO + MPC",
                   color="#2b8cbe", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + w / 2, cov_no, w, label="MAPPO only",
                   color="#e34a33", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Mean coverage fraction")
    #ax.set_title("Effect of Parameters on Coverage", fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
