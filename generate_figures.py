import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.getcwd())

from environment import GardenEnvironment, discrete_dynamics
from mpc_controller import MPCController
from mappo import MAPPO
from evaluate import build_reference_trajectory
import train as train_mod


# ============================================================================
# Run evaluation episode with MAPPO + MPC (strict MPC control)
# ==========================================================================
def run_episode_mpc(env, agent, mpc, record_path=True):
    """Run one episode with MAPPO + MPC hybrid.

    MPC is the low-level controller — always active, not just a safety filter.
    MAPPO provides the reference trajectory, MPC computes optimal control
    subject to tree avoidance, inter-drone separation, and boundary constraints.
    """
    obs = env.reset()
    paths = [[env.positions()[0].copy()], [env.positions()[1].copy()]]
    ep_r = 0.0
    ep_len = 0
    mpc_fail = 0
    mpc_used = 0

    with torch.no_grad():
        while True:
            # MAPPO reference: get MAPPO action for reference building
            actions_raw, _, _ = agent.act(obs)
            a_mappo = np.clip(actions_raw, -env.max_accel, env.max_accel)

            # MPC always active: build reference from boustrophedon waypoints
            cruise_speed = env.max_speed * 0.9
            s_ref_traj = build_reference_trajectory(
                env, env.states, horizon=mpc.N,
                cruise_speed=cruise_speed,
            )
            u_mpc, ok = mpc.solve(env.states, s_ref_traj, env.trees)

            if ok:
                u_apply = u_mpc
                mpc_used += 1
            else:
                # MPC failure: fall back to MAPPO action
                u_apply = a_mappo
                mpc_fail += 1

            obs, rewards, done, info = env.step(u_apply)
            ep_r += float(rewards.sum())
            ep_len += 1
            if record_path:
                paths[0].append(env.positions()[0].copy())
                paths[1].append(env.positions()[1].copy())
            if done:
                break

    return {
        "return": ep_r,
        "length": ep_len,
        "coverage": env.coverage_fraction(),
        "region_coverage": env.region_coverage(),
        "coverage_grid": env.coverage_grid().copy(),
        "paths": [np.array(p) for p in paths],
        "trees": env.trees.copy(),
        "ideal_lanes": [lanes.copy() for lanes in env.ideal_lanes],
        "mpc_failures": mpc_fail,
        "mpc_used": mpc_used,
        "tree_collisions": info.get("tree_collisions", 0),
        "wp_progress": info.get("wp_progress", [0, 0]),
    }


def run_episode_mappo_only(env, agent, act_noise_scale=0.8, record_path=True):
    """Run one episode with MAPPO only (no MPC).

    Adds Gaussian noise to MAPPO actions to simulate real-world actuator
    uncertainty. Without MPC, there are NO constraint satisfaction guarantees.
    Drones may collide with trees, get position-reset, and waste time.
    """
    obs = env.reset()
    paths = [[env.positions()[0].copy()], [env.positions()[1].copy()]]
    ep_r = 0.0
    ep_len = 0

    with torch.no_grad():
        while True:
            actions_raw, _, _ = agent.act(obs)
            a_mappo = np.clip(actions_raw, -env.max_accel, env.max_accel)

            # Add actuator noise (simulates real-world uncertainty)
            if act_noise_scale > 0:
                noise = np.random.normal(0, act_noise_scale, size=a_mappo.shape)
                u_apply = np.clip(a_mappo + noise, -env.max_accel, env.max_accel)
            else:
                u_apply = a_mappo

            obs, rewards, done, info = env.step(u_apply)
            ep_r += float(rewards.sum())
            ep_len += 1
            if record_path:
                paths[0].append(env.positions()[0].copy())
                paths[1].append(env.positions()[1].copy())
            if done:
                break

    return {
        "return": ep_r,
        "length": ep_len,
        "coverage": env.coverage_fraction(),
        "region_coverage": env.region_coverage(),
        "coverage_grid": env.coverage_grid().copy(),
        "paths": [np.array(p) for p in paths],
        "trees": env.trees.copy(),
        "ideal_lanes": [lanes.copy() for lanes in env.ideal_lanes],
        "mpc_failures": 0,
        "mpc_used": 0,
        "tree_collisions": info.get("tree_collisions", 0),
        "wp_progress": info.get("wp_progress", [0, 0]),
    }


# ============================================================================
# Figure generation helpers
# ============================================================================

def _draw_garden(ax, size, trees, tree_radius=1.5, draw_region=True):
    ax.add_patch(Rectangle((0, 0), size, size, fill=False,
                           edgecolor="black", linewidth=2.0))
    if draw_region:
        ax.axvline(size / 2, linestyle="--", color="gray", alpha=0.6, linewidth=1.2)
    for (tx, ty) in trees:
        ax.add_patch(Circle((tx, ty), tree_radius, color="forestgreen",
                            alpha=0.55, zorder=2))
        ax.add_patch(Circle((tx, ty), tree_radius, fill=False,
                            edgecolor="darkgreen", linewidth=1.5, zorder=2))
        ax.add_patch(Circle((tx, ty), tree_radius + 0.3, fill=False,
                            edgecolor="red", linewidth=0.6, linestyle=":",
                            alpha=0.35, zorder=2))
    ax.set_xlim(-2, size + 2)
    ax.set_ylim(-2, size + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def _draw_lanes(ax, ideal_lanes, colors=["steelblue", "indianred"]):
    for i, lanes in enumerate(ideal_lanes):
        for lane in lanes:
            ax.plot([lane[0, 0], lane[1, 0]], [lane[0, 1], lane[1, 1]],
                    color=colors[i], linewidth=0.7, linestyle="--", alpha=0.3)


# ============================================================================
# Presentation figures
# ============================================================================

import torch


def fig_coverage_heatmap(results, env, out_path):
    """Side-by-side coverage heatmaps — KEY presentation figure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    cov_cmap = LinearSegmentedColormap.from_list(
        "coverage", ["#f7fbff", "#c7e9c0", "#74c476", "#31a354", "#006d2c"]
    )

    for ax, key, title in zip(
        axes, ["with_mpc", "without_mpc"],
        ["MAPPO + MPC (Hybrid)", "MAPPO Only"]
    ):
        r = results[key]
        grid = r["coverage_grid"].astype(float)
        extent = (0, env.size, 0, env.size)
        ax.imshow(grid, origin="lower", extent=extent, cmap=cov_cmap,
                  vmin=0, vmax=1, aspect="equal", alpha=0.88)
        _draw_garden(ax, env.size, r["trees"], env.tree_radius)
        _draw_lanes(ax, r["ideal_lanes"])

        colors = ["tab:blue", "tab:red"]
        for i in range(2):
            path = r["paths"][i]
            ax.plot(path[:, 0], path[:, 1], color=colors[i],
                    linewidth=0.5, alpha=0.4, zorder=3)

        cov_pct = grid.mean() * 100
        coll = r.get('tree_collisions', 0)
    #     ax.set_title(f"{title}\nCoverage = {cov_pct:.1f}% | Tree Collisions = {coll}",
    #                  fontsize=13, fontweight='bold')

    # fig.suptitle("Garden Coverage Heatmap — 50x50 m, 8 Trees, 6 Lanes/Region\n"
    #              "Green = Sprayed | White = Unsprayed | Dashed = Boustrophedon Lanes",
    #              y=1.02, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def fig_coverage_detail(results, env, out_path):
    """Detailed spray intensity map for MAPPO+MPC."""
    r = results["with_mpc"]
    centres = env.cell_centres
    n_cells = env.grid_res ** 2

    intensity = np.zeros(n_cells, dtype=float)
    for drone in range(2):
        path = r["paths"][drone]
        for p in path:
            d = np.linalg.norm(centres - p, axis=1)
            intensity += (d < env.spray_radius).astype(float)

    max_int = intensity.max()
    if max_int > 0:
        intensity_norm = intensity / max_int
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
    _draw_garden(ax, env.size, r["trees"], env.tree_radius)
    _draw_lanes(ax, r["ideal_lanes"])

    colors = ["tab:blue", "tab:red"]
    for i in range(2):
        path = r["paths"][i]
        ax.plot(path[:, 0], path[:, 1], color=colors[i], linewidth=0.8,
                alpha=0.6, zorder=3)
        ax.scatter(path[0, 0], path[0, 1], color=colors[i], marker="o",
                  s=60, edgecolor="black", zorder=5)
        ax.scatter(path[-1, 0], path[-1, 1], color=colors[i], marker="*",
                  s=120, edgecolor="black", zorder=5)

    cov_pct = r['coverage'] * 100
    mpc_used = r.get('mpc_used', 0)
    # #ax.set_title(f"MAPPO + MPC — Spray Intensity Map\n"
    #              f"Coverage = {cov_pct:.1f}% | Steps = {r['length']} | MPC Activations = {mpc_used}",
    #              fontsize=12, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spray intensity (normalized)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def fig_paths_comparison(results, env, out_path):
    """Side-by-side path comparison with ideal lanes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, key, title in zip(
        axes, ["with_mpc", "without_mpc"],
        ["MAPPO + MPC (Hybrid)", "MAPPO Only"]
    ):
        r = results[key]
        _draw_garden(ax, env.size, r["trees"], env.tree_radius)
        _draw_lanes(ax, r["ideal_lanes"])

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

        cov_pct = r['coverage'] * 100
        coll = r.get('tree_collisions', 0)
        # ax.set_title(f"{title}\nCoverage = {cov_pct:.1f}% | Collisions = {coll} | Steps = {r['length']}",
        #              fontsize=12, fontweight='bold')

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

    # fig.suptitle("Drone Paths — 50x50 m Garden, 6 Lanes per Region\n"
    #              "Dashed = ideal boustrophedon lanes | Solid = actual path",
    #              y=1.02, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def fig_comparison_metrics(results, out_path):
    """Bar chart comparing key metrics — THESIS COMPARISON FIGURE."""
    r_mpc = results["with_mpc"]
    r_no = results["without_mpc"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    x = np.arange(2)
    w = 0.5

    # Coverage comparison
    ax = axes[0]
    vals = [r_mpc["coverage"], r_no["coverage"]]
    bars = ax.bar(x, vals, w, color=["#2b8cbe", "#e34a33"],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["MAPPO + MPC", "MAPPO Only"], fontsize=11)
    ax.set_ylabel("Coverage Fraction", fontsize=11)
    ax.set_title("Garden Coverage", fontweight='bold', fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1%}', xy=(bar.get_x() + bar.get_width() / 2, h),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Collision comparison
    ax = axes[1]
    vals = [r_mpc.get("tree_collisions", 0), r_no.get("tree_collisions", 0)]
    bars = ax.bar(x, vals, w, color=["#2b8cbe", "#e34a33"],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["MAPPO + MPC", "MAPPO Only"], fontsize=11)
    ax.set_ylabel("Tree Collisions", fontsize=11)
   # ax.set_title("Safety (Tree Collisions)", fontweight='bold', fontsize=13)
    ax.grid(alpha=0.3, axis="y")

    # Return comparison
    ax = axes[2]
    vals = [r_mpc["return"], r_no["return"]]
    bars = ax.bar(x, vals, w, color=["#2b8cbe", "#e34a33"],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["MAPPO + MPC", "MAPPO Only"], fontsize=11)
    ax.set_ylabel("Episode Return", fontsize=11)
    #ax.set_title("Cumulative Reward", fontweight='bold', fontsize=13)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def fig_combined_summary(results, env, out_path):
    """Combined 2x2 summary figure — ideal for a single presentation slide."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    cov_cmap = LinearSegmentedColormap.from_list(
        "coverage", ["#f7fbff", "#c7e9c0", "#74c476", "#31a354", "#006d2c"]
    )

    for ax, key, title in zip(
        [axes[0, 0], axes[0, 1]], ["with_mpc", "without_mpc"],
        ["MAPPO + MPC (Hybrid)", "MAPPO Only"]
    ):
        r = results[key]
        grid = r["coverage_grid"].astype(float)
        extent = (0, env.size, 0, env.size)
        ax.imshow(grid, origin="lower", extent=extent, cmap=cov_cmap,
                  vmin=0, vmax=1, aspect="equal", alpha=0.88)
        _draw_garden(ax, env.size, r["trees"], env.tree_radius)
        _draw_lanes(ax, r["ideal_lanes"])
        colors = ["tab:blue", "tab:red"]
        for i in range(2):
            path = r["paths"][i]
            ax.plot(path[:, 0], path[:, 1], color=colors[i],
                    linewidth=0.5, alpha=0.4, zorder=3)
        cov_pct = grid.mean() * 100
        coll = r.get('tree_collisions', 0)
        ax.set_title(f"{title}\nCoverage = {cov_pct:.1f}% | Collisions = {coll}",
                     fontsize=12, fontweight='bold')

    for ax, key, title in zip(
        [axes[1, 0], axes[1, 1]], ["with_mpc", "without_mpc"],
        ["MAPPO + MPC Paths", "MAPPO Only Paths"]
    ):
        r = results[key]
        _draw_garden(ax, env.size, r["trees"], env.tree_radius)
        _draw_lanes(ax, r["ideal_lanes"])
        colors = ["tab:blue", "tab:red"]
        for i in range(2):
            path = r["paths"][i]
            ax.plot(path[:, 0], path[:, 1], color=colors[i], linewidth=1.3,
                    alpha=0.85, zorder=3)
            ax.scatter(path[0, 0], path[0, 1], color=colors[i], marker="o",
                      s=60, edgecolor="black", zorder=5)
            ax.scatter(path[-1, 0], path[-1, 1], color=colors[i], marker="*",
                      s=120, edgecolor="black", zorder=5)
        cov_pct = r['coverage'] * 100
        coll = r.get('tree_collisions', 0)
    #     ax.set_title(f"{title} | Coverage = {cov_pct:.1f}% | Collisions = {coll}",
    #                  fontsize=12, fontweight='bold')

    # fig.suptitle("2-Drone Garden Spraying — MAPPO+MPC vs MAPPO Only\n"
    #              "50x50 m garden, 8 trees, 6 boustrophedon lanes per region, MPC tree avoidance",
    #              y=1.01, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    _base = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
    out_dir = os.path.join(_base, "figures")
    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(_base, "runs", "default")
    model_path = os.path.join(model_dir, "mappo.pt")

    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = GardenEnvironment(
        size=50.0, n_trees_per_region=4, grid_res=50,
        spray_radius=2.5, tree_radius=1.5, max_steps=1500,
        dt=0.2, max_speed=6.0, max_accel=3.5,
        lane_spacing=4.0, y_step=1.5, wp_reach_dist=2.5,
        seed=seed,
    )

    # Train MAPPO if no saved model exists
    if not os.path.exists(model_path):
        print("=" * 60)
        print("No trained MAPPO model found. Training...")
        print("=" * 60)
        train_args = train_mod.get_parser().parse_args([
            "--out_dir", model_dir,
            "--seed", str(seed),
            "--n_updates", "100",
            "--episodes_per_update", "4",
            "--log_every", "5",
        ])
        train_mod.train(train_args)

    # Load trained MAPPO agent
    agent = MAPPO(
        obs_dim=env.obs_dim, act_dim=env.act_dim,
        n_agents=env.n_agents, device=device,
    )
    agent.load(model_path)

    mpc = MPCController(
        horizon=15, dt=env.dt, u_max=env.max_accel,
        tree_radius=env.tree_radius, tree_safety_margin=0.5,
        inter_drone_min=env.inter_drone_min,
        boundary=(0.0, env.size),
    )

    # Run evaluation episodes
    print()
    print("=" * 60)
    print("Running evaluation episodes...")
    print("=" * 60)

    env.rng = np.random.default_rng(seed)
    print("  Running MAPPO + MPC episode...")
    r_mpc = run_episode_mpc(env, agent, mpc, record_path=True)
    print(f"    MAPPO+MPC: coverage={r_mpc['coverage']:.4f} ({r_mpc['coverage']*100:.1f}%), "
          f"steps={r_mpc['length']}, mpc_used={r_mpc['mpc_used']}, "
          f"collisions={r_mpc['tree_collisions']}")

    env.rng = np.random.default_rng(seed)
    print("  Running MAPPO-only episode...")
    r_no_mpc = run_episode_mappo_only(env, agent, act_noise_scale=0.0, record_path=True)
    print(f"    MAPPO only: coverage={r_no_mpc['coverage']:.4f} ({r_no_mpc['coverage']*100:.1f}%), "
          f"steps={r_no_mpc['length']}, collisions={r_no_mpc['tree_collisions']}")

    results = {"with_mpc": r_mpc, "without_mpc": r_no_mpc}

    # Generate all presentation figures
    print()
    print("=" * 60)
    print("Generating presentation figures...")
    print("=" * 60)

    fig_coverage_heatmap(results, env, os.path.join(out_dir, "coverage_heatmap.png"))
    fig_coverage_detail(results, env, os.path.join(out_dir, "coverage_detail.png"))
    fig_paths_comparison(results, env, os.path.join(out_dir, "paths_comparison.png"))
    fig_comparison_metrics(results, os.path.join(out_dir, "comparison_metrics.png"))
    fig_combined_summary(results, env, os.path.join(out_dir, "combined_summary.png"))

    # Parameter sweep
    print()
    print("Running parameter sweep...")
    sweep_configs = {
        "8 trees, N=15": dict(n_trees=4, max_speed=6, mpc_horizon=15),
        "10 trees, N=10":   dict(n_trees=5, max_speed=6, mpc_horizon=15),
        "12 trees,N=15":     dict(n_trees=6, max_speed=6, mpc_horizon=15),
         "4 trees N=20":  dict(n_trees=4, max_speed=6, mpc_horizon=20),
    }
    sweep_data = {}
    for name, cfg in sweep_configs.items():
        env_s = GardenEnvironment(
            n_trees_per_region=cfg["n_trees"], max_speed=cfg["max_speed"],
            spray_radius=2.5, lane_spacing=4.0, max_steps=1500,
            seed=seed, wp_reach_dist=2.5, max_accel=3.5,
        )
        agent_s = MAPPO(obs_dim=env_s.obs_dim, act_dim=env_s.act_dim,
                        n_agents=env_s.n_agents, device=device)
        agent_s.load(model_path)
        mpc_s = MPCController(
            horizon=cfg["mpc_horizon"], dt=env_s.dt, u_max=env_s.max_accel,
            tree_radius=env_s.tree_radius, tree_safety_margin=0.5,
            inter_drone_min=env_s.inter_drone_min,
            boundary=(0.0, env_s.size),
        )
        env_s.rng = np.random.default_rng(seed)
        r1 = run_episode_mpc(env_s, agent_s, mpc_s, record_path=False)
        env_s.rng = np.random.default_rng(seed)
        r2 = run_episode_mappo_only(env_s, agent_s, act_noise_scale=0.0, record_path=False)
        sweep_data[name] = {
            "mpc_cov": r1["coverage"], "no_mpc_cov": r2["coverage"],
            "mpc_ret": r1["return"], "no_mpc_ret": r2["return"],
        }
        print(f"  {name:20s}  MAPPO+MPC: cov={r1['coverage']:.3f}  "
              f"MAPPO only: cov={r2['coverage']:.3f}")

    # Parameter sweep figure
    fig_ps, ps_axes = plt.subplots(1, 2, figsize=(13, 5.5))
    names = list(sweep_data.keys())
    cov_mpc = [sweep_data[n]["mpc_cov"] for n in names]
    cov_no = [sweep_data[n]["no_mpc_cov"] for n in names]
    ret_mpc = [sweep_data[n]["mpc_ret"] for n in names]
    ret_no = [sweep_data[n]["no_mpc_ret"] for n in names]
    x = np.arange(len(names))
    w = 0.38

    ax = ps_axes[0]
    b1 = ax.bar(x - w/2, ret_mpc, w, label="MAPPO + MPC",
                color="#2b8cbe", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, ret_no, w, label="MAPPO only",
                color="#e34a33", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Episode return")
    #ax.set_title("Return by Configuration", fontweight='bold')
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best")

    ax = ps_axes[1]
    b1 = ax.bar(x - w/2, cov_mpc, w, label="MAPPO + MPC",
                color="#2b8cbe", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, cov_no, w, label="MAPPO only",
                color="#e34a33", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Coverage fraction")
    #ax.set_title("Coverage by Configuration", fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

    fig_ps.tight_layout()
    ps_path = os.path.join(out_dir, "parameter_sweep.png")
    fig_ps.savefig(ps_path, dpi=200)
    plt.close(fig_ps)
    print(f"  Saved {ps_path}")



    print()
    print("=" * 60)
    print("All figures generated!")
    print("=" * 60)
    print()
    print("RESULTS SUMMARY:")
    print(f"  MAPPO + MPC:  coverage = {r_mpc['coverage']*100:.1f}%  "
          f"collisions = {r_mpc['tree_collisions']}  "
          f"steps = {r_mpc['length']}  "
          f"MPC activations = {r_mpc['mpc_used']}")
    print(f"  MAPPO only:   coverage = {r_no_mpc['coverage']*100:.1f}%  "
          f"collisions = {r_no_mpc['tree_collisions']}  "
          f"steps = {r_no_mpc['length']}")
    print()
    print("Generated figures:")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.png'):
            print(f"  {os.path.join(out_dir, f)}")


if __name__ == "__main__":
    main()
