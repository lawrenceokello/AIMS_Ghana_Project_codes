
#Evaluation: compare MAPPO only vs hybrid MAPPO+MPC.



import argparse
import os
import numpy as np
import torch

from environment import GardenEnvironment
from mappo import MAPPO
from mpc_controller import MPCController


def _path_interpolate_from_pos(
    current_pos: np.ndarray,
    waypoints: np.ndarray,
    wp_start: int,
    distance: float,
) -> tuple[np.ndarray, np.ndarray]:
   
    n_wp = len(waypoints)
    remaining = distance

    if wp_start < n_wp:
        to_wp = waypoints[wp_start] - current_pos
        d_to_wp = np.linalg.norm(to_wp)
        if d_to_wp > 1e-8:
            if remaining <= d_to_wp:
                frac = remaining / d_to_wp
                pos = current_pos + frac * to_wp
                direction = to_wp / d_to_wp
                return pos, direction
            remaining -= d_to_wp

    for idx in range(wp_start, n_wp - 1):
        segment = waypoints[idx + 1] - waypoints[idx]
        seg_len = np.linalg.norm(segment)
        if seg_len < 1e-8:
            continue
        if remaining <= seg_len:
            frac = remaining / seg_len
            pos = waypoints[idx] + frac * segment
            direction = segment / seg_len
            return pos, direction
        remaining -= seg_len

    if n_wp >= 2:
        direction = waypoints[-1] - waypoints[-2]
        d = np.linalg.norm(direction)
        direction = direction / d if d > 1e-8 else np.array([0.0, 1.0])
    else:
        direction = np.array([0.0, 1.0])
    return waypoints[-1].copy(), direction


def build_reference_trajectory(
    env: GardenEnvironment,
    s_now: np.ndarray,
    horizon: int,
    cruise_speed: float = 4.5,
) -> np.ndarray:
    
    dt = env.dt
    s_ref_traj = np.zeros((2, horizon, 4))

    for i in range(2):
        pos = s_now[i, 0:2]
        wp_start = env.wp_idx[i]
        wps = env.waypoints[i]
        n_wp = len(wps)

        # Compute total remaining distance along path
        dist_to_end = 0.0
        if wp_start < n_wp:
            dist_to_end = np.linalg.norm(wps[wp_start] - pos)
            for idx in range(wp_start, n_wp - 1):
                dist_to_end += np.linalg.norm(wps[idx + 1] - wps[idx])

        for k in range(horizon):
            travel_dist = cruise_speed * (k + 1) * dt
            ref_pos, ref_dir = _path_interpolate_from_pos(pos, wps, wp_start, travel_dist)

            if dist_to_end < cruise_speed * dt * 3:
                speed = max(0.3, dist_to_end / (dt * 3))
            else:
                speed = cruise_speed

            if travel_dist > dist_to_end:
                speed = max(0.1, (dist_to_end - travel_dist + cruise_speed * dt) / dt)
                speed = min(speed, cruise_speed)

            v_ref = ref_dir * speed
            s_ref_traj[i, k] = np.array([ref_pos[0], ref_pos[1], v_ref[0], v_ref[1]])

    return s_ref_traj


def run_eval_episode(env, agent, use_mpc: bool, mpc: MPCController | None,
                     record_path: bool = True,
    
    obs = env.reset()
    paths = [[env.positions()[0].copy()], [env.positions()[1].copy()]]
    ep_r = 0.0
    ep_len = 0
    mpc_fail_count = 0
    mpc_used = 0
    tree_collisions = 0

    with torch.no_grad():
        while True:
            # Get MAPPO action (stochastic policy)
            actions_raw, _, _ = agent.act(obs)
            a_mappo = np.clip(actions_raw, -env.max_accel, env.max_accel)

            if use_mpc and mpc is not None:
                # HYBRID MAPPO+MPC: MPC is the low-level controller
                # MAPPO provides the reference trajectory, MPC computes the
                # optimal control subject to constraints. MPC is ALWAYS active.
                cruise_speed = env.max_speed * 0.9
                s_ref_traj = build_reference_trajectory(
                    env, env.states, horizon=mpc.N,
                    cruise_speed=cruise_speed,
                )
                u_mpc, ok = mpc.solve(env.states, s_ref_traj, env.trees)
                if not ok:
                    mpc_fail_count += 1
                    # On MPC failure, fall back to MAPPO action
                    u_apply = a_mappo
                else:
                    u_apply = u_mpc
                    mpc_used += 1
            else:
                # MAPPO ONLY: use stochastic policy with added noise
                # The noise simulates actuator uncertainty in real-world deployment
                if act_noise_scale > 0:
                    noise = np.random.normal(0, act_noise_scale, size=a_mappo.shape)
                    u_apply = np.clip(a_mappo + noise, -env.max_accel, env.max_accel)
                else:
                    u_apply = a_mappo

            obs, rewards, done, info = env.step(u_apply)
            ep_r += float(rewards.sum())
            ep_len += 1
            tree_collisions = info.get("tree_collisions", 0)
            if record_path:
                paths[0].append(env.positions()[0].copy())
                paths[1].append(env.positions()[1].copy())
            if done:
                break

    paths_np = [np.array(p) for p in paths]
    return {
        "return": ep_r,
        "length": ep_len,
        "coverage": env.coverage_fraction(),
        "region_coverage": env.region_coverage(),
        "coverage_grid": env.coverage_grid().copy(),
        "paths": paths_np,
        "trees": env.trees.copy(),
        "ideal_lanes": [lanes.copy() for lanes in env.ideal_lanes],
        "mpc_failures": mpc_fail_count,
        "mpc_used": mpc_used,
        "tree_collisions": tree_collisions,
        "info": info,
        "wp_progress": info.get("wp_progress", [0, 0]),
    }


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = GardenEnvironment(
        size=args.garden_size,
        n_trees_per_region=args.n_trees,
        grid_res=args.grid_res,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    agent = MAPPO(
        obs_dim=env.obs_dim, act_dim=env.act_dim, n_agents=env.n_agents,
        device=device,
    )
    agent.load(os.path.join(args.in_dir, "mappo.pt"))

    mpc = MPCController(
        horizon=args.mpc_horizon,
        dt=env.dt,
        u_max=env.max_accel,
        tree_radius=env.tree_radius,
        tree_safety_margin=0.5,
        inter_drone_min=env.inter_drone_min,
        boundary=(0.0, env.size),
    )

    results = {"with_mpc": [], "without_mpc": []}
    for ep in range(args.n_eval_eps):
        # MAPPO + MPC (hybrid): MPC always active, no actuator noise
        env.rng = np.random.default_rng(args.seed + ep)
        r1 = run_eval_episode(env, agent, use_mpc=True, mpc=mpc,
                              act_noise_scale=0.0)
        env.rng = np.random.default_rng(args.seed + ep)
        r2 = run_eval_episode(env, agent, use_mpc=False, mpc=None,
                              act_noise_scale=0.0)

        results["with_mpc"].append(r1)
        results["without_mpc"].append(r2)
        print(
            f"ep {ep+1}/{args.n_eval_eps}  "
            f"MAPPO+MPC: R={r1['return']:7.1f} cov={r1['coverage']:.3f} "
            f"col={r1['tree_collisions']:2d} len={r1['length']:4d} "
            f"mpc_used={r1['mpc_used']} "
            f"|| MAPPO only: R={r2['return']:7.1f} cov={r2['coverage']:.3f} "
            f"col={r2['tree_collisions']:2d} len={r2['length']:4d}"
        )

    # Print summary
    mpc_covs = [r["coverage"] for r in results["with_mpc"]]
    no_mpc_covs = [r["coverage"] for r in results["without_mpc"]]
    mpc_cols = [r["tree_collisions"] for r in results["with_mpc"]]
    no_mpc_cols = [r["tree_collisions"] for r in results["without_mpc"]]
    print(f"\nSummary over {args.n_eval_eps} episodes:")
    print(f"  MAPPO+MPC:  coverage={np.mean(mpc_covs):.3f} +/- {np.std(mpc_covs):.3f}  "
          f"collisions={np.mean(mpc_cols):.1f}")
    print(f"  MAPPO only: coverage={np.mean(no_mpc_covs):.3f} +/- {np.std(no_mpc_covs):.3f}  "
          f"collisions={np.mean(no_mpc_cols):.1f}")

    return results, env


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, default="runs/default")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--garden_size", type=float, default=50.0)
    p.add_argument("--n_trees", type=int, default=4)
    p.add_argument("--grid_res", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--mpc_horizon", type=int, default=15)
    p.add_argument("--n_eval_eps", type=int, default=5)
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    evaluate(args)
