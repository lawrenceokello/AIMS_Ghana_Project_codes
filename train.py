import argparse
import os
import time
import numpy as np
import torch

from environment import GardenEnvironment
from mappo import MAPPO, RolloutBuffer


def run_episode(env: GardenEnvironment, agent: MAPPO, buf: RolloutBuffer | None): # env=drone simulation environment , 
                                     #MAPPO selecting actions,evaluating  policy, producing controls, buf Storage buffer for PPO training data.
    obs = env.reset()  #Starts a new episode, tree postions, places drones,generates trees,clears coverage,resets counters.      
    ep_r = 0.0 # episode reward
    ep_len = 0  # time steps
    while True:
        actions, logps, value = agent.act(obs)  ## actions from the observations
        next_obs, rewards, done, info = env.step(actions) ## applies actions to the environment

        if buf is not None:
            buf.add( obs=obs.copy(),jobs=obs.reshape(-1).copy(),a=actions.copy(),lp=logps.copy(),
                r=rewards.copy(),v=value, d=float(done),
            )  # stores observation, joint observation, actions, log probabilities, rewards, value V(s)

        obs = next_obs # use the current observation to update 
        ep_r += float(rewards.sum())  #This accumulates total reward across the entire episode.
        ep_len += 1
        if done:
            break
    return ep_r, ep_len, env.coverage_fraction(), info


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = GardenEnvironment(size=args.garden_size,n_trees_per_region=args.n_trees,grid_res=args.grid_res,
        max_steps=args.max_steps,lane_spacing=args.lane_spacing, spray_radius=args.spray_radius,
        max_speed=args.max_speed,max_accel=args.max_accel, seed=args.seed,
    )
    agent = MAPPO(obs_dim=env.obs_dim,act_dim=env.act_dim,n_agents=env.n_agents, lr=args.lr,clip_eps=args.clip_eps,
        gamma=args.gamma, lam=args.lam,epochs=args.ppo_epochs, minibatch=args.minibatch, device=device,
    )

    history = { "episode": [], "return": [], "length": [], "coverage": [], "pi_loss": [], "v_loss": [],
    }

    buf = RolloutBuffer()
    episodes_per_update = args.episodes_per_update
    n_updates = args.n_updates

    t0 = time.time()
    ep = 0
    for update in range(1, n_updates + 1):
        buf.reset()
        upd_returns, upd_lengths, upd_cov = [], [], []
        for _ in range(episodes_per_update):
            ep += 1
            ep_r, ep_len, cov, info = run_episode(env, agent, buf)
            upd_returns.append(ep_r)
            upd_lengths.append(ep_len)
            upd_cov.append(cov)
            history["episode"].append(ep)
            history["return"].append(ep_r)
            history["length"].append(ep_len)
            history["coverage"].append(cov)

        stats = agent.update(buf, last_value=0.0)
        for _ in range(episodes_per_update):
            history["pi_loss"].append(stats["pi_loss"])
            history["v_loss"].append(stats["v_loss"])

        if update % args.log_every == 0 or update == 1:
            elapsed = time.time() - t0
            print(
                f"upd {update:4d}/{n_updates}  ep {ep:5d}  "
                f"R={np.mean(upd_returns):7.2f}  "
                f"len={np.mean(upd_lengths):5.1f}  "
                f"cov={np.mean(upd_cov):.3f}  "
                f"pi_loss={stats['pi_loss']:.3f}  "
                f"v_loss={stats['v_loss']:.2f}  "
                f"t={elapsed:.0f}s"
            )

    os.makedirs(args.out_dir, exist_ok=True)
    agent.save(os.path.join(args.out_dir, "mappo.pt"))
    np.savez( os.path.join(args.out_dir, "history.npz"),episode=np.array(history["episode"]),ret=np.array(history["return"]),
        length=np.array(history["length"]),coverage=np.array(history["coverage"]),  pi_loss=np.array(history["pi_loss"]),
        v_loss=np.array(history["v_loss"]),
    )
    print(f"\nSaved checkpoint and history to {args.out_dir}")


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="runs/default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--garden_size", type=float, default=50.0)
    p.add_argument("--n_trees", type=int, default=4)
    p.add_argument("--grid_res", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--lane_spacing", type=float, default=4.0)
    p.add_argument("--spray_radius", type=float, default=2.5)
    p.add_argument("--max_speed", type=float, default=6.0)
    p.add_argument("--max_accel", type=float, default=3.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ppo_epochs", type=int, default=6)
    p.add_argument("--minibatch", type=int, default=128)
    p.add_argument("--episodes_per_update", type=int, default=4)
    p.add_argument("--n_updates", type=int, default=200)
    p.add_argument("--log_every", type=int, default=10)
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
