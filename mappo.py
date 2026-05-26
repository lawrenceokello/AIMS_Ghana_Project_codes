from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ============================================================================
# Networks
# ============================================================================

def mlp(sizes: list[int], act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    min_std = -2.0
    max_std = 0.5

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def forward(self, obs): # Based on observation, drone uses Guassian distribution action 
        mu = self.net(obs)#passes observation through network to compute the mean
        std = torch.exp(torch.clamp(self.log_std, self.min_std, self.max_std)) # clamps std
        return Normal(mu, std)

    def act(self, obs):   # defines the choice of action for the drones
        dist = self.forward(obs)# gives the distribution choic
        a = dist.sample()  # samples actions instead of being deterministic
        logp = dist.log_prob(a).sum(-1)
        return a, logp

    def evaluate(self, obs, a):
        dist = self.forward(obs)
        logp = dist.log_prob(a).sum(-1)
        return logp


class CentralCritic(nn.Module):

    def __init__(self, joint_obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp([joint_obs_dim, hidden, hidden, 1])

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ============================================================================
# Rollout buffer
# ============================================================================

class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.joint_obs = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs, jobs, a, lp, r, v, d):
        self.obs.append(obs); self.joint_obs.append(jobs)
        self.actions.append(a); self.logps.append(lp)
        self.rewards.append(r); self.values.append(v); self.dones.append(d)

    def as_tensors(self, device):
        return {
            "obs": torch.as_tensor(np.array(self.obs), dtype=torch.float32, device=device),
            "joint_obs": torch.as_tensor(np.array(self.joint_obs), dtype=torch.float32, device=device),
            "actions": torch.as_tensor(np.array(self.actions), dtype=torch.float32, device=device),
            "logps": torch.as_tensor(np.array(self.logps), dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(np.array(self.rewards), dtype=torch.float32, device=device),
            "values": torch.as_tensor(np.array(self.values), dtype=torch.float32, device=device),
            "dones": torch.as_tensor(np.array(self.dones), dtype=torch.float32, device=device),
        }


# ============================================================================
# Generalized Advantage Estimation
# ============================================================================

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, last_value=0.0):
    T = len(rewards)
    team_r = rewards.sum(axis=1)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = team_r[t] + gamma * next_v * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


# ============================================================================
# MAPPO agent
# ============================================================================

class MAPPO:
    def __init__( self, obs_dim: int, act_dim: int,n_agents: int, lr: float = 3e-4,  clip_eps: float = 0.2,
        gamma: float = 0.99, lam: float = 0.95, c_vf: float = 0.5,c_ent: float = 0.01, epochs: int = 6,
        minibatch: int = 64, device: str = "cpu", ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.c_vf = c_vf
        self.c_ent = c_ent
        self.epochs = epochs
        self.minibatch = minibatch
        self.device = device

        self.actor = Actor(obs_dim, act_dim).to(device) # creating the actor neural network and sends it to CPU
        self.critic = CentralCritic(obs_dim * n_agents).to(device) #creating the value neural network and sends it to neural network
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        ) # Recalls all the weights and the biases, Creates the optimization algorithm that updates neural-network parameters during training.

    @torch.no_grad() # Do not compute gradients inside this function.” During execution, no learning its only execution
    def act(self, obs_np: np.ndarray):  # observation in NumPy format
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device) # change numpy to tensor, computation is done CPU, GPU
        a, lp = self.actor.act(obs) # calls action and log probability
        joint_obs = obs.reshape(1, -1)
        v = self.critic(joint_obs).item()  # The Critic predicts:V(s) which estimates expected future reward from the current state.
        return a.cpu().numpy(), lp.cpu().numpy(), v  # returns results in numpy not tensors

    def update(self, buf: RolloutBuffer, last_value: float = 0.0): #  extracts rollout data,  ,  normalizes 
        rewards = np.array(buf.rewards)                #, converts everything into tensors, prepares data for neural-network optimization.
        values = np.array(buf.values)
        dones = np.array(buf.dones)   # converts to numpy

        adv, returns = compute_gae(rewards, values, dones, self.gamma, self.lam, last_value) # computes advantages and returns
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) # normalizes  advantages

        data = buf.as_tensors(self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device) 
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)  #onverts everything into tensors

        T = data["obs"].shape[0]
        idxs = np.arange(T)
        stats = {"pi_loss": [], "v_loss": [], "approx_kl": []} # store pi_loss, v_loss, kl= how new policy differs from the old policy.

        for _ in range(self.epochs):
            np.random.shuffle(idxs)   # Randomly shuffles sample ordering
            for start in range(0, T, self.minibatch):
                mb = idxs[start:start + self.minibatch] # Select minibatch indices

                obs_mb = data["obs"][mb] # minibatch observation
                act_mb = data["actions"][mb] # minibatch action
                logp_old_mb = data["logps"][mb]# minibatch log probability

                B = obs_mb.shape[0]    #this extracts the sample size from the observation list details
                obs_flat = obs_mb.reshape(B * self.n_agents, self.obs_dim)  #Actor network can process all agents as one large batch.
                act_flat = act_mb.reshape(B * self.n_agents, self.act_dim)
                logp_old_flat = logp_old_mb.reshape(B * self.n_agents)

                logp = self.actor.evaluate(obs_flat, act_flat)
                ratio = torch.exp(logp - logp_old_flat)
                adv_per_agent = adv_t[mb].unsqueeze(1).expand(-1, self.n_agents).reshape(-1)
                surr1 = ratio * adv_per_agent                                   # product of r_t(0).A_t
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_per_agent # clipped function
                pi_loss = -torch.min(surr1, surr2).mean()  # pytorch choose the minimum of the two

                joint_obs_mb = data["joint_obs"][mb]  #Extract minibatch joint observations
                v_pred = self.critic(joint_obs_mb)  #Critic prediction
                v_loss = F.mse_loss(v_pred, ret_t[mb])  # compute the mean square error (V(s)-R_t)^2

                loss = pi_loss + self.c_vf * v_loss

                self.opt.zero_grad()  # resets all gradients to zero before computing new ones.
                loss.backward()   # Compute gradients
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5) # Gradient clipping
                self.opt.step()  # Use the gradients to modify the network parameters.

                with torch.no_grad():
                    approx_kl = (logp_old_flat - logp).mean().item()
                stats["pi_loss"].append(pi_loss.item())
                stats["v_loss"].append(v_loss.item())
                stats["approx_kl"].append(approx_kl)

        return {k: float(np.mean(v)) for k, v in stats.items()}  # Converts tensor into Python scalar.

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
