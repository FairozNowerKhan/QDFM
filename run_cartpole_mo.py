import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algos.qdfm_mo import QDFM_MO
from utils import set_seed

try:
    import gymnasium as gym
except ImportError:
    gym = None


class SimpleDataset:
    def __init__(self, obs, act_idx, rew, next_obs, done):
        self.obs = obs
        self.act_idx = act_idx
        self.rew = rew
        self.next_obs = next_obs
        self.done = done
        self.N = obs.shape[0]


def collect_cartpole_mo_dataset(n_episodes=500, device="cpu"):
    env = gym.make("CartPole-v1")
    obs_l, act_l, rew_l, nobs_l, done_l = [], [], [], [], []
    for _ in range(n_episodes):
        o, _ = env.reset()
        done = False
        prev_a = 0
        while not done:
            a = env.action_space.sample()
            o2, r, term, trunc, _ = env.step(a)
            switch_penalty = -1.0 if a != prev_a else 0.0
            r_vec = np.array([r, switch_penalty], dtype=np.float32)
            obs_l.append(o)
            act_l.append(a)
            rew_l.append(r_vec)
            nobs_l.append(o2)
            done_l.append(float(term or trunc))
            prev_a = a
            o = o2
            done = term or trunc
    return SimpleDataset(
        obs=torch.as_tensor(np.array(obs_l, dtype=np.float32), device=device),
        act_idx=torch.as_tensor(np.array(act_l, dtype=np.int64), device=device),
        rew=torch.as_tensor(np.array(rew_l, dtype=np.float32), device=device),
        next_obs=torch.as_tensor(np.array(nobs_l, dtype=np.float32), device=device),
        done=torch.as_tensor(np.array(done_l, dtype=np.float32), device=device),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="results_cartpole_mo")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    ds = collect_cartpole_mo_dataset(n_episodes=500, device=args.device)
    obs_dim = ds.obs.shape[1]
    K = 2

    method = QDFM_MO(obs_dim, K, n_objectives=2, omega_dim=2, gamma=0.99,
                     beta=10.0, M=16, device=args.device)
    method.train(ds, k1=3000, k2=20000, k3=10000, batch=256)

    env = gym.make("CartPole-v1")
    omega_list = [np.array([w, 1.0 - w]) for w in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]]
    results = []

    for omega in omega_list:
        task_returns, smooth_returns = [], []
        for _ in range(200):
            o, _ = env.reset()
            done = False
            ep_task, ep_smooth = 0.0, 0.0
            prev_a = 0
            steps = 0
            while not done and steps < 500:
                a = method.act(o, omega=omega)
                o, r, term, trunc, _ = env.step(a)
                ep_task += r
                if a != prev_a:
                    ep_smooth -= 1.0
                prev_a = a
                done = term or trunc
                steps += 1
            task_returns.append(ep_task)
            smooth_returns.append(ep_smooth)
        results.append({
            "omega": omega.tolist(),
            "task_return": float(np.mean(task_returns)),
            "smooth_return": float(np.mean(smooth_returns)),
        })

    with open(out / f"metrics_cartpole_mo_s{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)

    print("CartPole MO experiment complete")


if __name__ == "__main__":
    main()
