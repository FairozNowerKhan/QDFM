import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.fork_env import MultiObjectiveFork, generate_fork_dataset
from algos.qdfm_mo import QDFM_MO
from utils import set_seed


class SimpleDataset:
    def __init__(self, obs, act_idx, rew, next_obs, done):
        self.obs = obs
        self.act_idx = act_idx
        self.rew = rew
        self.next_obs = next_obs
        self.done = done
        self.N = obs.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="results_fork")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    env = MultiObjectiveFork()
    raw = generate_fork_dataset(env, n_traj_per_goal=200)

    ds = SimpleDataset(
        obs=torch.as_tensor(raw["obs"], device=args.device),
        act_idx=torch.as_tensor(raw["actions"], device=args.device, dtype=torch.long),
        rew=torch.as_tensor(raw["rewards"], device=args.device),
        next_obs=torch.as_tensor(raw["next_obs"], device=args.device),
        done=torch.as_tensor(raw["done"], device=args.device),
    )

    method = QDFM_MO(env.obs_dim, env.num_actions, env.n_objectives,
                     omega_dim=2, gamma=0.99, beta=10.0, M=16, device=args.device)
    method.train(ds, k1=3000, k2=20000, k3=10000, batch=256)

    omega_list = [
        np.array([1.0, 0.0]),
        np.array([0.5, 0.5]),
        np.array([0.0, 1.0]),
    ]

    for omega in omega_list:
        visit_counts = np.zeros((env.gs, env.gs), dtype=np.int32)
        for _ in range(200):
            o, _ = env.reset()
            for _ in range(env.max_steps):
                x, y = env.get_pos()
                visit_counts[y, x] += 1
                a = method.act(o, omega=omega)
                o, r, term, trunc, info = env.step(a)
                if term or trunc:
                    break
        tag = f"fork_omega{omega[0]:.1f}_{omega[1]:.1f}_s{args.seed}"
        np.save(out / f"visits_{tag}.npy", visit_counts)

    print("Fork experiment complete")


if __name__ == "__main__":
    main()
