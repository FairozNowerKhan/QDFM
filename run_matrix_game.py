import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.matrix_game import TwoAgentMatrixGame, generate_matrix_game_dataset
from bench.nets import MLP, CategoricalPolicy
from bench.nets_marl import PerAgentRateNet, VectorQ
from bench.ctmc import bridge_sample, target_rate_vector
from utils import set_seed


class MARLDataset:
    def __init__(self, obs, a1, a2, rew, next_obs, done, device):
        self.obs = torch.as_tensor(obs, device=device)
        self.a1 = torch.as_tensor(a1, device=device, dtype=torch.long)
        self.a2 = torch.as_tensor(a2, device=device, dtype=torch.long)
        self.rew = torch.as_tensor(rew, device=device)
        self.next_obs = torch.as_tensor(next_obs, device=device)
        self.done = torch.as_tensor(done, device=device)
        self.N = obs.shape[0]


def sample_marl_batch(ds, batch_size):
    idx = torch.randint(0, ds.N, (batch_size,), device=ds.obs.device)
    return ds.obs[idx], ds.a1[idx], ds.a2[idx], ds.rew[idx], ds.next_obs[idx], ds.done[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="results_matrix")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    raw = generate_matrix_game_dataset(n_episodes=2000)
    ds = MARLDataset(raw["obs"], raw["a1"], raw["a2"], raw["rewards"],
                     raw["next_obs"], raw["done"], args.device)

    env = TwoAgentMatrixGame()
    n_pref = 11
    omega_list = [np.array([i/(n_pref-1), 1.0 - i/(n_pref-1)]) for i in range(n_pref)]

    results = []
    for omega in omega_list:
        ep_returns = []
        for _ in range(100):
            o, _ = env.reset()
            a1 = np.random.choice(3)
            a2 = np.random.choice(3)
            o2, r, term, trunc, _ = env.step([a1, a2])
            scal = (omega * r).sum()
            ep_returns.append(r.tolist())
        results.append({
            "omega": omega.tolist(),
            "mean_return": np.mean(ep_returns, axis=0).tolist(),
        })

    with open(out / f"metrics_matrix_s{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Matrix game experiment complete")


if __name__ == "__main__":
    main()
