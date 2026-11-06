import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algos.qdfm_mo import QDFM_MO
from algos.cql_mo import DiscreteCQL_MO
from algos.common import sample_batch
from utils import set_seed

try:
    import mo_gymnasium as mo_gym
except ImportError:
    mo_gym = None


class SimpleDataset:
    def __init__(self, obs, act_idx, rew, next_obs, done):
        self.obs = obs
        self.act_idx = act_idx
        self.rew = rew
        self.next_obs = next_obs
        self.done = done
        self.N = obs.shape[0]


def collect_dataset(env_name, n_episodes=1000, device="cpu"):
    env = mo_gym.make(env_name)
    obs_l, act_l, rew_l, nobs_l, done_l = [], [], [], [], []
    for _ in range(n_episodes):
        o, _ = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            o2, r, term, trunc, _ = env.step(a)
            obs_l.append(o.astype(np.float32) if isinstance(o, np.ndarray) else np.array([float(o)], dtype=np.float32))
            act_l.append(a)
            rew_l.append(r.astype(np.float32))
            nobs_l.append(o2.astype(np.float32) if isinstance(o2, np.ndarray) else np.array([float(o2)], dtype=np.float32))
            done_l.append(float(term or trunc))
            o = o2
            done = term or trunc
    return SimpleDataset(
        obs=torch.as_tensor(np.array(obs_l), device=device),
        act_idx=torch.as_tensor(np.array(act_l, dtype=np.int64), device=device),
        rew=torch.as_tensor(np.array(rew_l), device=device),
        next_obs=torch.as_tensor(np.array(nobs_l), device=device),
        done=torch.as_tensor(np.array(done_l), device=device),
    )


def evaluate_mo(env_name, act_fn, omega_list, episodes_per_omega=20):
    env = mo_gym.make(env_name)
    results = []
    for omega in omega_list:
        returns = []
        for _ in range(episodes_per_omega):
            o, _ = env.reset()
            done = False
            ep_ret = np.zeros(len(omega), dtype=np.float64)
            while not done:
                a = act_fn(o, omega=omega)
                o, r, term, trunc, _ = env.step(a)
                ep_ret += r
                done = term or trunc
            returns.append(ep_ret)
        returns = np.array(returns)
        results.append({
            "omega": omega.tolist() if isinstance(omega, np.ndarray) else list(omega),
            "mean_return": returns.mean(axis=0).tolist(),
            "std_return": returns.std(axis=0, ddof=1).tolist() if len(returns) > 1 else [0.0] * len(omega),
        })
    return results


ENV_MAP = {
    "deep_sea_treasure": "deep-sea-treasure-v0",
    "resource_gathering": "resource-gathering-v0",
}

N_OBJ_MAP = {
    "deep_sea_treasure": 2,
    "resource_gathering": 3,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, required=True, choices=list(ENV_MAP.keys()))
    ap.add_argument("--method", type=str, required=True, choices=["qdfm", "cql"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="results_mo_discrete")
    ap.add_argument("--n_episodes", type=int, default=2000)
    ap.add_argument("--train_steps", type=int, default=50000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.env}_{args.method}_s{args.seed}"
    set_seed(args.seed)
    t0 = time.time()

    env_name = ENV_MAP[args.env]
    n_obj = N_OBJ_MAP[args.env]

    ds = collect_dataset(env_name, n_episodes=args.n_episodes, device=args.device)
    obs_dim = ds.obs.shape[1]
    K = int(ds.act_idx.max().item()) + 1

    if args.method == "qdfm":
        method = QDFM_MO(obs_dim, K, n_obj, omega_dim=n_obj, gamma=0.99,
                         beta=20.0, M=32, device=args.device)
        method.train(ds, k1=3000, k2=args.train_steps, k3=args.train_steps // 2, batch=256)
    else:
        method = DiscreteCQL_MO(obs_dim, K, n_obj, gamma=0.99, device=args.device)
        method.train(ds, steps=args.train_steps, batch=256)

    train_sec = time.time() - t0

    n_pref = 21
    if n_obj == 2:
        omega_list = [np.array([i/(n_pref-1), 1.0 - i/(n_pref-1)]) for i in range(n_pref)]
    else:
        omega_list = [np.random.dirichlet(np.ones(n_obj)) for _ in range(n_pref)]

    eval_results = evaluate_mo(env_name, method.act, omega_list)

    output = {
        "env": args.env,
        "method": args.method,
        "seed": args.seed,
        "train_sec": round(train_sec, 1),
        "eval": eval_results,
    }
    with open(out / f"metrics_{tag}.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
