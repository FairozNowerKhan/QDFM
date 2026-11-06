import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import minari
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discretize import build_centroids, assign_to_centroids
from eval_policy import evaluate_discrete_policy
from algos.qdfm import QDFM
from algos.cql import DiscreteCQL
from algos.iql import DiscreteIQL
from algos.awac import AWAC
from algos.bcq import DiscreteBCQ
from algos.brac import DiscreteBRAC
from algos.greedyq import GreedyQ
from utils import set_seed


class OfflineDataset:
    def __init__(self, obs, act_idx, rew, next_obs, done):
        self.obs = obs
        self.act_idx = act_idx
        self.rew = rew
        self.next_obs = next_obs
        self.done = done
        self.N = obs.shape[0]


def load_minari_transitions(dataset_name):
    ds = minari.load_dataset(dataset_name, download=True)
    obs_l, act_l, rew_l, nobs_l, done_l = [], [], [], [], []
    for ep in ds:
        o = ep.observations
        a = ep.actions
        r = ep.rewards
        term = ep.terminations
        trunc = ep.truncations
        obs_l.append(o[:-1])
        act_l.append(a)
        rew_l.append(r)
        nobs_l.append(o[1:])
        done_l.append(np.logical_or(term, trunc).astype(np.float32))
    return {
        "obs": np.concatenate(obs_l, axis=0).astype(np.float32),
        "actions": np.concatenate(act_l, axis=0).astype(np.float32),
        "rewards": np.concatenate(rew_l, axis=0).astype(np.float32),
        "next_obs": np.concatenate(nobs_l, axis=0).astype(np.float32),
        "done": np.concatenate(done_l, axis=0).astype(np.float32),
    }


def make_dataset(dataset_name, K_act, device, seed=0):
    raw = load_minari_transitions(dataset_name)
    centroids = build_centroids(raw["actions"], K=K_act, seed=seed)
    act_idx = assign_to_centroids(raw["actions"], centroids)
    ds = OfflineDataset(
        obs=torch.as_tensor(raw["obs"], device=device),
        act_idx=torch.as_tensor(act_idx, device=device, dtype=torch.long),
        rew=torch.as_tensor(raw["rewards"], device=device),
        next_obs=torch.as_tensor(raw["next_obs"], device=device),
        done=torch.as_tensor(raw["done"], device=device),
    )
    return ds, centroids


DATASET_MAP = {
    "halfcheetah_medium": "mujoco/halfcheetah/medium-v0",
    "halfcheetah_expert": "mujoco/halfcheetah/expert-v0",
    "hopper_medium": "mujoco/hopper/medium-v0",
    "hopper_expert": "mujoco/hopper/expert-v0",
    "walker2d_medium": "mujoco/walker2d/medium-v0",
    "walker2d_expert": "mujoco/walker2d/expert-v0",
}

ENV_MAP = {
    "halfcheetah": "HalfCheetah-v5",
    "hopper": "Hopper-v5",
    "walker2d": "Walker2d-v5",
}


def make_method(name, obs_dim, K, device):
    d = device
    h = (256, 256)
    if name == "qdfm":
        return QDFM(obs_dim, K, omega_dim=1, hidden=h, gamma=0.99,
                     beta=20.0, M=64, Krenew=50, h=0.05, N=20, device=d)
    if name == "cql":
        return DiscreteCQL(obs_dim, K, hidden=h, gamma=0.99, alpha=1.0, device=d)
    if name == "iql":
        return DiscreteIQL(obs_dim, K, hidden=h, gamma=0.99, device=d)
    if name == "awac":
        return AWAC(obs_dim, K, hidden=h, gamma=0.99, device=d)
    if name == "bcq":
        return DiscreteBCQ(obs_dim, K, hidden=h, gamma=0.99, device=d)
    if name == "brac":
        return DiscreteBRAC(obs_dim, K, hidden=h, gamma=0.99, device=d)
    if name == "greedyq":
        return GreedyQ(obs_dim, K, hidden=h, gamma=0.99, device=d)
    raise ValueError(name)


def train_method(method, ds, name, e_critic=500000, e_qdfm=350000):
    if name == "qdfm":
        method.train(ds, k1=150000, k2=e_critic, k3=e_qdfm, batch=256)
    elif name == "brac":
        method.train(ds, steps=1000000, batch=256, mu_steps=20000)
    else:
        method.train(ds, steps=1000000, batch=256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True,
                    choices=list(DATASET_MAP.keys()))
    ap.add_argument("--method", type=str, required=True,
                    choices=["qdfm","cql","iql","awac","bcq","brac","greedyq"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--K_act", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="results_mujoco")
    ap.add_argument("--e_critic", type=int, default=500000)
    ap.add_argument("--e_qdfm", type=int, default=350000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.task}_{args.method}_K{args.K_act}_s{args.seed}"
    t0 = time.time()
    set_seed(args.seed)

    dataset_name = DATASET_MAP[args.task]
    env_name_key = args.task.rsplit("_", 1)[0]
    env_name = ENV_MAP[env_name_key]

    ds, centroids = make_dataset(dataset_name, args.K_act, args.device, seed=args.seed)
    obs_dim = ds.obs.shape[1]

    method = make_method(args.method, obs_dim, args.K_act, args.device)
    train_method(method, ds, args.method, args.e_critic, args.e_qdfm)
    train_sec = time.time() - t0

    env = gym.make(env_name)
    mean_ret, std_ret, act_ms = evaluate_discrete_policy(
        env, method.act, centroids, args.device, episodes=10, max_steps=1000)

    results = {
        "task": args.task,
        "method": args.method,
        "seed": args.seed,
        "K_act": args.K_act,
        "mean_return": float(mean_ret),
        "std_return": float(std_ret),
        "act_time_ms": float(act_ms),
        "train_sec": round(train_sec, 1),
    }

    with open(out / f"metrics_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
