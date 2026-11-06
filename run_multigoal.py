import argparse, json, os, sys, time, random
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multigoal_env import MultiGoalGridworld, generate_expert_dataset
from algos.qdfm import QDFM
from algos.cql import DiscreteCQL
from algos.iql import DiscreteIQL
from algos.awac import AWAC
from algos.bcq import DiscreteBCQ
from algos.brac import DiscreteBRAC
from algos.greedyq import GreedyQ
from utils import set_seed

NUM_ACTIONS = 4
OBS_DIM = 2


class SimpleDataset:
    def __init__(self, obs, act_idx, rew, next_obs, done):
        self.obs = obs
        self.act_idx = act_idx
        self.rew = rew
        self.next_obs = next_obs
        self.done = done
        self.N = obs.shape[0]


def build_dataset(env, device, n_traj_per_goal=250):
    raw = generate_expert_dataset(env, n_traj_per_goal=n_traj_per_goal)
    return SimpleDataset(
        obs=torch.as_tensor(raw["obs"], device=device),
        act_idx=torch.as_tensor(raw["actions"], device=device, dtype=torch.long),
        rew=torch.as_tensor(raw["rewards"], device=device),
        next_obs=torch.as_tensor(raw["next_obs"], device=device),
        done=torch.as_tensor(raw["done"], device=device),
    )


def make_method(name, device):
    d = device
    h = (256, 256)
    if name == "qdfm":
        return QDFM(OBS_DIM, NUM_ACTIONS, omega_dim=1, hidden=h,
                     gamma=0.99, beta=5.0, M=16, Krenew=50,
                     h=0.05, N=20, device=d)
    if name == "cql":
        return DiscreteCQL(OBS_DIM, NUM_ACTIONS, hidden=h, gamma=0.99, alpha=1.0, device=d)
    if name == "iql":
        return DiscreteIQL(OBS_DIM, NUM_ACTIONS, hidden=h, gamma=0.99,
                           expectile=0.7, aw_temp=3.0, device=d)
    if name == "awac":
        return AWAC(OBS_DIM, NUM_ACTIONS, hidden=h, gamma=0.99, lam=1.0, device=d)
    if name == "bcq":
        return DiscreteBCQ(OBS_DIM, NUM_ACTIONS, hidden=h, gamma=0.99, cand_topk=4, device=d)
    if name == "brac":
        return DiscreteBRAC(OBS_DIM, NUM_ACTIONS, hidden=h, gamma=0.99, alpha=1.0, device=d)
    if name == "greedyq":
        return GreedyQ(OBS_DIM, NUM_ACTIONS, hidden=h, gamma=0.99, device=d)
    raise ValueError(name)


def train_method(method, ds, name):
    if name == "qdfm":
        method.train(ds, k1=3000, k2=20000, k3=10000, batch=256)
    elif name == "brac":
        method.train(ds, steps=30000, batch=256, mu_steps=10000)
    else:
        method.train(ds, steps=30000, batch=256)


@torch.no_grad()
def evaluate(env, act_fn, episodes=200, max_steps=50):
    visit_counts = np.zeros((env.gs, env.gs), dtype=np.int32)
    branch_actions = []
    per_episode = []
    for _ in range(episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        trap = False
        goal_id = -1
        for _ in range(max_steps):
            x, y = env.get_pos()
            visit_counts[y, x] += 1
            a = act_fn(obs)
            if x == 5 and y <= 2:
                branch_actions.append(int(a))
            obs, r, term, trunc, info = env.step(a)
            ep_ret += r
            if info["trap"]:
                trap = True
            if info["goal_id"] >= 0:
                goal_id = info["goal_id"]
            if term or trunc:
                break
        per_episode.append({"return": float(ep_ret), "trap": trap, "goal_id": int(goal_id)})
    returns = [e["return"] for e in per_episode]
    trap_rate = sum(1 for e in per_episode if e["trap"]) / episodes
    goals_hit = set(e["goal_id"] for e in per_episode if e["goal_id"] >= 0)
    mode_cov = len(goals_hit) / env.num_goals
    if branch_actions:
        counts = np.bincount(branch_actions, minlength=NUM_ACTIONS)
        probs = counts / counts.sum()
        log_probs = np.log(probs + 1e-10)
        entropy = float(-np.sum(probs * log_probs))
    else:
        entropy = 0.0
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "trap_rate": float(trap_rate),
        "mode_coverage": float(mode_cov),
        "branch_entropy": float(entropy),
        "goals_reached": sorted(goals_hit),
        "visit_counts": visit_counts,
        "per_episode": per_episode,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, required=True,
                    choices=["qdfm","cql","iql","awac","bcq","brac","greedyq"])
    ap.add_argument("--K", type=int, required=True, choices=[2,3,4,5])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="results_multigoal")
    ap.add_argument("--eval_episodes", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"K{args.K}_{args.method}_s{args.seed}"
    t0 = time.time()
    set_seed(args.seed)

    env = MultiGoalGridworld(K=args.K)
    ds = build_dataset(env, device=args.device, n_traj_per_goal=250)

    method = make_method(args.method, args.device)
    train_method(method, ds, args.method)
    train_sec = time.time() - t0

    res = evaluate(env, method.act, episodes=args.eval_episodes)

    np.save(out / f"visits_{tag}.npy", res["visit_counts"])

    metrics = {k: res[k] for k in
               ["mean_return","std_return","trap_rate","mode_coverage",
                "branch_entropy","goals_reached"]}
    metrics["method"] = args.method
    metrics["K"] = args.K
    metrics["seed"] = args.seed
    metrics["train_sec"] = round(train_sec, 1)
    with open(out / f"metrics_{tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out / f"episodes_{tag}.json", "w") as f:
        json.dump(res["per_episode"], f)


if __name__ == "__main__":
    main()
