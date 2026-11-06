import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench.nets import CategoricalPolicy
from bench.ctmc import ctmc_sample_batch
from utils import set_seed

try:
    import gymnasium as gym
except ImportError:
    gym = None


class FixedRateNet:
    def __init__(self, policy, alpha=5.0):
        self.policy = policy
        self.alpha = alpha

    def __call__(self, s, omega, t, a):
        probs = self.policy.probs(s)
        rates = self.alpha * probs
        return rates

    def parameters(self):
        return self.policy.parameters()

    def to(self, device):
        self.policy = self.policy.to(device)
        return self


def train_reference_policy(env_name, device, steps=10000):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    K = env.action_space.n
    pi = CategoricalPolicy(obs_dim, K).to(device)
    opt = torch.optim.Adam(pi.parameters(), lr=1e-3)

    obs_l, act_l = [], []
    for _ in range(200):
        o, _ = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            o2, r, term, trunc, _ = env.step(a)
            obs_l.append(o)
            act_l.append(a)
            o = o2
            done = term or trunc

    obs_t = torch.as_tensor(np.array(obs_l, dtype=np.float32), device=device)
    act_t = torch.as_tensor(np.array(act_l, dtype=np.int64), device=device)
    N = obs_t.shape[0]

    for _ in range(steps):
        idx = torch.randint(0, N, (256,), device=device)
        logp = torch.log_softmax(pi.logits(obs_t[idx]), dim=-1)
        loss = torch.nn.functional.nll_loss(logp, act_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    return pi, K, obs_dim


def evaluate_policy(env_name, act_fn, episodes=200, max_steps=500):
    env = gym.make(env_name)
    returns = []
    for _ in range(episodes):
        o, _ = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < max_steps:
            a = act_fn(o)
            o, r, term, trunc, _ = env.step(a)
            ep_ret += r
            done = term or trunc
            steps += 1
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="CartPole-v1",
                    choices=["CartPole-v1", "Acrobot-v1"])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default="results_ctmc_check")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    pi, K, obs_dim = train_reference_policy(args.env, args.device)

    results = {}

    def random_act(o):
        return np.random.randint(0, K)

    m, s = evaluate_policy(args.env, random_act)
    results["random"] = {"mean": m, "std": s}

    def sample_act(o):
        s_t = torch.as_tensor(o, dtype=torch.float32, device=args.device).unsqueeze(0)
        return int(pi.sample(s_t).item())
    m, s = evaluate_policy(args.env, sample_act)
    results["sample_pi"] = {"mean": m, "std": s}

    def argmax_act(o):
        s_t = torch.as_tensor(o, dtype=torch.float32, device=args.device).unsqueeze(0)
        return int(pi.mode(s_t).item())
    m, s = evaluate_policy(args.env, argmax_act)
    results["argmax_pi"] = {"mean": m, "std": s}

    for alpha in [1.0, 2.0, 5.0, 10.0]:
        rate_net = FixedRateNet(pi, alpha=alpha)
        omega_dim = 1

        def ctmc_act(o, rn=rate_net):
            s_t = torch.as_tensor(o, dtype=torch.float32, device=args.device).unsqueeze(0)
            omega = torch.ones(1, omega_dim, device=args.device)
            a = ctmc_sample_batch(rn, pi, s_t, omega, h=0.05, N=20)
            return int(a.item())

        m, s = evaluate_policy(args.env, ctmc_act)
        results[f"ctmc_alpha{alpha}"] = {"mean": m, "std": s}

    with open(out / f"metrics_ctmc_{args.env}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"CTMC check on {args.env} complete")


if __name__ == "__main__":
    main()
