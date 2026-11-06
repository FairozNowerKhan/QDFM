import time
import numpy as np
import torch


@torch.no_grad()
def evaluate_discrete_policy(env, policy_act_fn, centroids, device, episodes=10, max_steps=1000):
    returns = []
    act_times = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < max_steps:
            t0 = time.time()
            a_idx = policy_act_fn(obs)
            t1 = time.time()
            act_times.append((t1 - t0) * 1000.0)
            a_cont = centroids[int(a_idx)]
            obs, rew, term, trunc, _ = env.step(a_cont)
            done = bool(term or trunc)
            ep_ret += float(rew)
            steps += 1
        returns.append(ep_ret)
    returns = np.asarray(returns, dtype=np.float64)
    act_times = np.asarray(act_times, dtype=np.float64)
    return returns.mean(), returns.std(ddof=1) if len(returns) > 1 else 0.0, act_times.mean()
