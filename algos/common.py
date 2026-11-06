import torch
import torch.nn.functional as F


def sample_batch(ds, batch_size):
    idx = torch.randint(0, ds.N, (batch_size,), device=ds.obs.device)
    return (
        ds.obs[idx],
        ds.act_idx[idx],
        ds.rew[idx],
        ds.next_obs[idx],
        ds.done[idx],
    )


def dqn_target(q_next_all, done, rew, gamma):
    v = q_next_all.max(dim=1).values
    y = rew + gamma * (1.0 - done) * v
    return y


def soft_value_from_q(q_vals, beta):
    w = torch.softmax(beta * q_vals, dim=-1)
    return (w * q_vals).sum(dim=-1)
