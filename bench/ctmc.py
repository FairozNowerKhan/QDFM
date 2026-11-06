import torch


def bridge_sample(A0, A1, t_u):
    u = torch.rand_like(t_u)
    pick1 = (u < t_u)
    At = torch.where(pick1, A1, A0)
    return At


def target_rate_vector(K, A0, A1, At, t_u, eps=1e-4):
    B = At.shape[0]
    rates = torch.zeros(B, K, device=At.device)
    mask = (At == A0) & (A0 != A1)
    if mask.any():
        lam = 1.0 / (1.0 - t_u[mask] + eps)
        rates[mask, A1[mask]] = lam
    return rates


@torch.no_grad()
def ctmc_sample_batch(rate_net, policy_mu, s, omega, h=0.05, N=20, a0=None):
    device = s.device
    if a0 is None:
        a = policy_mu.sample(s)
    else:
        a = a0.clone()

    B = a.shape[0]
    for n in range(N):
        t = torch.full((B, 1), float(n) * h, device=device)
        rates = rate_net(s, omega, t, a)
        rates.scatter_(1, a.view(-1, 1), 0.0)
        lam = rates.sum(dim=1)
        stay_prob = torch.clamp(1.0 - h * lam, 0.0, 1.0)
        u = torch.rand(B, device=device)
        stay = (u < stay_prob)
        jump = ~stay
        if jump.any():
            probs = rates[jump] / (lam[jump].view(-1, 1) + 1e-8)
            a_new = torch.multinomial(probs, num_samples=1).squeeze(-1)
            a[jump] = a_new
    return a
