import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256), act=nn.ReLU):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), act()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, K, hidden=(256, 256)):
        super().__init__()
        self.mlp = MLP(obs_dim, K, hidden=hidden)

    def logits(self, s):
        return self.mlp(s)

    def probs(self, s):
        return torch.softmax(self.logits(s), dim=-1)

    @torch.no_grad()
    def sample(self, s):
        p = self.probs(s)
        return torch.multinomial(p, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def mode(self, s):
        return self.logits(s).argmax(dim=-1)


class DiscreteQ(nn.Module):
    def __init__(self, obs_dim, K, hidden=(256, 256)):
        super().__init__()
        self.mlp = MLP(obs_dim, K, hidden=hidden)

    def forward(self, s):
        return self.mlp(s)


class RateNet(nn.Module):
    def __init__(self, obs_dim, K, omega_dim=1, hidden=(256, 256)):
        super().__init__()
        self.K = K
        self.in_dim = obs_dim + omega_dim + 1 + K
        self.mlp = MLP(self.in_dim, K, hidden=hidden)

    def forward(self, s, omega, t, a_cur_idx):
        B = s.shape[0]
        a_one = torch.zeros(B, self.K, device=s.device)
        a_one.scatter_(1, a_cur_idx.view(-1, 1), 1.0)
        x = torch.cat([s, omega, t, a_one], dim=-1)
        rates = torch.relu(self.mlp(x))
        return rates
