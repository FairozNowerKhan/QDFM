import torch
import torch.nn as nn
import torch.nn.functional as F
from bench.nets import DiscreteQ, CategoricalPolicy, MLP
from .common import sample_batch


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=(256,256)):
        super().__init__()
        self.mlp = MLP(obs_dim, 1, hidden=hidden)

    def forward(self, s):
        return self.mlp(s).squeeze(-1)


def expectile_loss(diff, tau):
    w = torch.where(diff > 0, tau, 1.0 - tau)
    return (w * diff.pow(2)).mean()


class DiscreteIQL:
    name = "iql"

    def __init__(self, obs_dim, K, lr=3e-4, hidden=(256,256), gamma=0.99,
                 expectile=0.7, aw_temp=3.0, device="cuda"):
        self.device = torch.device(device)
        self.K = K
        self.q = DiscreteQ(obs_dim, K, hidden=hidden).to(self.device)
        self.tgt = DiscreteQ(obs_dim, K, hidden=hidden).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.v = ValueNet(obs_dim, hidden=hidden).to(self.device)
        self.pi = CategoricalPolicy(obs_dim, K, hidden=hidden).to(self.device)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.gamma = gamma
        self.expectile = expectile
        self.aw_temp = aw_temp

    @torch.no_grad()
    def act(self, obs_np):
        s = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        return int(self.pi.mode(s).item())

    def train(self, ds, steps=100000, batch=256, target_tau=0.005, log_every=5000):
        for it in range(1, steps+1):
            s, a, r, sp, done = sample_batch(ds, batch)
            with torch.no_grad():
                v_next = self.v(sp)
                y = r + self.gamma * (1.0 - done) * v_next
            q_sa = self.q(s).gather(1, a.view(-1,1)).squeeze(1)
            loss_q = F.mse_loss(q_sa, y)
            self.opt_q.zero_grad()
            loss_q.backward()
            self.opt_q.step()
            with torch.no_grad():
                q_sa_det = self.q(s).gather(1, a.view(-1,1)).squeeze(1)
            v_s = self.v(s)
            loss_v = expectile_loss(q_sa_det - v_s, self.expectile)
            self.opt_v.zero_grad()
            loss_v.backward()
            self.opt_v.step()
            with torch.no_grad():
                adv = q_sa_det - self.v(s)
                w = torch.exp(adv / self.aw_temp).clamp(max=100.0)
            logp = torch.log_softmax(self.pi.logits(s), dim=-1)
            logp_a = logp.gather(1, a.view(-1,1)).squeeze(1)
            loss_pi = -(w * logp_a).mean()
            self.opt_pi.zero_grad()
            loss_pi.backward()
            self.opt_pi.step()
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.tgt.parameters()):
                    pt.data.mul_(1.0 - target_tau).add_(target_tau * p.data)
