import torch
import torch.nn.functional as F
from bench.nets import DiscreteQ, CategoricalPolicy
from .common import sample_batch, soft_value_from_q


class AWAC:
    name = "awac"

    def __init__(self, obs_dim, K, lr=3e-4, hidden=(256,256), gamma=0.99, lam=1.0, device="cuda"):
        self.device = torch.device(device)
        self.K = K
        self.q = DiscreteQ(obs_dim, K, hidden=hidden).to(self.device)
        self.tgt = DiscreteQ(obs_dim, K, hidden=hidden).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.pi = CategoricalPolicy(obs_dim, K, hidden=hidden).to(self.device)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam

    @torch.no_grad()
    def act(self, obs_np):
        s = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        return int(self.pi.mode(s).item())

    def train(self, ds, steps=100000, batch=256, target_tau=0.005, log_every=5000):
        for it in range(1, steps+1):
            s, a, r, sp, done = sample_batch(ds, batch)
            q_sa = self.q(s).gather(1, a.view(-1,1)).squeeze(1)
            with torch.no_grad():
                q_next = self.tgt(sp)
                y = r + self.gamma * (1.0 - done) * q_next.max(dim=1).values
            loss_q = F.mse_loss(q_sa, y)
            self.opt_q.zero_grad()
            loss_q.backward()
            self.opt_q.step()
            with torch.no_grad():
                q_all = self.q(s)
                pi_probs = self.pi.probs(s)
                v = (pi_probs * q_all).sum(dim=1)
                adv = q_sa - v
                w = torch.exp(adv / self.lam).clamp(max=100.0)
            logp = torch.log_softmax(self.pi.logits(s), dim=-1)
            logp_a = logp.gather(1, a.view(-1,1)).squeeze(1)
            loss_pi = -(w * logp_a).mean()
            self.opt_pi.zero_grad()
            loss_pi.backward()
            self.opt_pi.step()
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.tgt.parameters()):
                    pt.data.mul_(1.0 - target_tau).add_(target_tau * p.data)
