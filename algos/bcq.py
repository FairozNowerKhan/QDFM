import torch
import torch.nn.functional as F
from bench.nets import DiscreteQ, CategoricalPolicy
from .common import sample_batch


class DiscreteBCQ:
    name = "bcq"

    def __init__(self, obs_dim, K, lr=3e-4, hidden=(256,256), gamma=0.99, cand_topk=10, device="cuda"):
        self.device = torch.device(device)
        self.K = K
        self.q = DiscreteQ(obs_dim, K, hidden=hidden).to(self.device)
        self.tgt = DiscreteQ(obs_dim, K, hidden=hidden).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.mu = CategoricalPolicy(obs_dim, K, hidden=hidden).to(self.device)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.opt_mu = torch.optim.Adam(self.mu.parameters(), lr=lr)
        self.gamma = gamma
        self.cand_topk = cand_topk

    @torch.no_grad()
    def act(self, obs_np):
        s = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        probs = self.mu.probs(s).squeeze(0)
        topk = torch.topk(probs, k=min(self.cand_topk, self.K)).indices
        q = self.q(s).squeeze(0)
        a = topk[torch.argmax(q[topk])]
        return int(a.item())

    def train(self, ds, steps=100000, batch=256, target_tau=0.005, log_every=5000):
        for it in range(1, steps+1):
            s, a, r, sp, done = sample_batch(ds, batch)
            logp = torch.log_softmax(self.mu.logits(s), dim=-1)
            loss_mu = F.nll_loss(logp, a)
            self.opt_mu.zero_grad()
            loss_mu.backward()
            self.opt_mu.step()
            q_sa = self.q(s).gather(1, a.view(-1,1)).squeeze(1)
            with torch.no_grad():
                q_next_all = self.tgt(sp)
                probs = self.mu.probs(sp)
                topk_idx = torch.topk(probs, k=min(self.cand_topk, self.K), dim=1).indices
                q_cand = q_next_all.gather(1, topk_idx)
                v = q_cand.max(dim=1).values
                y = r + self.gamma * (1.0 - done) * v
            loss_q = F.mse_loss(q_sa, y)
            self.opt_q.zero_grad()
            loss_q.backward()
            self.opt_q.step()
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.tgt.parameters()):
                    pt.data.mul_(1.0 - target_tau).add_(target_tau * p.data)
