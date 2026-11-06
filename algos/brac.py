import torch
import torch.nn.functional as F
from bench.nets import DiscreteQ, CategoricalPolicy
from .common import sample_batch


class DiscreteBRAC:
    name = "brac"

    def __init__(self, obs_dim, K, device="cuda", lr=3e-4, hidden=(256,256),
                 gamma=0.99, alpha=1.0, tau=0.005):
        self.device = torch.device(device)
        self.K = K
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.q = DiscreteQ(obs_dim, K, hidden=hidden).to(self.device)
        self.q_tgt = DiscreteQ(obs_dim, K, hidden=hidden).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.pi = CategoricalPolicy(obs_dim, K, hidden=hidden).to(self.device)
        self.mu = CategoricalPolicy(obs_dim, K, hidden=hidden).to(self.device)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.opt_mu = torch.optim.Adam(self.mu.parameters(), lr=lr)

    def train_mu(self, ds, steps=20000, batch=256):
        for _ in range(steps):
            s, a, _, _, _ = sample_batch(ds, batch)
            logp = torch.log_softmax(self.mu.logits(s), dim=-1)
            loss = F.nll_loss(logp, a)
            self.opt_mu.zero_grad()
            loss.backward()
            self.opt_mu.step()

    @torch.no_grad()
    def act(self, obs_np):
        s = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.pi.logits(s)
        probs = torch.softmax(logits, dim=-1)
        a = torch.argmax(probs, dim=-1)
        return int(a.item())

    def train(self, ds, steps=100000, batch=256, mu_steps=20000):
        self.train_mu(ds, steps=mu_steps, batch=batch)
        for _ in range(steps):
            s, a, r, sp, done = sample_batch(ds, batch)
            with torch.no_grad():
                q_next = self.q_tgt(sp)
                log_pi_next = torch.log_softmax(self.pi.logits(sp), dim=-1)
                pi_next = torch.softmax(self.pi.logits(sp), dim=-1)
                log_mu_next = torch.log_softmax(self.mu.logits(sp), dim=-1)
                v_next = (pi_next * (q_next - self.alpha * (log_pi_next - log_mu_next))).sum(dim=-1)
                y = r + self.gamma * (1.0 - done) * v_next
            q_all = self.q(s)
            q_sa = q_all.gather(1, a.view(-1, 1)).squeeze(1)
            loss_q = F.mse_loss(q_sa, y)
            self.opt_q.zero_grad()
            loss_q.backward()
            self.opt_q.step()
            q_detached = self.q(s).detach()
            log_pi = torch.log_softmax(self.pi.logits(s), dim=-1)
            pi = torch.softmax(self.pi.logits(s), dim=-1)
            log_mu = torch.log_softmax(self.mu.logits(s), dim=-1)
            actor_obj = (pi * (q_detached - self.alpha * (log_pi - log_mu))).sum(dim=-1).mean()
            loss_pi = -actor_obj
            self.opt_pi.zero_grad()
            loss_pi.backward()
            self.opt_pi.step()
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.q_tgt.parameters()):
                    pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
