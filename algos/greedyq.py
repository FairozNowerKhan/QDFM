import torch
import torch.nn.functional as F
from bench.nets import DiscreteQ
from .common import sample_batch, dqn_target


class GreedyQ:
    name = "greedyq"

    def __init__(self, obs_dim, K, lr=3e-4, hidden=(256,256), gamma=0.99, device="cuda"):
        self.device = torch.device(device)
        self.K = K
        self.q = DiscreteQ(obs_dim, K, hidden=hidden).to(self.device)
        self.tgt = DiscreteQ(obs_dim, K, hidden=hidden).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma

    @torch.no_grad()
    def act(self, obs_np):
        s = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        q = self.q(s).squeeze(0)
        return int(torch.argmax(q).item())

    def train(self, ds, steps=100000, batch=256, target_tau=0.005, log_every=5000):
        for it in range(1, steps+1):
            s, a, r, sp, done = sample_batch(ds, batch)
            q_sa = self.q(s).gather(1, a.view(-1,1)).squeeze(1)
            with torch.no_grad():
                q_next = self.tgt(sp)
                y = dqn_target(q_next, done, r, self.gamma)
            loss = F.mse_loss(q_sa, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.tgt.parameters()):
                    pt.data.mul_(1.0 - target_tau).add_(target_tau * p.data)
