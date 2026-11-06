import torch
import torch.nn.functional as F
from bench.nets_marl import VectorQ
from .common import sample_batch


class DiscreteCQL_MO:
    name = "cql_mo"

    def __init__(self, obs_dim, K, n_objectives, lr=3e-4, hidden=(256,256),
                 gamma=0.99, alpha=1.0, device="cuda"):
        self.device = torch.device(device)
        self.K = K
        self.n_obj = n_objectives
        self.q = VectorQ(obs_dim, K, n_objectives, hidden=hidden).to(self.device)
        self.tgt = VectorQ(obs_dim, K, n_objectives, hidden=hidden).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.alpha = alpha

    @torch.no_grad()
    def act(self, obs_np, omega=None):
        s = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        if omega is None:
            omega = torch.ones(self.n_obj, device=self.device) / self.n_obj
        else:
            omega = torch.as_tensor(omega, device=self.device, dtype=torch.float32)
        q_vec = self.q(s).squeeze(0)
        q_scalar = (q_vec * omega.unsqueeze(0)).sum(dim=-1)
        return int(torch.argmax(q_scalar).item())

    def train(self, ds, steps=100000, batch=256, target_tau=0.005):
        for it in range(1, steps+1):
            s, a, r, sp, done = sample_batch(ds, batch)
            omega = torch.distributions.Dirichlet(
                torch.ones(self.n_obj, device=self.device)).sample((batch,))
            q_vec_all = self.q(s)
            q_scalar_all = (q_vec_all * omega.unsqueeze(1)).sum(dim=-1)
            q_sa = q_scalar_all.gather(1, a.view(-1,1)).squeeze(1)
            with torch.no_grad():
                q_vec_next = self.tgt(sp)
                q_scalar_next = (q_vec_next * omega.unsqueeze(1)).sum(dim=-1)
                v_next = q_scalar_next.max(dim=1).values
                r_scalar = (r.view(batch, -1) * omega).sum(dim=-1) if r.dim() > 1 else r
                y = r_scalar + self.gamma * (1.0 - done) * v_next
            bellman = F.mse_loss(q_sa, y)
            cql_pen = (torch.logsumexp(q_scalar_all, dim=1) - q_sa).mean()
            loss = bellman + self.alpha * cql_pen
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.tgt.parameters()):
                    pt.data.mul_(1.0 - target_tau).add_(target_tau * p.data)
