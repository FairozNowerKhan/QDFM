import torch
import torch.nn.functional as F

from bench.nets import RateNet, DiscreteQ, CategoricalPolicy
from bench.ctmc import bridge_sample, target_rate_vector, ctmc_sample_batch
from .common import sample_batch, soft_value_from_q


class QDFM:
    name = "qdfm"

    def __init__(self, obs_dim, K, omega_dim=1, lr=3e-4, hidden=(256,256),
                 gamma=0.99, beta=20.0, M=64, Krenew=50, h=0.05, N=20, device="cuda"):
        self.device = torch.device(device)
        self.K = K
        self.omega_dim = omega_dim
        self.u = RateNet(obs_dim, K, omega_dim=omega_dim, hidden=hidden).to(self.device)
        self.q = DiscreteQ(obs_dim, K, hidden=hidden).to(self.device)
        self.q_tgt = DiscreteQ(obs_dim, K, hidden=hidden).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.mu = CategoricalPolicy(obs_dim, K, hidden=hidden).to(self.device)
        self.opt_u = torch.optim.Adam(self.u.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.opt_mu = torch.optim.Adam(self.mu.parameters(), lr=lr)
        self.gamma = gamma
        self.beta = beta
        self.M = M
        self.Krenew = Krenew
        self.h = h
        self.N = N

    def _sample_omega(self, B):
        if self.omega_dim == 1:
            return torch.ones(B, 1, device=self.device)
        w = torch.distributions.Dirichlet(
            torch.ones(self.omega_dim, device=self.device)).sample((B,))
        return w

    @torch.no_grad()
    def act(self, obs_np):
        s = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        omega = torch.ones(1, self.omega_dim, device=self.device)
        a = ctmc_sample_batch(self.u, self.mu, s, omega, h=self.h, N=self.N)
        return int(a.item())

    def train_mu(self, ds, steps=20000, batch=256):
        for _ in range(steps):
            s, a, _, _, _ = sample_batch(ds, batch)
            logp = torch.log_softmax(self.mu.logits(s), dim=-1)
            loss = F.nll_loss(logp, a)
            self.opt_mu.zero_grad()
            loss.backward()
            self.opt_mu.step()

    def train(self, ds, k1=5000, k2=50000, k3=30000, batch=256, q_tau=0.005, log_every=5000):
        self.train_mu(ds, steps=20000, batch=batch)

        for it in range(1, k1+1):
            s, a_data, _, _, _ = sample_batch(ds, batch)
            omega = self._sample_omega(batch)
            A0 = torch.randint(0, self.K, (batch,), device=self.device)
            A1 = self.mu.sample(s)
            t = torch.rand(batch, device=self.device)
            At = bridge_sample(A0, A1, t)
            t_in = t.view(-1, 1)
            u_tgt = target_rate_vector(self.K, A0, A1, At, t)
            u_hat = self.u(s, omega, t_in, At)
            loss = F.mse_loss(u_hat, u_tgt)
            self.opt_u.zero_grad()
            loss.backward()
            self.opt_u.step()

        support_cache = None
        for it in range(1, k2+1):
            s, a, r, sp, done = sample_batch(ds, batch)
            omega = self._sample_omega(batch)
            if (it % self.Krenew) == 1 or support_cache is None:
                sp_rep = sp.repeat_interleave(self.M, dim=0)
                a_supp = self.mu.sample(sp_rep).view(batch, self.M)
                support_cache = a_supp
            a_supp = support_cache
            with torch.no_grad():
                q_next_all = self.q_tgt(sp)
                q_supp = q_next_all.gather(1, a_supp)
                V = soft_value_from_q(q_supp, beta=self.beta)
                y = r + self.gamma * (1.0 - done) * V
            q_all = self.q(s)
            q_sa = q_all.gather(1, a.view(-1,1)).squeeze(1)
            loss_q = F.mse_loss(q_sa, y)
            self.opt_q.zero_grad()
            loss_q.backward()
            self.opt_q.step()
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.q_tgt.parameters()):
                    pt.data.mul_(1.0 - q_tau).add_(q_tau * p.data)

        endpoint_cache = None
        for it in range(1, k3+1):
            s, a0_data, _, _, _ = sample_batch(ds, batch)
            omega = self._sample_omega(batch)
            A0 = a0_data
            if (it % self.Krenew) == 1 or endpoint_cache is None:
                s_rep = s.repeat_interleave(self.M, dim=0)
                o_rep = omega.repeat_interleave(self.M, dim=0)
                A1 = ctmc_sample_batch(self.u, self.mu, s_rep, o_rep, h=self.h, N=self.N)
                endpoint_cache = A1.view(batch, self.M)
            A1_supp = endpoint_cache
            with torch.no_grad():
                q_all = self.q(s)
                q_supp = q_all.gather(1, A1_supp)
                w = torch.softmax(self.beta * q_supp, dim=1)
            t = torch.rand(batch, self.M, device=self.device)
            At = bridge_sample(
                A0.view(-1,1).expand(-1,self.M).reshape(-1),
                A1_supp.reshape(-1),
                t.reshape(-1)
            ).view(batch, self.M)
            s_rep = s.repeat_interleave(self.M, dim=0)
            o_rep = omega.repeat_interleave(self.M, dim=0)
            A0_rep = A0.view(-1,1).expand(-1,self.M).reshape(-1)
            A1_rep = A1_supp.reshape(-1)
            At_rep = At.reshape(-1)
            t_rep = t.reshape(-1)
            u_tgt = target_rate_vector(self.K, A0_rep, A1_rep, At_rep, t_rep)
            u_hat = self.u(s_rep, o_rep, t_rep.view(-1,1), At_rep)
            per = (u_hat - u_tgt).pow(2).mean(dim=1).view(batch, self.M)
            loss = (w * per).sum(dim=1).mean()
            self.opt_u.zero_grad()
            loss.backward()
            self.opt_u.step()
