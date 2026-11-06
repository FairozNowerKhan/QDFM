import torch
import torch.nn.functional as F

from bench.nets import RateNet, CategoricalPolicy
from bench.nets_marl import VectorQ
from bench.ctmc import bridge_sample, target_rate_vector, ctmc_sample_batch
from .common import sample_batch


class QDFM_MO:
    name = "qdfm_mo"

    def __init__(self, obs_dim, K, n_objectives, omega_dim=None, lr=3e-4,
                 hidden=(256,256), gamma=0.99, beta=20.0, M=64, Krenew=50,
                 h=0.05, N=20, device="cuda"):
        self.device = torch.device(device)
        self.K = K
        self.n_obj = n_objectives
        self.omega_dim = omega_dim if omega_dim else n_objectives
        self.u = RateNet(obs_dim, K, omega_dim=self.omega_dim, hidden=hidden).to(self.device)
        self.q = VectorQ(obs_dim, K, n_objectives, hidden=hidden).to(self.device)
        self.q_tgt = VectorQ(obs_dim, K, n_objectives, hidden=hidden).to(self.device)
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
        w = torch.distributions.Dirichlet(
            torch.ones(self.omega_dim, device=self.device)).sample((B,))
        return w

    def _scalarize_q(self, q_vec, omega):
        return (q_vec * omega.unsqueeze(1)).sum(dim=-1)

    @torch.no_grad()
    def act(self, obs_np, omega=None):
        s = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        if omega is None:
            omega = torch.ones(1, self.omega_dim, device=self.device) / self.omega_dim
        else:
            omega = torch.as_tensor(omega, device=self.device, dtype=torch.float32).unsqueeze(0)
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

    def train(self, ds, k1=5000, k2=50000, k3=30000, batch=256, q_tau=0.005):
        self.train_mu(ds, steps=20000, batch=batch)

        for it in range(1, k1+1):
            s, a_data, _, _, _ = sample_batch(ds, batch)
            omega = self._sample_omega(batch)
            A0 = torch.randint(0, self.K, (batch,), device=self.device)
            A1 = self.mu.sample(s)
            t = torch.rand(batch, device=self.device)
            At = bridge_sample(A0, A1, t)
            u_tgt = target_rate_vector(self.K, A0, A1, At, t)
            u_hat = self.u(s, omega, t.view(-1,1), At)
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
                q_vec_next = self.q_tgt(sp)
                q_scalar_all = self._scalarize_q(q_vec_next, omega)
                q_supp = q_scalar_all.gather(1, a_supp)
                w_soft = torch.softmax(self.beta * q_supp, dim=-1)
                V = (w_soft * q_supp).sum(dim=-1)
                r_scalar = (r.view(batch, -1) * omega).sum(dim=-1) if r.dim() > 1 else r
                y = r_scalar + self.gamma * (1.0 - done) * V
            q_vec = self.q(s)
            q_scalar = self._scalarize_q(q_vec, omega)
            q_sa = q_scalar.gather(1, a.view(-1,1)).squeeze(1)
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
                q_vec = self.q(s)
                q_scalar = self._scalarize_q(q_vec, omega)
                q_supp = q_scalar.gather(1, A1_supp)
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
