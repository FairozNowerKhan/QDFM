import torch
import torch.nn as nn
from bench.nets import MLP


class PerAgentRateNet(nn.Module):
    def __init__(self, obs_dim, K_per_agent, n_agents, omega_dim=1, hidden=(256, 256)):
        super().__init__()
        self.n_agents = n_agents
        self.K = K_per_agent
        in_dim = obs_dim + omega_dim + 1 + n_agents * K_per_agent
        self.nets = nn.ModuleList([
            MLP(in_dim, K_per_agent, hidden=hidden) for _ in range(n_agents)
        ])

    def forward(self, s, omega, t, joint_a_idx):
        B = s.shape[0]
        a_one = torch.zeros(B, self.n_agents * self.K, device=s.device)
        for i in range(self.n_agents):
            offset = i * self.K
            a_one.scatter_(1, joint_a_idx[:, i:i+1] + offset, 1.0)
        x = torch.cat([s, omega, t, a_one], dim=-1)
        rates = []
        for i in range(self.n_agents):
            r_i = torch.relu(self.nets[i](x))
            rates.append(r_i)
        return rates


class VectorQ(nn.Module):
    def __init__(self, obs_dim, K, n_objectives, hidden=(256, 256)):
        super().__init__()
        self.K = K
        self.n_obj = n_objectives
        self.mlp = MLP(obs_dim, K * n_objectives, hidden=hidden)

    def forward(self, s):
        out = self.mlp(s)
        B = s.shape[0]
        return out.view(B, self.K, self.n_obj)


class QMIXMixer(nn.Module):
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, agent_qs, state):
        B = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, -1)
        b1 = self.hyper_b1(state).unsqueeze(1)
        h = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).unsqueeze(-1)
        b2 = self.hyper_b2(state).unsqueeze(1)
        out = torch.bmm(h, w2) + b2
        return out.squeeze(-1).squeeze(-1)
