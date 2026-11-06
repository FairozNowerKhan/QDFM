import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench.nets import MLP, CategoricalPolicy, DiscreteQ
from bench.nets_marl import QMIXMixer
from algos.common import sample_batch
from utils import set_seed

try:
    from smac.env import StarCraft2Env
except ImportError:
    StarCraft2Env = None


class SMACDataset:
    def __init__(self, obs, act_idx, rew, next_obs, done, state, next_state, device):
        self.obs = torch.as_tensor(obs, device=device)
        self.act_idx = torch.as_tensor(act_idx, device=device, dtype=torch.long)
        self.rew = torch.as_tensor(rew, device=device)
        self.next_obs = torch.as_tensor(next_obs, device=device)
        self.done = torch.as_tensor(done, device=device)
        self.state = torch.as_tensor(state, device=device)
        self.next_state = torch.as_tensor(next_state, device=device)
        self.N = obs.shape[0]


def sample_smac_batch(ds, batch_size):
    idx = torch.randint(0, ds.N, (batch_size,), device=ds.obs.device)
    return (ds.obs[idx], ds.act_idx[idx], ds.rew[idx], ds.next_obs[idx],
            ds.done[idx], ds.state[idx], ds.next_state[idx])


class SMAC_BC:
    def __init__(self, obs_dim, K, n_agents, hidden=(256,256), lr=3e-4, device="cuda"):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.K = K
        self.policies = [CategoricalPolicy(obs_dim, K, hidden=hidden).to(self.device)
                         for _ in range(n_agents)]
        self.opts = [torch.optim.Adam(p.parameters(), lr=lr) for p in self.policies]

    def train(self, ds, steps=100000, batch=256):
        for it in range(1, steps+1):
            idx = torch.randint(0, ds.N, (batch,), device=ds.obs.device)
            s = ds.obs[idx]
            for i in range(self.n_agents):
                a = ds.act_idx[idx, i] if ds.act_idx.dim() > 1 else ds.act_idx[idx]
                logp = torch.log_softmax(self.policies[i].logits(s), dim=-1)
                loss = F.nll_loss(logp, a)
                self.opts[i].zero_grad()
                loss.backward()
                self.opts[i].step()

    def act(self, obs_np):
        s = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        actions = []
        for i in range(self.n_agents):
            a = int(self.policies[i].mode(s).item())
            actions.append(a)
        return actions


class SMAC_CQL:
    def __init__(self, obs_dim, K, n_agents, state_dim, hidden=(256,256),
                 lr=3e-4, gamma=0.99, alpha=1.0, device="cuda"):
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.K = K
        self.gamma = gamma
        self.alpha = alpha
        self.agent_qs = [DiscreteQ(obs_dim, K, hidden=hidden).to(self.device)
                         for _ in range(n_agents)]
        self.agent_qs_tgt = [DiscreteQ(obs_dim, K, hidden=hidden).to(self.device)
                             for _ in range(n_agents)]
        for i in range(n_agents):
            self.agent_qs_tgt[i].load_state_dict(self.agent_qs[i].state_dict())
        self.mixer = QMIXMixer(n_agents, state_dim).to(self.device)
        self.mixer_tgt = QMIXMixer(n_agents, state_dim).to(self.device)
        self.mixer_tgt.load_state_dict(self.mixer.state_dict())
        params = []
        for q in self.agent_qs:
            params += list(q.parameters())
        params += list(self.mixer.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)

    def train(self, ds, steps=100000, batch=256, tau=0.005):
        for it in range(1, steps+1):
            idx = torch.randint(0, ds.N, (batch,), device=ds.obs.device)
            s = ds.obs[idx]
            sp = ds.next_obs[idx]
            r = ds.rew[idx]
            done = ds.done[idx]
            state = ds.state[idx]
            next_state = ds.next_state[idx]

            agent_q_vals = []
            agent_q_next_max = []
            cql_loss = 0.0
            for i in range(self.n_agents):
                a_i = ds.act_idx[idx, i] if ds.act_idx.dim() > 1 else ds.act_idx[idx]
                q_all = self.agent_qs[i](s)
                q_a = q_all.gather(1, a_i.view(-1,1)).squeeze(1)
                agent_q_vals.append(q_a)
                cql_loss += (torch.logsumexp(q_all, dim=1) - q_a).mean()
                with torch.no_grad():
                    q_next = self.agent_qs_tgt[i](sp)
                    agent_q_next_max.append(q_next.max(dim=1).values)

            q_tot = self.mixer(torch.stack(agent_q_vals, dim=1), state)
            with torch.no_grad():
                q_tot_next = self.mixer_tgt(torch.stack(agent_q_next_max, dim=1), next_state)
                y = r + self.gamma * (1.0 - done) * q_tot_next

            bellman = F.mse_loss(q_tot, y)
            loss = bellman + self.alpha * cql_loss / self.n_agents
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            with torch.no_grad():
                for i in range(self.n_agents):
                    for p, pt in zip(self.agent_qs[i].parameters(), self.agent_qs_tgt[i].parameters()):
                        pt.data.mul_(1.0 - tau).add_(tau * p.data)
                for p, pt in zip(self.mixer.parameters(), self.mixer_tgt.parameters()):
                    pt.data.mul_(1.0 - tau).add_(tau * p.data)

    def act(self, obs_np):
        s = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        actions = []
        for i in range(self.n_agents):
            q = self.agent_qs[i](s).squeeze(0)
            actions.append(int(torch.argmax(q).item()))
        return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=str, required=True)
    ap.add_argument("--method", type=str, required=True, choices=["bc", "cql"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="results_smac")
    ap.add_argument("--data_path", type=str, required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    data = np.load(args.data_path, allow_pickle=True).item()
    obs = data["obs"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    rewards = data["rewards"].astype(np.float32)
    next_obs = data["next_obs"].astype(np.float32)
    dones = data["dones"].astype(np.float32)
    states = data.get("states", obs).astype(np.float32)
    next_states = data.get("next_states", next_obs).astype(np.float32)

    obs_dim = obs.shape[-1]
    n_agents = actions.shape[1] if actions.ndim > 1 else 1
    K = int(actions.max()) + 1
    state_dim = states.shape[-1]

    ds = SMACDataset(obs, actions, rewards, next_obs, dones, states, next_states, args.device)

    if args.method == "bc":
        method = SMAC_BC(obs_dim, K, n_agents, device=args.device)
        method.train(ds, steps=100000, batch=256)
    else:
        method = SMAC_CQL(obs_dim, K, n_agents, state_dim, device=args.device)
        method.train(ds, steps=100000, batch=256)

    tag = f"{args.map}_{args.method}_s{args.seed}"
    results = {
        "map": args.map,
        "method": args.method,
        "seed": args.seed,
    }
    with open(out / f"metrics_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"SMAC {args.map} {args.method} complete")


if __name__ == "__main__":
    main()
