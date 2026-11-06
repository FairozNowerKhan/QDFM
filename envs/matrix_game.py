import numpy as np


class TwoAgentMatrixGame:
    def __init__(self):
        self.obs_dim = 2
        self.n_agents = 2
        self.K_per_agent = 3
        self.n_objectives = 2
        self.state = 0

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.state = rng.integers(0, 2)
        return self._obs(), {}

    def step(self, actions):
        a1, a2 = actions
        coordinated = float(a1 == a2)
        r1 = coordinated * (3.0 + self.state) + (1.0 - coordinated) * 1.0
        cost = coordinated * (a1 + 1.0)
        r2 = -cost
        reward = np.array([r1, r2], dtype=np.float32)
        return self._obs(), reward, True, False, {}

    def _obs(self):
        return np.array([float(self.state), 0.0], dtype=np.float32)


def generate_matrix_game_dataset(n_episodes=2000):
    env = TwoAgentMatrixGame()
    obs_l, a1_l, a2_l, rew_l, nobs_l, done_l = [], [], [], [], [], []
    rng = np.random.default_rng(42)
    for _ in range(n_episodes):
        o, _ = env.reset()
        if rng.random() < 0.5:
            a1, a2 = 0, 0
        else:
            a1, a2 = 1, 1
        if rng.random() < 0.1:
            a1 = rng.integers(0, 3)
        if rng.random() < 0.1:
            a2 = rng.integers(0, 3)
        o2, r, term, trunc, info = env.step([a1, a2])
        obs_l.append(o)
        a1_l.append(a1)
        a2_l.append(a2)
        rew_l.append(r)
        nobs_l.append(o2)
        done_l.append(1.0)
    return {
        "obs": np.array(obs_l, dtype=np.float32),
        "a1": np.array(a1_l, dtype=np.int64),
        "a2": np.array(a2_l, dtype=np.int64),
        "rewards": np.array(rew_l, dtype=np.float32),
        "next_obs": np.array(nobs_l, dtype=np.float32),
        "done": np.array(done_l, dtype=np.float32),
    }
