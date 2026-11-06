import numpy as np

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
_DELTAS = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}


class MultiObjectiveFork:
    def __init__(self, grid_size=5, max_steps=30):
        self.gs = grid_size
        self.max_steps = max_steps
        self.start = (2, 0)
        self.goal_a = (0, 4)
        self.goal_b = (4, 4)
        self.trap = (2, 2)
        self.obs_dim = 2
        self.num_actions = 4
        self.n_objectives = 2
        self.x, self.y, self.steps = 2, 0, 0

    def reset(self, seed=None):
        self.x, self.y = self.start
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        dx, dy = _DELTAS[action]
        self.x = max(0, min(self.gs - 1, self.x + dx))
        self.y = max(0, min(self.gs - 1, self.y + dy))
        self.steps += 1
        info = {}
        if (self.x, self.y) == self.trap:
            return self._obs(), np.array([-10.0, -10.0], dtype=np.float32), True, False, info
        if (self.x, self.y) == self.goal_a:
            return self._obs(), np.array([10.0, 0.0], dtype=np.float32), True, False, info
        if (self.x, self.y) == self.goal_b:
            return self._obs(), np.array([0.0, 10.0], dtype=np.float32), True, False, info
        truncated = self.steps >= self.max_steps
        return self._obs(), np.array([0.0, 0.0], dtype=np.float32), False, truncated, info

    def _obs(self):
        return np.array([self.x / (self.gs - 1), self.y / (self.gs - 1)], dtype=np.float32)

    def get_pos(self):
        return self.x, self.y


def generate_fork_dataset(env, n_traj_per_goal=200):
    obs_l, act_l, rew_l, nobs_l, done_l = [], [], [], [], []
    for goal in [env.goal_a, env.goal_b]:
        for _ in range(n_traj_per_goal):
            o, _ = env.reset()
            gx, gy = goal
            for _ in range(env.max_steps):
                x, y = env.get_pos()
                if x == 2 and y >= 1:
                    a = LEFT if gx < 2 else RIGHT
                elif x != gx:
                    a = LEFT if gx < x else RIGHT
                elif y < gy:
                    a = UP
                else:
                    a = UP
                o2, r, term, trunc, info = env.step(a)
                obs_l.append(o)
                act_l.append(a)
                rew_l.append(r)
                nobs_l.append(o2)
                done_l.append(float(term or trunc))
                o = o2
                if term or trunc:
                    break
    return {
        "obs": np.array(obs_l, dtype=np.float32),
        "actions": np.array(act_l, dtype=np.int64),
        "rewards": np.array(rew_l, dtype=np.float32),
        "next_obs": np.array(nobs_l, dtype=np.float32),
        "done": np.array(done_l, dtype=np.float32),
    }
