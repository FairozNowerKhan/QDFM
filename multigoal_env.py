import numpy as np

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
_DELTAS = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}

_GOAL_X = {
    2: [2, 8],
    3: [1, 5, 9],
    4: [0, 3, 7, 10],
    5: [0, 2, 5, 8, 10],
}


class MultiGoalGridworld:
    def __init__(self, K=2, grid_size=11, max_steps=50,
                 goal_reward=10.0, trap_reward=-10.0):
        assert K in _GOAL_X
        self.K = K
        self.gs = grid_size
        self.max_steps = max_steps
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.start = (5, 0)
        self.trap_x = 5
        self.trap_ys = set(range(3, 10))
        self.goals = [(gx, 10) for gx in _GOAL_X[K]]
        self.goal_set = set(self.goals)
        self.num_goals = K
        self.obs_dim = 2
        self.num_actions = 4
        self.x, self.y, self.steps = 5, 0, 0

    def reset(self, seed=None):
        self.x, self.y = self.start
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        dx, dy = _DELTAS[action]
        self.x = max(0, min(self.gs - 1, self.x + dx))
        self.y = max(0, min(self.gs - 1, self.y + dy))
        self.steps += 1
        info = {"trap": False, "goal_id": -1}
        if self.x == self.trap_x and self.y in self.trap_ys:
            info["trap"] = True
            return self._obs(), self.trap_reward, True, False, info
        for gi, (gx, gy) in enumerate(self.goals):
            if self.x == gx and self.y == gy:
                info["goal_id"] = gi
                return self._obs(), self.goal_reward, True, False, info
        truncated = self.steps >= self.max_steps
        return self._obs(), 0.0, False, truncated, info

    def _obs(self):
        return np.array([self.x / 10.0, self.y / 10.0], dtype=np.float32)

    def get_pos(self):
        return self.x, self.y


def _expert_action(x, y, goal_x, goal_y=10):
    if goal_x == 5:
        safe_x = 4
        if x == 5 and y >= 2:
            return LEFT
        if x > safe_x:
            return LEFT
        if x < safe_x:
            return RIGHT
        if y < goal_y:
            return UP
        return RIGHT
    if x == 5 and y >= 2:
        return LEFT if goal_x < 5 else RIGHT
    if x != goal_x:
        return LEFT if goal_x < x else RIGHT
    if y < goal_y:
        return UP
    return UP


def generate_expert_dataset(env, n_traj_per_goal=250):
    obs_l, act_l, rew_l, nobs_l, done_l = [], [], [], [], []
    for gi, (gx, gy) in enumerate(env.goals):
        for _ in range(n_traj_per_goal):
            o, _ = env.reset()
            for _ in range(env.max_steps):
                a = _expert_action(env.x, env.y, gx, gy)
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
