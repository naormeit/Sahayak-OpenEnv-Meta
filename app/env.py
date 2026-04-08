import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SahayakEnv(gym.Env):
    def __init__(self, level=1):
        super().__init__()
        self.level = level
        self.size = 10
        self.action_space = spaces.Discrete(4) # 0:Up, 1:Down, 2:Left, 3:Right
        self.observation_space = spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=np.float32)
        self.reset()

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.float32)

    def state(self):
        """MANDATORY for OpenEnv Spec"""
        return {
            "agent_pos": self.agent_pos.tolist() if isinstance(self.agent_pos, np.ndarray) else list(self.agent_pos),
            "goal_pos": self.goal_pos,
            "obstacles": self.obstacles,
            "steps": self.steps,
            "level": self.level
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        self.steps = 0
        # Simple obstacle logic for level 2+
        self.obstacles = [[2, 2], [5, 5]] if self.level > 1 else []
        return self._get_obs(), {"status": "success"}

    def step(self, action):
        self.steps += 1
        # 0: Up, 1: Down, 2: Left, 3: Right
        if action == 0 and self.agent_pos[1] < self.size-1: self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[1] > 0: self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[0] > 0: self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.size-1: self.agent_pos[0] += 1
        
        terminated = bool(self.agent_pos == self.goal_pos)
        truncated = bool(self.steps >= 50)
        reward = 1.0 if terminated else -0.1
        return self._get_obs(), reward, terminated, truncated, {}