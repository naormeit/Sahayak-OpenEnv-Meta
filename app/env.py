import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, List, Dict

class SahayakEnv(gym.Env):
    def __init__(self, level=1):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=9, shape=(2,), dtype=np.float32)
        
        # SYNCED WITH YAML IDs
        self.tasks = ["path_efficiency", "goal_convergence", "instruction_adherence"]
        self.current_task = self.tasks[0]
        
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([9, 9])
        self.steps = 0

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.float32)

    def reset(self, *, seed=None, options=None, task_id=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        self.steps = 0
        if task_id in self.tasks:
            self.current_task = task_id
        return self._get_obs(), {"task": self.current_task, "tasks": self.tasks}

    def grader(self, observation: Any, action: Any, task_id: str = None) -> float:
        t_id = task_id or self.current_task
        obs = np.clip(np.array(observation), 0, 9)
        dist = np.sum(np.abs(obs - self.goal_pos))
        progress = 1.0 - (dist / 18.0)
        
        if t_id == "path_efficiency":
            score = 0.5 + (progress * 0.4) - (self.steps / 100.0)
        elif t_id == "goal_convergence":
            score = 0.2 + (progress * 0.7)
        else: # instruction_adherence
            score = 0.4 + (progress * 0.4)
            
        return float(np.clip(score, 0.1, 0.9))

    def step(self, action: int):
        self.steps += 1
        if action == 0 and self.agent_pos[1] < 9: self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[1] > 0: self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[0] > 0: self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < 9: self.agent_pos[0] += 1
        
        obs = self._get_obs()
        reward = self.grader(obs, action)
        done = bool(np.array_equal(self.agent_pos, self.goal_pos) or self.steps >= 50)
        return obs, reward, done, False, {"task": self.current_task, "tasks": self.tasks}