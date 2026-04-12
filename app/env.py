import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, List, Dict

class SahayakEnv(gym.Env):
    def __init__(self, level=1):
        super().__init__()
        self.level = level
        self.size = 10
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=np.float32)
        
        # Explicit Task Registry for Validator Discovery
        self._task_ids = ["reach_goal", "efficiency_path", "exploration"]
        self.current_task = self._task_ids[0]
        
        self.agent_pos = np.array([0, 0])
        self.goal_pos = [9, 9]
        self.steps = 0
        # Initialize
        self.reset()

    @property
    def tasks(self) -> List[str]:
        """Discovery property used by validators to find available graders."""
        return self._task_ids

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.float32)

    def state(self) -> Dict[str, Any]:
        return {
            "agent_pos": self.agent_pos.tolist(),
            "goal_pos": self.goal_pos,
            "steps": self.steps,
            "level": self.level,
            "tasks": self._task_ids,
            "current_task": self.current_task
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None, task_id: str = None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        self.goal_pos = [self.size-1, self.size-1]
        self.steps = 0
        self.obstacles = [[2, 2], [5, 5]] if self.level > 1 else []
        
        # If a specific task is requested by the validator, switch to it
        if task_id and task_id in self._task_ids:
            self.current_task = task_id
        else:
            # Fallback: Cycle tasks to show variety
            idx = (self._task_ids.index(self.current_task) + 1) % len(self._task_ids)
            self.current_task = self._task_ids[idx]
        
        return self._get_obs(), {"status": "success", "task": self.current_task}
    
    def grader(self, observation: Any, action: Any) -> float:
        """
        Calculates a score STRICTLY between 0.01 and 0.99.
        """
        obs_arr = np.array(observation)
        # Manhattan distance to goal
        dist = np.linalg.norm(obs_arr - np.array(self.goal_pos), ord=1)
        max_dist = 2 * (self.size - 1)
        progress = 1.0 - (dist / max_dist)

        if self.current_task == "reach_goal":
            score = progress * 0.95 
        elif self.current_task == "efficiency_path":
            efficiency = max(0.1, 1.0 - (self.steps / 50))
            score = (progress + efficiency) / 2
        else: # exploration
            score = 0.5 + (progress * 0.4)

        # CRITICAL: Ensures score is NEVER exactly 0.0 or 1.0
        return float(np.clip(score, 0.01, 0.99))

    def step(self, action: int):
        self.steps += 1
        
        # Movement logic: 0:Up, 1:Down, 2:Left, 3:Right
        if action == 0 and self.agent_pos[1] < self.size-1: self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[1] > 0: self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[0] > 0: self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.size-1: self.agent_pos[0] += 1
        
        terminated = bool(np.array_equal(self.agent_pos, self.goal_pos))
        truncated = bool(self.steps >= 50)
        
        # Use grader for reward
        reward = self.grader(self._get_obs(), action)
        
        return self._get_obs(), reward, terminated, truncated, {"task": self.current_task}