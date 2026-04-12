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
        
        # EXPLICIT Task IDs - Don't change these
        self.tasks = ["reach_goal", "efficiency_path", "exploration"]
        self.current_task = self.tasks[0]
        
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([9, 9])
        self.steps = 0
        self.reset()

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.float32)

    def state(self) -> Dict[str, Any]:
        return {
            "agent_pos": self.agent_pos.tolist(),
            "goal_pos": self.goal_pos.tolist(),
            "steps": self.steps,
            "current_task": self.current_task,
            "tasks": self.tasks
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None, task_id: str = None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        self.steps = 0
        
        if task_id in self.tasks:
            self.current_task = task_id
            
        # Return observation and a rich info dict
        return self._get_obs(), {"status": "success", "task": self.current_task, "tasks": self.tasks}
    
    def grader(self, observation: Any, action: Any) -> float:
        """Robust scoring strictly within (0.05, 0.95) to avoid any float errors."""
        obs_arr = np.array(observation)
        # Manhattan distance
        dist = np.sum(np.abs(obs_arr - self.goal_pos))
        max_dist = 18.0 
        
        # Progress calculation
        progress = 1.0 - (dist / max_dist)

        if self.current_task == "reach_goal":
            score = 0.1 + (progress * 0.8) # Range: 0.1 to 0.9
        elif self.current_task == "efficiency_path":
            step_penalty = max(0, self.steps / 50.0)
            score = 0.5 + (progress * 0.2) - (step_penalty * 0.1)
        else: # exploration
            score = 0.3 + (progress * 0.5)

        # FINAL CAPPING: Impossible to be 0 or 1
        return float(np.clip(score, 0.05, 0.95))

    def step(self, action: int):
        self.steps += 1
        
        # 0:Up, 1:Down, 2:Left, 3:Right
        if action == 0 and self.agent_pos[1] < self.size-1: self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[1] > 0: self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[0] > 0: self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.size-1: self.agent_pos[0] += 1
        
        terminated = bool(np.array_equal(self.agent_pos, self.goal_pos))
        truncated = bool(self.steps >= 50)
        
        # Reward comes directly from the grader
        reward = self.grader(self._get_obs(), action)
        
        return self._get_obs(), reward, terminated, truncated, {"task": self.current_task, "tasks": self.tasks}