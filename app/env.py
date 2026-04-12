import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Safer import for OpenEnv types
try:
    from openenv import Observation, Action
except ImportError:
    # Fallback if the library structure differs on Hugging Face
    from typing import Any
    Observation = Any
    Action = Any

class SahayakEnv(gym.Env):
    def __init__(self, level=1):
        super().__init__()
        self.level = level
        self.size = 10
        self.action_space = spaces.Discrete(4) # 0:Up, 1:Down, 2:Left, 3:Right
        self.observation_space = spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=np.float32)
        
        # OpenEnv Phase 2 Requirement: Define at least 3 tasks
        self.tasks = ["reach_goal", "efficiency_path", "exploration"]
        self.current_task = self.tasks[0]
        
        self.agent_pos = np.array([0, 0])
        self.goal_pos = [9, 9]
        self.obstacles = []
        self.steps = 0
        self.reset()

    def _get_obs(self):
        return np.array(self.agent_pos, dtype=np.float32)

    def state(self):
        """MANDATORY for OpenEnv Spec"""
        return {
            "agent_pos": self.agent_pos.tolist(),
            "goal_pos": self.goal_pos,
            "obstacles": self.obstacles,
            "steps": self.steps,
            "level": self.level,
            "tasks": self.tasks # Include tasks in state
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Handle Gymnasium compliance
        super().reset(seed=seed)
        
        self.agent_pos = np.array([0, 0])
        self.goal_pos = [self.size-1, self.size-1]
        self.steps = 0
        self.obstacles = [[2, 2], [5, 5]] if self.level > 1 else []
        
        # Cycle through tasks
        self.current_task = self.tasks[self.steps % 3]
        
        # Must return (observation, info_dict)
        observation = self._get_obs()
        info = {"status": "success", "task": self.current_task}
        return observation, info
    
    def grader(self, observation: Observation, action: Action) -> float:
        """
        MANDATORY Phase 2 Logic:
        1. Must support at least 3 tasks.
        2. Scores must be strictly (0, 1) range.
        """
        # Calculate Manhattan distance to goal
        dist = np.linalg.norm(self.agent_pos - np.array(self.goal_pos), ord=1)
        max_dist = 2 * (self.size - 1)
        
        # Normalized progress score (closer = higher score)
        progress = 1.0 - (dist / max_dist)

        if self.current_task == "reach_goal":
            score = progress * 0.95 # Max 0.95
        elif self.current_task == "efficiency_path":
            # Penalty for high step count
            efficiency = max(0.1, 1.0 - (self.steps / 50))
            score = (progress + efficiency) / 2
        else: # exploration
            score = 0.5 + (progress * 0.4)

        # Force strictly between 0 and 1
        return float(max(0.01, min(0.99, score)))

    def step(self, action):
        self.steps += 1
        
        # Movement logic
        if action == 0 and self.agent_pos[1] < self.size-1: self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[1] > 0: self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[0] > 0: self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.size-1: self.agent_pos[0] += 1
        
        terminated = bool(np.array_equal(self.agent_pos, self.goal_pos))
        truncated = bool(self.steps >= 50)
        
        # Use grader to determine the reward for Phase 2 compliance
        reward = self.grader(self._get_obs(), action)
        
        return self._get_obs(), reward, terminated, truncated, {"task": self.current_task}