from fastapi import FastAPI
from app.env import SahayakEnv
from pydantic import BaseModel
from typing import Optional, List
import numpy as np

app = FastAPI()
env = SahayakEnv(level=1)

class StepRequest(BaseModel):
    action: int

class ResetRequest(BaseModel):
    level: int = 1
    task_id: Optional[str] = None

class GradeRequest(BaseModel):
    observation: List[float]
    action: int
    task_id: Optional[str] = None

@app.get("/")
def root():
    return {
        "status": "Sahayak AI Live", 
        "level": env.level, 
        "task": env.current_task,
        "available_tasks": env.tasks
    }

@app.get("/tasks")
def get_tasks():
    # MANDATORY: Allows validator to discover your 3 tasks
    return {"tasks": env.tasks}

@app.get("/state")
def get_state():
    return env.state()

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global env
    env.level = req.level
    obs, info = env.reset(task_id=req.task_id)
    return {
        "observation": obs.tolist() if isinstance(obs, np.ndarray) else obs, 
        "info": info
    }

@app.post("/step")
def step(req: StepRequest):
    obs, reward, terminated, truncated, info = env.step(req.action)
    return {
        "observation": obs.tolist() if isinstance(obs, np.ndarray) else obs,
        "reward": float(reward),
        "done": bool(terminated or truncated),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info
    }

@app.post("/grade")
def grade(req: GradeRequest):
    # MANDATORY: Allows validator to test graders independently
    if req.task_id:
        env.current_task = req.task_id
    score = env.grader(req.observation, req.action)
    return {"score": float(score)}