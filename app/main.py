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
    return {"status": "live", "tasks": env.tasks, "current": env.current_task}

@app.get("/tasks")
def get_tasks():
    # Returns the bare list the validator is looking for
    return {"tasks": env.tasks}

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    obs, info = env.reset(task_id=req.task_id)
    return {
        "observation": obs.tolist(), 
        "info": info,
        "task": env.current_task
    }

@app.post("/step")
def step(req: StepRequest):
    obs, reward, terminated, truncated, info = env.step(req.action)
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "done": bool(terminated or truncated),
        "info": info
    }

@app.post("/grade")
def grade(req: GradeRequest):
    # This endpoint specifically allows the bot to test your scores
    temp_task = env.current_task
    if req.task_id:
        env.current_task = req.task_id
    
    score = env.grader(req.observation, req.action)
    env.current_task = temp_task # Reset back
    return {"score": float(score), "task": req.task_id or env.current_task}