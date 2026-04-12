from fastapi import FastAPI
from app.env import SahayakEnv
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()
env = SahayakEnv()

class StepRequest(BaseModel):
    action: int

class ResetRequest(BaseModel):
    task_id: Optional[str] = None

class GradeRequest(BaseModel):
    observation: List[float]
    action: int
    task_id: Optional[str] = None

@app.get("/")
def root():
    return {"status": "ok", "tasks": env.tasks, "current": env.current_task}

@app.get("/tasks")
def tasks_list():
    return {"tasks": env.tasks}

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    obs, info = env.reset(task_id=req.task_id)
    return {"observation": obs.tolist(), "info": info}

@app.post("/step")
def step(req: StepRequest):
    obs, reward, done, _, info = env.step(req.action)
    return {"observation": obs.tolist(), "reward": reward, "done": done, "info": info}

@app.post("/grade")
def grade(req: GradeRequest):
    score = env.grader(req.observation, req.action, task_id=req.task_id)
    return {"score": score, "task_id": req.task_id or env.current_task}