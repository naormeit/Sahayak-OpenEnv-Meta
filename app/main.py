from fastapi import FastAPI
from app.env import SahayakEnv
from pydantic import BaseModel
import numpy as np

app = FastAPI()
# Initialize the environment
env = SahayakEnv(level=1)

class StepRequest(BaseModel):
    action: int

class ResetRequest(BaseModel):
    level: int = 1

@app.get("/")
def root():
    return {"status": "Sahayak AI Live", "level": env.level, "task": env.current_task}

@app.get("/state")
def get_state():
    return env.state()

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global env
    # Update level if changed
    env.level = req.level
    obs, info = env.reset()
    
    # Ensure observation is a list for JSON compatibility
    return {
        "observation": obs.tolist() if isinstance(obs, np.ndarray) else obs, 
        "info": info
    }

@app.post("/step")
def step(req: StepRequest):
    obs, reward, terminated, truncated, info = env.step(req.action)
    
    # Phase 2 Critical: Ensure reward is a standard float and scores are in range
    return {
        "observation": obs.tolist() if isinstance(obs, np.ndarray) else obs,
        "reward": float(reward),
        "done": bool(terminated or truncated),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info
    }