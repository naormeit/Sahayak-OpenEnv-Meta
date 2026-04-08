from fastapi import FastAPI
from app.env import SahayakEnv
from pydantic import BaseModel

app = FastAPI()
env = SahayakEnv(level=1)

class StepRequest(BaseModel):
    action: int

@app.get("/")
def root():
    return {"status": "Sahayak AI Live", "level": env.level}

@app.get("/state")
def get_state():
    return env.state()

@app.post("/reset")
def reset(level: int = 1):
    global env
    env = SahayakEnv(level=level)
    obs, info = env.reset()
    return {"observation": obs.tolist(), "info": info}

@app.post("/step")
def step(req: StepRequest):
    obs, reward, terminated, truncated, info = env.step(req.action)
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "done": terminated or truncated,
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    }