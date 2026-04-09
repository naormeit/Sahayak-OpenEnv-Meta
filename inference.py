import os
import requests
from openai import OpenAI

# 1. Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://0.0.0.0:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8b")
API_KEY = os.getenv("API_KEY")

client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("API_KEY")
)

def clamp_score(score):
    """Ensures score is strictly between 0 and 1 (Phase 2 requirement)."""
    return max(0.01, min(0.99, float(score)))

def run_inference():
    print("[START]")
    
    try:
        reset_resp = requests.post(f"{API_BASE_URL}/reset", json={"level": 1})
        state = reset_resp.json().get("observation")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return

    done = False
    step_count = 0
    total_reward = 0
    hallucination_count = 0

    while not done and step_count < 50:
        prompt = f"Agent at {state}. Goal (9,9). Action (0:Up, 1:Down, 2:Left, 3:Right). Return ONLY the digit."
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5
            )
            
            content = (response.choices[0].message.content or "0").strip()
            if len(content) > 1: hallucination_count += 1
            
            action = int(content[0])
        except Exception:
            action = 0 
            hallucination_count += 1

        try:
            step_resp = requests.post(f"{API_BASE_URL}/step", json={"action": action})
            res = step_resp.json()
            
            state = res.get("observation")
            reward = res.get("reward")
            done = res.get("done")
            total_reward += reward
            step_count += 1

            print(f"[STEP] Step: {step_count} | Action: {action} | Reward: {reward} | State: {state}")
        except Exception as e:
            print(f"Step failed: {e}")
            break

    # --- PHASE 2 GRADER LOGIC ---

    # Task 1: Efficiency (Steps taken vs Max Steps)
    efficiency = clamp_score(1 - (step_count / 50))
    
    # Task 2: Convergence (Did it finish?)
    convergence = 0.95 if done else 0.05
    
    # Task 3: Adherence (Hallucination rate)
    adherence = clamp_score(1 - (hallucination_count / (step_count or 1)))

    print(f"[END] Total Reward: {total_reward}")
    print(f"TASK_SCORE:path_efficiency:{efficiency}")
    print(f"TASK_SCORE:goal_convergence:{convergence}")
    print(f"TASK_SCORE:instruction_adherence:{adherence}")

if __name__ == "__main__":
    if not API_KEY:
        print("Missing API_KEY! Evaluation cannot proceed through the proxy.")
    else:
        run_inference()