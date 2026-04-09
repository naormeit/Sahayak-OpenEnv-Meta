import os
import requests
from openai import OpenAI

# 1. Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://0.0.0.0:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8b")
API_KEY = os.getenv("API_KEY")

# 2. Initialize OpenAI Client (OpenEnv Requirement)
client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("API_KEY")
)

def run_inference():
    print("[START]")
    
    # Reset environment
    try:
        # We use the same API_BASE_URL to talk to your FastAPI server
        reset_resp = requests.post(f"{API_BASE_URL}/reset", json={"level": 1})
        state = reset_resp.json().get("observation")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return

    done = False
    step_count = 0
    total_reward = 0

    while not done and step_count < 50:
        # Ask Llama for action
        prompt = f"Agent at {state}. Goal (9,9). Action (0:Up, 1:Down, 2:Left, 3:Right). Return ONLY the digit."
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5
            )
            
            # Robust extraction: handles 'None' or empty content
            content = response.choices[0].message.content or "0"
            action = int(content.strip()[0])
        except Exception:
            action = 0 # Fallback for hallucinations or API issues

        # Take Step
        try:
            step_resp = requests.post(f"{API_BASE_URL}/step", json={"action": action})
            res = step_resp.json()
            
            state = res.get("observation")
            reward = res.get("reward")
            done = res.get("done")
            total_reward += reward
            step_count += 1

            # MANDATORY Logging format for evaluation scoring
            print(f"[STEP] Step: {step_count} | Action: {action} | Reward: {reward} | State: {state}")
        except Exception as e:
            print(f"Step failed: {e}")
            break

    print(f"[END] Total Reward: {total_reward}")

if __name__ == "__main__":
    # The validator requires API_KEY to be present for the proxy to work
    if not API_KEY:
        print("Missing API_KEY! Evaluation cannot proceed through the proxy.")
    else:
        run_inference()