import os
import requests
from openai import OpenAI

# 1. Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://0.0.0.0:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8b")
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Initialize OpenAI Client (OpenEnv Requirement)
client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key=HF_TOKEN
)

def run_inference():
    print("[START]")
    
    # Reset environment
    try:
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

            # MANDATORY Logging format
            print(f"[STEP] Step: {step_count} | Action: {action} | Reward: {reward} | State: {state}")
        except Exception as e:
            print(f"Step failed: {e}")
            break

    print(f"[END] Total Reward: {total_reward}")

if __name__ == "__main__":
    if not HF_TOKEN:
        print("Missing HF_TOKEN! Set it in your environment variables.")
    else:
        run_inference()