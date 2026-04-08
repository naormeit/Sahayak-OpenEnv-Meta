import os
import time
import requests
from openai import OpenAI

# 1. Setup Environment Variables (These must be in your HF Space Settings)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-8b")
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Initialize OpenAI Client (Mandatory per Checklist)
client = OpenAI(
    base_url=f"https://api-inference.huggingface.co/v1/", # Standard HF OpenAI-compat endpoint
    api_key=HF_TOKEN
)

def run_inference():
    print("[START]") # Mandatory Log Tag
    
    # Reset the environment to start
    reset_resp = requests.post(f"{API_BASE_URL}/reset", json={"level": 1})
    state = reset_resp.json().get("observation")
    
    total_reward = 0
    done = False
    step_count = 0

    while not done and step_count < 50:
        # 3. Ask Llama-3 for the next move
        prompt = f"You are an RL agent in a 10x10 grid. Current position: {state}. Goal is (9,9). Options: 0:Up, 1:Down, 2:Left, 3:Right. Output ONLY the number of the best action."
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )
        
        try:
            # Extract action number from LLM response
            action = int(response.choices[0].message.content.strip())
        except:
            action = 0 # Fallback to Up if LLM hallucinates

        # 4. Take the step in your Environment
        step_resp = requests.post(f"{API_BASE_URL}/step", json={"action": action})
        result = step_resp.json()
        
        state = result.get("observation")
        reward = result.get("reward")
        done = result.get("done")
        total_reward += reward
        step_count += 1

        # 5. Mandatory STEP logging format
        print(f"[STEP] Step: {step_count} | Action: {action} | Reward: {reward} | State: {state}")

    print(f"[END] Total Reward: {total_reward}") # Mandatory Log Tag

if __name__ == "__main__":
    if not HF_TOKEN:
        print("Error: HF_TOKEN not found in environment variables.")
    else:
        run_inference()