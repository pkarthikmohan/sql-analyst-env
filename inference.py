"""
inference.py — Baseline AI agent for SQL Analyst OpenEnv
---------------------------------------------------------
Uses the OpenAI client (pointed at any compatible LLM via API_BASE_URL)
to solve all 3 tasks by interacting with the running FastAPI environment.
"""
import os
import sys
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# ── Configuration ─────────────────────────────────────────────────────────────

ENV_BASE_URL = "https://p-karthik-mohan-sql-analyst-env.hf.space"   # live HF Space
MAX_ATTEMPTS = 5                         # max SQL attempts per task
TASK_IDS     = [1, 2, 3]                 # tasks to solve

API_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME   = "llama-3.1-8b-instant"
HF_TOKEN     = os.environ.get("HF_TOKEN")

# ── OpenAI Client ─────────────────────────────────────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "no-key-needed",
)

# ── Environment API helpers ───────────────────────────────────────────────────

def env_reset(task_id: int) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    r.raise_for_status()
    return r.json()

def env_step(sql: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json={"action": sql})
    r.raise_for_status()
    return r.json()

def env_state() -> dict:
    r = requests.get(f"{ENV_BASE_URL}/state")
    r.raise_for_status()
    return r.json()

def wait_for_server(retries: int = 10, delay: float = 2.0):
    """Wait until the FastAPI server is ready."""
    print("Waiting for environment server...")
    for i in range(retries):
        try:
            r = requests.get(f"{ENV_BASE_URL}/docs")
            if r.status_code == 200:
                print("Server is up!")
                return
        except requests.ConnectionError:
            pass
        time.sleep(delay)
    print("ERROR: Server did not start in time.")
    sys.exit(1)

# ── LLM SQL Generator ─────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    return """You are an expert SQL analyst. Your job is to write correct SQLite queries.

Rules:
- Only write SELECT or WITH (CTE) statements. Never INSERT, UPDATE, DELETE, or DROP.
- Always match the exact column names specified in the task.
- Use proper SQLite syntax. RANK() OVER (...) is supported in SQLite 3.25+.
- Return ONLY the raw SQL query — no explanation, no markdown, no backticks.
- If a previous attempt scored less than 1.0, study the feedback and fix the query.
"""

def build_user_prompt(
    task_description: str,
    schema: str,
    hint: str,
    attempt: int,
    previous_attempts: list,
) -> str:
    prompt = f"""Task:
{task_description}

Database schema:
{schema}

Hint: {hint}

Attempt number: {attempt}
"""
    if previous_attempts:
        prompt += "\nPrevious attempts:\n"
        for i, prev in enumerate(previous_attempts):
            prompt += f"--- Attempt {i+1} ---\nSQL: {prev['sql']}\nReward: {prev['reward']}\n\n"
        
    prompt += "\nWrite the corrected SQL query now:"
    return prompt

def ask_llm(task_description: str, schema: str, hint: str,
            attempt: int, previous_attempts: list) -> str:
    """Call the LLM and return a SQL string."""
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user",   "content": build_user_prompt(
            task_description, schema, hint, attempt, previous_attempts
        )},
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    sql = response.choices[0].message.content.strip()

    # Strip markdown fences if model wraps in ```sql ... ```
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[-1].rsplit("\n", 1)[0].replace("```", "").strip()

    return sql

# ── Main Agent Loop ───────────────────────────────────────────────────────────

def solve_task(task_id: int) -> dict:
    """Run the agent on a single task. Returns final result dict."""
    print(f"\n{'='*60}")
    print(f"TASK {task_id}")
    print('='*60)

    # Reset environment
    reset_resp = env_reset(task_id)
    obs        = reset_resp["observation"]
    task_desc  = obs["task_description"]
    schema     = obs["schema"]
    hint       = obs["hint"]
    difficulty = obs["difficulty"]

    print(f"Difficulty : {difficulty.upper()}")
    print(f"Task       : {task_desc}\n")

    previous_attempts = []
    best_reward = 0.0
    final_sql   = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"Attempt {attempt}/{MAX_ATTEMPTS}...")
        sql = ask_llm(task_desc, schema, hint, attempt, previous_attempts)
        print(f"Generated SQL: {sql}")
        
        step_resp = env_step(sql)
        reward = step_resp["reward"]
        print(f"Reward: {reward}")
        
        best_reward = max(best_reward, reward)
        final_sql = sql
        
        if reward >= 1.0:
            print("Task solved successfully!")
            break
            
        previous_attempts.append({"sql": sql, "reward": reward})

    return {
        "task_id":      task_id,
        "difficulty":   difficulty,
        "best_reward":  best_reward,
        "attempts":     len(previous_attempts) + (1 if best_reward >= 1.0 else 0),
        "final_sql":    final_sql,
        "solved":       best_reward >= 1.0,
    }


def main():
    print("SQL Analyst OpenEnv — Baseline Inference Agent")
    print(f"Model      : {MODEL_NAME}")
    print(f"API Base   : {API_BASE_URL}")
    print(f"Env Server : {ENV_BASE_URL}")

    wait_for_server()

    results = []
    for task_id in TASK_IDS:
        res = solve_task(task_id)
        results.append(res)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print('='*60)

    total_score = 0.0
    solved_count = 0
    for r in results:
        total_score += r["best_reward"]
        if r["solved"]:
            solved_count += 1
        print(f"Task {r['task_id']} ({r['difficulty']}): Reward = {r['best_reward']:.2f} | Solved = {r['solved']}")

    avg_score = total_score / len(results) if results else 0
    print(f"\nTasks solved   : {solved_count} / {len(results)}")
    print(f"Average reward : {avg_score:.3f} / 1.000")


if __name__ == "__main__":
    main()