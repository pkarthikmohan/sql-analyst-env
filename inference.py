"""
inference.py — Baseline AI agent for SQL Analyst OpenEnv
"""
import os
import sys
import json
import time
import requests
from openai import OpenAI

ENV_BASE_URL = "https://p-karthik-mohan-sql-analyst-env.hf.space"
MAX_ATTEMPTS = 5
TASK_IDS     = [1, 2, 3, 4, 5]

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "no-key-needed",
)

def env_reset(task_id: int) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    r.raise_for_status()
    return r.json()

def env_step(sql: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json={"action": sql})
    r.raise_for_status()
    return r.json()

def wait_for_server(retries: int = 10, delay: float = 2.0):
    print("Waiting for environment server...")
    for i in range(retries):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                print("Server is ready.\n")
                return
        except requests.exceptions.ConnectionError:
            pass
        print(f"  Not ready yet... ({i+1}/{retries})")
        time.sleep(delay)
    print("ERROR: Server did not start in time.")
    sys.exit(1)

def build_system_prompt() -> str:
    return """You are an expert SQL analyst. Your job is to write correct SQLite queries.

Rules:
- Only write SELECT or WITH (CTE) statements. Never INSERT, UPDATE, DELETE, or DROP.
- Always match the exact column names specified in the task.
- Always filter WHERE status = 'completed' unless told otherwise.
- Use STRFTIME('%Y', order_date) for year filtering in SQLite.
- RANK() OVER (...) is supported in SQLite 3.25+.
- Return ONLY the raw SQL query — no explanation, no markdown, no backticks.
- If a previous attempt scored less than 1.0, study the feedback and fix the query.
"""

def build_user_prompt(task_description, schema, hint, attempt, previous_attempts):
    prompt = f"""Task:
{task_description}

Database schema:
{schema}

Hint: {hint}

Attempt number: {attempt}
"""
    if previous_attempts:
        prompt += "\nYour previous attempts and their scores:\n"
        for prev in previous_attempts[-3:]:
            prompt += f"""
  Attempt {prev['attempt']}:
    SQL:    {prev['sql']}
    Reward: {prev['reward']} / 1.0
    Columns expected : {prev['details'].get('expected_columns', [])}
    Columns you gave : {prev['details'].get('agent_columns', [])}
    Rows expected    : {prev['details'].get('expected_row_count', '?')}
    Rows you gave    : {prev['details'].get('agent_row_count', '?')}
"""
    prompt += "\nWrite the corrected SQL query now:"
    return prompt

def ask_llm(task_description, schema, hint, attempt, previous_attempts):
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
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()
    return sql

def solve_task(task_id: int) -> dict:
    print(f"\n{'='*60}")
    print(f"TASK {task_id}")
    print('='*60)

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
        print(f"  Attempt {attempt}/{MAX_ATTEMPTS} — asking LLM...")
        sql = ask_llm(task_desc, schema, hint, attempt, previous_attempts)
        print(f"  SQL: {sql[:120]}{'...' if len(sql) > 120 else ''}")

        step_resp = env_step(sql)
        reward    = step_resp["reward"]
        done      = step_resp["done"]
        details   = step_resp["observation"].get("reward_breakdown", {})

        print(f"  Reward: {reward:.3f}  "
              f"(cols={details.get('column_score',0):.2f}  "
              f"rows={details.get('row_score',0):.2f}  "
              f"vals={details.get('value_score',0):.2f})")

        best_reward = max(best_reward, reward)
        final_sql   = sql

        previous_attempts.append({
            "attempt": attempt,
            "sql":     sql,
            "reward":  reward,
            "details": details,
        })

        if done:
            print(f"  PERFECT SCORE on attempt {attempt}!")
            break
        elif reward >= 0.8:
            print(f"  Score is close ({reward:.3f}). Trying to improve...")
        else:
            print(f"  Score is low ({reward:.3f}). Refining query...")

    return {
        "task_id":     task_id,
        "difficulty":  difficulty,
        "best_reward": best_reward,
        "attempts":    len(previous_attempts),
        "final_sql":   final_sql,
        "solved":      best_reward >= 1.0,
    }

def main():
    print("SQL Analyst OpenEnv — Baseline Inference Agent")
    print(f"Model      : {MODEL_NAME}")
    print(f"API Base   : {API_BASE_URL}")
    print(f"Env Server : {ENV_BASE_URL}")

    wait_for_server()

    results = []
    for task_id in TASK_IDS:
        result = solve_task(task_id)
        results.append(result)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print('='*60)

    total_score = 0.0
    for r in results:
        status = "SOLVED" if r["solved"] else f"best={r['best_reward']:.3f}"
        print(f"  Task {r['task_id']} ({r['difficulty']:6s})  {status}  "
              f"in {r['attempts']} attempt(s)")
        total_score += r["best_reward"]

    avg_score = total_score / len(results)
    print(f"\n  Average reward : {avg_score:.3f} / 1.000")
    print(f"  Tasks solved   : {sum(1 for r in results if r['solved'])} / {len(results)}")

    with open("results.json", "w") as f:
        json.dump({
            "results":      results,
            "avg_score":    round(avg_score, 3),
            "tasks_solved": sum(1 for r in results if r["solved"]),
        }, f, indent=2)
    print(f"\n  Results saved to results.json")

if __name__ == "__main__":
    main()