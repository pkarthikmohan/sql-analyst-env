"""
inference.py — Baseline AI agent for SQL Analyst OpenEnv
---------------------------------------------------------
Uses the OpenAI client (pointed at any compatible LLM via API_BASE_URL)
to solve all 3 tasks by interacting with the running FastAPI environment.

Environment variables required:
  API_BASE_URL  — LLM API base URL  (e.g. https://api.openai.com/v1)
  MODEL_NAME    — model to use       (e.g. gpt-4o-mini)
  HF_TOKEN      — Hugging Face token (used as the API key)

Usage:
  python inference.py
"""

import os
import sys
import json
import time
import requests
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

ENV_BASE_URL = "http://127.0.0.1:7860"   # where FastAPI server is running
MAX_ATTEMPTS = 5                         # max SQL attempts per task
TASK_IDS     = [1, 2, 3]                # tasks to solve

API_BASE_URL = "https://api.groq.com/openai/v1"
MODEL_NAME   = "llama-3.1-8b-instant"
HF_TOKEN     = "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"

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
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                print("Server is ready.\n")
                return
        except requests.exceptions.ConnectionError:
            pass
        print(f"  Not ready yet, retrying in {delay}s... ({i+1}/{retries})")
        time.sleep(delay)
    print("ERROR: Server did not start in time. Is uvicorn running?")
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
        prompt += "\nYour previous attempts and their scores:\n"
        for prev in previous_attempts[-3:]:   # show last 3 only
            prompt += f"""
  Attempt {prev['attempt']}:
    SQL:    {prev['sql']}
    Reward: {prev['reward']} / 1.0
    Columns expected : {prev['details'].get('expected_columns', [])}
    Columns you gave : {prev['details'].get('agent_columns', [])}
    Rows expected    : {prev['details'].get('expected_row_count', '?')}
    Rows you gave    : {prev['details'].get('agent_row_count', '?')}
    Sub-scores       : columns={prev['details'].get('column_score', 0):.2f}  rows={prev['details'].get('row_score', 0):.2f}  values={prev['details'].get('value_score', 0):.2f}
"""
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
        temperature=0.0,       # deterministic — we want correct SQL, not creative SQL
        max_tokens=512,
    )
    sql = response.choices[0].message.content.strip()

    # Strip markdown fences if model wraps in ```sql ... ```
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

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
        print(f"  Attempt {attempt}/{MAX_ATTEMPTS} — asking LLM...")

        # Get SQL from LLM
        sql = ask_llm(task_desc, schema, hint, attempt, previous_attempts)
        print(f"  SQL: {sql[:120]}{'...' if len(sql) > 120 else ''}")

        # Submit to environment
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

        # Store attempt for LLM feedback
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
        "task_id":      task_id,
        "difficulty":   difficulty,
        "best_reward":  best_reward,
        "attempts":     len(previous_attempts),
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
        result = solve_task(task_id)
        results.append(result)

    # ── Final Summary ─────────────────────────────────────────────────────────
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

    # Save results to file (useful for judges / CI)
    output_path = "results.json"
    with open(output_path, "w") as f:
        json.dump({
            "results":     results,
            "avg_score":   round(avg_score, 3),
            "tasks_solved": sum(1 for r in results if r["solved"]),
        }, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    return avg_score


if __name__ == "__main__":
    main()