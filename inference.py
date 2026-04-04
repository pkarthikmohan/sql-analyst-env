"""
inference.py — Baseline AI agent for SQL Analyst OpenEnv
---------------------------------------------------------
Stdout format (mandatory):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<sql> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import sys
import json
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ENV_BASE_URL = "https://p-karthik-mohan-sql-analyst-env.hf.space"
MAX_ATTEMPTS = 5
TASK_IDS     = [1, 2, 3, 4, 5, 6, 7, 8]
BENCHMARK    = "sql-analyst-env"

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "no-key-needed",
)

# ── Mandatory stdout log functions ────────────────────────────────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    action_clean = action.replace("\n", " ").strip()[:120]
    error_val    = error if error else "null"
    done_val     = str(done).lower()
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps}  rewards={rewards_str}", flush=True)

# ── Environment helpers ───────────────────────────────────────────────────────

def env_reset(task_id):
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    r.raise_for_status()
    return r.json()

def env_step(sql):
    r = requests.post(f"{ENV_BASE_URL}/step", json={"action": sql})
    r.raise_for_status()
    return r.json()

def wait_for_server(retries=10, delay=2.0):
    print("Waiting for environment server...", flush=True)
    for i in range(retries):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=3)
            if r.status_code == 200:
                print("Server is ready.\n", flush=True)
                return
        except requests.exceptions.ConnectionError:
            pass
        print(f"  Not ready yet... ({i+1}/{retries})", flush=True)
        time.sleep(delay)
    print("ERROR: Server did not start in time.", flush=True)
    sys.exit(1)

# ── LLM ───────────────────────────────────────────────────────────────────────

def build_system_prompt():
    return """You are an expert SQL analyst. Your job is to write correct SQLite queries.

Rules:
- Only write SELECT or WITH (CTE) statements. Never INSERT, UPDATE, DELETE, or DROP.
- Always match the exact column names specified in the task.
- Always filter WHERE status = 'completed' unless told otherwise.
- Use STRFTIME('%Y', order_date) for year filtering in SQLite.
- Use STRFTIME('%Y-%m', order_date) for year-month formatting.
- RANK() OVER (...) and LAG() OVER (...) are supported in SQLite 3.25+.
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
        {"role": "user",   "content": build_user_prompt(task_description, schema, hint, attempt, previous_attempts)},
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
        sql = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()
    return sql

# ── Task solver ───────────────────────────────────────────────────────────────

def solve_task(task_id):
    task_name  = f"sql-task-{task_id}"
    reset_resp = env_reset(task_id)
    obs        = reset_resp["observation"]
    task_desc  = obs["task_description"]
    schema     = obs["schema"]
    hint       = obs["hint"]
    difficulty = obs["difficulty"]

    print(f"\n{'='*60}", flush=True)
    print(f"TASK {task_id} ({difficulty.upper()})", flush=True)
    print(f"Task: {task_desc}\n", flush=True)

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    previous_attempts = []
    all_rewards       = []
    best_reward       = 0.0
    final_sql         = ""
    steps_taken       = 0
    success           = False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"  Attempt {attempt}/{MAX_ATTEMPTS} — asking LLM...", flush=True)
        error = None
        sql   = ""

        try:
            sql = ask_llm(task_desc, schema, hint, attempt, previous_attempts)
            print(f"  SQL: {sql[:120]}{'...' if len(sql) > 120 else ''}", flush=True)
        except Exception as e:
            error = str(e)
            log_step(step=attempt, action="", reward=0.0, done=False, error=error)
            all_rewards.append(0.0)
            steps_taken = attempt
            continue

        try:
            step_resp = env_step(sql)
            reward    = step_resp["reward"]
            done      = step_resp["done"]
            details   = step_resp["observation"].get("reward_breakdown", {})
        except Exception as e:
            error = str(e)
            log_step(step=attempt, action=sql, reward=0.0, done=False, error=error)
            all_rewards.append(0.0)
            steps_taken = attempt
            continue

        all_rewards.append(reward)
        steps_taken = attempt
        best_reward = max(best_reward, reward)
        final_sql   = sql

        print(f"  Reward: {reward:.3f}  (cols={details.get('column_score',0):.2f}  rows={details.get('row_score',0):.2f}  vals={details.get('value_score',0):.2f})", flush=True)

        log_step(step=attempt, action=sql, reward=reward, done=done, error=error)

        previous_attempts.append({
            "attempt": attempt,
            "sql":     sql,
            "reward":  reward,
            "details": details,
        })

        if done:
            print(f"  PERFECT SCORE on attempt {attempt}!", flush=True)
            success = True
            break
        elif reward >= 0.8:
            print(f"  Score is close ({reward:.3f}). Trying to improve...", flush=True)
        else:
            print(f"  Score is low ({reward:.3f}). Refining query...", flush=True)

    log_end(success=success, steps=steps_taken, score=best_reward, rewards=all_rewards)

    return {
        "task_id":     task_id,
        "task_name":   task_name,
        "difficulty":  difficulty,
        "best_reward": best_reward,
        "attempts":    steps_taken,
        "final_sql":   final_sql,
        "solved":      success,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("SQL Analyst OpenEnv — Baseline Inference Agent", flush=True)
    print(f"Model      : {MODEL_NAME}", flush=True)
    print(f"API Base   : {API_BASE_URL}", flush=True)
    print(f"Env Server : {ENV_BASE_URL}", flush=True)

    wait_for_server()

    results     = []
    total_score = 0.0

    for task_id in TASK_IDS:
        result = solve_task(task_id)
        results.append(result)
        total_score += result["best_reward"]

    print(f"\n{'='*60}", flush=True)
    print("FINAL RESULTS", flush=True)
    print('='*60, flush=True)

    for r in results:
        status = "SOLVED" if r["solved"] else f"best={r['best_reward']:.3f}"
        print(f"  Task {r['task_id']} ({r['difficulty']:6s})  {status}  in {r['attempts']} attempt(s)", flush=True)

    avg_score    = total_score / len(results)
    tasks_solved = sum(1 for r in results if r["solved"])

    print(f"\n  Average reward : {avg_score:.3f} / 1.000", flush=True)
    print(f"  Tasks solved   : {tasks_solved} / {len(results)}", flush=True)

    with open("results.json", "w") as f:
        json.dump({"results": results, "avg_score": round(avg_score, 3), "tasks_solved": tasks_solved}, f, indent=2)
    print(f"\n  Results saved to results.json", flush=True)

if __name__ == "__main__":
    main()