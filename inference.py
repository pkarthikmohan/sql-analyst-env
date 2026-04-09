"""
inference.py — Baseline AI agent for SQL Analyst OpenEnv

STDOUT FORMAT (mandatory):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<sql> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

All debug output goes to stderr. Stdout has only [START], [STEP], [END] lines.
"""

import os
import sys
import json
import time
import requests

# ── Configuration ─────────────────────────────────────────────────────────────

ENV_BASE_URL = "https://p-karthik-mohan-sql-analyst-env.hf.space"
MAX_ATTEMPTS = 1
BENCHMARK    = "sql-analyst-env"

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.environ.get("HF_TOKEN")

client = None

# ── Stdout log functions (mandatory format) ───────────────────────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    action_clean = str(action).replace("\n", " ").strip()[:120]
    error_val    = error if error else "null"
    done_val     = str(done).lower()
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def debug(msg):
    print(msg, file=sys.stderr, flush=True)

# ── Environment helpers ───────────────────────────────────────────────────────

def env_reset(task_id):
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(sql):
    r = requests.post(f"{ENV_BASE_URL}/step", json={"action": sql}, timeout=30)
    r.raise_for_status()
    return r.json()

def wait_for_server(retries=10, delay=3.0):
    debug("Waiting for environment server...")
    for i in range(retries):
        try:
            r = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                debug("Server is ready.")
                return
        except Exception:
            pass
        debug(f"  Not ready yet... ({i+1}/{retries})")
        time.sleep(delay)
    debug("ERROR: Server did not start in time.")
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
    task_name = f"sql-task-{task_id}"

    try:
        reset_resp = env_reset(task_id)
    except Exception as e:
        debug(f"ERROR resetting task {task_id}: {e}")
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        log_step(step=1, action="", reward=0.0, done=False, error=str(e))
        log_end(success=False, steps=1, rewards=[0.0])
        return {"task_id": task_id, "task_name": task_name, "difficulty": "unknown",
                "best_reward": 0.0, "attempts": 1, "solved": False}

    obs        = reset_resp["observation"]
    task_desc  = obs["task_description"]
    schema     = obs["schema"]
    hint       = obs["hint"]
    difficulty = obs["difficulty"]

    debug(f"\n{'='*60}")
    debug(f"TASK {task_id} ({difficulty.upper()})")
    debug(f"Task: {task_desc}\n")

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    previous_attempts = []
    all_rewards       = []
    best_reward       = 0.0
    steps_taken       = 0
    success           = False

    for attempt in range(1, MAX_ATTEMPTS + 1):
        debug(f"  Attempt {attempt}/{MAX_ATTEMPTS} — asking LLM...")
        error = None
        sql   = ""

        try:
            sql = ask_llm(task_desc, schema, hint, attempt, previous_attempts)
            debug(f"  SQL: {sql[:120]}{'...' if len(sql) > 120 else ''}")
        except Exception as e:
            error = str(e)
            debug(f"  LLM error: {error}")
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
            debug(f"  Step error: {error}")
            log_step(step=attempt, action=sql, reward=0.0, done=False, error=error)
            all_rewards.append(0.0)
            steps_taken = attempt
            continue

        all_rewards.append(reward)
        steps_taken = attempt
        best_reward = max(best_reward, reward)

        debug(f"  Reward: {reward:.3f}  (cols={details.get('column_score',0):.2f}  rows={details.get('row_score',0):.2f}  vals={details.get('value_score',0):.2f})")

        log_step(step=attempt, action=sql, reward=reward, done=done, error=error)

        previous_attempts.append({
            "attempt": attempt,
            "sql":     sql,
            "reward":  reward,
            "details": details,
        })

        if done:
            debug(f"  PERFECT SCORE on attempt {attempt}!")
            success = True
            break
        elif reward >= 0.8:
            debug(f"  Score is close ({reward:.3f}). Trying to improve...")
        else:
            debug(f"  Score is low ({reward:.3f}). Refining query...")

    log_end(success=success, steps=steps_taken, rewards=all_rewards)

    return {
        "task_id":     task_id,
        "task_name":   task_name,
        "difficulty":  difficulty,
        "best_reward": best_reward,
        "attempts":    steps_taken,
        "solved":      success,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global client

    try:
        from openai import OpenAI
        api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or "no-key-needed"
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=api_key,
        )
    except Exception as e:
        debug(f"ERROR initializing OpenAI client: {e}")
        sys.exit(1)

    debug("SQL Analyst OpenEnv — Baseline Inference Agent")
    debug(f"Model      : {MODEL_NAME}")
    debug(f"API Base   : {API_BASE_URL}")
    debug(f"Env Server : {ENV_BASE_URL}")

    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])
    else:
        task_id = int(os.environ.get("TASK_ID", "1"))

    wait_for_server()

    result = solve_task(task_id)

    debug(f"\nRESULT: Task {result['task_id']} ({result['difficulty']}) — {'SOLVED' if result['solved'] else 'best=' + str(round(result['best_reward'], 3))}")

    with open("results.json", "w") as f:
        json.dump({
            "results":      [result],
            "avg_score":    round(result["best_reward"], 3),
            "tasks_solved": 1 if result["solved"] else 0,
        }, f, indent=2)
    debug("Results saved to results.json")

if __name__ == "__main__":
    main()