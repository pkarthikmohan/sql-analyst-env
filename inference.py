"""
inference.py — Baseline AI agent for SQL Analyst OpenEnv

Required environment variables:
    API_BASE_URL  LLM API endpoint
    MODEL_NAME    Model identifier
    HF_TOKEN      HuggingFace / API key

Stdout format:
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import sys
import json
import time
import requests
from typing import List, Optional
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

ENV_BASE_URL = "https://p-karthik-mohan-sql-analyst-env.hf.space"
MAX_ATTEMPTS = 5
BENCHMARK    = "sql-analyst-env"

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.environ.get("HF_TOKEN")

TASKS = [1, 2, 3, 4, 5, 6, 7, 8]
SUCCESS_SCORE_THRESHOLD = 0.5

# Initialize client at top level like the sample
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "no-key-needed",
)

# ── Stdout log functions (mandatory format) ───────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_clean = str(action).replace("\n", " ").strip()[:120]
    error_val    = error if error else "null"
    done_val     = str(done).lower()
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def debug(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

# ── Environment helpers ───────────────────────────────────────────────────────

def env_reset(task_id: int) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(sql: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json={"action": sql}, timeout=30)
    r.raise_for_status()
    return r.json()

def wait_for_server(retries: int = 10, delay: float = 3.0) -> None:
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

def build_system_prompt() -> str:
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

def ask_llm(task_description, schema, hint, attempt, previous_attempts) -> str:
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

# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_id: int) -> None:
    task_name  = f"sql-task-{task_id}"
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False
    last_error: Optional[str] = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = env_reset(task_id)
        obs        = reset_resp["observation"]
        task_desc  = obs["task_description"]
        schema     = obs["schema"]
        hint       = obs["hint"]
        difficulty = obs["difficulty"]

        debug(f"\n{'='*60}")
        debug(f"TASK {task_id} ({difficulty.upper()})")
        debug(f"Task: {task_desc}\n")

        previous_attempts = []

        for attempt in range(1, MAX_ATTEMPTS + 1):
            debug(f"  Attempt {attempt}/{MAX_ATTEMPTS} — asking LLM...")
            last_error = None
            sql = ""

            try:
                sql = ask_llm(task_desc, schema, hint, attempt, previous_attempts)
                debug(f"  SQL: {sql[:120]}{'...' if len(sql) > 120 else ''}")
            except Exception as e:
                last_error = str(e)
                debug(f"  LLM error: {last_error}")
                log_step(step=attempt, action="", reward=0.0, done=False, error=last_error)
                rewards.append(0.0)
                steps_taken = attempt
                continue

            try:
                step_resp = env_step(sql)
                reward    = step_resp["reward"]
                done      = step_resp["done"]
                details   = step_resp["observation"].get("reward_breakdown", {})
            except Exception as e:
                last_error = str(e)
                debug(f"  Step error: {last_error}")
                log_step(step=attempt, action=sql, reward=0.0, done=False, error=last_error)
                rewards.append(0.0)
                steps_taken = attempt
                continue

            rewards.append(reward)
            steps_taken = attempt

            debug(f"  Reward: {reward:.3f}  (cols={details.get('column_score',0):.2f}  rows={details.get('row_score',0):.2f}  vals={details.get('value_score',0):.2f})")

            log_step(step=attempt, action=sql, reward=reward, done=done, error=last_error)

            previous_attempts.append({
                "attempt": attempt,
                "sql":     sql,
                "reward":  reward,
                "details": details,
            })

            if done:
                debug(f"  PERFECT SCORE on attempt {attempt}!")
                break
            elif reward >= 0.8:
                debug(f"  Score is close ({reward:.3f}). Trying to improve...")
            else:
                debug(f"  Score is low ({reward:.3f}). Refining query...")

    except Exception as e:
        last_error = str(e)
        debug(f"ERROR in task {task_id}: {last_error}")

    finally:
        # Clamp score strictly between 0 and 1 — required by OpenEnv spec
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(1e-6, min(score, 1 - 1e-6))
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    debug("SQL Analyst OpenEnv — Baseline Inference Agent")
    debug(f"Model      : {MODEL_NAME}")
    debug(f"API Base   : {API_BASE_URL}")
    debug(f"Env Server : {ENV_BASE_URL}")

    wait_for_server()

    for task_id in TASKS:
        run_task(task_id)

if __name__ == "__main__":
    main()