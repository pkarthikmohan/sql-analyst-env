import os
import re

# 1. Update requirements.txt
req_txt = "requirements.txt"
with open(req_txt, "r") as f:
    reqs = f.read()
if "gradio" not in reqs:
    with open(req_txt, "a") as f:
        f.write("\ngradio==4.44.0\npandas\n")

# 2. Create the Gradio UI file (ui.py)
ui_code = """
import gradio as gr
import requests
import pandas as pd
import json
import uuid

ENV_BASE_URL = "http://127.0.0.1:7860"

def new_session():
    return str(uuid.uuid4())

def load_task(task_id, sid):
    try:
        task_num = int(task_id.split()[1])
        r = requests.post(f"{ENV_BASE_URL}/api/reset", json={"task_id": task_num, "session_id": sid})
        if r.status_code != 200:
            return f"Error: {r.text}", "", "", pd.DataFrame(), f"Error loading task {task_num}"
            
        data = r.json()
        obs = data["observation"]
        return obs["task_description"], obs["schema"], obs["hint"], pd.DataFrame(), "Task loaded. Write SQL below."
    except Exception as e:
        return str(e), "", "", pd.DataFrame(), "Error loading task"

def run_sql(sql, sid):
    if not sql.strip():
        return pd.DataFrame(), "Please enter a SQL query.", "Error"
    try:
        r = requests.post(f"{ENV_BASE_URL}/api/step", json={"action": sql, "session_id": sid})
        data = r.json()
        obs = data.get("observation", {})
        reward = data.get("reward", 0.0)
        done = data.get("done", False)
        
        df = pd.DataFrame(obs.get("result_preview", []))
        breakdown = obs.get("reward_breakdown", {})
        
        feedback = f"🎯 Reward: {reward:.2f} / 1.0\\n"
        if breakdown:
            feedback += f"Columns: {breakdown.get('column_score',0):.2f}, Rows: {breakdown.get('row_score',0):.2f}, Values: {breakdown.get('value_score',0):.2f}"
        if "error" in obs:
            feedback += f"\\n\\n⚠️ Error: {obs['error']}"
        
        return df, feedback, "✅ SOLVED!" if done else "Keep trying!"
    except Exception as e:
         return pd.DataFrame(), str(e), "Error"

def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
        gr.Markdown("# 📊 SQL Analyst OpenEnv - Hackathon Edition")
        gr.Markdown("Test Human or AI performance on realistic E-Commerce SQL Data tasks. [API served at `/api`]")
        
        sid = gr.State(new_session)
        
        with gr.Row():
            with gr.Column(scale=1):
                task_dropdown = gr.Dropdown(choices=["Task 1 (Easy)", "Task 2 (Medium)", "Task 3 (Hard)", "Task 4 (Medium)", "Task 5 (Hard)"], value="Task 1 (Easy)", label="Select Task")
                btn_load = gr.Button("🔄 Load Task")
                
                desc = gr.Textbox(label="Business Question", interactive=False, lines=2)
                hint = gr.Textbox(label="Hint", interactive=False)
                schema = gr.Code(label="Database Schema", language="sql", interactive=False)
            
            with gr.Column(scale=2):
                sql_input = gr.Code(label="SQL Editor", language="sql", lines=10)
                btn_run = gr.Button("🚀 Run SQL", variant="primary")
                
                status_out = gr.Markdown("Ready.")
                feedback_out = gr.Textbox(label="Feedback & Score", interactive=False)
                grid_out = gr.Dataframe(label="Result Preview (First 5 Rows)")

        btn_load.click(load_task, inputs=[task_dropdown, sid], outputs=[desc, schema, hint, grid_out, feedback_out])
        btn_run.click(run_sql, inputs=[sql_input, sid], outputs=[grid_out, feedback_out, status_out])
        
    return demo
"""
with open("ui.py", "w", encoding="utf-8") as f:
    f.write(ui_code.strip() + "\n")

# 3. Rewrite main.py (adding concurrency fixes, security, and mounting gradio)
main_py_code = """
import sqlite3
import os
import json
import re
from datetime import datetime
from typing import Any, Optional
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import gradio as gr
from ui import build_ui

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join("data", "ecommerce.db")
# Under /api for direct agent access, root for UI
api_app = FastAPI(title="SQL Analyst OpenEnv API", version="1.1.0")

# ── Pydantic Models ───────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    session_id: str = "default"
    action: str

class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict

class ResetRequest(BaseModel):
    task_id: int
    session_id: str = "default"

class ResetResponse(BaseModel):
    observation: dict
    info: dict

class StateRequest(BaseModel):
    session_id: str = "default"

class StateResponse(BaseModel):
    session_id: str
    task_id: Optional[int]
    task_description: Optional[str]
    schema_info: str
    attempts: int
    best_reward: float
    history: list

# ── Task Definitions ──────────────────────────────────────────────────────────

TASKS = {
    1: {
        "description": "Find the total number of completed orders placed in the year 2024. Return a single number with column name: total_orders",
        "difficulty": "easy",
        "hint": "Use COUNT with WHERE filters on status and order_date",
        "answer_query": "SELECT COUNT(*) AS total_orders FROM orders WHERE status = 'completed' AND order_date LIKE '2024%'",
    },
    2: {
        "description": "Find the top 5 customers by total revenue (sum of total_amount for completed orders only). Return columns: first_name, last_name, total_revenue. Order by total_revenue descending.",
        "difficulty": "medium",
        "hint": "JOIN orders with customers, GROUP BY customer, filter completed, ORDER and LIMIT",
        "answer_query": "SELECT c.first_name, c.last_name, ROUND(SUM(o.total_amount), 2) AS total_revenue FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE o.status = 'completed' GROUP BY o.customer_id ORDER BY total_revenue DESC LIMIT 5",
    },
    3: {
        "description": "For each product category, calculate the total revenue (completed orders only) and rank categories by revenue using a window function. Return columns: category, total_revenue, revenue_rank. Order by revenue_rank ascending.",
        "difficulty": "hard",
        "hint": "Use SUM with GROUP BY inside a CTE, then apply RANK() OVER (ORDER BY ...) on the result",
        "answer_query": "WITH category_revenue AS ( SELECT p.category, SUM(o.total_amount) AS total_revenue FROM orders o JOIN products p ON o.product_id = p.product_id WHERE o.status = 'completed' GROUP BY p.category ) SELECT category, total_revenue, RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank FROM category_revenue ORDER BY revenue_rank ASC",
    },
    4: {
        "description": "Find the average price of products in each category, but only for categories that have more than 2 products. Return columns: category, avg_price.",
        "difficulty": "medium",
        "hint": "Use GROUP BY with HAVING COUNT(...) > 2.",
        "answer_query": "SELECT category, ROUND(AVG(price), 2) AS avg_price FROM products GROUP BY category HAVING COUNT(product_id) > 2",
    },
    5: {
        "description": "Identify customers who have ordered products from both the 'Electronics' and 'Clothing' categories. Return columns: customer_id, first_name.",
        "difficulty": "hard",
        "hint": "Use INTERSECT on two queries, or GROUP BY customer HAVING COUNT(DISTINCT category) = 2.",
        "answer_query": "SELECT DISTINCT c.customer_id, c.first_name FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN products p ON o.product_id = p.product_id WHERE p.category = 'Electronics' INTERSECT SELECT DISTINCT c.customer_id, c.first_name FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN products p ON o.product_id = p.product_id WHERE p.category = 'Clothing'",
    },
}

# ── Sessions ──────────────────────────────────────────────────────────────────

sessions = {}

def get_session(sid: str):
    if sid not in sessions:
        sessions[sid] = {
            "task_id": None, "task": None, 
            "expected_rows": None, "expected_columns": None,
            "attempts": 0, "best_reward": 0.0, "history": []
        }
    return sessions[sid]

# ── Database Helpers ──────────────────────────────────────────────────────────

def progress_handler():
    raise sqlite3.OperationalError("Query execution aborted: Timed out or exceeded instruction limits. Hint: Too complex CROSS JOIN?")

def get_connection():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail=f"Database not found at {DB_PATH}.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Security feature: Prevent DOS
    conn.set_progress_handler(progress_handler, 500000)
    return conn

def get_schema_info() -> str:
    conn = get_connection()
    cur = conn.cursor()
    schema_parts = []
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r["name"] for r in cur.fetchall()]
    for table in tables:
        cur.execute(f"PRAGMA table_info({table})")
        cols = cur.fetchall()
        col_defs = ", ".join(f"{c['name']} {c['type']}" for c in cols)
        cur.execute(f"SELECT COUNT(*) AS n FROM {table}")
        count = cur.fetchone()["n"]
        schema_parts.append(f"  {table} ({col_defs})  -- {count} rows")
    conn.close()
    return "Tables:\\n" + "\\n".join(schema_parts)

def run_query(sql: str) -> tuple[list[dict], list[str]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql)
    columns = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    conn.close()
    return rows, columns

def compute_expected(sid: str):
    session = get_session(sid)
    task = session["task"]
    rows, columns = run_query(task["answer_query"])
    session["expected_rows"] = rows
    session["expected_columns"] = columns

# ── Reward Function ───────────────────────────────────────────────────────────

def compute_reward(sid: str, agent_rows: list[dict], agent_cols: list[str]) -> tuple[float, dict]:
    session = get_session(sid)
    expected_rows = session["expected_rows"]
    expected_cols = session["expected_columns"]
    details = {}

    agent_cols_lower   = [c.lower() for c in agent_cols]
    expected_cols_lower = [c.lower() for c in expected_cols]
    col_matches = sum(1 for c in expected_cols_lower if c in agent_cols_lower)
    col_score = (col_matches / len(expected_cols_lower)) * 0.30 if expected_cols_lower else 0.0
    details["column_score"] = round(col_score, 3)

    expected_count = len(expected_rows)
    agent_count    = len(agent_rows)
    if expected_count == 0:
        row_score = 0.30 if agent_count == 0 else 0.0
    else:
        row_ratio = min(agent_count, expected_count) / max(agent_count, expected_count)
        row_score = row_ratio * 0.30
    details["row_score"] = round(row_score, 3)

    if not expected_rows or not agent_rows:
        value_score = 0.0
    else:
        def normalize(v):
            if v is None: return ""
            try: return str(round(float(v), 1))
            except: return str(v).strip().lower()

        matched_cells = 0
        total_cells   = len(expected_rows) * len(expected_cols_lower)

        for exp_row, agt_row in zip(expected_rows, agent_rows):
            for col in expected_cols_lower:
                exp_val = normalize(exp_row.get(col) or exp_row.get(col.upper()))
                agt_val = normalize(agt_row.get(col) or agt_row.get(col.upper()))
                if not agt_val:
                    exp_idx = expected_cols_lower.index(col)
                    if exp_idx < len(agent_cols):
                        pos_col = agent_cols[exp_idx]
                        agt_val = normalize(agt_row.get(pos_col))
                if exp_val == agt_val:
                    matched_cells += 1

        value_score = (matched_cells / total_cells) * 0.40 if total_cells > 0 else 0.0
    details["value_score"] = round(value_score, 3)

    total = round(col_score + row_score + value_score, 3)
    details["total_reward"] = total
    return total, details

# ── Endpoints ─────────────────────────────────────────────────────────────────

@api_app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    if req.task_id not in TASKS: raise HTTPException(status_code=400, detail="Invalid task_id")
    session = get_session(req.session_id)
    session["task_id"]    = req.task_id
    session["task"]       = TASKS[req.task_id]
    session["attempts"]   = 0
    session["best_reward"] = 0.0
    session["history"]    = []
    compute_expected(req.session_id)
    observation = {
        "task_id":          req.task_id,
        "difficulty":       session["task"]["difficulty"],
        "task_description": session["task"]["description"],
        "schema":           get_schema_info(),
        "hint":             session["task"]["hint"],
    }
    return ResetResponse(observation=observation, info={"message": f"Task {req.task_id} loaded."})

@api_app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    session = get_session(req.session_id)
    if session["task_id"] is None: raise HTTPException(status_code=400, detail="Call /reset first.")
    session["attempts"] += 1
    sql = req.action.strip()

    if not re.match(r"^\\s*(SELECT|WITH)\\b", sql, re.IGNORECASE):
        return StepResponse(
            observation={"error": "Only SELECT or WITH allowed."}, reward=0.0, done=False,
            info={"attempt": session["attempts"], "message": "Rejected"}
        )

    try:
        agent_rows, agent_cols = run_query(sql)
    except Exception as e:
        session["history"].append({"attempt": session["attempts"], "sql": sql, "reward": 0.0, "error": str(e)})
        return StepResponse(
            observation={"error": str(e), "sql_submitted": sql}, reward=0.0, done=False,
            info={"attempt": session["attempts"], "message": "SQL Error"}
        )

    reward, details = compute_reward(req.session_id, agent_rows, agent_cols)
    session["best_reward"] = max(session["best_reward"], reward)
    done = reward >= 1.0
    session["history"].append({"attempt": session["attempts"], "sql": sql, "reward": reward, "details": details})

    observation = {
        "task_id": session["task_id"], "task_description": session["task"]["description"],
        "sql_submitted": sql, "result_preview": agent_rows[:5], "result_row_count": len(agent_rows),
        "reward_breakdown": details,
    }
    return StepResponse(
        observation=observation, reward=reward, done=done,
        info={"attempt": session["attempts"], "best_reward": session["best_reward"]}
    )

@api_app.get("/state", response_model=StateResponse)
def state(req: StateRequest):
    session = get_session(req.session_id)
    return StateResponse(
        session_id=req.session_id,
        task_id=session["task_id"],
        task_description=session["task"]["description"] if session["task"] else None,
        schema_info=get_schema_info(),
        attempts=session["attempts"],
        best_reward=session["best_reward"],
        history=session["history"],
    )

@api_app.get("/")
def root():
    return {"message": "API running at /api. Try the UI at root (handled by wrapper)!"}

# ── Server Setup ──────────────────────────────────────────────────────────────

demo = build_ui()
app = gr.mount_gradio_app(api_app, demo, path="/")
"""
with open("main.py", "w", encoding="utf-8") as f:
    f.write(main_py_code.strip() + "\n")

# 4. Modify inference.py to use session IDs
with open("inference.py", "r", encoding="utf-8") as f:
    inf = f.read()

inf = inf.replace('ENV_BASE_URL = "http://127.0.0.1:7860"', 'ENV_BASE_URL = "http://127.0.0.1:7860/api"')
inf = inf.replace("TASK_IDS     = [1, 2, 3]", "TASK_IDS     = [1, 2, 3, 4, 5]")
inf = inf.replace('def env_reset(task_id: int) -> dict:', 'def env_reset(task_id: int, session_id: str) -> dict:')
inf = inf.replace('json={"task_id": task_id}', 'json={"task_id": task_id, "session_id": session_id}')

inf = inf.replace('def env_step(sql: str) -> dict:', 'def env_step(sql: str, session_id: str) -> dict:')
inf = inf.replace('json={"action": sql}', 'json={"action": sql, "session_id": session_id}')

inf = inf.replace("reset_resp = env_reset(task_id)", 'session_id = f"baseline_{task_id}"\n    reset_resp = env_reset(task_id, session_id)')
inf = inf.replace("step_resp = env_step(sql)", 'step_resp = env_step(sql, session_id)')

inf = inf.replace('r = requests.get(f"{ENV_BASE_URL}/state")', 'r = requests.get(f"{ENV_BASE_URL}/state", json={"session_id": "baseline_1"})')

# health check fix for inference wait_for_server
inf = re.sub(r'requests\.get\(f"\{ENV_BASE_URL\}/health", timeout=3\)', 'requests.get(f"{ENV_BASE_URL}/", timeout=3)', inf)

with open("inference.py", "w", encoding="utf-8") as f:
    f.write(inf)

print("Done generating make_awesome.")
