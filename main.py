import sqlite3
import os
import json
import re
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DB_PATH = os.path.join("data", "ecommerce.db")
app = FastAPI(title="SQL Analyst OpenEnv", version="1.0.0")

class StepRequest(BaseModel):
    action: str

class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict

class ResetRequest(BaseModel):
    task_id: int

class ResetResponse(BaseModel):
    observation: dict
    info: dict

class StateResponse(BaseModel):
    task_id: int
    task_description: str
    schema_info: str
    attempts: int
    best_reward: float
    history: list

TASKS = {
    1: {
        "description": (
            "Find the total number of completed orders placed in the year 2024. "
            "Return a single number with column name: total_orders"
        ),
        "difficulty": "easy",
        "hint": "Use COUNT with WHERE filters on status and order_date",
        "answer_query": """
            SELECT COUNT(*) AS total_orders
            FROM orders
            WHERE status = 'completed'
              AND order_date LIKE '2024%'
        """,
    },
    2: {
        "description": (
            "Find the top 5 customers by total revenue (sum of total_amount for completed orders only). "
            "Return columns: first_name, last_name, total_revenue. "
            "Order by total_revenue descending."
        ),
        "difficulty": "medium",
        "hint": "JOIN orders with customers, GROUP BY customer, filter completed, ORDER and LIMIT",
        "answer_query": """
            SELECT c.first_name, c.last_name,
                   ROUND(SUM(o.total_amount), 2) AS total_revenue
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE o.status = 'completed'
            GROUP BY o.customer_id
            ORDER BY total_revenue DESC
            LIMIT 5
        """,
    },
    3: {
        "description": (
            "For each product category, calculate the total revenue (completed orders only) "
            "and rank categories by revenue using a window function. "
            "Return columns: category, total_revenue, revenue_rank. "
            "Order by revenue_rank ascending."
        ),
        "difficulty": "hard",
        "hint": "Use SUM with GROUP BY inside a CTE, then apply RANK() OVER (ORDER BY ...) on the result",
        "answer_query": """
            WITH category_revenue AS (
                SELECT p.category,
                       SUM(o.total_amount) AS total_revenue
                FROM orders o
                JOIN products p ON o.product_id = p.product_id
                WHERE o.status = 'completed'
                GROUP BY p.category
            )
            SELECT category,
                   total_revenue,
                   RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank
            FROM category_revenue
            ORDER BY revenue_rank ASC
        """,
    },
    4: {
        "description": (
            "Find the average price of products in each category, "
            "but only for categories that have more than 2 products. "
            "Return columns: category, avg_price."
        ),
        "difficulty": "medium",
        "hint": "Use GROUP BY with HAVING COUNT(...) > 2",
        "answer_query": """
            SELECT category, ROUND(AVG(price), 2) AS avg_price
            FROM products
            GROUP BY category
            HAVING COUNT(product_id) > 2
        """,
    },
    5: {
        "description": (
            "Identify customers who have ordered products from both the Electronics "
            "and Clothing categories. Return columns: customer_id, first_name."
        ),
        "difficulty": "hard",
        "hint": "Use INTERSECT on two queries filtering by category",
        "answer_query": """
            SELECT DISTINCT c.customer_id, c.first_name
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            JOIN products p ON o.product_id = p.product_id
            WHERE p.category = 'Electronics'
            INTERSECT
            SELECT DISTINCT c.customer_id, c.first_name
            FROM customers c
            JOIN orders o ON c.customer_id = o.customer_id
            JOIN products p ON o.product_id = p.product_id
            WHERE p.category = 'Clothing'
        """,
    },
}

session = {
    "task_id": None,
    "task": None,
    "expected_rows": None,
    "expected_columns": None,
    "attempts": 0,
    "best_reward": 0.0,
    "history": [],
}

def get_connection():
    if not os.path.exists(DB_PATH):
        raise HTTPException(
            status_code=500,
            detail=f"Database not found at {DB_PATH}. Run seed.py first."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
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
    return "Tables:\n" + "\n".join(schema_parts)

def run_query(sql: str) -> tuple[list[dict], list[str]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql)
    columns = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    conn.close()
    return rows, columns

def compute_expected():
    task = session["task"]
    rows, columns = run_query(task["answer_query"])
    session["expected_rows"] = rows
    session["expected_columns"] = columns

def compute_reward(agent_rows: list[dict], agent_cols: list[str]) -> tuple[float, dict]:
    expected_rows = session["expected_rows"]
    expected_cols = session["expected_columns"]
    details = {}

    agent_cols_lower    = [c.lower() for c in agent_cols]
    expected_cols_lower = [c.lower() for c in expected_cols]
    col_matches = sum(1 for c in expected_cols_lower if c in agent_cols_lower)
    col_score = (col_matches / len(expected_cols_lower)) * 0.30 if expected_cols_lower else 0.0
    details["column_score"] = round(col_score, 3)
    details["expected_columns"] = expected_cols
    details["agent_columns"] = agent_cols

    expected_count = len(expected_rows)
    agent_count    = len(agent_rows)
    if expected_count == 0:
        row_score = 0.30 if agent_count == 0 else 0.0
    else:
        row_ratio = min(agent_count, expected_count) / max(agent_count, expected_count)
        row_score = row_ratio * 0.30
    details["row_score"] = round(row_score, 3)
    details["expected_row_count"] = expected_count
    details["agent_row_count"]    = agent_count

    if not expected_rows or not agent_rows:
        value_score = 0.0
    else:
        def normalize(v):
            if v is None:
                return ""
            try:
                return str(round(float(v), 1))
            except (ValueError, TypeError):
                return str(v).strip().lower()

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

@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    if req.task_id not in TASKS:
        raise HTTPException(status_code=400, detail="task_id must be 1, 2, 3, 4, or 5")

    session["task_id"]     = req.task_id
    session["task"]        = TASKS[req.task_id]
    session["attempts"]    = 0
    session["best_reward"] = 0.0
    session["history"]     = []

    compute_expected()

    schema = get_schema_info()
    observation = {
        "task_id":          req.task_id,
        "difficulty":       session["task"]["difficulty"],
        "task_description": session["task"]["description"],
        "schema":           schema,
        "hint":             session["task"]["hint"],
    }
    return ResetResponse(
        observation=observation,
        info={"message": f"Task {req.task_id} loaded. Use POST /step with your SQL query."}
    )

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    if session["task_id"] is None:
        raise HTTPException(status_code=400, detail="Call /reset first to load a task.")

    session["attempts"] += 1
    sql = req.action.strip()

    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        return StepResponse(
            observation={"error": "Only SELECT or WITH (CTE) statements are allowed."},
            reward=0.0,
            done=False,
            info={"attempt": session["attempts"], "message": "Rejected: not a SELECT/WITH query."}
        )

    try:
        agent_rows, agent_cols = run_query(sql)
    except Exception as e:
        entry = {
            "attempt": session["attempts"],
            "sql": sql,
            "reward": 0.0,
            "error": str(e),
        }
        session["history"].append(entry)
        return StepResponse(
            observation={"error": str(e), "sql_submitted": sql},
            reward=0.0,
            done=False,
            info={"attempt": session["attempts"], "message": "SQL execution error."}
        )

    reward, details = compute_reward(agent_rows, agent_cols)
    session["best_reward"] = max(session["best_reward"], reward)
    done = reward >= 1.0

    entry = {
        "attempt":   session["attempts"],
        "sql":       sql,
        "reward":    reward,
        "details":   details,
        "timestamp": datetime.now().isoformat(),
    }
    session["history"].append(entry)

    observation = {
        "task_id":          session["task_id"],
        "task_description": session["task"]["description"],
        "sql_submitted":    sql,
        "result_preview":   agent_rows[:5],
        "result_row_count": len(agent_rows),
        "reward_breakdown": details,
    }

    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info={
            "attempt":     session["attempts"],
            "best_reward": session["best_reward"],
            "message":     "Perfect score! Task complete." if done else "Keep refining your query.",
        }
    )

@app.get("/state", response_model=StateResponse)
def state():
    if session["task_id"] is None:
        raise HTTPException(status_code=400, detail="No active task. Call /reset first.")

    return StateResponse(
        task_id=session["task_id"],
        task_description=session["task"]["description"],
        schema_info=get_schema_info(),
        attempts=session["attempts"],
        best_reward=session["best_reward"],
        history=session["history"],
    )

@app.get("/")
def root():
    return {
        "name":    "SQL Analyst OpenEnv",
        "version": "1.0.0",
        "tasks":   {k: {"difficulty": v["difficulty"], "description": v["description"]} for k, v in TASKS.items()},
        "endpoints": ["/reset", "/step", "/state"],
    }

@app.get("/health")
def health():
    return {"status": "ok", "db_exists": os.path.exists(DB_PATH)}