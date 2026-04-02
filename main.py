import sqlite3
import os
import re
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DB_PATH = os.path.join("data", "ecommerce.db")
app = FastAPI(title="SQL Analyst OpenEnv", version="2.0.0")

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
    6: {
        "description": (
            "Calculate the month-over-month revenue growth percentage for completed orders in 2024. "
            "For each month show total revenue and percentage change vs previous month. "
            "Return columns: month, total_revenue, prev_revenue, growth_pct. "
            "Order by month ascending. Round growth_pct to 2 decimal places. "
            "For the first month, prev_revenue and growth_pct should be NULL."
        ),
        "difficulty": "expert",
        "hint": "Use LAG() window function to get previous month revenue, then calculate (current - prev) / prev * 100",
        "answer_query": """
            WITH monthly AS (
                SELECT STRFTIME('%m', order_date) AS month,
                       ROUND(SUM(total_amount), 2) AS total_revenue
                FROM orders
                WHERE status = 'completed'
                  AND order_date LIKE '2024%'
                GROUP BY month
            )
            SELECT month,
                   total_revenue,
                   LAG(total_revenue) OVER (ORDER BY month) AS prev_revenue,
                   ROUND(
                       (total_revenue - LAG(total_revenue) OVER (ORDER BY month))
                       / LAG(total_revenue) OVER (ORDER BY month) * 100,
                   2) AS growth_pct
            FROM monthly
            ORDER BY month ASC
        """,
    },
    7: {
        "description": (
            "For each city, find the single best-selling product by total quantity sold "
            "from completed orders. "
            "Return columns: city, product_name, total_quantity. "
            "Order by city ascending. "
            "If two products tie, return the one with the lower product_id."
        ),
        "difficulty": "expert",
        "hint": "Use RANK() OVER (PARTITION BY city ORDER BY total_quantity DESC, product_id ASC) in a CTE, then filter WHERE rank = 1",
        "answer_query": """
            WITH city_product AS (
                SELECT c.city,
                       p.product_name,
                       p.product_id,
                       SUM(o.quantity) AS total_quantity,
                       RANK() OVER (
                           PARTITION BY c.city
                           ORDER BY SUM(o.quantity) DESC, p.product_id ASC
                       ) AS rnk
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                JOIN products  p ON o.product_id  = p.product_id
                WHERE o.status = 'completed'
                GROUP BY c.city, p.product_id
            )
            SELECT city, product_name, total_quantity
            FROM city_product
            WHERE rnk = 1
            ORDER BY city ASC
        """,
    },
    8: {
        "description": (
            "Find customers whose total spending in the second half of 2024 (July-December) "
            "was strictly greater than their total spending in the first half of 2024 (January-June). "
            "Only consider completed orders. "
            "Return columns: customer_id, first_name, last_name, h1_revenue, h2_revenue. "
            "Order by h2_revenue descending."
        ),
        "difficulty": "expert",
        "hint": "Use conditional SUM with CASE WHEN to split spending by half-year, then filter WHERE h2 > h1",
        "answer_query": """
            WITH half_year AS (
                SELECT c.customer_id,
                       c.first_name,
                       c.last_name,
                       ROUND(SUM(CASE
                           WHEN STRFTIME('%m', o.order_date) BETWEEN '01' AND '06'
                           THEN o.total_amount ELSE 0 END), 2) AS h1_revenue,
                       ROUND(SUM(CASE
                           WHEN STRFTIME('%m', o.order_date) BETWEEN '07' AND '12'
                           THEN o.total_amount ELSE 0 END), 2) AS h2_revenue
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.status = 'completed'
                  AND o.order_date LIKE '2024%'
                GROUP BY c.customer_id
            )
            SELECT customer_id, first_name, last_name, h1_revenue, h2_revenue
            FROM half_year
            WHERE h2_revenue > h1_revenue
            ORDER BY h2_revenue DESC
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

def compute_reward(agent_rows, agent_cols):
    expected_rows = session["expected_rows"]
    expected_cols = session["expected_columns"]
    details = {}

    agent_cols_lower    = [c.lower() for c in agent_cols]
    expected_cols_lower = [c.lower() for c in expected_cols]
    col_matches = sum(1 for c in expected_cols_lower if c in agent_cols_lower)
    col_score = (col_matches / len(expected_cols_lower)) * 0.30 if expected_cols_lower else 0.0
    details["column_score"]    = round(col_score, 3)
    details["expected_columns"] = expected_cols
    details["agent_columns"]    = agent_cols

    expected_count = len(expected_rows)
    agent_count    = len(agent_rows)
    if expected_count == 0:
        row_score = 0.30 if agent_count == 0 else 0.0
    else:
        row_ratio = min(agent_count, expected_count) / max(agent_count, expected_count)
        row_score = row_ratio * 0.30
    details["row_score"]          = round(row_score, 3)
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
                        agt_val = normalize(agt_row.get(agent_cols[exp_idx]))
                if exp_val == agt_val:
                    matched_cells += 1
        value_score = (matched_cells / total_cells) * 0.40 if total_cells > 0 else 0.0
    details["value_score"]  = round(value_score, 3)
    details["total_reward"] = round(col_score + row_score + value_score, 3)
    return details["total_reward"], details

@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    if req.task_id not in TASKS:
        raise HTTPException(status_code=400, detail="task_id must be 1–8")
    session["task_id"]     = req.task_id
    session["task"]        = TASKS[req.task_id]
    session["attempts"]    = 0
    session["best_reward"] = 0.0
    session["history"]     = []
    compute_expected()
    observation = {
        "task_id":          req.task_id,
        "difficulty":       session["task"]["difficulty"],
        "task_description": session["task"]["description"],
        "schema":           get_schema_info(),
        "hint":             session["task"]["hint"],
    }
    return ResetResponse(
        observation=observation,
        info={"message": f"Task {req.task_id} loaded. Use POST /step with your SQL query."}
    )

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    if session["task_id"] is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    session["attempts"] += 1
    sql = req.action.strip()

    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        return StepResponse(
            observation={"error": "Only SELECT or WITH allowed."},
            reward=0.0, done=False,
            info={"attempt": session["attempts"], "message": "Rejected."}
        )

    try:
        agent_rows, agent_cols = run_query(sql)
    except Exception as e:
        session["history"].append({"attempt": session["attempts"], "sql": sql, "reward": 0.0, "error": str(e)})
        return StepResponse(
            observation={"error": str(e)}, reward=0.0, done=False,
            info={"attempt": session["attempts"], "message": "SQL error."}
        )

    reward, details = compute_reward(agent_rows, agent_cols)
    session["best_reward"] = max(session["best_reward"], reward)
    done = reward >= 1.0
    session["history"].append({
        "attempt": session["attempts"], "sql": sql,
        "reward": reward, "details": details,
        "timestamp": datetime.now().isoformat(),
    })

    return StepResponse(
        observation={
            "task_id":          session["task_id"],
            "sql_submitted":    sql,
            "result_preview":   agent_rows[:5],
            "result_row_count": len(agent_rows),
            "reward_breakdown": details,
        },
        reward=reward, done=done,
        info={
            "attempt":     session["attempts"],
            "best_reward": session["best_reward"],
            "message":     "Perfect score!" if done else "Keep refining.",
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
        "version": "2.0.0",
        "tasks":   {k: {"difficulty": v["difficulty"], "description": v["description"]} for k, v in TASKS.items()},
        "endpoints": ["/reset", "/step", "/state", "/health"],
    }

@app.get("/health")
def health():
    return {"status": "ok", "db_exists": os.path.exists(DB_PATH)}