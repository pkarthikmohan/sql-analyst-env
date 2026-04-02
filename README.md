---
title: SQL Analyst OpenEnv
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 📊 SQL Analyst OpenEnv

> A real-world OpenEnv environment where an AI agent must write correct SQL queries
> against a live e-commerce database to answer business analytics questions.

**Live Demo:** https://p-karthik-mohan-sql-analyst-env.hf.space/docs

---

## What Is This?

-This environment simulates the daily work of a **data analyst at an e-commerce company**.
The AI agent receives a natural language business question, explores the database schema,
and must produce a correct SQL query to answer it.

Unlike toy environments, this is a task that real analysts perform every day —
making it a meaningful benchmark for AI reasoning and code generation.

---

## Baseline Agent Results

| Metric | Score |
|---|---|
| Tasks solved | 6 / 8 |
| Average reward | 0.985 / 1.000 |
| Model used | llama-3.1-8b-instant (Groq) |
| Time to complete | Under 3 minutes |

---

## Tasks

Eight tasks of increasing difficulty, each graded by an automated SQL result comparator.

### Task 1 — Easy
**"How many completed orders were placed in 2024?"**
- Requires: COUNT, WHERE, date filtering
- Expected output: single row — total_orders

### Task 2 — Medium
**"Find the top 5 customers by total revenue from completed orders."**
- Requires: JOIN, GROUP BY, SUM, ORDER BY, LIMIT
- Expected output: 5 rows — first_name, last_name, total_revenue

### Task 3 — Hard
**"Rank product categories by total revenue using a window function."**
- Requires: CTE, JOIN, GROUP BY, RANK() OVER (...)
- Expected output: all categories — category, total_revenue, revenue_rank

### Task 4 — Medium
**"Find average product price per category, only for categories with more than 2 products."**
- Requires: GROUP BY, AVG, HAVING
- Expected output: category, avg_price

### Task 5 — Hard
**"Find customers who ordered from both Electronics and Clothing categories."**
- Requires: INTERSECT, multiple JOINs
- Expected output: customer_id, first_name

### Task 6 — Expert
**"Calculate month-over-month revenue growth % for 2024."**
- Requires: LAG() window function, CTE, STRFTIME
- Expected output: month, total_revenue, prev_revenue, growth_pct

### Task 7 — Expert
**"For each city, find the best-selling product by quantity."**
- Requires: RANK() OVER (PARTITION BY city ...), CTE
- Expected output: city, product_name, total_quantity

### Task 8 — Expert
**"Find customers who spent more in H2 2024 than H1 2024."**
- Requires: CASE WHEN, conditional SUM, HAVING
- Expected output: customer_id, first_name, last_name, h1_revenue, h2_revenue

---

## API Endpoints

Base URL: https://p-karthik-mohan-sql-analyst-env.hf.space

### POST /reset
Load a task. Always call this first.

Request:
```json
{"task_id": 1}
```

Response:
```json
{
  "observation": {
    "task_id": 1,
    "difficulty": "easy",
    "task_description": "Find the total number of completed orders...",
    "schema": "Tables:\n  customers (...)\n  products (...)\n  orders (...)",
    "hint": "Use COUNT with WHERE filters on status and order_date"
  },
  "info": {"message": "Task 1 loaded."}
}
```

### POST /step
Submit a SQL query. Returns reward 0.0-1.0 and feedback.

Request:
```json
{"action": "SELECT COUNT(*) AS total_orders FROM orders WHERE status = 'completed'"}
```

Response:
```json
{
  "observation": {
    "result_preview": [{"total_orders": 312}],
    "reward_breakdown": {
      "column_score": 0.30,
      "row_score": 0.30,
      "value_score": 0.40,
      "total_reward": 1.0
    }
  },
  "reward": 1.0,
  "done": true
}
```

### GET /state
Get current session — task info, all attempts, best score.

### GET /health
Returns {"status": "ok", "db_exists": true} when server is ready.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| task_description | string | Natural language business question |
| schema | string | All table names, columns, types, row counts |
| hint | string | Guidance on which SQL constructs to use |
| result_preview | array | First 5 rows of the agent query result |
| result_row_count | integer | Total rows returned by agent query |
| reward_breakdown | object | Sub-scores for columns, rows, values |

---

## Action Space

| Property | Value |
|---|---|
| Type | string |
| Format | Valid SQLite SELECT or WITH statement |
| Restrictions | No INSERT, UPDATE, DELETE, DROP |

Example actions:
```sql
-- Easy
SELECT COUNT(*) AS total_orders FROM orders WHERE status = 'completed'

-- Medium
SELECT c.first_name, c.last_name, SUM(o.total_amount) AS total_revenue
FROM orders o JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY o.customer_id ORDER BY total_revenue DESC LIMIT 5

-- Expert
WITH monthly AS (
  SELECT STRFTIME('%Y-%m', order_date) AS month, SUM(total_amount) AS total_revenue
  FROM orders WHERE status = 'completed' AND order_date LIKE '2024%'
  GROUP BY month
)
SELECT month, total_revenue,
  LAG(total_revenue) OVER (ORDER BY month) AS prev_revenue,
  ROUND((total_revenue - LAG(total_revenue) OVER (ORDER BY month))
    / LAG(total_revenue) OVER (ORDER BY month) * 100, 2) AS growth_pct
FROM monthly ORDER BY month ASC
```

---

## Reward Function

Partial credit score from 0.0 to 1.0 with three components.
Even an imperfect query receives meaningful feedback — not just pass/fail.

| Component | Weight | How it is measured |
|---|---|---|
| Column names | 0.30 | Fraction of expected column names present |
| Row count | 0.30 | Ratio of returned rows vs expected rows |
| Cell values | 0.40 | Fraction of cells matching expected values |

---

## Database Schema

A realistic e-commerce dataset with 600 orders, 100 customers, 30 products.

```
customers  (customer_id, first_name, last_name, email, city, signup_date)  — 100 rows
products   (product_id, product_name, category, price, stock)              — 30 rows
orders     (order_id, customer_id, product_id, quantity, total_amount, order_date, status) — 600 rows
```

All orders are dated in 2024. Status values: completed, pending, cancelled.
10 product categories: Electronics, Clothing, Books, Home & Garden, Sports, Beauty, Toys, Food & Grocery, Automotive, Music.

---

## Running the Baseline Agent

```bash
pip install openai requests python-dotenv

export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-8b-instant
export HF_TOKEN=your_groq_api_key_here

python inference.py
```

Expected output:
```
SQL Analyst OpenEnv — Baseline Inference Agent
Model      : llama-3.1-8b-instant
Tasks solved   : 6 / 8
Average reward : 0.985 / 1.000
Results saved to results.json
```

---

## Local Setup

```bash
git clone https://huggingface.co/spaces/P-Karthik-Mohan/sql-analyst-env
cd sql-analyst-env
pip install -r requirements.txt
python seed.py
uvicorn main:app --host 0.0.0.0 --port 7860
```

---

## Docker

```bash
docker build -t sql-analyst-env .
docker run -p 7860:7860 sql-analyst-env
```

---

## Hardware Requirements

| Resource | Requirement |
|---|---|
| CPU | 1-2 vCPU |
| RAM | 8GB |
| GPU | Not required |
| Disk | ~50MB |
| Inference time | Under 5 minutes for all 8 tasks |

---

## Project Structure

```
sql-analyst-env/
├── main.py          # FastAPI server — /reset, /step, /state endpoints
├── seed.py          # Database seeder — creates ecommerce.db
├── inference.py     # Baseline AI agent using OpenAI-compatible client
├── openenv.yaml     # OpenEnv specification file
├── Dockerfile       # Container for Hugging Face Spaces
├── requirements.txt # Python dependencies
└── data/
    └── ecommerce.db # SQLite database (auto-created by seed.py)
```

---

## Built With

- FastAPI — REST API framework
- SQLite — embedded database, no setup required
- OpenAI Python client — compatible with Groq, Together, OpenAI
- Docker — containerized deployment
- Hugging Face Spaces — hosting