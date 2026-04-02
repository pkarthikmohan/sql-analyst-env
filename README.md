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

This environment simulates the daily work of a **data analyst at an e-commerce company**.
The AI agent receives a natural language business question, explores the database schema,
and must produce a correct SQL query to answer it.

Unlike toy environments, this is a task that real analysts perform every day —
making it a meaningful benchmark for AI reasoning and code generation.

---

## Tasks

Three tasks of increasing difficulty, each graded by an automated SQL result comparator.

### Task 1 — Easy
**"How many completed orders were placed in 2024?"**
- Requires: COUNT, WHERE, date filtering
- Tests: basic aggregation and filtering
- Expected output: single row, single column

### Task 2 — Medium
**"Find the top 5 customers by total revenue from completed orders."**
- Requires: JOIN, GROUP BY, SUM, ORDER BY, LIMIT
- Tests: multi-table joins and aggregation
- Expected output: 5 rows with first_name, last_name, total_revenue

### Task 3 — Hard
**"Rank product categories by total revenue using a window function."**
- Requires: CTE (WITH), JOIN, GROUP BY, RANK() OVER (...)
- Tests: advanced SQL — CTEs and window functions
- Expected output: all categories with total_revenue and revenue_rank

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
  "info": {"message": "Task 1 loaded. Use POST /step with your SQL query."}
}
```

### POST /step
Submit a SQL query. Returns reward and feedback.

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

-- Hard
WITH rev AS (
  SELECT p.category, SUM(o.total_amount) AS total_revenue
  FROM orders o JOIN products p ON o.product_id = p.product_id
  WHERE o.status = 'completed' GROUP BY p.category
)
SELECT category, total_revenue, RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank
FROM rev ORDER BY revenue_rank ASC
```

---

## Reward Function

Partial credit score from 0.0 to 1.0 with three components:

| Component | Weight | How it is measured |
|---|---|---|
| Column names | 0.30 | Fraction of expected column names present |
| Row count | 0.30 | Ratio of returned rows vs expected rows |
| Cell values | 0.40 | Fraction of cells matching expected values |

Even an imperfect query receives meaningful feedback — not just pass/fail.
This gives the agent a gradient signal to improve from.

---

## Database Schema

A realistic e-commerce dataset with 600 orders, 100 customers, 30 products.

```
customers  (customer_id, first_name, last_name, email, city, signup_date)
products   (product_id, product_name, category, price, stock)
orders     (order_id, customer_id, product_id, quantity, total_amount, order_date, status)
```

All orders are dated in 2024. Status values: completed, pending, cancelled.

---

## Running the Baseline Agent

```bash
pip install openai requests

export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_api_key_here

python inference.py
```

Expected output:
```
Tasks solved   : 3 / 3
Average reward : 1.000 / 1.000
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

- CPU: 1-2 vCPU
- RAM: 8GB
- GPU: Not required
- Inference time: Under 5 minutes for all 3 tasks

---

## Project Structure

```
sql-analyst-env/
├── main.py          # FastAPI server
├── seed.py          # Database seeder
├── inference.py     # Baseline AI agent
├── openenv.yaml     # OpenEnv specification
├── Dockerfile       # Container for HF Spaces
├── requirements.txt # Python dependencies
└── data/
    └── ecommerce.db # SQLite database
```