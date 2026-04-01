---
title: SQL Analyst OpenEnv
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# SQL Analyst OpenEnv

A real-world OpenEnv environment where an AI agent writes SQL queries against an e-commerce database to answer business questions.

## Tasks

| ID | Difficulty | Description |
|----|-----------|-------------|
| 1  | Easy      | Count completed orders in 2024 |
| 2  | Medium    | Top 5 customers by revenue (JOIN + GROUP BY) |
| 3  | Hard      | Category revenue ranking (CTE + window function) |

## API Endpoints

- `POST /reset` — load a task
- `POST /step` — submit a SQL query, get reward 0.0–1.0
- `GET /state` — current session state
- `GET /docs` — interactive API documentation

## Quick Start

```bash
# Reset to task 1
curl -X POST https://p-karthik-mohan-sql-analyst-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1}'

# Submit a SQL query
curl -X POST https://p-karthik-mohan-sql-analyst-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": "SELECT COUNT(*) AS total_orders FROM orders WHERE status = 'completed'"}'
```

## Reward Function

Partial credit scoring — 0.0 to 1.0:

- **0.30** — correct column names
- **0.30** — correct row count
- **0.40** — correct cell values

## Running inference.py

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_api_key_here
python inference.py
```

## Database Schema

```
customers  (customer_id, first_name, last_name, email, city, signup_date)  — 100 rows
products   (product_id, product_name, category, price, stock)              — 30 rows
orders     (order_id, customer_id, product_id, quantity, total_amount, order_date, status) — 600 rows
```

## Hardware Requirements

- 2 vCPU, 8GB RAM minimum
- No GPU required
- SQLite — no external database needed