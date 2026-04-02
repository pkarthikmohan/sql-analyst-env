import requests
import json

base = "https://p-karthik-mohan-sql-analyst-env.hf.space"

sql = "SELECT c.customer_id, c.first_name, c.last_name, SUM(CASE WHEN STRFTIME('%m', o.order_date) BETWEEN '01' AND '06' THEN o.total_amount ELSE 0 END) AS h1_revenue, SUM(CASE WHEN STRFTIME('%m', o.order_date) BETWEEN '07' AND '12' THEN o.total_amount ELSE 0 END) AS h2_revenue FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE o.status = 'completed' AND o.order_date LIKE '2024%' GROUP BY c.customer_id HAVING h2_revenue > h1_revenue ORDER BY h2_revenue DESC"

requests.post(f"{base}/reset", json={"task_id": 8})
r = requests.post(f"{base}/step", json={"action": sql})
data = r.json()

print("Reward:", data["reward"])
print("Breakdown:", json.dumps(data["observation"]["reward_breakdown"], indent=2))
print("First 2 rows agent got:")
for row in data["observation"]["result_preview"][:2]:
    print(" ", row)
