import sqlite3
import random
from datetime import datetime, timedelta
import os

DB_PATH = os.path.join("data", "ecommerce.db")

FIRST_NAMES = ["Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry",
               "Irene", "Jack", "Karen", "Liam", "Mona", "Nate", "Olivia", "Paul",
               "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
               "Yara", "Zoe", "Aaron", "Bella", "Chris", "Diana"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
              "Davis", "Wilson", "Taylor", "Anderson", "Thomas", "Jackson", "White",
              "Harris", "Martin", "Thompson", "Robinson", "Clark", "Lewis"]

CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
          "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
          "Mumbai", "Delhi", "Chennai", "Bangalore", "Hyderabad", "Kolkata"]

CATEGORIES = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports",
              "Beauty", "Toys", "Food & Grocery", "Automotive", "Music"]

PRODUCTS = [
    ("Wireless Headphones", "Electronics", 79.99),
    ("Bluetooth Speaker", "Electronics", 49.99),
    ("USB-C Hub", "Electronics", 34.99),
    ("Mechanical Keyboard", "Electronics", 129.99),
    ("Gaming Mouse", "Electronics", 59.99),
    ("Webcam HD", "Electronics", 89.99),
    ("Monitor 24inch", "Electronics", 249.99),
    ("Phone Stand", "Electronics", 14.99),
    ("Running Shoes", "Clothing", 89.99),
    ("Denim Jacket", "Clothing", 64.99),
    ("Cotton T-Shirt", "Clothing", 19.99),
    ("Yoga Pants", "Clothing", 44.99),
    ("Winter Coat", "Clothing", 149.99),
    ("Baseball Cap", "Clothing", 24.99),
    ("Python Programming", "Books", 39.99),
    ("Data Science Handbook", "Books", 49.99),
    ("Machine Learning Guide", "Books", 54.99),
    ("Cook Book Deluxe", "Books", 29.99),
    ("History of AI", "Books", 34.99),
    ("Garden Hose 50ft", "Home & Garden", 44.99),
    ("Plant Pots Set", "Home & Garden", 29.99),
    ("LED Desk Lamp", "Home & Garden", 39.99),
    ("Yoga Mat", "Sports", 34.99),
    ("Resistance Bands", "Sports", 19.99),
    ("Dumbbell Set 20kg", "Sports", 79.99),
    ("Jump Rope", "Sports", 12.99),
    ("Face Moisturizer", "Beauty", 24.99),
    ("Shampoo Pro", "Beauty", 14.99),
    ("Perfume Set", "Beauty", 59.99),
    ("Building Blocks", "Toys", 34.99),
]

STATUSES = ["completed", "completed", "completed", "pending", "cancelled"]


def create_tables(conn):
    conn.executescript("""
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE customers (
            customer_id   INTEGER PRIMARY KEY,
            first_name    TEXT NOT NULL,
            last_name     TEXT NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            city          TEXT NOT NULL,
            signup_date   TEXT NOT NULL
        );

        CREATE TABLE products (
            product_id    INTEGER PRIMARY KEY,
            product_name  TEXT NOT NULL,
            category      TEXT NOT NULL,
            price         REAL NOT NULL,
            stock         INTEGER NOT NULL
        );

        CREATE TABLE orders (
            order_id      INTEGER PRIMARY KEY,
            customer_id   INTEGER NOT NULL,
            product_id    INTEGER NOT NULL,
            quantity      INTEGER NOT NULL,
            total_amount  REAL NOT NULL,
            order_date    TEXT NOT NULL,
            status        TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id)  REFERENCES products(product_id)
        );
    """)
    print("Tables created.")


def seed_customers(conn, n=100):
    used_emails = set()
    rows = []
    for i in range(1, n + 1):
        fn = random.choice(FIRST_NAMES)
        ln = random.choice(LAST_NAMES)
        base_email = f"{fn.lower()}.{ln.lower()}{i}@example.com"
        while base_email in used_emails:
            base_email = f"{fn.lower()}.{ln.lower()}{i}_{random.randint(10,99)}@example.com"
        used_emails.add(base_email)
        city = random.choice(CITIES)
        days_ago = random.randint(30, 730)
        signup = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        rows.append((i, fn, ln, base_email, city, signup))
    conn.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?,?)", rows
    )
    print(f"Inserted {n} customers.")


def seed_products(conn):
    rows = []
    for i, (name, cat, price) in enumerate(PRODUCTS, start=1):
        stock = random.randint(0, 200)
        rows.append((i, name, cat, price, stock))
    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?)", rows
    )
    print(f"Inserted {len(PRODUCTS)} products.")


def seed_orders(conn, n=600):
    rows = []
    base_date = datetime(2024, 1, 1)
    for i in range(1, n + 1):
        cust_id = random.randint(1, 100)
        prod_id = random.randint(1, len(PRODUCTS))
        qty = random.randint(1, 5)
        price = PRODUCTS[prod_id - 1][2]
        total = round(price * qty, 2)
        days_offset = random.randint(0, 364)
        order_date = (base_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
        status = random.choice(STATUSES)
        rows.append((i, cust_id, prod_id, qty, total, order_date, status))
    conn.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?,?,?)", rows
    )
    print(f"Inserted {n} orders.")


def main():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    seed_customers(conn)
    seed_products(conn)
    seed_orders(conn)
    conn.commit()

    # Quick sanity check
    cur = conn.cursor()
    print("\n--- Sanity Check ---")
    for table in ["customers", "products", "orders"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        print(f"  {table}: {cur.fetchone()[0]} rows")

    # Preview a joined query
    cur.execute("""
        SELECT c.first_name, c.last_name, p.product_name, o.total_amount, o.order_date
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN products  p ON o.product_id  = p.product_id
        LIMIT 5
    """)
    print("\n--- Sample joined rows ---")
    for row in cur.fetchall():
        print(" ", row)

    conn.close()
    print(f"\nDatabase saved to: {DB_PATH}")


if __name__ == "__main__":
    main()