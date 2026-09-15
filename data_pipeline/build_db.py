import sqlite3
import pandas as pd

# Load the cleaned CSV from the scraping step
df = pd.read_csv("data_pipeline/books_raw.csv")

# Connect to (or create) the SQLite database file
conn = sqlite3.connect("data_pipeline/zepto_books.db")
cursor = conn.cursor()

# Create the categories table
cursor.execute("""
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE
)
""")

# Create the books table, with a foreign key to categories
cursor.execute("""
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
)
""")

# Insert unique categories
unique_categories = df["category"].unique()

for cat_name in unique_categories:
    cursor.execute(
        "INSERT OR IGNORE INTO categories (category_name) VALUES (?)",
        (cat_name,)
    )

conn.commit()
print(f"Inserted {len(unique_categories)} categories.")

# Build a lookup: category_name -> category_id
cursor.execute("SELECT category_id, category_name FROM categories")
category_lookup = {name: cat_id for cat_id, name in cursor.fetchall()}

# Insert books, using the lookup to attach the right category_id
for _, row in df.iterrows():
    cursor.execute(
        """
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            int(row["in_stock"]),
            category_lookup[row["category"]]
        )
    )

conn.commit()
print(f"Inserted {len(df)} books.")

conn.close()

