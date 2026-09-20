import sqlite3

conn = sqlite3.connect("data_pipeline/zepto_books.db")
cursor = conn.cursor()

output_lines = []

def run_query(label, sql):
    output_lines.append(f"\n--- {label} ---")
    output_lines.append(sql)
    cursor.execute(sql)
    rows = cursor.fetchall()
    for row in rows:
        output_lines.append(str(row))
    return rows

# 1. SELECT + WHERE
run_query(
    "Books rated 5 stars",
    "SELECT title, rating FROM books WHERE rating = 5"
)

# 2. ORDER BY + LIMIT
run_query(
    "Top 5 most expensive books (INR)",
    "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 5"
)

# 3. DISTINCT
run_query(
    "Distinct rating values present",
    "SELECT DISTINCT rating FROM books"
)

# 4. BETWEEN
run_query(
    "Books priced between £20 and £30",
    "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 20 AND 30"
)

# 5. JOIN across both tables
run_query(
    "Books per category, ordered by rating (via JOIN)",
    """
    SELECT categories.category_name, books.title, books.rating
    FROM books
    JOIN categories ON books.category_id = categories.category_id
    ORDER BY categories.category_name, books.rating DESC
    """
)

conn.close()

# Print to terminal AND save to a file
full_output = "\n".join(output_lines)
print(full_output)

with open("data_pipeline/query_results.txt", "w", encoding="utf-8") as f:
    f.write(full_output)

print("\nSaved to data_pipeline/query_results.txt")