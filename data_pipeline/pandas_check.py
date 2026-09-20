import sqlite3
import pandas as pd

conn = sqlite3.connect("data_pipeline/zepto_books.db")

# --- Approach 1: pd.read_sql on two separate queries, then compare to a JOIN done in SQL ---
sql_join_result = pd.read_sql(
    """
    SELECT categories.category_name, books.title, books.rating
    FROM books
    JOIN categories ON books.category_id = categories.category_id
    ORDER BY books.title
    """,
    conn
)

# --- Approach 2: pull each table separately, then join them with pandas itself ---
categories_df = pd.read_sql("SELECT * FROM categories", conn)
books_df = pd.read_sql("SELECT * FROM books", conn)

pandas_merge_result = pd.merge(
    books_df, categories_df, on="category_id"
)[["category_name", "title", "rating"]].sort_values("title").reset_index(drop=True)

sql_join_result = sql_join_result.sort_values("title").reset_index(drop=True)

print("SQL JOIN result (first 5):")
print(sql_join_result.head())

print("\npandas merge result (first 5):")
print(pandas_merge_result.head())

print("\nAre they equivalent?", sql_join_result.equals(pandas_merge_result))

conn.close()
