"""Build the normalized SQLite database for Module 1 from the cleaned
scraped CSV (books_raw.csv), and load it with categories + books data.

Safe to re-run: existing rows are cleared before each load so the
script never produces duplicate rows.
"""
import sqlite3
import pandas as pd

CSV_PATH = "data_pipeline/books_raw.csv"
DB_PATH = "data_pipeline/zepto_books.db"


def create_tables(cursor):
    """Create the categories and books tables if they don't already exist."""
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        category_id INTEGER PRIMARY KEY,
        category_name TEXT UNIQUE
    )
    """)
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


def clear_tables(cursor):
    """Delete existing rows so this script can be re-run safely without
    producing duplicate books/categories."""
    cursor.execute("DELETE FROM books")
    cursor.execute("DELETE FROM categories")


def insert_categories(cursor, df):
    """Insert each unique category name once. Returns a name->id lookup."""
    unique_categories = df["category"].unique()
    for cat_name in unique_categories:
        cursor.execute(
            "INSERT OR IGNORE INTO categories (category_name) VALUES (?)",
            (cat_name,)
        )
    cursor.execute("SELECT category_id, category_name FROM categories")
    return {name: cat_id for cat_id, name in cursor.fetchall()}, len(unique_categories)


def insert_books(cursor, df, category_lookup):
    """Insert every book row, resolving its category name to a category_id."""
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


def main():
    df = pd.read_csv(CSV_PATH)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    create_tables(cursor)
    clear_tables(cursor)
    conn.commit()

    category_lookup, n_categories = insert_categories(cursor, df)
    conn.commit()
    print(f"Inserted {n_categories} categories.")

    insert_books(cursor, df, category_lookup)
    conn.commit()
    print(f"Inserted {len(df)} books.")

    conn.close()


if __name__ == "__main__":
    main()
