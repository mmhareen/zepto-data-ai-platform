# Data Pipeline — Module 1

Scrapes book listings from books.toscrape.com, cleans and enriches the data,
loads it into a normalized SQLite database, and queries it with both SQL and pandas.

## Setup

From the repo root:
\`\`\`
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

## How to run

Run these in order from the repo root (with the venv active):

\`\`\`
python data_pipeline\scrape.py       # scrapes books.toscrape.com -> books_raw.csv
python data_pipeline\build_db.py     # builds zepto_books.db from books_raw.csv
python data_pipeline\queries.py      # runs 5 SQL queries -> query_results.txt
python data_pipeline\pandas_check.py # compares pd.read_sql vs pd.merge
\`\`\`

## Design decisions

- **Scope:** scraped the first 3 categories listed on the site's sidebar
  (Travel, Mystery, Historical Fiction), following pagination within each
  category, for a total of 69 books — comfortably over the required 60.
- **Currency conversion:** price_inr is computed using the required fixed
  baseline rate of 1 GBP = 105.50 INR, as specified by the assignment. This
  is a fixed project-defined constant, not a live market rate, so it requires
  no API call or date reference.
- **Rating conversion:** the site encodes star ratings as a CSS class name
  (e.g. `star-rating Two`) rather than as visible text or a number, so a
  dictionary lookup (`{"One": 1, ..., "Five": 5}`) was used to convert it
  to an integer.
- **Missing/malformed data:** no rows failed to parse — every book on the
  scraped pages had a complete title, price, rating, and availability
  field, so no imputation or row-dropping was necessary.
- **Schema:** two tables, `categories` (category_id PK, category_name) and
  `books` (book_id PK, ...various fields..., category_id FK referencing
  categories). This normalizes category names instead of repeating them on
  every book row.
- **Verification:** the JOIN query's result was reproduced independently
  using `pd.merge` on DataFrames pulled from each table separately, and
  confirmed identical to the SQL JOIN's result via `df.equals()`.

## Files

- `scrape.py` — scraping logic (requests + BeautifulSoup)
- `build_db.py` — creates the SQLite schema and loads cleaned data into it
- `queries.py` — 5 required SQL queries, output saved to `query_results.txt`
- `pandas_check.py` — pd.read_sql vs pd.merge equivalence check
- `books_raw.csv` — cleaned scraped data (intermediate output)
- `zepto_books.db` — the SQLite database
- `query_results.txt` — saved output of the 5 SQL queries