# Zepto Data & AI Platform

End-to-end data pipeline, predictive analytics, and a local LangGraph RAG
support assistant, built as a capstone project for Zepto's analytics guild.

This repository contains three independently-graded modules that together
tell one story: a data-engineering pipeline feeds clean structured data,
an analytics pipeline profiles and predicts on customer/passenger-style
data end to end, and a support assistant puts a grounded GenAI service in
front of company policy documents.

## Modules

| Module | Path | Marks | Status |
|---|---|---|---|
| Data Pipeline | [`/data_pipeline`](./data_pipeline/README.md) | 25 | Complete |
| Analytics | [`/analytics`](./analytics/README.md) | 50 | In progress |
| Support Assistant | [`/support_assistant`](./support_assistant/README.md) | 25 | Not started |

## Setup

Each module uses the same virtual environment and a single consolidated
`requirements.txt` at the repo root.

\`\`\`
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

## How to run each module

### Data Pipeline (`/data_pipeline`)
\`\`\`
python data_pipeline\scrape.py
python data_pipeline\build_db.py
python data_pipeline\queries.py
python data_pipeline\pandas_check.py
\`\`\`
See [`data_pipeline/README.md`](./data_pipeline/README.md) for full design
decisions and query output.

### Analytics (`/analytics`)
_To be added._

### Support Assistant (`/support_assistant`)
_To be added._

## Design decision summary

- **Data Pipeline:** scraped 69 books across 3 categories from
  books.toscrape.com, cleaned into typed columns, converted GBP to INR
  using the required fixed rate (1 GBP = 105.50 INR), and loaded into a
  normalized two-table SQLite schema. Full details in the module README.
- **Analytics:** _to be added as this module is built._
- **Support Assistant:** _to be added as this module is built._