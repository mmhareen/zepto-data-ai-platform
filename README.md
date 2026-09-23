# Zepto Data & AI Platform

![Python](https://img.shields.io/badge/python-3.12%2F3.13-blue)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

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
| Analytics | [`/analytics`](./analytics/README.md) | 50 | Complete |
| Support Assistant | [`/support_assistant`](./support_assistant/README.md) | 25 | Complete |

## Setup

Each module uses the same virtual environment and a single consolidated
`requirements.txt` at the repo root.

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## How to run each module

### Data Pipeline (`/data_pipeline`)

    python data_pipeline\scrape.py
    python data_pipeline\build_db.py
    python data_pipeline\queries.py
    python data_pipeline\pandas_check.py

See [`data_pipeline/README.md`](./data_pipeline/README.md) for full design
decisions and query output.

### Analytics (`/analytics`)

    python analytics\01_load_and_profile.py
    python analytics\02_univariate.py
    python analytics\03_bivariate.py
    python analytics\04_data_story.py
    python analytics\05_standardization_check.py
    python analytics\06_modeling.py

See [`analytics/README.md`](./analytics/README.md) for the full EDA
write-up, model comparison tables, and final recommendation.

### Support Assistant (`/support_assistant`)

    python support_assistant\build_index.py
    uvicorn support_assistant.main:app --reload

Or via Docker:

    docker build -t zepto-support-assistant .
    docker run -p 7860:7860 zepto-support-assistant

See [`support_assistant/README.md`](./support_assistant/README.md) for the
full RAG architecture write-up and example API calls.

## Design decision summary

- **Data Pipeline:** scraped 69 books across 3 categories from
  books.toscrape.com, cleaned into typed columns, converted GBP to INR
  using the required fixed rate (1 GBP = 105.50 INR), and loaded into a
  normalized two-table SQLite schema. Full details in the module README.
- **Analytics:** cleaned the Titanic dataset per a missing-value threshold
  rule, ran full univariate/bivariate/multivariate EDA with a 4-chart data
  story, then built a leak-safe classification pipeline (Logistic
  Regression, Decision Tree, Random Forest) plus a fare-prediction
  regression side-task. Random Forest was selected as the recommended
  deployment model based on accuracy/F1/recall. Full details, all metrics,
  and the final recommendation are in the module README.
- **Support Assistant:** built a RAG pipeline over 8 Zepto policy
  documents, embedded locally with sentence-transformers and stored in
  ChromaDB, orchestrated with a 3-node LangGraph (intent classification,
  retrieval+answer, direct answer), served via FastAPI, and containerized
  with Docker. Graded entirely through a deterministic MOCK_LLM=1 offline
  mode — no API key or network LLM call required. Full architecture
  write-up in the module README.