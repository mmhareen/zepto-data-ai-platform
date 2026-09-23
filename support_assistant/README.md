# Support Assistant — Module 3

A grounded RAG (Retrieval-Augmented Generation) support assistant for Zepto policy questions, orchestrated with LangGraph, served via FastAPI, and containerized with Docker. Graded entirely through a deterministic offline mock mode (MOCK_LLM) — no API key or network LLM call required.

## Setup

From the repo root:

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## How to run

**1. Build the vector index (one-time, or whenever docs change):**

    python support_assistant\build_index.py

**2. Run the API locally:**

    uvicorn support_assistant.main:app --reload

Then POST to http://127.0.0.1:8000/ask with {"query": "..."}.

**3. Run via Docker instead:**

    docker build -t zepto-support-assistant .
    docker run -p 7860:7860 zepto-support-assistant

Then POST to http://127.0.0.1:7860/ask.

MOCK_LLM defaults to mock mode (1) if left unset — this is what is graded. Setting MOCK_LLM=0 would route generation through a real LLM (optional, ungraded extension, not implemented in this baseline).

## Example calls (MOCK_LLM at default)

**Policy question (triggers retrieval):**
Request: {"query": "What is your delivery policy?"}

    {
      "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
      "sources": ["doc_01", "doc_02", "doc_05"],
      "confidence": 1.0
    }

**General question (no retrieval):**
Request: {"query": "What is the capital of France?"}

    {
      "answer": "I can only answer questions about Zepto policies right now.",
      "sources": [],
      "confidence": 1.0
    }

## Architecture: the RAG pipeline

Diagram of the request flow:

    Client
      |
      |  POST /ask {"query": "..."}
      v
    FastAPI (main.py) -- validates AskRequest
      |
      v
    LangGraph (graph.py)
      |
      [classify_intent] -- keyword match?
        |                  |
      policy            general
        v                  v
      [retrieve_and_answer]   [direct_answer]
        |                          |
        v                          |
      ChromaDB (embeddings          |
      from 8 policy docs)           |
        |                           |
        v                           v
      answer / sources / confidence
      validated via SupportResponse (schemas.py)
        |
        v
      JSON response to client

**Ingestion:** 8 plain-text policy documents live in support_assistant/docs/ (doc_01.txt through doc_08.txt), one Zepto policy per file (delivery, returns, membership, tracking, cancellation, damaged items, gift cards, support hours). Each document is used whole as a single chunk, given its short length. build_index.py reads all 8 files and pairs each with a human-readable title via a doc_titles dictionary.

**Embedding:** build_index.py uses sentence-transformers (all-MiniLM-L6-v2) to convert each document's text into a 384-dimensional vector locally (no API call). These embeddings, along with the raw text and title metadata, are stored in a persistent ChromaDB collection named zepto_policies, saved to disk at support_assistant/chroma_db/.

**Retrieval:** happens inside the retrieve_and_answer node in graph.py. The incoming query is embedded with the same model, then collection.query(...) finds the top-3 most similar stored chunks via cosine similarity. This step runs identically regardless of MOCK_LLM, since embedding and ChromaDB need no API key or network access.

**Generation:** also happens inside retrieve_and_answer (for policy questions) or direct_answer (for general questions), both in graph.py. This is the one stage that branches on MOCK_LLM:
- Default (mock, MOCK_LLM=1 or unset) — graded baseline: retrieve_and_answer returns a canned string built from the top retrieved chunk (f"Based on the retrieved context: {snippet}"); direct_answer returns a fixed canned string. No LLM call is made in either case; sources/confidence are populated deterministically by code.
- Optional MOCK_LLM=0 extension (not implemented in this baseline): retrieve_and_answer would instead prompt a real LLM using the structured template in prompt_template.py, grounded only in the retrieved chunks; direct_answer would prompt the LLM directly with no retrieval.

**Routing:** classify_intent (also mock-mode keyword-based, matching against 8 fixed keywords) decides which of the two generation nodes a query is sent to, via a LangGraph conditional edge. This routing decision itself does not depend on MOCK_LLM — only what each destination node does once it gets there.

**Data flow, end to end:** a query arrives at the FastAPI /ask endpoint (main.py) → validated into an AskRequest → passed into the compiled LangGraph (graph.py) as the initial state → classify_intent sets intent → the conditional edge routes to retrieve_and_answer (which also queries ChromaDB) or direct_answer → either node populates answer/sources/confidence, validated via the SupportResponse Pydantic schema (schemas.py) → the final state is returned to FastAPI, which sends it back as the HTTP response.

## Files
- docs/doc_01.txt … doc_08.txt — policy corpus
- build_index.py — embeds and stores the corpus in ChromaDB
- test_retrieval.py — manual retrieval sanity check
- prompt_template.py — structured prompt (used by the optional real-LLM path)
- schemas.py — Pydantic SupportResponse output schema
- graph.py — LangGraph StateGraph: 3 nodes, conditional routing
- main.py — FastAPI app, POST /ask endpoint
- chroma_db/ — persisted vector store