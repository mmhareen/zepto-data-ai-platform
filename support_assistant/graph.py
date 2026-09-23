"""LangGraph pipeline for the Zepto support assistant.

Defines a 3-node graph: classify_intent -> (retrieve_and_answer | direct_answer).
Every node's generation step branches on the MOCK_LLM environment variable.
MOCK_LLM unset or "1" (the default, graded baseline) uses deterministic,
rule-based logic with no LLM call. MOCK_LLM=0 is an optional, ungraded
extension not implemented in this baseline.
"""
import os
from typing import TypedDict
import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from .schemas import SupportResponse

MOCK_LLM = os.environ.get("MOCK_LLM", "1")

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="support_assistant/chroma_db")
collection = client.get_or_create_collection(name="zepto_policies")

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership",
    "tracking", "cancel", "gift card", "support hours"
]

TOP_K_RESULTS = 3
SNIPPET_LENGTH = 200


class GraphState(TypedDict):
    """Shared state passed between every node in the graph."""
    query: str
    intent: str
    answer: str
    sources: list
    confidence: float


def classify_intent(state: GraphState) -> GraphState:
    """Classify the incoming query as policy_question or general_question.

    Mock mode (default): keyword heuristic against POLICY_KEYWORDS, no LLM call.
    """
    query_lower = state["query"].lower()

    if MOCK_LLM == "1":
        if any(keyword in query_lower for keyword in POLICY_KEYWORDS):
            intent = "policy_question"
        else:
            intent = "general_question"
    else:
        raise NotImplementedError(
            "MOCK_LLM=0 (real-LLM classification) is an optional extension "
            "not implemented in this baseline."
        )

    return {**state, "intent": intent}


def retrieve_and_answer(state: GraphState) -> GraphState:
    """Retrieve the top matching policy chunks and answer from them.

    Retrieval (embedding + ChromaDB query) always runs for real, in both
    modes. Only the final answer text branches on MOCK_LLM.
    """
    query = state["query"]

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=TOP_K_RESULTS)

    top_chunk_ids = results["ids"][0]
    top_chunk_texts = results["documents"][0]

    if MOCK_LLM == "1":
        top_chunk_snippet = top_chunk_texts[0][:SNIPPET_LENGTH]
        answer_text = f"Based on the retrieved context: {top_chunk_snippet}"
        confidence_value = 1.0
    else:
        raise NotImplementedError(
            "MOCK_LLM=0 (real-LLM generation) is an optional extension "
            "not implemented in this baseline."
        )

    validated = SupportResponse(
        answer=answer_text,
        sources=list(top_chunk_ids),
        confidence=confidence_value
    )

    return {**state, **validated.model_dump()}


def direct_answer(state: GraphState) -> GraphState:
    """Answer general (non-policy) questions with no retrieval.

    Mock mode (default): fixed canned string, no LLM call.
    """
    if MOCK_LLM == "1":
        answer_text = "I can only answer questions about Zepto policies right now."
        confidence_value = 1.0
    else:
        raise NotImplementedError(
            "MOCK_LLM=0 (real-LLM generation) is an optional extension "
            "not implemented in this baseline."
        )

    validated = SupportResponse(
        answer=answer_text,
        sources=[],
        confidence=confidence_value
    )

    return {**state, **validated.model_dump()}


def route_after_classification(state: GraphState) -> str:
    """Conditional edge: route to retrieval or direct answer based on intent."""
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    else:
        return "direct_answer"


graph_builder = StateGraph(GraphState)

graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("retrieve_and_answer", retrieve_and_answer)
graph_builder.add_node("direct_answer", direct_answer)

graph_builder.set_entry_point("classify_intent")

graph_builder.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

graph_builder.add_edge("retrieve_and_answer", END)
graph_builder.add_edge("direct_answer", END)

app_graph = graph_builder.compile()
