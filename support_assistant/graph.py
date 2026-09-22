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


class GraphState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list
    confidence: float


def classify_intent(state: GraphState) -> GraphState:
    query_lower = state["query"].lower()

    if MOCK_LLM == "1":
        if any(keyword in query_lower for keyword in POLICY_KEYWORDS):
            intent = "policy_question"
        else:
            intent = "general_question"
    else:
        intent = "policy_question"

    return {**state, "intent": intent}


def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)

    top_chunk_ids = results["ids"][0]
    top_chunk_texts = results["documents"][0]

    if MOCK_LLM == "1":
        top_chunk_snippet = top_chunk_texts[0][:200]
        answer_text = f"Based on the retrieved context: {top_chunk_snippet}"
        confidence_value = 1.0
    else:
        answer_text = "Real-LLM path not implemented in this baseline."
        confidence_value = 0.0

    validated = SupportResponse(
        answer=answer_text,
        sources=list(top_chunk_ids),
        confidence=confidence_value
    )

    return {**state, **validated.model_dump()}


def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM == "1":
        answer_text = "I can only answer questions about Zepto policies right now."
        confidence_value = 1.0
    else:
        answer_text = "Real-LLM path not implemented in this baseline."
        confidence_value = 0.0

    validated = SupportResponse(
        answer=answer_text,
        sources=[],
        confidence=confidence_value
    )

    return {**state, **validated.model_dump()}


def route_after_classification(state: GraphState) -> str:
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


if __name__ == "__main__":
    test_queries = [
        "What is your delivery policy?",
        "What's the capital of France?"
    ]

    for q in test_queries:
        result = app_graph.invoke({"query": q, "intent": "", "answer": "", "sources": [], "confidence": 0.0})
        print(f"\nQuery: {q}")
        print(f"Intent: {result['intent']}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Confidence: {result['confidence']}")
