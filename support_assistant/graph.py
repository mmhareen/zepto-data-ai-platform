"""LangGraph pipeline for the Zepto support assistant.

Defines a 3-node graph: classify_intent -> (retrieve_and_answer | direct_answer).
Every node's generation step branches on the MOCK_LLM environment variable.
MOCK_LLM unset or "1" (the default, graded baseline) uses deterministic,
rule-based logic with no LLM call. MOCK_LLM=0 is an optional, ungraded
extension: a real LLM call is not implemented in this baseline, but the
retry-with-corrective-instruction scaffolding required by the assignment
is present below so the structure exists even though it can never be
exercised while MOCK_LLM defaults to "1".
"""
import os
from typing import TypedDict
import chromadb
from pydantic import ValidationError
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
MAX_LLM_RETRIES = 2


class GraphState(TypedDict):
    """Shared state passed between every node in the graph."""
    query: str
    intent: str
    answer: str
    sources: list
    confidence: float


def _call_real_llm(prompt: str) -> str:
    """Placeholder for the optional MOCK_LLM=0 extension's actual LLM call.

    Not implemented in this baseline -- no real LLM is wired in.
    """
    raise NotImplementedError(
        "MOCK_LLM=0 (real-LLM call) is an optional extension not "
        "implemented in this baseline."
    )


def _call_llm_and_validate(prompt: str, sources: list) -> SupportResponse:
    """Optional MOCK_LLM=0 extension: call a real LLM and validate its
    raw output against the SupportResponse schema, retrying up to
    MAX_LLM_RETRIES times with a corrective instruction appended to the
    prompt if validation fails, before giving up and returning a clearly
    marked error response.

    This scaffolding is required by the assignment to be present in code
    even though _call_real_llm always raises NotImplementedError in this
    baseline, so the retry loop itself never actually gets to run while
    MOCK_LLM defaults to "1".
    """
    last_error = None
    current_prompt = prompt

    for attempt in range(MAX_LLM_RETRIES + 1):
        raw_output = _call_real_llm(current_prompt)
        try:
            return SupportResponse.model_validate_json(raw_output)
        except (ValidationError, ValueError) as exc:
            last_error = exc
            current_prompt = (
                f"{current_prompt}\n\nYour previous response was invalid "
                f"({exc}). Respond again using exactly the required JSON "
                f"schema: {{\"answer\": str, \"sources\": [str], "
                f"\"confidence\": float}}."
            )

    return SupportResponse(
        answer=f"Error: LLM failed to produce a valid response after "
               f"{MAX_LLM_RETRIES} retries ({last_error}).",
        sources=sources,
        confidence=0.0
    )


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
        # Optional MOCK_LLM=0 extension: would call the LLM to classify.
        # Not implemented in this baseline.
        intent = _call_real_llm(state["query"])

    return {**state, "intent": intent}


def retrieve_and_answer(state: GraphState) -> GraphState:
    """Retrieve the top matching policy chunks and answer from them.

    Retrieval (embedding + ChromaDB query) always runs for real, in both
    modes. Only the final answer text branches on MOCK_LLM.
    """
    query = state["query"]

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=TOP_K_RESULTS)

    top_chunk_ids = list(results["ids"][0])
    top_chunk_texts = results["documents"][0]

    if MOCK_LLM == "1":
        top_chunk_snippet = top_chunk_texts[0][:SNIPPET_LENGTH]
        answer_text = f"Based on the retrieved context: {top_chunk_snippet}"
        validated = SupportResponse(
            answer=answer_text,
            sources=top_chunk_ids,
            confidence=1.0
        )
    else:
        # Optional MOCK_LLM=0 extension: would prompt a real LLM using the
        # structured template in prompt_template.py, grounded in
        # top_chunk_texts, and validate/retry via _call_llm_and_validate.
        prompt = f"Context: {top_chunk_texts}\nQuestion: {query}"
        validated = _call_llm_and_validate(prompt, sources=top_chunk_ids)

    return {**state, **validated.model_dump()}


def direct_answer(state: GraphState) -> GraphState:
    """Answer general (non-policy) questions with no retrieval.

    Mock mode (default): fixed canned string, no LLM call.
    """
    if MOCK_LLM == "1":
        validated = SupportResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0
        )
    else:
        # Optional MOCK_LLM=0 extension: would prompt the LLM directly,
        # no retrieval, validated/retried via _call_llm_and_validate.
        validated = _call_llm_and_validate(state["query"], sources=[])

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
