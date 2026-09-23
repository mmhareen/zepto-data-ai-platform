"""FastAPI wrapper exposing the LangGraph support assistant via POST /ask."""
from fastapi import FastAPI
from pydantic import BaseModel
from .graph import app_graph
from .schemas import SupportResponse

app = FastAPI(title="Zepto Support Assistant")


class AskRequest(BaseModel):
    query: str


@app.post("/ask", response_model=SupportResponse)
def ask(request: AskRequest) -> SupportResponse:
    """Run a query through the graph and return a validated response.

    Falls back to a safe error response instead of a raw 500 if anything
    in the graph raises (e.g. an unimplemented MOCK_LLM=0 path).
    """
    initial_state = {
        "query": request.query,
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0
    }
    try:
        result = app_graph.invoke(initial_state)
    except Exception as exc:
        return SupportResponse(
            answer=f"Sorry, something went wrong processing your request: {exc}",
            sources=[],
            confidence=0.0
        )

    return SupportResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )
