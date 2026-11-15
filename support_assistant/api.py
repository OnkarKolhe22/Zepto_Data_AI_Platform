
from fastapi import FastAPI
from pydantic import BaseModel

from main import ask_zepto, AnswerResponse

app = FastAPI(
    title="Zepto Support Assistant",
    description="A policy-based customer support assistant using RAG.",
    version="1.0.0"
)


class AskRequest(BaseModel):
    query: str


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest):
    return ask_zepto(request.query)
