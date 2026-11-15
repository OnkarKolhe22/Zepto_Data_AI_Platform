
import os
from pathlib import Path
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from prompts import ZEPT0_PROMPT


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"

MOCK_LLM = os.getenv("MOCK_LLM", "1")

# Required keyword list from the assignment
POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


# --------------------------------------------------
# 2. Pydantic response schema
# --------------------------------------------------

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# --------------------------------------------------
# 3. LangGraph state
# --------------------------------------------------

class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


# --------------------------------------------------
# 4. Load embedding model and ChromaDB
# --------------------------------------------------

print("Loading Zepto support assistant resources...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(
    path=str(DB_DIR)
)

collection = chroma_client.get_collection(
    name="zepto_policies"
)

print("Support assistant resources loaded!")


# --------------------------------------------------
# 5. Optional real LLM helper
# --------------------------------------------------

def call_real_llm(prompt: str) -> str:
    """
    Optional Groq integration.
    Used only when MOCK_LLM=0.
    """

    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Set GROQ_API_KEY to use the optional real LLM."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


# --------------------------------------------------
# 6. Node: classify_intent
# --------------------------------------------------

def classify_intent(state: AssistantState):

    query = state["query"]
    lower_query = query.lower()

    if MOCK_LLM == "1":
        # Deterministic offline keyword classification
        if any(keyword in lower_query for keyword in POLICY_KEYWORDS):
            intent = "policy_question"
        else:
            intent = "general_question"

    else:
        # Optional real LLM classification
        prompt = f"""
Classify the customer question as exactly one of:
policy_question or general_question.

Customer question: {query}

Return only the classification.
"""

        intent = call_real_llm(prompt).strip().lower()

        if intent not in ["policy_question", "general_question"]:
            intent = "general_question"

    print("Detected intent:", intent)

    return {"intent": intent}


# --------------------------------------------------
# 7. Node: retrieve_and_answer
# --------------------------------------------------

def retrieve_and_answer(state: AssistantState):

    query = state["query"]

    # Embed the customer's question
    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    # Retrieve the three most similar policy documents
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    retrieved_documents = results["documents"][0]
    retrieved_ids = results["ids"][0]

    if not retrieved_documents:
        return {
            "answer": "I could not find a matching Zepto policy.",
            "sources": [],
            "confidence": 0.0
        }

    # Build context for the prompt
    context = "\n\n".join(retrieved_documents)

    if MOCK_LLM == "1":

        # Required deterministic mock answer
        top_chunk_snippet = retrieved_documents[0][:200]

        answer = (
            "Based on the retrieved context: "
            + top_chunk_snippet
        )

    else:

        # Optional real LLM answer with up to 2 retries
        prompt = ZEPT0_PROMPT.format(
            question=query,
            context=context
        )

        answer = None
        validation_error = None

        for attempt in range(3):
            try:
                current_prompt = prompt

                if attempt > 0:
                    current_prompt += (
                        "\nPrevious output was invalid. "
                        "Return a corrected, concise answer."
                    )

                raw_answer = call_real_llm(current_prompt)

                # Validate the generated answer
                parsed = AnswerResponse.model_validate_json(
                    raw_answer
                )

                answer = parsed.answer
                validation_error = None
                break

            except Exception as error:
                validation_error = error

        if validation_error is not None:
            raise ValueError(
                "Real LLM response failed validation "
                "after 3 attempts."
            ) from validation_error

    return {
        "answer": answer,
        "sources": retrieved_ids,
        "confidence": 1.0
    }


# --------------------------------------------------
# 8. Node: direct_answer
# --------------------------------------------------

def direct_answer(state: AssistantState):

    if MOCK_LLM == "1":

        answer = (
            "I can only answer questions about Zepto "
            "policies right now."
        )

    else:

        prompt = f"""
You are a helpful Zepto customer support assistant.

Answer this general customer question briefly:
{state["query"]}
"""

        answer = call_real_llm(prompt)

    return {
        "answer": answer,
        "sources": [],
        "confidence": 1.0
    }


# --------------------------------------------------
# 9. Conditional routing
# --------------------------------------------------

def route_question(state: AssistantState):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# --------------------------------------------------
# 10. Build LangGraph
# --------------------------------------------------

workflow = StateGraph(AssistantState)

workflow.add_node(
    "classify_intent",
    classify_intent
)

workflow.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

workflow.add_node(
    "direct_answer",
    direct_answer
)

workflow.add_edge(START, "classify_intent")

workflow.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

graph = workflow.compile()


# --------------------------------------------------
# 11. Run assistant and validate response
# --------------------------------------------------

def ask_zepto(query: str) -> AnswerResponse:

    result = graph.invoke({
        "query": query
    })

    response = AnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"]
    )

    return response


# --------------------------------------------------
# 12. Test the workflow
# --------------------------------------------------

if __name__ == "__main__":

    test_questions = [
        "How long does Zepto delivery take?",
        "Who invented Python?"
    ]

    for question in test_questions:

        print("\nCustomer question:", question)

        response = ask_zepto(question)

        print(response.model_dump_json(indent=2))
