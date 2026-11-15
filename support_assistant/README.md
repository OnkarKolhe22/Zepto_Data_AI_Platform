
# Zepto Support Assistant

## 1. Project Overview

The Zepto Support Assistant is a Retrieval-Augmented Generation (RAG) application designed to answer customer questions using eight official Zepto policy documents.

The application uses local sentence-transformer embeddings, ChromaDB for vector storage, LangGraph for workflow orchestration, and FastAPI to expose a REST API.

## 2. Technology Stack

- Python 3.12
- Sentence Transformers (all-MiniLM-L6-v2)
- ChromaDB
- LangGraph
- FastAPI
- Pydantic
- Docker
- Groq (optional real LLM mode)

## 3. Project Structure

```text
support_assistant/
│
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── chroma_db/
├── create_docs.py
├── ingest.py
├── prompts.py
├── main.py
├── api.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 4. Architecture

The application follows a Retrieval-Augmented Generation (RAG) architecture.

### Step 1: Document Ingestion

Eight Zepto policy documents are stored in the docs folder.

### Step 2: Embedding Generation

The all-MiniLM-L6-v2 sentence-transformer model converts policy documents into numerical embeddings.

### Step 3: Vector Database

ChromaDB stores the document embeddings and metadata in a persistent local database.

### Step 4: Intent Classification

LangGraph classifies incoming questions into:

- policy_question
- general_question

### Step 5: Retrieval

For policy questions, the application converts the query into an embedding and retrieves the top three most relevant policy chunks from ChromaDB using cosine similarity.

### Step 6: Answer Generation

The retrieved context is passed to the answer-generation workflow.

A structured prompt defines the role, context, task, response format, length, negative constraints, and few-shot examples.

### Step 7: API Response

FastAPI exposes the assistant through the POST /ask endpoint.

Pydantic validates the response structure.

## 5. LangGraph Workflow

The application uses a StateGraph with a TypedDict state.

It contains three main nodes:

1. classify_intent
2. retrieve_and_answer
3. direct_answer

Conditional routing sends policy questions to retrieval and general questions to the direct-answer node.

## 6. Mock LLM Mode

The application supports a deterministic offline mode using:

MOCK_LLM=1

In mock mode:

- Intent classification uses keyword matching.
- Policy questions perform actual embedding and ChromaDB retrieval.
- The answer uses the first retrieved document chunk.
- Confidence is set to 1.0.
- General questions receive a fixed response.

The default mode is MOCK_LLM=1.

An optional real LLM mode can be enabled using MOCK_LLM=0 and a configured GROQ_API_KEY.

## 7. Running the Application Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run document ingestion:

```bash
python ingest.py
```

Start the API:

```bash
uvicorn api:app --reload
```

Open Swagger UI:

http://127.0.0.1:8000/docs

## 8. API Documentation

### Endpoint

POST /ask

### Request Format

```json
{
  "query": "How long does Zepto delivery take?"
}
```

### Response Format

```json
{
  "answer": "Answer based on retrieved context",
  "sources": ["doc_01", "doc_08", "doc_04"],
  "confidence": 1.0
}
```

## 9. Docker Deployment

Build the Docker image:

```bash
docker build -t zepto-support-assistant .
```

Run the Docker container:

```bash
docker run -d --name zepto-support -p 8000:8000 -e MOCK_LLM=1 zepto-support-assistant
```

Check running containers:

```bash
docker ps
```

Open the API documentation:

http://localhost:8000/docs

## 10. API Testing Results

### Test 1: Policy Question

Request:

```json
{
  "query": "How long does Zepto delivery take?"
}
```

Result:

- HTTP Status: 200 OK
- Retrieved sources: doc_01, doc_08, doc_04
- Confidence: 1.0
- Answer begins with: Based on the retrieved context.

### Test 2: General Question

Request:

```json
{
  "query": "Who invented Python?"
}
```

Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

Result: HTTP 200 OK.

Both API tests were successfully executed using the Docker container.

## 11. Conclusion

The Zepto Support Assistant demonstrates a complete RAG pipeline with document ingestion, local embedding generation, vector database retrieval, intent classification, LangGraph orchestration, structured response validation, and FastAPI deployment.

The application also supports deterministic offline testing through MOCK_LLM mode and containerized deployment using Docker.
