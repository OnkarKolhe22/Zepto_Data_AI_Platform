
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Project folders
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "chroma_db"

# Load the local embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded!")

# Connect to persistent ChromaDB
client = chromadb.PersistentClient(path=str(DB_DIR))

# Rebuild the collection to avoid duplicate records
try:
    client.delete_collection("zepto_policies")
except Exception:
    pass

collection = client.create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)

# Read and embed all policy documents
documents = []
document_ids = []
metadatas = []

for file_path in sorted(DOCS_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")

    documents.append(text)
    document_ids.append(file_path.stem)
    metadatas.append({"source": file_path.name})

print(f"Found {len(documents)} policy documents.")

# Generate embeddings
embeddings = model.encode(documents).tolist()

# Store documents and vectors in ChromaDB
collection.add(
    ids=document_ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("All policy documents embedded and stored!")
print("Documents in database:", collection.count())

# Test retrieval
question = "How long does Zepto delivery take?"
query_embedding = model.encode([question]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

print("\nSearch question:", question)
print("\nTop matching documents:")

for doc_id, document, distance in zip(
    results["ids"][0],
    results["documents"][0],
    results["distances"][0]
):
    print("\nDocument ID:", doc_id)
    print("Cosine distance:", round(distance, 4))
    print("Content:", document[:200], "...")
