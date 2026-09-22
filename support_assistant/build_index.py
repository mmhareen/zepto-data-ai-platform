import os
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_FOLDER = "support_assistant/docs"

# Map each doc ID to a human-readable topic name
doc_titles = {
    "doc_01": "Delivery Policy",
    "doc_02": "Returns & Refunds",
    "doc_03": "Membership Tiers",
    "doc_04": "Order Tracking",
    "doc_05": "Order Cancellation Policy",
    "doc_06": "Damaged or Missing Items",
    "doc_07": "Gift Cards",
    "doc_08": "Customer Support Hours"
}

documents = []
doc_ids = []
metadatas = []

for filename in sorted(os.listdir(DOCS_FOLDER)):
    if filename.endswith(".txt"):
        filepath = os.path.join(DOCS_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        doc_id = filename.replace(".txt", "")
        documents.append(text)
        doc_ids.append(doc_id)
        metadatas.append({"title": doc_titles[doc_id]})

print(f"Loaded {len(documents)} documents: {doc_ids}")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents).tolist()

print(f"Generated {len(embeddings)} embeddings, each of length {len(embeddings[0])}")

client = chromadb.PersistentClient(path="support_assistant/chroma_db")
collection = client.get_or_create_collection(name="zepto_policies")

collection.add(
    ids=doc_ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print(f"Stored {collection.count()} documents in ChromaDB collection 'zepto_policies'")