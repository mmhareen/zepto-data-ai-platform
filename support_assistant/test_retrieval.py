import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="support_assistant/chroma_db")
collection = client.get_or_create_collection(name="zepto_policies")

test_query = "How long do I have to return a damaged item?"
query_embedding = model.encode([test_query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

print(f"Query: {test_query}\n")
for i in range(len(results["ids"][0])):
    doc_id = results["ids"][0][i]
    title = results["metadatas"][0][i]["title"]
    distance = results["distances"][0][i]
    snippet = results["documents"][0][i][:100]
    print(f"{i+1}. {doc_id} ({title}) — distance: {distance:.4f}")
    print(f"   {snippet}...\n")