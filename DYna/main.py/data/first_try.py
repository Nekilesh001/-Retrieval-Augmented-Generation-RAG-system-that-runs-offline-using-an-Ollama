import os
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

DATA_DIR = "data/"
CHUNK_SIZE = 300

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load or create FAISS index
index_path = "vector_store.faiss"
chunks = []

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    with open("chunks.txt", "r", encoding="utf-8") as f:
        chunks = f.read().split("\n\n")
else:
    index = None

# Load new documents
def load_documents():
    texts = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        elif file.endswith(".pdf"):
            reader = PdfReader(path)
            content = "\n".join([page.extract_text() or "" for page in reader.pages])
            texts.append(content)
    return texts

# Split into chunks
def split_into_chunks(texts):
    new_chunks = []
    for text in texts:
        for i in range(0, len(text), CHUNK_SIZE):
            new_chunks.append(text[i:i+CHUNK_SIZE])
    return new_chunks

# Add new chunks to FAISS
def update_vector_store(new_chunks):
    global index
    embeddings = embedding_model.encode(new_chunks)
    if index is None:
        dimension = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    chunks.extend(new_chunks)
    faiss.write_index(index, index_path)
    with open("chunks.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunks))

# Retrieve top-k chunks
def retrieve(query, top_k=3):
    query_vec = embedding_model.encode([query])[0].astype("float32")
    D, I = index.search(np.array([query_vec]), top_k)
    return [chunks[i] for i in I[0]]

# Ask question to Ollama (local)
def ask_ollama(context, question, model="llama2"):
    prompt = f"""Use the context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt}
    )
    return response.json()["response"]

# ========= Main Chat =========
if __name__ == "__main__":
    print("📚 Dynamic RAG system started.")
    print("🧠 Loading all documents...")
    texts = load_documents()
    new_chunks = split_into_chunks(texts)
    update_vector_store(new_chunks)
    last_files = set(os.listdir(DATA_DIR))

    print("\n✅ Ready to answer your questions. Type 'exit' to quit.")
    while True:
        query = input("\n🔎 Your Question: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Detect new files
        current_files = set(os.listdir(DATA_DIR))
        if current_files != last_files:
            print("📥 New files detected. Updating knowledge base...")
            texts = load_documents()
            new_chunks = split_into_chunks(texts)
            update_vector_store(new_chunks)
            last_files = current_files
            print("✅ Knowledge base updated.")

        relevant_chunks = retrieve(query)
        context = "\n\n".join(relevant_chunks)
        answer = ask_ollama(context, query)
        print("\n🤖 Ollama's Answer:\n", answer)
