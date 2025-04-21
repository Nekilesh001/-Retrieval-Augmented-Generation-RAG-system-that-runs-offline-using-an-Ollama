# -Retrieval-Augmented-Generation-RAG-system-that-runs-offline-using-an-Ollama

Overview

This project implements a fully dynamic, offline RAG system leveraging an Ollama (Llama‑family) model. It allows you to:

Ingest .txt and .pdf documents into a local folder (data/).

Automatically split documents into fixed‑size chunks for efficient retrieval.

Embed each chunk using a SentenceTransformer model (all-MiniLM-L6-v2).

Index embeddings in FAISS for semantic search.

Monitor the data/ folder for new documents and re‑index on‑the‑fly.

Retrieve relevant chunks based on a user query.

Generate answers using a local Ollama (Llama‑family) model without requiring internet access.
