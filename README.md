📚 Sanskrit Story Retrieval System (RAG)

🌟 Overview

A hybrid Retrieval-Augmented Generation (RAG) system for Sanskrit stories, combining semantic embeddings and keyword matching. It retrieves contextually relevant answers from a curated corpus using FAISS and a quantized Qwen model, fully optimized for CPU deployment.

⚡ Key Features

Hybrid Search: Dense embeddings + BM25 keyword matching

Offline & Efficient: Runs locally on CPU using quantized Qwen

Contextual Responses: LLM generates coherent answers from retrieved passages

Fast Retrieval: FAISS-powered vector search

RRF Fusion: Combines results from vector and keyword search for better accuracy

🛠 Technical Components
Component	Details
Embeddings	sentence-transformers/all-MiniLM-L6-v2, 384-dim
Vector Store	FAISS, cosine similarity, CPU-optimized
Language Model	Qwen GGUF, llama-cpp-python, 2048-token context
Keyword Matcher	BM25 (k1=1.5, b=0.75)
Text Splitter	RecursiveCharacterTextSplitter, 500-char chunks, 100-char overlap

📁 Folder Structure

RAG_Sanskrit_Kashish/
├── code/
│   ├── app.py           # Query interface & RAG execution
│   ├── ingest.py        # Document ingestion & indexing
│   ├── utils.py         # Helper functions
│   ├── faiss_index/     # Stored vector index
│   ├── qwen.gguf        # CPU-compatible LLM model
│   └── requirements.txt
│
├── data/
│   ├── devbhakta.txt
│   ├── ghantakarna.txt
│   ├── kalidasa.txt
│   ├── murkhabhriya.txt
│   └── sheetam.txt
│
├── venv/                # Python virtual environment
├── README.md
└── report/
    └── Sanskrit_RAG_Report.pdf

🎯 Highlights

CPU-friendly: No GPU required, runs locally

Hybrid retrieval: Combines semantic understanding & exact keyword match

Context-aware answers: Preserves story context with chunked embeddings

Scalable: Easily add more Sanskrit stories without major changes
