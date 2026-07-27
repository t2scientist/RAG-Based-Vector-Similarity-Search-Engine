# VectorDB (Python Edition)

A Python port of the C++ VectorDB project. Same idea, same algorithms,
same REST API, same web UI — just Python instead of C++.

Implements **HNSW**, **KD-Tree**, and **Brute Force** search side by side,
plus a **RAG pipeline** powered by a local LLM via Ollama.

## Files

```
vectordb_py/
├── app.py            ← Everything: algorithms + Flask REST API
├── index.html         ← Frontend (unchanged from the original)
├── requirements.txt   ← Python dependencies
└── README.md
```

Only two dependencies: `flask` (web server) and `requests` (talks to Ollama).
Everything else — HNSW, KD-Tree, Brute Force, distance metrics, the text
chunker — is written by hand in plain Python, same as the original C++.

## Setup

1. Install Python 3.9+.
2. Install Ollama from https://ollama.com, then:
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run:
   ```
   python app.py
   ```
5. Open http://localhost:8080

## REST API

Identical to the C++ version — see the main project README for the full
endpoint reference (`/search`, `/insert`, `/delete/:id`, `/items`,
`/benchmark`, `/hnsw-info`, `/doc/insert`, `/doc/list`, `/doc/delete/:id`,
`/doc/search`, `/doc/ask`, `/status`, `/stats`).
