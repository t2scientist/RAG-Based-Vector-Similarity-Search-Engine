# AI-Powered Vector Database & RAG Similarity Search Engine

A lightweight **Vector Database** built from scratch in **Python** that performs semantic similarity search using **HNSW**, **KD-Tree**, and **Brute Force** search algorithms. The project also implements a complete **Retrieval-Augmented Generation (RAG)** pipeline using **Ollama** for local embeddings and LLM inference.

> This project demonstrates how modern AI search systems such as Pinecone, Chroma, Weaviate, and Milvus work internally.

---

## Features

- Custom Vector Database implementation
- HNSW, KD-Tree and Brute Force search algorithms
- Cosine, Euclidean and Manhattan distance metrics
- Semantic document search
- Automatic document chunking
- Local embedding generation using Ollama
- Retrieval-Augmented Generation (RAG)
- Flask REST API
- Interactive Web Interface

---

# Project Architecture

```
                User Uploads Document
                        │
                        ▼
               Document Chunking
                        │
                        ▼
      Ollama (nomic-embed-text Embedding Model)
                        │
                        ▼
              768-Dimensional Embeddings
                        │
                        ▼
              Custom Vector Database
          (HNSW / KD-Tree / Brute Force)
                        │
                        ▼
                Similarity Search
                        │
                        ▼
           Top-K Relevant Document Chunks
                        │
                        ▼
           Ollama (Llama 3.2 Language Model)
                        │
                        ▼
                  AI Generated Answer
```

---

# How It Works

## 1. Document Ingestion

The user uploads or pastes text into the application.

Long documents are automatically divided into smaller overlapping chunks to improve retrieval quality.

---

## 2. Embedding Generation

Each chunk is converted into a **768-dimensional vector embedding** using:

- **nomic-embed-text**

Embeddings capture the semantic meaning of text rather than just keywords.

---

## 3. Vector Storage

The generated vectors are stored inside a custom in-memory vector database.

Each document chunk is indexed using:

- HNSW
- KD-Tree
- Brute Force

---

## 4. Similarity Search

When the user asks a question:

- The question is converted into an embedding.
- The vector database searches for the most similar document chunks.
- The top matching chunks are retrieved.

---

## 5. Retrieval-Augmented Generation (RAG)

The retrieved chunks are passed to **Llama 3.2** as additional context.

The language model generates an answer using only the retrieved information, reducing hallucinations and improving accuracy.

---

# Search Algorithms

## HNSW (Hierarchical Navigable Small World)

A graph-based Approximate Nearest Neighbor search algorithm.

- Extremely fast
- Optimized for high-dimensional vectors
- Used in production vector databases like Pinecone, Weaviate, Milvus and Chroma

---

## KD-Tree

A tree-based search structure that recursively partitions vector space.

Advantages

- Fast for low-dimensional data

Limitations

- Performance decreases significantly for high-dimensional embeddings.

---

## Brute Force

The simplest nearest-neighbor search.

Every stored vector is compared with the query vector.

Advantages

- Exact results

Limitations

- Slow for large datasets

---

# Distance Metrics

The project supports three similarity metrics.

### Cosine Similarity

Measures similarity based on vector direction.

Best for semantic search.

---

### Euclidean Distance

Measures straight-line distance between vectors.

---

### Manhattan Distance

Measures distance by summing coordinate differences.

---

# Technologies Used

- Python
- Flask
- HTML
- CSS
- JavaScript
- Ollama
- HNSW
- KD-Tree
- Vector Embeddings
- REST API

---

# REST API

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | `/doc/insert` | Insert a document |
| GET | `/doc/list` | List all documents |
| DELETE | `/doc/delete/<id>` | Delete a document |
| POST | `/doc/ask` | Ask questions using RAG |
| GET | `/status` | Check Ollama status |

---

# Project Flow

```
User

↓

Upload Document

↓

Chunking

↓

Embedding Generation

↓

Vector Database

↓

User Question

↓

Question Embedding

↓

Similarity Search

↓

Top-K Chunks

↓

Llama 3.2

↓

Generated Answer
```

---

# Why Vector Database?

Traditional databases perform keyword matching.

Vector databases perform **semantic search**, allowing the system to retrieve information based on meaning instead of exact words.

Example

Question

```
Who is the ruler of the forest?
```

Document

```
The lion is the king of the jungle.
```

Although the words are different, semantic search correctly retrieves the relevant document.

---

# Future Improvements

- Hybrid Search (BM25 + Vector Search)
- Cross-Encoder Re-ranking
- PDF Upload Support
- Persistent Vector Storage
- User Authentication
- Docker Deployment
- Cloud Deployment
- Streaming LLM Responses

---

# Learning Outcomes

Through this project I gained practical experience with:

- Vector Databases
- Semantic Search
- Retrieval-Augmented Generation (RAG)
- Approximate Nearest Neighbor Search
- HNSW Graph Indexing
- Embedding Models
- Flask Backend Development
- REST APIs
- Local LLM Deployment using Ollama

---

# Project Highlights

- Built a Vector Database from scratch in Python.
- Implemented HNSW, KD-Tree and Brute Force search algorithms.
- Developed a complete Retrieval-Augmented Generation pipeline.
- Integrated Ollama for local embeddings and LLM inference.
- Created a Flask-based REST API with an interactive web interface.
