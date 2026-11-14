# DocuMind

**A document Q&A system with citations — built from scratch.**

Give it a PDF, ask questions in plain language, and get answers that cite the exact page they came from.

Built without LangChain or LlamaIndex, to understand every part of the RAG pipeline.

---

## ✨ Features

- **Semantic search** — finds meaning, not just keywords
- **Hybrid search** — vector **+** BM25 keyword, so exact names and entities are found
- **Cross-encoder reranking** — a second model re-scores candidates for precision
- **Grounded answers** — the LLM answers from YOUR document, not its training data
- **Citations** — every answer shows the source page numbers
- **Honest refusals** — says "I don't know" when the document doesn't contain the answer
- **Measurable quality** — an evaluation harness with recall@3, MRR, and keyword coverage
- **Framework comparison** — the same pipeline in LangChain (`pr-docu-mind-langchain/`)
- **Vector-store comparison** — NumPy vs FAISS, with a scaling benchmark

---

## 🧠 How it works

```
PDF → Extract → Chunk → Embed → Store → Hybrid search → Rerank → Generate → Cite
```

| Step | What it does | Why it matters |
|---|---|---|
| **Extract** | Reads the PDF page by page (PyMuPDF) | Preserves page numbers for citations |
| **Chunk** | Splits text into 450-token pieces (50 overlap) | Fits the model limit; precise retrieval |
| **Embed** | Turns chunks into 384-number vectors (bge-small) | Makes meaning searchable |
| **Store** | Saves vectors + metadata to disk (NumPy or FAISS) | Embed once, search forever |
| **Hybrid search** | Vector (meaning) + BM25 keyword (exact terms) | Each covers the other's weakness |
| **Rerank** | Cross-encoder re-scores the candidates | The production quality lift |
| **Threshold** | Refuses when the best score is too low | No confident wrong answers |
| **Generate** | LLM answers from the chunks (Groq) | Grounded, factual answers |
| **Cite** | Shows the source pages | Verifiable, trustworthy |

---

## 🛠 Tech Stack

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.14 | Modern, supported |
| PDF extraction | PyMuPDF | Fast, free, open source |
| Embeddings | BAAI/bge-small-en-v1.5 | 512-token context, 384 dims, ~130 MB |
| Keyword search | BM25 (from scratch) | Exact terms, names, entities |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Small, fast, large quality lift |
| Vector store | NumPy (default) / FAISS (comparison) | NumPy shows the math; FAISS scales |
| LLM | Groq API (`openai/gpt-oss-120b`) | Free tier, strong model |
| Interface | CLI (`main.py`) | Simple, demoable |

**All free. No credit card required.**

---

## 📁 Project Structure

```
pr-docu-mind/
├── main.py                  # CLI entry point
├── build_faiss_index.py     # builds the FAISS index
├── src/
│   ├── pdf_extractor.py     # PDF → text (page by page)
│   ├── chunker.py           # text → 450-token chunks
│   ├── embeddings.py        # text → 384-dim vectors
│   ├── vector_store.py      # NumPy store + search
│   ├── vector_store_faiss.py# FAISS store (same interface)
│   ├── hybrid_search.py     # vector + BM25 fusion
│   ├── reranker.py          # cross-encoder reranking
│   ├── retriever.py         # ties retrieval together
│   └── llm.py               # grounded answer generation
├── eval/
│   ├── set.json             # evaluation questions + expected pages
│   ├── evaluate.py          # recall@3, MRR, keyword coverage
│   ├── compare_stores.py    # NumPy vs FAISS
│   └── scale_benchmark.py   # where FAISS actually wins
├── data/                    # PDFs go here (sample.pdf included)
├── storage/                 # generated NumPy store (gitignored)
├── storage_faiss/           # generated FAISS index (gitignored)
├── .env                     # API key (never committed)
├── .env.example             # shows which variables are needed
└── requirements.txt
```

---

## 🚀 Installation

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it (PowerShell)
venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Configure your key
copy .env.example .env
#    then edit .env and paste your Groq key
```

---

## 💬 Usage

```bash
# Build the knowledge base (first run)
python test_vector_store.py

# Ask questions
python main.py
```

**Example:**

```
Your question: What are the safety recommendations for spacecraft?

ANSWER
----
The panel recommended re-examining mission objectives, adopting an
integrated testing approach, using a Design Reference Mission,
engaging an independent integrator, and developing an integrated
acquisition strategy.

SOURCES
----
1. Page 37 (score: 0.7684)
2. Page 29 (score: 0.7463)
3. Page 31 (score: 0.7391)
```

---

## 📊 Evaluation

```bash
python -m eval.evaluate
```

```
Questions:        6
Hits:             6
Recall@3:         1.00
MRR:              1.00
Keyword coverage: 100.00%
```

---

## 🔬 Comparisons

### LangChain (Step 11)

The same pipeline rebuilt with LangChain lives in `pr-docu-mind-langchain/`.
LangChain removed ~70% of the code — but its character-based splitter produced
**164 chunks instead of our 136**, and you lose visibility into chunking,
scoring, and metadata. **Use it to ship faster; understand the internals to debug it.**

### NumPy vs FAISS (Step 13)

```bash
python -m eval.compare_stores      # correctness + speed
python -m eval.scale_benchmark     # scaling
```

| Corpus size | NumPy | FAISS Flat | FAISS IVF |
|---|---|---|---|
| 136 | 0.024 ms | 0.024 ms | 0.011 ms |
| 10,000 | 2.6 ms | 3.2 ms | 0.06 ms |
| 100,000 | 17.0 ms | 28.6 ms | **0.16 ms** |

**Surprise:** FAISS *Flat* is slower than NumPy — it is the same brute force.
The real win is the **approximate** indexes (IVF/HNSW): ~100× faster at 100k vectors.

---

## 🎯 Key Decisions

| Decision | Choice | Why |
|---|---|---|
| Chunk size | 450 tokens | Fits the 512-token model limit |
| Overlap | 50 tokens | Preserves context at boundaries |
| Embedding model | bge-small | Balance of size, speed, quality |
| Hybrid weights | 0.7 vector / 0.3 keyword | Semantic dominates; keyword rescues entities |
| Reranker | ms-marco-MiniLM | Ordering only — its scores aren't calibrated |
| Threshold | 0.6 on the hybrid score | Calibrated; blocks irrelevant questions |
| Temperature | 0.1 | Factual, grounded answers |
| Vector store | NumPy (default) | Fast enough below ~10k chunks; no dependency |

---

## 🗺 What's Next

- [x] Hybrid search (vector + keyword)
- [x] Reranking (cross-encoder)
- [x] Framework comparison (LangChain)
- [x] Vector-store comparison (NumPy vs FAISS)
- [ ] Deployment (FastAPI + Docker + Render)

---

## 📄 License

MIT — use it, learn from it, build on it.
