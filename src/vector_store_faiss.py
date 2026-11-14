"""FAISS vector store — the industry standard for similarity search.

Same interface as the NumPy VectorStore, so the rest of the pipeline
(retriever, reranker, LLM) works unchanged. Swap them and compare.
"""

import json
from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:
    """A vector store backed by FAISS.

    IndexFlatIP = exact search using inner product. Because our embeddings
    are normalised, inner product == cosine similarity.
    """

    def __init__(self, storage_dir: str = "storage_faiss", dim: int = 384):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []

    def add(self, chunks: list[dict]):
        """Add chunks (each with an 'embedding') to the index."""
        embeddings = np.array([c["embedding"] for c in chunks], dtype="float32")
        self.index.add(embeddings)
        self.metadata.extend(
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "page_number": c["page_number"],
                "token_count": c["token_count"],
            }
            for c in chunks
        )

    def save(self):
        """Persist the index and metadata to disk."""
        faiss.write_index(self.index, str(self.storage_dir / "index.faiss"))
        with open(self.storage_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self):
        """Load the index and metadata from disk."""
        self.index = faiss.read_index(str(self.storage_dir / "index.faiss"))
        with open(self.storage_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search for the most similar chunks."""
        query = np.array([query_embedding], dtype="float32")
        scores, indices = self.index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 when there are fewer than top_k
                continue
            results.append({"metadata": self.metadata[int(idx)], "score": float(score)})
        return results

    def __len__(self) -> int:
        return self.index.ntotal
