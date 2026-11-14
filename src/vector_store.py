"""Vector store for DocuMind (NumPy implementation)."""

import json
from pathlib import Path
import numpy as np


class VectorStore:
    """A simple vector store using NumPy."""

    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = None
        self.metadata = []

    def add(self, chunks: list[dict]):
        """Add chunks with embeddings to the store."""
        new_embeddings = np.array([c["embedding"] for c in chunks], dtype="float32")
        new_metadata = [
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "page_number": c["page_number"],
                "token_count": c["token_count"],
            }
            for c in chunks
        ]
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.metadata.extend(new_metadata)

    def save(self):
        """Save embeddings and metadata to disk."""
        np.save(self.storage_dir / "embeddings.npy", self.embeddings)
        with open(self.storage_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self):
        """Load embeddings and metadata from disk."""
        self.embeddings = np.load(self.storage_dir / "embeddings.npy")
        with open(self.storage_dir / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search for the most similar chunks."""
        query = np.array(query_embedding, dtype="float32")
        scores = self.embeddings @ query
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [
            {
                "metadata": self.metadata[i],
                "score": float(scores[i]),
            }
            for i in top_indices
        ]

    def __len__(self):
        return len(self.metadata)
