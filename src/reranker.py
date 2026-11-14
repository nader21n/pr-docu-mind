"""Reranking with a cross-encoder.

A cross-encoder reads the question and a document TOGETHER and outputs a
relevance score. This is stronger than vector search (which compares two
separate embeddings) because the model sees the actual words on both sides.
"""

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Reranks retrieved candidates by true relevance."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
        """Add a 'rerank_score' to each candidate and return the top_k.

        Scores are unbounded (roughly -11 to +11 for this model): the higher,
        the more relevant. Irrelevant pairs score near zero or negative.
        """
        if not candidates:
            return []
        pairs = [(query, c["metadata"]["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]
