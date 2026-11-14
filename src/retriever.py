"""Retrieval for DocuMind."""

from src.embeddings import embed_text
from src.vector_store import VectorStore
from src.hybrid_search import hybrid_search
from src.reranker import Reranker


class Retriever:
    """Retrieves relevant chunks from the vector store."""

    def __init__(
        self,
        storage_dir: str = "storage",
        top_k: int = 5,
        min_score: float = 0.0,
        use_reranker: bool = True,
    ):
        self.store = VectorStore(storage_dir)
        self.store.load()
        self.top_k = top_k
        self.min_score = min_score
        self.reranker = Reranker() if use_reranker else None

    def retrieve(self, question: str) -> list[dict]:
        """Retrieve the most relevant chunks for a question.

        Candidates come from hybrid search, then a cross-encoder reranks
        them. The threshold applies to the reranker score.
        """
        question_embedding = embed_text(question)
        candidates = hybrid_search(self.store, question, question_embedding, top_k=10)

        if self.reranker:
            candidates = self.reranker.rerank(question, candidates, top_k=self.top_k)
            score_key = "rerank_score"
        else:
            score_key = "score"

        # The threshold uses the calibrated hybrid score; the cross-encoder
        # only changes the ORDER, not the cutoff.
        filtered = [r for r in candidates if r["score"] >= self.min_score]

        return [
            {
                "text": r["metadata"]["text"],
                "page_number": r["metadata"]["page_number"],
                "score": r[score_key],
            }
            for r in filtered
        ]
