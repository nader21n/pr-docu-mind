"""Hybrid search: vector + keyword (BM25 with stemming)."""

import math
import re
from collections import Counter

# Common English words that carry no meaning for search.
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "into",
    "through",
    "during",
    "before",
    "after",
    "to",
    "from",
    "in",
    "on",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "what",
    "when",
    "where",
    "who",
    "which",
    "why",
    "how",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "as",
    "we",
    "you",
    "they",
    "he",
    "she",
    "i",
    "our",
    "their",
}


def _stem(word: str) -> str:
    """Simple stemmer: strip common suffixes so 'chairs' matches 'chair'."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def _tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, drop stopwords, and stem."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [_stem(w) for w in text.split() if w and w not in STOPWORDS]


def bm25_search(store, query: str, top_k: int = 3) -> list[dict]:
    """Keyword search using BM25."""
    k1 = 1.2
    b = 0.75
    docs = [m["text"] for m in store.metadata]
    tokenized = [_tokenize(d) for d in docs]
    doc_lens = [len(t) for t in tokenized]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 1
    N = len(docs)
    query_terms = _tokenize(query)
    scores = []
    for i, tokens in enumerate(tokenized):
        tf = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            df = sum(1 for t in tokenized if term in t)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            freq = tf[term]
            denom = freq + k1 * (1 - b + b * doc_lens[i] / avgdl)
            score += idf * (freq * (k1 + 1)) / denom
        scores.append(score)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [
        {"metadata": store.metadata[i], "score": float(scores[i]), "index": i}
        for i in order[:top_k]
    ]


def hybrid_search(
    store, query: str, query_embedding: list[float], top_k: int = 3, alpha: float = 0.7
) -> list[dict]:
    """Combine vector and keyword search.
    Each result carries:
      - vector_score: raw cosine similarity (absolute, 0-1) or None
      - keyword_score: normalised BM25 score (0-1)
      - score: combined ranking score
    """
    candidate_k = 30
    vector_results = store.search(query_embedding, top_k=candidate_k)
    keyword_results = bm25_search(store, query, top_k=candidate_k)
    k_max = max((r["score"] for r in keyword_results), default=1.0) or 1.0
    combined = {}
    for r in vector_results:
        idx = store.metadata.index(r["metadata"])
        combined[idx] = {
            "metadata": r["metadata"],
            "vector_score": r["score"],
            "keyword_score": 0.0,
            "score": alpha * r["score"],
        }
    for r in keyword_results:
        idx = r["index"]
        nk = r["score"] / k_max
        if idx in combined:
            combined[idx]["keyword_score"] = nk
            combined[idx]["score"] += (1 - alpha) * nk
        else:
            combined[idx] = {
                "metadata": r["metadata"],
                "vector_score": None,
                "keyword_score": nk,
                "score": (1 - alpha) * nk,
            }
    ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]
