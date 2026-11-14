"""Compare the NumPy vector store against the FAISS vector store.

Checks:
  1. Do they return the SAME results? (correctness)
  2. How fast is each? (speed)

Run:  python -m eval.compare_stores
"""

import time
import warnings

warnings.filterwarnings("ignore")

from src.embeddings import embed_text
from src.vector_store import VectorStore
from src.vector_store_faiss import FaissVectorStore

QUESTIONS = [
    "When was the Aerospace Safety Advisory Panel established?",
    "What are the safety recommendations for spacecraft?",
    "What is the Design Reference Mission?",
    "What issues affect the Artemis III schedule?",
    "Who chairs the Aerospace Safety Advisory Panel?",
    "What is the recipe for chocolate cake?",
]


def main():
    numpy_store = VectorStore("storage")
    numpy_store.load()

    faiss_store = FaissVectorStore("storage_faiss")
    faiss_store.load()

    print("=" * 70)
    print("  NumPy vs FAISS — comparison")
    print("=" * 70)
    print(f"  NumPy store: {len(numpy_store)} vectors")
    print(f"  FAISS store: {len(faiss_store)} vectors")
    print()

    same = 0
    for q in QUESTIONS:
        embedding = embed_text(q)
        np_res = numpy_store.search(embedding, top_k=3)
        fa_res = faiss_store.search(embedding, top_k=3)
        np_pages = [r["metadata"]["page_number"] for r in np_res]
        fa_pages = [r["metadata"]["page_number"] for r in fa_res]
        match = np_pages == fa_pages
        same += match
        print(f"  [{'same' if match else 'DIFF'}] {q[:50]}")
        print(f"         NumPy: {np_pages}")
        print(f"         FAISS: {fa_pages}")

    print()
    print(f"  Agreement: {same}/{len(QUESTIONS)}")
    print()

    # Speed
    print("  Speed (10 searches each):")
    start = time.time()
    for _ in range(10):
        for q in QUESTIONS:
            numpy_store.search(embed_text(q), top_k=3)
    np_time = time.time() - start

    start = time.time()
    for _ in range(10):
        for q in QUESTIONS:
            faiss_store.search(embed_text(q), top_k=3)
    fa_time = time.time() - start

    print(f"    NumPy: {np_time:.3f}s")
    print(f"    FAISS: {fa_time:.3f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
