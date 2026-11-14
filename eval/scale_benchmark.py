"""Show where FAISS actually wins: scale.

Tiles the real embeddings to simulate larger corpora, then compares
brute-force NumPy against FAISS exact (Flat) and approximate (IVF).

Run:  python -m eval.scale_benchmark
"""

import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import faiss

from src.embeddings import embed_text
from src.vector_store import VectorStore

QUERY = "What are the safety recommendations for spacecraft?"


def build_arrays(n):
    """Tile the real 136 vectors until we have n."""
    store = VectorStore("storage")
    store.load()
    base = store.embeddings.astype("float32")
    reps = int(np.ceil(n / len(base)))
    return np.tile(base, (reps, 1))[:n]


def time_it(fn, repeats=50):
    start = time.time()
    for _ in range(repeats):
        fn()
    return (time.time() - start) / repeats * 1000  # ms


def main():
    q = np.array(embed_text(QUERY), dtype="float32")

    print("=" * 72)
    print("  Scaling: NumPy vs FAISS")
    print("=" * 72)
    print(
        f"  {'corpus':>9} | {'NumPy ms':>10} | {'FAISS Flat ms':>15} | {'FAISS IVF ms':>14}"
    )
    print("-" * 72)

    for n in [136, 1_000, 10_000, 100_000]:
        data = build_arrays(n)
        dim = data.shape[1]

        def numpy_search():
            scores = data @ q
            return np.argsort(scores)[-3:][::-1]

        flat = faiss.IndexFlatIP(dim)
        flat.add(data)

        def flat_search():
            return flat.search(np.array([q]), 3)

        nlist = max(1, int(np.sqrt(n)))
        quantizer = faiss.IndexFlatIP(dim)
        ivf = faiss.IndexIVFFlat(quantizer, dim, nlist)
        ivf.train(data)
        ivf.add(data)

        def ivf_search():
            return ivf.search(np.array([q]), 3)

        print(
            f"  {n:>9,} | {time_it(numpy_search):>10.3f} "
            f"| {time_it(flat_search):>15.3f} | {time_it(ivf_search):>14.3f}"
        )

    print("-" * 72)
    print("  Average ms per query (lower is better).")
    print("=" * 72)


if __name__ == "__main__":
    main()
