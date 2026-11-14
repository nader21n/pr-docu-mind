"""Test the vector store."""

from src.pdf_extractor import extract_text_from_pdf
from src.chunker import chunk_text
from src.embeddings import embed_text, embed_chunks
from src.vector_store import VectorStore


def main():
    # 1. Extract and chunk
    pages = extract_text_from_pdf("data/sample.pdf")
    all_chunks = []
    for page in pages:
        chunks = chunk_text(page["text"])
        for chunk in chunks:
            chunk["page_number"] = page["page_number"]
            all_chunks.append(chunk)
    # 2. Embed ALL chunks
    embed_chunks(all_chunks)
    # 3. Create the store and add chunks
    store = VectorStore("storage")
    store.add(all_chunks)
    print(f"Store size: {len(store)} chunks")
    # 4. Save to disk
    store.save()
    print("Saved to storage/")
    # 5. Create a NEW store and load from disk
    new_store = VectorStore("storage")
    new_store.load()
    print(f"Loaded: {len(new_store)} chunks")
    # 6. Search
    query = "spacecraft safety recommendations"
    query_embedding = embed_text(query)
    results = new_store.search(query_embedding, top_k=3)
    print()
    print(f"Query: {query}")
    print(f"Top 3 results:")
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        print(
            f"  {i}. score={r['score']:.4f} | page {meta['page_number']} | {meta['text'][:80]}..."
        )


if __name__ == "__main__":
    main()
