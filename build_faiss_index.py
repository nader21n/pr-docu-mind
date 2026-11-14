"""Build the FAISS index from data/sample.pdf.

Run once after installing faiss-cpu:
    python build_faiss_index.py
"""

import warnings

warnings.filterwarnings("ignore")

from src.chunker import chunk_text
from src.embeddings import embed_chunks
from src.pdf_extractor import extract_text_from_pdf
from src.vector_store_faiss import FaissVectorStore


def main():
    print("1. Extracting text from PDF...")
    pages = extract_text_from_pdf("data/sample.pdf")
    print(f"   {len(pages)} pages")

    print("2. Chunking (per page)...")
    all_chunks = []
    for page in pages:
        for chunk in chunk_text(page["text"]):
            chunk["page_number"] = page["page_number"]
            all_chunks.append(chunk)
    print(f"   {len(all_chunks)} chunks")

    print("3. Embedding (this takes a moment)...")
    embed_chunks(all_chunks)

    print("4. Building FAISS index...")
    store = FaissVectorStore(
        storage_dir="storage_faiss", dim=len(all_chunks[0]["embedding"])
    )
    store.add(all_chunks)
    store.save()

    print(f"   Done. {len(store)} vectors stored in storage_faiss/")


if __name__ == "__main__":
    main()
