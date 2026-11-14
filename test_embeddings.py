"""Test the embedding function."""

from src.pdf_extractor import extract_text_from_pdf
from src.chunker import chunk_text
from src.embeddings import embed_text, embed_chunks


def main():
    pages = extract_text_from_pdf("data/sample.pdf")

    all_chunks = []
    for page in pages:
        chunks = chunk_text(page["text"])
        for chunk in chunks:
            chunk["page_number"] = page["page_number"]
            all_chunks.append(chunk)

    # Embed the first 5 chunks
    sample = all_chunks[:5]
    embed_chunks(sample)

    print(f"Total chunks: {len(all_chunks)}")
    print(f"Embedding dimension: {len(sample[0]['embedding'])}")
    print(f"First embedding (first 5 numbers): {sample[0]['embedding'][:5]}")
    print()
    print("--- First chunk text (first 200 chars) ---")
    print(sample[0]["text"][:200])


if __name__ == "__main__":
    main()
