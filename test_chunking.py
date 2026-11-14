"""Test the text chunking function."""

from src.pdf_extractor import extract_text_from_pdf
from src.chunker import chunk_text


def main():
    pages = extract_text_from_pdf("data/sample.pdf")

    all_chunks = []
    for page in pages:
        chunks = chunk_text(page["text"])
        for chunk in chunks:
            chunk["page_number"] = page["page_number"]
            all_chunks.append(chunk)

    print(f"Total chunks: {len(all_chunks)}")
    print(f"First chunk tokens: {all_chunks[0]['token_count']}")
    print()
    print("--- First chunk (first 300 chars) ---")
    print(all_chunks[0]["text"][:300])


if __name__ == "__main__":
    main()
