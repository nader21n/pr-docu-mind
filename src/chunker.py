"""Text chunking for DocuMind."""

from transformers import AutoTokenizer


# Load the embedding model's tokenizer
TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

# Chunk settings
CHUNK_SIZE = 450
CHUNK_OVERLAP = 50


def chunk_text(text: str) -> list[dict]:
    """Split text into overlapping chunks, measured in tokens.

    Args:
        text: The text to chunk.

    Returns:
        A list of dictionaries, each with:
            - "chunk_id": a unique ID for the chunk
            - "text": the chunk's text
            - "token_count": the number of tokens in the chunk
    """
    tokens = TOKENIZER.encode(text, add_special_tokens=False)

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": TOKENIZER.decode(chunk_tokens, skip_special_tokens=True),
                "token_count": len(chunk_tokens),
            }
        )

        chunk_id += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
