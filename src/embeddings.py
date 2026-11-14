"""Embeddings for DocuMind."""

from sentence_transformers import SentenceTransformer


# Load the embedding model
MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")

# The model's embedding dimension
DIMENSION = 384


def embed_text(text: str) -> list[float]:
    """Convert text into an embedding vector.

    Args:
        text: The text to embed.

    Returns:
        A list of 384 floats — the embedding vector.
    """
    embedding = MODEL.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add embeddings to a list of chunks.

    Args:
        chunks: A list of chunk dicts (with "text" key).

    Returns:
        The same list, with an "embedding" key added to each.
    """
    for chunk in chunks:
        chunk["embedding"] = embed_text(chunk["text"])
    return chunks
