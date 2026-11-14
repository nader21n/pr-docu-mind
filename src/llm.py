"""LLM answer generation for DocuMind."""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"


def build_prompt(question: str, contexts: list[dict]) -> str:
    """Build a prompt with the question and retrieved context."""
    context_text = "\n\n".join(
        f"[Page {c['page_number']}] {c['text']}" for c in contexts
    )
    return f"""You are a helpful assistant answering questions based on the provided document.
Context from the document:
{context_text}
Question: {question}
Answer based only on the context above. If the context doesn't contain the answer, say "I don't know"."""


def generate_answer(question: str, contexts: list[dict]) -> str:
    """Generate an answer using the LLM.
    Args:
        question: The user's question.
        contexts: The retrieved chunks.
    Returns:
        The LLM's answer.
    """
    prompt = build_prompt(question, contexts)
    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=500,
        temperature=0.1,
    )
    return response.choices[0].message.content
