"""Test the LLM answer generation."""

from src.retriever import Retriever
from src.llm import generate_answer


def main():
    retriever = Retriever(top_k=3)

    question = "What are the safety recommendations for spacecraft?"
    contexts = retriever.retrieve(question)

    print(f"Question: {question}")
    print(f"Retrieved {len(contexts)} chunks")
    print()

    answer = generate_answer(question, contexts)

    print("--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
