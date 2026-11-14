"""DocuMind — Document Q&A with citations."""

from src.retriever import Retriever
from src.llm import generate_answer


def main():
    print("=" * 60)
    print("  DocuMind — Ask questions about your PDF")
    print("=" * 60)
    print()
    retriever = Retriever(top_k=3, min_score=0.6)
    while True:
        question = input("\nYour question (or 'quit' to exit): ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        print("\nThinking...")
        contexts = retriever.retrieve(question)
        # If nothing is relevant enough, say "I don't know"
        if not contexts:
            print("\n" + "=" * 60)
            print("I don't know — the document doesn't contain information about that.")
            print("=" * 60)
            continue
        answer = generate_answer(question, contexts)
        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)
        print("\n" + "=" * 60)
        print("SOURCES")
        print("=" * 60)
        for i, c in enumerate(contexts, 1):
            print(f"{i}. Page {c['page_number']} (score: {c['score']:.4f})")
            print(f"   {c['text'][:100]}...")
            print()


if __name__ == "__main__":
    main()
