"""Test the retriever."""

from src.retriever import Retriever


def main():
    retriever = Retriever(top_k=3)

    question = "What are the safety recommendations for spacecraft?"
    results = retriever.retrieve(question)

    print(f"Question: {question}")
    print(f"Found {len(results)} relevant chunks:")
    print()

    for i, r in enumerate(results, 1):
        print(f"{i}. score={r['score']:.4f} | page {r['page_number']}")
        print(f"   {r['text'][:120]}...")
        print()


if __name__ == "__main__":
    main()
