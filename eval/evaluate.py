"""Evaluate DocuMind."""

import json
from src.retriever import Retriever
from src.llm import generate_answer


def main():
    with open("eval/set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    retriever = Retriever(top_k=3, min_score=0.6, use_reranker=True)
    print("=" * 70)
    print("  DocuMind Evaluation")
    print("=" * 70)
    print()
    hits = 0
    reciprocal_ranks = []
    keyword_totals = 0
    keyword_hits = 0
    for item in eval_set:
        question = item["question"]
        expected_pages = item["expected_pages"]
        expected_keywords = item["expected_keywords"]
        contexts = retriever.retrieve(question)
        retrieved_pages = [c["page_number"] for c in contexts]
        # Retrieval: is an expected page in the results?
        if expected_pages:
            hit = any(p in retrieved_pages for p in expected_pages)
        else:
            # Unanswerable question: correct behavior is NO results
            hit = len(contexts) == 0

        rank = None
        for i, p in enumerate(retrieved_pages):
            if p in expected_pages:
                rank = i + 1
                break
        if hit:
            hits += 1
            if rank:
                reciprocal_ranks.append(1 / rank)
        # Answer: do the expected keywords appear?
        if contexts and expected_keywords:
            answer = generate_answer(question, contexts)
            found = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
            keyword_totals += len(expected_keywords)
            keyword_hits += len(found)
            keyword_pct = f"{len(found)}/{len(expected_keywords)}"
        else:
            keyword_pct = "-"
        status = "OK  " if hit else "MISS"
        pages_str = ",".join(str(p) for p in retrieved_pages) or "none"
        expected_str = ",".join(str(p) for p in expected_pages) or "none"
        print(f"[{status}] {question}")
        print(
            f"       retrieved: {pages_str} | expected: {expected_str} | keywords: {keyword_pct}"
        )
    # Summary
    total = len(eval_set)
    recall = hits / total
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    kw = keyword_hits / keyword_totals if keyword_totals else 0
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Questions:        {total}")
    print(f"  Hits:             {hits}")
    print(f"  Recall@3:         {recall:.2f}")
    print(f"  MRR:              {mrr:.2f}")
    print(f"  Keyword coverage: {kw:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
