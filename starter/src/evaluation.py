
"""Evaluate retrieved evidence and preserve per-query traces."""

from typing import Any
import re

REQUIRE_QUOTE = True


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate source/page matches and preserve Hit@1 and Hit@3 traces."""

    question_id = question.get("id")
    strategy = question.get("strategy")
    split = question.get("split")

    gold_source_id = question.get("gold_source_id")
    gold_page = question.get("gold_page")
    evidence_quote = question.get("evidence_quote")

    # Support alternative gold evidence when provided.
    # Each alternative may specify its own source, page, and evidence quote.
    alternatives = question.get("alternative_gold_evidence", [])

    gold_evidence = []

    if gold_source_id is not None:
        gold_evidence.append({
            "source": gold_source_id,
            "page": gold_page,
            "quote": evidence_quote
        })

    gold_evidence.extend(alternatives)

    def matches_gold(hit: dict[str, Any]) -> bool:
        """Check whether a retrieved hit matches accepted gold evidence."""

        for gold in gold_evidence:
            source_matches = (
                hit.get("document_id") == gold.get("source")
                or hit.get("source") == gold.get("source")
            )

            page_matches = (
                gold.get("page") is None
                or hit.get("page") == gold.get("page")
            )

            quote = gold.get("quote")
            quote_matches = (
                not REQUIRE_QUOTE
                or not quote
                or _squash(quote) in _squash(hit.get("text"))
            )

            if source_matches and page_matches and quote_matches:
                return True

        return False

    hit_matches = [
        matches_gold(hit)
        for hit in hits
    ]

    hit_at_1 = bool(hit_matches and hit_matches[0])
    evidence_hit_at_3 = any(hit_matches[:3])

    return {
        "question_id": question_id,
        "strategy": strategy,
        "split": split,
        "question": question.get("question"),
        "expected_answer": question.get("expected_answer"),

        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,

        "gold_evidence": gold_evidence,

        "retrieved_hits": [
            {
                "rank": index + 1,
                "chunk_id": hit.get("chunk_id"),
                "document_id": hit.get("document_id"),
                "source": hit.get("source"),
                "page": hit.get("page"),
                "score": hit.get("score"),
                "matches_gold": hit_matches[index],
                "text": hit.get("text")
            }
            for index, hit in enumerate(hits)
        ],

        # These require separate human evaluation.
        "human_grounding": None,
        "chunk_coherence": None
    }


def summarize(
    traces: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and split."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for trace in traces:
        strategy = trace.get("strategy")
        split = trace.get("split")

        if strategy is None or split is None:
            raise ValueError(
                "Every trace must contain strategy and split"
            )

        key = (strategy, split)
        grouped.setdefault(key, []).append(trace)

    summary = {}

    for (strategy, split), group in grouped.items():
        total = len(group)

        hit_at_1_count = sum(
            bool(trace.get("hit_at_1"))
            for trace in group
        )

        evidence_hit_at_3_count = sum(
            bool(trace.get("evidence_hit_at_3"))
            for trace in group
        )

        summary.setdefault(strategy, {})
        summary[strategy][split] = {
            "total_queries": total,
            "hit_at_1_count": hit_at_1_count,
            "hit_at_1": (
                hit_at_1_count / total if total else 0.0
            ),
            "evidence_hit_at_3_count": evidence_hit_at_3_count,
            "evidence_hit_at_3": (
                evidence_hit_at_3_count / total if total else 0.0
            )
        }

    return summary
