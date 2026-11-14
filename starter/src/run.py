
"""Run the Day 06 RAG retrieval benchmark."""

import json
import os
from pathlib import Path
from typing import Any

from networkx import hits

from .utils.exporter import export_chunks_jsonl

from .ingestion import load_pdf
from .chunking import fixed_size, by_paragraph
from .embeddings import MyEmbedder
from .store import PgVectorStore
from .evaluation import evaluate_query, summarize


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CORPUS_PATH = Path(
    os.getenv("CORPUS_PATH", BASE_DIR / "data" / "corpus.pdf")
)

BENCHMARK_PATH = Path(
    os.getenv("BENCHMARK_PATH", BASE_DIR / "data" / "benchmark.json")
)

OUTPUT_DIR = Path(
    os.getenv("OUTPUT_DIR", BASE_DIR / "outputs")
)

DATABASE_URL = os.getenv("DATABASE_URL")

DOCUMENT_ID = os.getenv(
    "DOCUMENT_ID",
    CORPUS_PATH.stem
)

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
MANUAL_SAMPLE_SIZE = 5

# ---------------------------------------------------------
# Check benchmark and corpus files
# ---------------------------------------------------------
def _squash(s: str) -> str:
    return " ".join((s or "").split()).lower()


def validate_benchmark_quotes(pages, questions) -> None:
    """Warn about evidence quotes that are not verbatim on the gold page."""
    by_page = {p.page: _squash(p.text) for p in pages}
    bad = []
    for q in questions:
        quote, page = q.get("evidence_quote"), q.get("gold_page")
        if quote and page is not None and _squash(quote) not in by_page.get(page, ""):
            bad.append(q["id"])
    if bad:
        print(f"[WARN] Quote not found verbatim on gold page: {bad}")
    else:
        print("All evidence quotes found on their gold pages.")


def is_narrowing_filter(f) -> bool:
    return bool(f) and any(k != "document_id" for k in f)


def print_filter_subset(traces) -> None:
    print("\n--- Filter-question subset (B vs C) ---")
    for strategy in ("B", "C"):
        sub = [t for t in traces
               if t["strategy"] == strategy and t.get("has_filter")]
        n = len(sub)
        if not n:
            print("No questions with a narrowing filter.")
            return
        h1 = sum(bool(t["hit_at_1"]) for t in sub)
        h3 = sum(bool(t["evidence_hit_at_3"]) for t in sub)
        print(f"{strategy}: n={n}  Hit@1={h1}/{n}  Evidence Hit@3={h3}/{n}")

# ---------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------

def load_benchmark(path: Path) -> list[dict[str, Any]]:
    """Load benchmark questions from JSON."""

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Support either a JSON list or an object containing questions.
    if isinstance(data, dict):
        data = data.get("questions")

    if not isinstance(data, list) or not data:
        raise ValueError(
            "Benchmark must contain a non-empty list of questions"
        )

    for index, question in enumerate(data):
        if not isinstance(question, dict):
            raise ValueError(
                f"Benchmark item {index} must be an object"
            )

        required_fields = ["id", "split", "question"]

        for field in required_fields:
            if not question.get(field):
                raise ValueError(
                    f"Question at index {index} is missing '{field}'"
                )

        if question["split"] not in ("dev", "test"):
            raise ValueError(
                f"Invalid split for question {question['id']}: "
                f"{question['split']}"
            )

    return data


def validate_corpus(pages: list[Any]) -> None:
    """Require at least 30 pages containing extractable text."""

    content_pages = [
        page for page in pages
        if page.text and page.text.strip()
    ]

    print(f"Total physical pages: {len(pages)}")
    print(f"Content pages: {len(content_pages)}")

    if len(content_pages) < 30:
        raise ValueError(
            f"Corpus must contain at least 30 content pages. "
            f"Found {len(content_pages)}."
        )


# ---------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------

def build_strategy_a(pages: list[Any]) -> list[Any]:
    """Strategy A: fixed-size chunking."""

    chunks = []

    for page in pages:
        chunks.extend(
            fixed_size(
                page,
                size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP
            )
        )

    chunks_a = [
        chunk
        for page in pages
        for chunk in fixed_size(page, size=900, overlap=150)
    ]

    export_chunks_jsonl(
        chunks_a,
        Path("../results/chunks/strategy_a.jsonl"),
    )

    return chunks


def build_strategy_b(pages: list[Any]) -> list[Any]:
    """Strategy B: paragraph-based chunking."""

    chunks = []

    for page in pages:
        chunks.extend(
            by_paragraph(
                page,
                max_size=CHUNK_SIZE
            )
        )

    chunks_b = [
        chunk
        for page in pages
        for chunk in by_paragraph(page, max_size=900)
    ]

    export_chunks_jsonl(
        chunks_b,
        Path("../results/chunks/strategy_b.jsonl"),
    )

    return chunks


# ---------------------------------------------------------
# Embedding and storage
# ---------------------------------------------------------

def embed_and_store(
    strategy: str,
    chunks: list[Any],
    embedder: MyEmbedder,
    store: PgVectorStore
) -> list[list[float]]:
    """Generate embeddings and store the complete strategy dataset."""

    if not chunks:
        raise ValueError(
            f"No chunks generated for strategy {strategy}"
        )

    texts = [chunk.text for chunk in chunks]

    vectors = embedder.embed_documents(texts)

    if len(vectors) != len(chunks):
        raise ValueError(
            f"Embedding count mismatch for strategy {strategy}"
        )

    store.upsert(
        strategy=strategy,
        chunks=chunks,
        vectors=vectors
    )

    print(
        f"[{strategy}] Stored {len(chunks)} chunks "
        f"with {len(vectors)} embeddings"
    )

    return vectors


# ---------------------------------------------------------
# Retrieval and evaluation
# ---------------------------------------------------------

def run_strategy(
    strategy: str,
    questions: list[dict[str, Any]],
    embedder: MyEmbedder,
    store: PgVectorStore
) -> list[dict[str, Any]]:
    """Run the same benchmark questions against one strategy."""

    traces = []

    for question in questions:
        query = question["question"]

        query_vector = embedder.embed_query(query)

        # Strategy C applies the question's metadata filter.
        filters = None

        if strategy == "C":
            filters = question.get("filter")

            if filters is not None and not isinstance(filters, dict):
                raise ValueError(
                    f"Filter for question {question['id']} "
                    f"must be a dictionary"
                )
            
        if strategy == "C":
            print(
                f"[DEBUG] Question {question['id']} "
                f"filter = {filters}"
            )

        hits = store.search(
            strategy=strategy,
            query_vector=query_vector,
            top_k=TOP_K,
            filters=filters
        )

        # Preserve strategy and split in each evaluation trace.
        evaluation_question = {
            **question,
            "strategy": strategy
        }

        trace = evaluate_query(
            question=evaluation_question,
            hits=hits
        )

        traces.append(trace)

    return traces


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

def save_json(data: Any, path: Path) -> None:
    """Save JSON output."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2
        )


def prepare_manual_samples(
    traces: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Prepare retrieval results for manual assessment."""

    samples = []

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for trace in traces:
        key = (
            trace["strategy"],
            trace["split"]
        )

        groups.setdefault(key, []).append(trace)

    for (strategy, split), group in groups.items():
        selected = group[:MANUAL_SAMPLE_SIZE]

        for trace in selected:
            samples.append({
                **trace,
                "manual_assessment": {
                    "human_grounding": None,
                    "chunk_coherence": None
                }
            })

    return samples


def print_summary(summary: dict[str, Any]) -> None:
    """Print retrieval metrics."""

    print("\n========== RETRIEVAL SUMMARY ==========")

    for strategy, splits in summary.items():
        print(f"\nStrategy: {strategy}")

        for split, metrics in splits.items():
            print(f"  Split: {split}")
            print(
                f"  Total queries: {metrics['total_queries']}"
            )
            print(
                f"  Hit@1: {metrics['hit_at_1']:.2%}"
            )
            print(
                f"  Evidence Hit@3: "
                f"{metrics['evidence_hit_at_3']:.2%}"
            )

    print("\n=======================================")


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main() -> None:
    """Execute the Day 06 retrieval evaluation pipeline."""

    if not DATABASE_URL:
        raise ValueError(
            "DATABASE_URL environment variable is required"
        )

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus PDF not found: {CORPUS_PATH}"
        )

    print("\n[1/5] Loading corpus and benchmark...")

    pages = load_pdf(
        path=CORPUS_PATH,
        document_id=DOCUMENT_ID,
        source=str(CORPUS_PATH)
    )

    validate_corpus(pages)

    questions = load_benchmark(BENCHMARK_PATH)

    validate_benchmark_quotes(pages, questions)
    for p in pages[:3]:
        print(f"[DEBUG] p{p.page} headings: {p.metadata.get('headings')}")

    dev_questions = [
        q for q in questions
        if q["split"] == "dev"
    ]

    test_questions = [
        q for q in questions
        if q["split"] == "test"
    ]

    if not dev_questions or not test_questions:
        raise ValueError(
            "Benchmark must contain both dev and test questions"
        )

    print(f"Dev questions: {len(dev_questions)}")
    print(f"Test questions: {len(test_questions)}")

    print("\n[2/5] Building chunking strategies...")

    chunks_a = build_strategy_a(pages)
    chunks_b = build_strategy_b(pages)

    print(f"Strategy A - Fixed size: {len(chunks_a)} chunks")
    print(f"Strategy B - Paragraph: {len(chunks_b)} chunks")
    print("Sections found:", sorted({c.metadata.get("section_id") or "-" for c in chunks_b}))

    print("\n[3/5] Initializing embedding model and vector store...")

    embedder = MyEmbedder(MODEL_NAME)

    store = PgVectorStore(
        database_url=DATABASE_URL,
        dimensions=embedder.dimensions
    )

    store.create_schema()

    # Strategy A: fixed-size chunking.
    embed_and_store(
        strategy="A",
        chunks=chunks_a,
        embedder=embedder,
        store=store
    )

    # Strategy B: paragraph-based chunking.
    vectors_b = embed_and_store(
        strategy="B",
        chunks=chunks_b,
        embedder=embedder,
        store=store
    )

    # Strategy C: same paragraph chunks and embeddings as B.
    # The difference is metadata filtering during retrieval.
    store.upsert(
        strategy="C",
        chunks=chunks_b,
        vectors=vectors_b
    )

    print(
        f"[C] Stored {len(chunks_b)} chunks "
        "for metadata-filtered retrieval"
    )

    print("\n[4/5] Running dev/test retrieval evaluation...")

    all_traces = []

    for strategy in ("A", "B", "C"):
        print(f"\nRunning strategy {strategy}...")

        for split, split_questions in (
            ("dev", dev_questions),
            ("test", test_questions)
        ):
            traces = run_strategy(
                strategy=strategy,
                questions=split_questions,
                embedder=embedder,
                store=store
            )

            all_traces.extend(traces)

            print(
                f"  {split}: evaluated {len(traces)} questions"
            )

    print("\n[5/5] Summarizing and saving results...")

    summary = summarize(all_traces)

    manual_samples = prepare_manual_samples(all_traces)

    save_json(
        all_traces,
        OUTPUT_DIR / "retrieval_traces.json"
    )

    save_json(
        summary,
        OUTPUT_DIR / "retrieval_summary.json"
    )

    save_json(
        manual_samples,
        OUTPUT_DIR / "manual_assessment_samples.json"
    )

    print_summary(summary)
    print_filter_subset(all_traces)

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")
    print(" - retrieval_traces.json")
    print(" - retrieval_summary.json")
    print(" - manual_assessment_samples.json")


if __name__ == "__main__":
    main()
