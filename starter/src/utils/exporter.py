
import json
from pathlib import Path
from dataclasses import asdict
from typing import Any

from ..models import Chunk


def export_chunks(
    chunks: list[Chunk],
    output_path: Path,
) -> None:
    """Export chunking results to a JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [asdict(chunk) for chunk in chunks]

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4,
        )


def export_chunks_jsonl(
    chunks: list[Chunk],
    output_path: Path,
) -> None:
    """Export chunks using JSON Lines format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(asdict(chunk), ensure_ascii=False) + "\n"
            )
