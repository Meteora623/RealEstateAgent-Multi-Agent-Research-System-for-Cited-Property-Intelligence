from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.ingestion.pipeline import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest corpus into hybrid retrieval indexes.")
    parser.add_argument("--semantic-chunking", action="store_true")
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default=None,
        choices=["fixed", "semantic", "section_semantic"],
    )
    parser.add_argument("--profile", type=str, default=None, choices=["dev", "benchmark"])
    args = parser.parse_args()

    settings = get_settings()
    pipeline = IngestionPipeline(settings)
    stats = pipeline.run(
        semantic_chunking=args.semantic_chunking if args.semantic_chunking else None,
        chunking_strategy=args.chunking_strategy,
        dataset_profile=args.profile,
    )
    print(stats.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
