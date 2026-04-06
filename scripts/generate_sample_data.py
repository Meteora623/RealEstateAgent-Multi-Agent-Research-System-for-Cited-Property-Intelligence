from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.ingestion.generator import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic real-estate corpus.")
    parser.add_argument("--num-properties", type=int, default=None)
    parser.add_argument("--docs-per-property", type=int, default=None)
    parser.add_argument("--eval-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--profile", type=str, default=None, choices=["dev", "benchmark"])
    args = parser.parse_args()

    settings = get_settings()
    profile = args.profile or settings.dataset_profile
    if profile == "benchmark":
        num_properties = args.num_properties or settings.benchmark_num_properties
        docs_per_property = args.docs_per_property or settings.benchmark_docs_per_property
        eval_queries = args.eval_queries or settings.benchmark_eval_queries
        seed = args.seed or settings.benchmark_seed
    else:
        num_properties = args.num_properties or 120
        docs_per_property = args.docs_per_property or 4
        eval_queries = args.eval_queries or 60
        seed = args.seed or 11

    stats = generate_synthetic_dataset(
        data_dir=settings.data_path,
        num_properties=num_properties,
        docs_per_property=docs_per_property,
        eval_queries=eval_queries,
        seed=seed,
    )
    print(f"Generated dataset: {stats}")


if __name__ == "__main__":
    main()
