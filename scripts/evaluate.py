from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.evaluation.runner import run_evaluation_sync


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline evaluation across retrieval configurations.")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--ragas-mode", type=str, default=None, choices=["auto", "official", "fallback"])
    parser.add_argument("--profile", type=str, default=None, choices=["dev", "benchmark"])
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    settings = get_settings()
    summary = run_evaluation_sync(
        settings=settings,
        run_ragas=not args.skip_ragas,
        ragas_mode=args.ragas_mode,
        dataset_profile=args.profile,
        max_queries=args.max_queries,
        output_tag=args.tag,
    )
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
