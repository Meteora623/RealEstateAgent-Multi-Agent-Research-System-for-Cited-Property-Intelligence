from __future__ import annotations

import argparse
import os

from app.core.config import get_settings
from app.evaluation.final_runner import run_final_canonical_pipeline


def main() -> None:
    # Some Windows Python stacks load duplicate OpenMP runtimes from transitive deps.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    parser = argparse.ArgumentParser(
        description="Run canonical final benchmark pipeline and write reports/final artifacts."
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--tag", type=str, default="final")
    parser.add_argument("--reuse-run-dir", type=str, default=None)
    args = parser.parse_args()

    settings = get_settings()
    result = run_final_canonical_pipeline(
        settings=settings,
        max_queries=args.max_queries,
        output_tag=args.tag,
        reuse_run_dir=args.reuse_run_dir,
    )
    import json

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
