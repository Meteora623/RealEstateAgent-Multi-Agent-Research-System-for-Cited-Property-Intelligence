from __future__ import annotations

from app.core.config import get_settings
from app.evaluation.runner import run_evaluation_sync


def main() -> None:
    settings = get_settings()
    summary = run_evaluation_sync(
        settings=settings,
        run_ragas=False,
        ragas_mode="fallback",
        dataset_profile="benchmark",
        output_tag="compare",
    )
    print(f"Comparison complete. Best config: {summary.best_configuration}")
    print(f"Artifacts: {summary.run_dir}")


if __name__ == "__main__":
    main()
