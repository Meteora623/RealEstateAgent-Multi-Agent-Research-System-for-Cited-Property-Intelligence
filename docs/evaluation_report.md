# Evaluation Reporting Guide

## What Gets Generated

Every evaluation run creates:

- `artifacts/eval/<run_id>/quality_metrics.json`
- `artifacts/eval/<run_id>/latency_metrics.json`
- `artifacts/eval/<run_id>/comparison_table.csv`
- `artifacts/eval/<run_id>/evaluation_report.md`
- `artifacts/eval/<run_id>/resume_claim_support.md`
- `artifacts/eval/<run_id>/official_ragas_gate.json`

And updates:

- `reports/corpus_stats.json`
- `reports/corpus_stats.md`
- `reports/evaluation_summary.md`
- `reports/resume_claim_support.md`

## Official RAGAs Gate

When `run_ragas=true`, the runner enforces a gate for the best config:

- metric source must be `official_ragas`
- numeric values required for:
  - `faithfulness`
  - `answer_relevancy`
  - `context_recall`

If `REQUIRE_OFFICIAL_RAGAS_BEST_CONFIG=true`, evaluation fails when this gate is not met.

## Resume Claim Policy

Use resume wording only from the `Verified` section in `reports/resume_claim_support.md`.
Claims without verified evidence should be treated as partial or unsupported.

