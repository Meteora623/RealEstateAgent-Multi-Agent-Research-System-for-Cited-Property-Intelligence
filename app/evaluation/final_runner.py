from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.evaluation.finalize import (
    build_evaluation_summary_json,
    build_final_report_markdown,
    build_project_summary_candidates,
    build_validated_claims_markdown,
    classify_claims,
    find_row,
    load_comparison_rows,
    load_json,
    write_json,
)
from app.evaluation.ragas_runner import compute_quality_metrics
from app.evaluation.runner import run_evaluation_sync
from app.schemas.evaluation import ConfigEvaluationResult

REQUIRED_CONFIGS = {
    "dense_only_baseline",
    "hybrid_baseline",
    "hybrid_rerank",
    "hybrid_rerank_semantic",
}
REQUIRED_RAGAS_METRICS = {"faithfulness", "answer_relevancy", "context_recall"}


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _validate_final_acceptance(
    *,
    corpus_stats: dict[str, Any],
    summary_payload: dict[str, Any],
    comparison_rows: list[dict[str, str]],
    quality_metrics: list[dict[str, Any]],
    latency_metrics: dict[str, Any],
) -> None:
    errors: list[str] = []
    scale = summary_payload.get("scale_metrics", {})
    docs = int(scale.get("total_indexed_documents", 0))
    chunks = int(scale.get("total_indexed_chunks", 0))
    queries = int(scale.get("total_evaluation_queries", 0))
    if not (docs >= 3000 or chunks >= 15000):
        errors.append(
            f"Scale acceptance failed: docs={docs}, chunks={chunks}. Need docs>=3000 or chunks>=15000."
        )
    if queries < 50:
        errors.append(f"Evaluation query acceptance failed: queries={queries}. Need >=50.")

    config_names = {row.get("config_name", "") for row in comparison_rows}
    missing_configs = REQUIRED_CONFIGS - config_names
    if missing_configs:
        errors.append(f"Missing required configs in comparison table: {sorted(missing_configs)}")

    official_gate = summary_payload.get("official_ragas_validation", {})
    if not official_gate.get("best_config_official_ragas_complete"):
        errors.append(
            "Official RAGAs gate failed for best config. "
            f"Reason: {official_gate.get('reason', 'unknown')}"
        )

    best_config = summary_payload.get("best_configuration", "")
    try:
        best_row = find_row(comparison_rows, best_config)
        if best_row.get("metric_source") != "official_ragas":
            errors.append(f"Best config `{best_config}` does not use official_ragas metric source.")
        missing_numeric = [
            metric
            for metric in REQUIRED_RAGAS_METRICS
            if _to_float(best_row.get(metric)) is None
        ]
        if missing_numeric:
            errors.append(
                f"Best config `{best_config}` missing numeric official RAGAs metrics: {missing_numeric}."
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Unable to locate best config row in comparison table: {exc}")

    sync_async = latency_metrics.get("sync_vs_async_tools", {})
    sync = sync_async.get("sync", {})
    async_ = sync_async.get("async", {})
    for bucket_name, payload in {"sync": sync, "async": async_}.items():
        for required_key in ["avg_ms", "p50_ms", "p95_ms", "per_agent_timing_ms"]:
            if required_key not in payload:
                errors.append(
                    f"Latency summary missing `{required_key}` under sync_vs_async_tools.{bucket_name}."
                )

    rerank_block = latency_metrics.get("rerank_on_vs_off", {})
    if "rerank_off" not in rerank_block or "rerank_on" not in rerank_block:
        errors.append("Latency summary missing rerank_on_vs_off.rerank_off/rerank_on sections.")

    if not corpus_stats:
        errors.append("Corpus stats payload is empty.")
    if len(quality_metrics) < 4:
        errors.append("Quality metrics payload has fewer than four configuration results.")

    if errors:
        raise RuntimeError("Final acceptance checks failed:\n- " + "\n- ".join(errors))


def run_final_canonical_pipeline(
    *,
    settings: Settings,
    max_queries: int | None = None,
    output_tag: str = "final",
    reuse_run_dir: str | None = None,
) -> dict[str, Any]:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for canonical final runs with official RAGAs enforcement."
        )

    if reuse_run_dir:
        run_dir = Path(reuse_run_dir)
        if not run_dir.exists():
            raise RuntimeError(f"reuse_run_dir does not exist: {run_dir}")
    else:
        eval_query_count = max(max_queries or settings.benchmark_eval_queries, 50)
        summary = run_evaluation_sync(
            settings=settings,
            run_ragas=True,
            ragas_mode="official",
            dataset_profile="benchmark",
            max_queries=eval_query_count,
            output_tag=output_tag,
        )
        run_dir = Path(summary.run_dir)
    final_dir = settings.reports_path / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    corpus_stats = load_json(run_dir / "corpus_stats.json")
    scale_metrics = load_json(run_dir / "scale_metrics.json")
    quality_metrics = load_json(run_dir / "quality_metrics.json")
    latency_metrics = load_json(run_dir / "latency_metrics.json")
    quality_samples = load_json(run_dir / "quality_samples.json")

    if not isinstance(quality_metrics, list):
        raise RuntimeError("quality_metrics.json must contain a list of config result rows.")
    if not isinstance(quality_samples, dict):
        raise RuntimeError("quality_samples.json must contain config->samples mapping.")

    updated_results: list[dict[str, Any]] = []
    for row in quality_metrics:
        config_name = str(row.get("config_name", ""))
        samples = quality_samples.get(config_name, [])
        official_metrics, metric_source, notes = compute_quality_metrics(
            samples=samples if isinstance(samples, list) else [],
            settings=settings,
            ragas_mode="official",
        )
        row["ragas_metrics"] = official_metrics
        row["metric_source"] = metric_source
        existing_notes = row.get("notes", [])
        if not isinstance(existing_notes, list):
            existing_notes = [str(existing_notes)]
        row["notes"] = [*existing_notes, *notes, "Canonical finalization reused prior retrieval artifacts."]
        updated_results.append(row)

    quality_metrics = updated_results
    config_models = [ConfigEvaluationResult.model_validate(item) for item in quality_metrics]
    best_config = _pick_best_configuration(config_models)
    official_gate = _build_official_gate(config_models, best_config)

    comparison_rows = _comparison_rows_from_config_models(config_models)
    _write_comparison_table_from_rows(run_dir / "comparison_table.csv", comparison_rows)
    write_json(run_dir / "quality_metrics.json", quality_metrics)
    write_json(run_dir / "official_ragas_gate.json", official_gate)

    evaluation_summary = build_evaluation_summary_json(
        scale_metrics=scale_metrics,
        comparison_rows=comparison_rows,
        quality_metrics=quality_metrics,
        latency_metrics=latency_metrics,
        best_config=best_config,
        official_gate=official_gate,
    )
    claims = classify_claims(
        summary=evaluation_summary,
        comparison_rows=comparison_rows,
        latency_metrics=latency_metrics,
    )
    bullet_candidates = build_project_summary_candidates(
        summary=evaluation_summary,
        comparison_rows=comparison_rows,
        claims=claims,
    )
    if len(bullet_candidates) < 3:
        raise RuntimeError(
            "Unable to produce at least 3 conservative project-summary lines from verified metrics."
        )

    _validate_final_acceptance(
        corpus_stats=corpus_stats,
        summary_payload=evaluation_summary,
        comparison_rows=comparison_rows,
        quality_metrics=quality_metrics,
        latency_metrics=latency_metrics,
    )

    corpus_stats_canonical = dict(corpus_stats)
    corpus_stats_canonical["total_evaluation_queries"] = scale_metrics.get("total_evaluation_queries", 0)
    corpus_stats_canonical["total_metadata_fields_used_for_filtering"] = scale_metrics.get(
        "total_metadata_fields_used_for_filtering",
        0,
    )
    write_json(final_dir / "corpus_stats.json", corpus_stats_canonical)
    write_json(final_dir / "evaluation_summary.json", evaluation_summary)
    shutil.copy2(run_dir / "comparison_table.csv", final_dir / "comparison_table.csv")

    latency_summary = {
        "best_configuration": best_config,
        "sync_vs_async_tools": latency_metrics.get("sync_vs_async_tools", {}),
        "rerank_on_vs_off": latency_metrics.get("rerank_on_vs_off", {}),
        "per_config_latency_ms": {
            row.get("config_name", "unknown"): row.get("latency_metrics", {}) for row in quality_metrics
        },
        "timings_by_stage_path": str((run_dir / "timings_by_stage.json").as_posix()),
    }
    write_json(final_dir / "latency_summary.json", latency_summary)
    (final_dir / "validated_claims.md").write_text(
        build_validated_claims_markdown(claims=claims, bullet_candidates=bullet_candidates),
        encoding="utf-8",
    )
    (final_dir / "final_report.md").write_text(
        build_final_report_markdown(
            summary=evaluation_summary,
            comparison_rows=comparison_rows,
            claims=claims,
            bullet_candidates=bullet_candidates,
        ),
        encoding="utf-8",
    )

    result = {
        "canonical_reports_dir": str(final_dir.as_posix()),
        "source_run_dir": str(run_dir.as_posix()),
        "best_configuration": best_config,
        "required_files": {
            "corpus_stats": str((final_dir / "corpus_stats.json").as_posix()),
            "evaluation_summary": str((final_dir / "evaluation_summary.json").as_posix()),
            "comparison_table": str((final_dir / "comparison_table.csv").as_posix()),
            "latency_summary": str((final_dir / "latency_summary.json").as_posix()),
            "validated_claims": str((final_dir / "validated_claims.md").as_posix()),
            "final_report": str((final_dir / "final_report.md").as_posix()),
        },
        "acceptance_checks": {
            "raw_documents": int(scale_metrics.get("total_indexed_documents", 0)),
            "indexed_chunks": int(scale_metrics.get("total_indexed_chunks", 0)),
            "evaluation_queries": int(scale_metrics.get("total_evaluation_queries", 0)),
            "best_config_official_ragas_complete": bool(
                evaluation_summary["official_ragas_validation"].get("best_config_official_ragas_complete")
            ),
        },
    }
    write_json(final_dir / "finalization_manifest.json", result)
    return result


def _pick_best_configuration(results: list[ConfigEvaluationResult]) -> str:
    scored: list[tuple[str, float]] = []
    for result in results:
        ragas_values = [
            value
            for key, value in result.ragas_metrics.items()
            if key in {"faithfulness", "answer_relevancy", "context_recall"} and isinstance(value, (int, float))
        ]
        quality_score = sum(ragas_values) / len(ragas_values) if ragas_values else 0.0
        retrieval_score = result.retrieval_metrics.get("hit_rate_at_k", 0.0) * 0.7 + result.retrieval_metrics.get(
            "mrr_at_k", 0.0
        ) * 0.3
        scored.append((result.config_name, quality_score * 0.6 + retrieval_score * 0.4))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[0][0] if scored else "n/a"


def _build_official_gate(
    results: list[ConfigEvaluationResult],
    best_configuration: str,
) -> dict[str, Any]:
    best = next((item for item in results if item.config_name == best_configuration), None)
    if best is None:
        return {"best_config_official_ragas_complete": False, "reason": "Best config not found."}
    if best.metric_source != "official_ragas":
        return {
            "best_config_official_ragas_complete": False,
            "reason": f"Best config `{best_configuration}` does not use official_ragas metrics.",
        }
    missing = [m for m in REQUIRED_RAGAS_METRICS if not isinstance(best.ragas_metrics.get(m), (int, float))]
    if missing:
        return {
            "best_config_official_ragas_complete": False,
            "reason": f"Missing numeric official metrics for {missing}",
        }
    return {"best_config_official_ragas_complete": True, "reason": "Official gate passed."}


def _comparison_rows_from_config_models(results: list[ConfigEvaluationResult]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in results:
        rows.append(
            {
                "config_name": row.config_name,
                "chunking_strategy": row.chunking_strategy,
                "retrieval_mode": row.retrieval_mode,
                "rerank": str(row.rerank),
                "top_k": str(row.top_k),
                "candidate_pool_size": str(row.candidate_pool_size),
                "fusion_alpha": str(row.fusion_alpha),
                "metric_source": row.metric_source,
                "faithfulness": str(row.ragas_metrics.get("faithfulness")),
                "answer_relevancy": str(row.ragas_metrics.get("answer_relevancy")),
                "context_recall": str(row.ragas_metrics.get("context_recall")),
                "context_precision": str(row.ragas_metrics.get("context_precision")),
                "hit_rate_at_k": str(row.retrieval_metrics.get("hit_rate_at_k")),
                "mrr_at_k": str(row.retrieval_metrics.get("mrr_at_k")),
                "citation_coverage": str(row.retrieval_metrics.get("citation_coverage")),
                "unsupported_claim_rate": str(row.retrieval_metrics.get("unsupported_claim_rate")),
                "avg_latency_ms": str(row.latency_metrics.get("avg_ms")),
                "p50_latency_ms": str(row.latency_metrics.get("p50_ms")),
                "p95_latency_ms": str(row.latency_metrics.get("p95_ms")),
            }
        )
    return rows


def _write_comparison_table_from_rows(path: Path, rows: list[dict[str, str]]) -> None:
    import csv

    fields = [
        "config_name",
        "chunking_strategy",
        "retrieval_mode",
        "rerank",
        "top_k",
        "candidate_pool_size",
        "fusion_alpha",
        "metric_source",
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
        "hit_rate_at_k",
        "mrr_at_k",
        "citation_coverage",
        "unsupported_claim_rate",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
