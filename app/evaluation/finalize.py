from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_comparison_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_row(rows: list[dict[str, str]], config_name: str) -> dict[str, str]:
    for row in rows:
        if row.get("config_name") == config_name:
            return row
    raise ValueError(f"Config row `{config_name}` not found in comparison table.")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _delta(best: float, baseline: float) -> float:
    return best - baseline


def _delta_pct(best: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return ((best - baseline) / baseline) * 100.0


def build_evaluation_summary_json(
    *,
    scale_metrics: dict[str, Any],
    comparison_rows: list[dict[str, str]],
    quality_metrics: list[dict[str, Any]],
    latency_metrics: dict[str, Any],
    best_config: str,
    official_gate: dict[str, Any],
) -> dict[str, Any]:
    baseline_row = find_row(comparison_rows, "dense_only_baseline")
    best_row = find_row(comparison_rows, best_config)
    deltas = {
        "faithfulness_delta": round(
            _delta(_to_float(best_row.get("faithfulness")), _to_float(baseline_row.get("faithfulness"))), 4
        ),
        "answer_relevancy_delta": round(
            _delta(
                _to_float(best_row.get("answer_relevancy")),
                _to_float(baseline_row.get("answer_relevancy")),
            ),
            4,
        ),
        "context_recall_delta": round(
            _delta(_to_float(best_row.get("context_recall")), _to_float(baseline_row.get("context_recall"))), 4
        ),
        "hit_rate_at_k_delta": round(
            _delta(_to_float(best_row.get("hit_rate_at_k")), _to_float(baseline_row.get("hit_rate_at_k"))), 4
        ),
        "mrr_at_k_delta": round(
            _delta(_to_float(best_row.get("mrr_at_k")), _to_float(baseline_row.get("mrr_at_k"))), 4
        ),
        "unsupported_claim_rate_delta": round(
            _delta(
                _to_float(best_row.get("unsupported_claim_rate")),
                _to_float(baseline_row.get("unsupported_claim_rate")),
            ),
            4,
        ),
    }

    return {
        "scale_metrics": scale_metrics,
        "best_configuration": best_config,
        "official_ragas_validation": official_gate,
        "config_results": quality_metrics,
        "baseline_to_best_deltas": deltas,
        "latency": latency_metrics,
    }


def classify_claims(
    *,
    summary: dict[str, Any],
    comparison_rows: list[dict[str, str]],
    latency_metrics: dict[str, Any],
) -> dict[str, list[str]]:
    scale = summary["scale_metrics"]
    deltas = summary["baseline_to_best_deltas"]
    official_gate = summary["official_ragas_validation"]
    best = summary["best_configuration"]
    best_row = find_row(comparison_rows, best)

    verified: list[str] = []
    partial: list[str] = []
    unsupported: list[str] = []

    docs = int(scale.get("total_indexed_documents", 0))
    chunks = int(scale.get("total_indexed_chunks", 0))
    if docs >= 3000 or chunks >= 15000:
        verified.append(f"Corpus scale reached {docs:,} docs and {chunks:,} indexed chunks.")
    else:
        unsupported.append(f"Corpus scale insufficient: {docs:,} docs and {chunks:,} chunks.")

    if official_gate.get("best_config_official_ragas_complete"):
        verified.append(
            f"Best config `{best}` has numeric official RAGAs for faithfulness, answer_relevancy, context_recall."
        )
    else:
        unsupported.append(
            "Official RAGAs gate failed for best config (missing numeric required official metrics)."
        )

    if deltas.get("hit_rate_at_k_delta", 0.0) > 0 and deltas.get("mrr_at_k_delta", 0.0) > 0:
        verified.append(
            f"Retrieval improved from baseline: hit_rate@k delta {deltas['hit_rate_at_k_delta']:+.4f}, "
            f"MRR@k delta {deltas['mrr_at_k_delta']:+.4f}."
        )
    elif deltas.get("hit_rate_at_k_delta", 0.0) > 0 or deltas.get("mrr_at_k_delta", 0.0) > 0:
        partial.append("Retrieval improved on one metric but not all key retrieval metrics.")
    else:
        unsupported.append("No baseline-to-best retrieval improvement.")

    sync_avg = _to_float(latency_metrics.get("sync_vs_async_tools", {}).get("sync", {}).get("avg_ms"))
    async_avg = _to_float(latency_metrics.get("sync_vs_async_tools", {}).get("async", {}).get("avg_ms"))
    reduction_pct = _to_float(latency_metrics.get("sync_vs_async_tools", {}).get("avg_latency_reduction_pct"))
    if sync_avg > 0 and async_avg > 0 and reduction_pct > 0:
        verified.append(
            f"Async execution reduced avg latency by {reduction_pct:.2f}% (sync {sync_avg:.2f}ms -> async {async_avg:.2f}ms)."
        )
    elif sync_avg > 0 and async_avg > 0:
        partial.append("Sync/async measured but no latency reduction.")
    else:
        unsupported.append("Latency A/B metrics unavailable.")

    if deltas.get("faithfulness_delta", 0.0) > 0 and deltas.get("context_recall_delta", 0.0) > 0:
        verified.append(
            f"Official quality deltas are positive for faithfulness ({deltas['faithfulness_delta']:+.4f}) and "
            f"context_recall ({deltas['context_recall_delta']:+.4f})."
        )
    elif official_gate.get("best_config_official_ragas_complete"):
        partial.append("Official RAGAs available but not all key quality deltas are positive.")

    return {"verified": verified, "partially_supported": partial, "unsupported": unsupported}


def build_resume_bullet_candidates(
    *,
    summary: dict[str, Any],
    comparison_rows: list[dict[str, str]],
    claims: dict[str, list[str]],
) -> list[str]:
    scale = summary["scale_metrics"]
    best = summary["best_configuration"]
    best_row = find_row(comparison_rows, best)
    baseline_row = find_row(comparison_rows, "dense_only_baseline")
    latency = summary["latency"]

    bullets: list[str] = []
    if any("Corpus scale reached" in item for item in claims["verified"]):
        bullets.append(
            f"Architected a LangGraph multi-agent real-estate research system indexing {int(scale['total_indexed_chunks']):,}+ property passages with citation-backed outputs."
        )
    if any("Retrieval improved from baseline" in item for item in claims["verified"]):
        bullets.append(
            f"Implemented hybrid retrieval and reranking, improving hit_rate@k from {baseline_row['hit_rate_at_k']} to {best_row['hit_rate_at_k']} and MRR@k from {baseline_row['mrr_at_k']} to {best_row['mrr_at_k']}."
        )
    if any("Async execution reduced avg latency" in item for item in claims["verified"]):
        red = _to_float(latency.get("sync_vs_async_tools", {}).get("avg_latency_reduction_pct"))
        bullets.append(
            f"Instrumented per-agent latency benchmarking and reduced average response latency by {red:.2f}% through async tool execution."
        )
    if any("Best config" in item for item in claims["verified"]):
        bullets.append(
            f"Ran official RAGAs evaluation for four retrieval configurations and selected `{summary['best_configuration']}` with numeric faithfulness, answer_relevancy, and context_recall."
        )
    if len(bullets) < 3 and any("Best config" in item for item in claims["verified"]):
        bullets.append(
            f"Benchmarked dense-only, hybrid, hybrid+rerank, and hybrid+rerank+semantic variants, selecting `{summary['best_configuration']}` using reproducible quality and retrieval metrics."
        )
    if len(bullets) < 3 and any("Retrieval improved from baseline" in item for item in claims["verified"]):
        bullets.append(
            "Produced baseline-to-best retrieval deltas with hit_rate@k and MRR@k improvements backed by canonical comparison artifacts."
        )
    deduped: list[str] = []
    for bullet in bullets:
        if bullet not in deduped:
            deduped.append(bullet)
    return deduped[:5]


def build_resume_claim_support_markdown(
    *,
    claims: dict[str, list[str]],
    bullet_candidates: list[str],
) -> str:
    lines = [
        "# Resume Claim Support",
        "",
        "## Verified claims",
    ]
    lines.extend([f"- {item}" for item in claims["verified"]] or ["- None"])
    lines.extend(["", "## Partially supported claims"])
    lines.extend([f"- {item}" for item in claims["partially_supported"]] or ["- None"])
    lines.extend(["", "## Unsupported claims"])
    lines.extend([f"- {item}" for item in claims["unsupported"]] or ["- None"])
    lines.extend(["", "## Conservative resume-safe bullet candidates"])
    lines.extend([f"- {item}" for item in bullet_candidates[:5]])
    return "\n".join(lines) + "\n"


def build_final_report_markdown(
    *,
    summary: dict[str, Any],
    comparison_rows: list[dict[str, str]],
    claims: dict[str, list[str]],
    bullet_candidates: list[str],
) -> str:
    scale = summary["scale_metrics"]
    best_name = summary["best_configuration"]
    best_row = find_row(comparison_rows, best_name)
    deltas = summary["baseline_to_best_deltas"]
    latency = summary["latency"]
    sync_to_async = latency.get("sync_vs_async_tools", {}).get("avg_latency_reduction_pct", 0.0)
    rerank_delta = latency.get("rerank_on_vs_off", {}).get("avg_latency_delta_pct", 0.0)

    conservative_wording = bullet_candidates[0] if bullet_candidates else "No verified resume wording available."
    lines = [
        "# Final Canonical Report",
        "",
        "| corpus_scale | best_config_name | official_ragas_values | retrieval_improvement_deltas | latency_improvement_deltas | conservative_verified_resume_wording |",
        "|---|---|---|---|---|---|",
        (
            f"| docs={int(scale['total_indexed_documents'])}, chunks={int(scale['total_indexed_chunks'])}, queries={int(scale['total_evaluation_queries'])} "
            f"| {best_name} "
            f"| faithfulness={best_row['faithfulness']}, answer_relevancy={best_row['answer_relevancy']}, context_recall={best_row['context_recall']} "
            f"| hit_rate@k={deltas['hit_rate_at_k_delta']:+.4f}, mrr@k={deltas['mrr_at_k_delta']:+.4f}, faithfulness={deltas['faithfulness_delta']:+.4f}, context_recall={deltas['context_recall_delta']:+.4f} "
            f"| sync_to_async={float(sync_to_async):+.2f}%, rerank_on_vs_off={float(rerank_delta):+.2f}% "
            f"| {conservative_wording} |"
        ),
    ]
    return "\n".join(lines) + "\n"
