from __future__ import annotations

import asyncio
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from app.core.config import Settings
from app.evaluation.dataset import load_eval_queries
from app.evaluation.ragas_runner import compute_quality_metrics
from app.graph.workflow import RealEstateResearchSystem
from app.ingestion.pipeline import IngestionPipeline
from app.schemas.api import QueryRequest
from app.schemas.evaluation import ConfigEvaluationResult, EvaluationRunSummary
from app.schemas.retrieval import MetadataFilter


@dataclass
class EvalConfig:
    name: str
    retrieval_mode: str
    rerank: bool
    chunking_strategy: str
    top_k: int
    candidate_pool_size: int
    fusion_alpha: float
    rerank_top_n: int
    async_tools: bool
    semantic_chunking: bool


DEFAULT_CONFIGS = [
    EvalConfig(
        name="dense_only_baseline",
        retrieval_mode="dense",
        rerank=False,
        chunking_strategy="fixed",
        top_k=4,
        candidate_pool_size=10,
        fusion_alpha=1.0,
        rerank_top_n=8,
        async_tools=False,
        semantic_chunking=False,
    ),
    EvalConfig(
        name="hybrid_baseline",
        retrieval_mode="hybrid",
        rerank=False,
        chunking_strategy="fixed",
        top_k=6,
        candidate_pool_size=24,
        fusion_alpha=0.55,
        rerank_top_n=12,
        async_tools=False,
        semantic_chunking=False,
    ),
    EvalConfig(
        name="hybrid_rerank",
        retrieval_mode="hybrid",
        rerank=True,
        chunking_strategy="semantic",
        top_k=8,
        candidate_pool_size=36,
        fusion_alpha=0.62,
        rerank_top_n=24,
        async_tools=True,
        semantic_chunking=True,
    ),
    EvalConfig(
        name="hybrid_rerank_semantic",
        retrieval_mode="hybrid",
        rerank=True,
        chunking_strategy="section_semantic",
        top_k=8,
        candidate_pool_size=28,
        fusion_alpha=0.58,
        rerank_top_n=20,
        async_tools=True,
        semantic_chunking=True,
    ),
]


class EvaluationRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ingestion = IngestionPipeline(settings)

    async def run(
        self,
        run_ragas: bool = True,
        ragas_mode: str | None = None,
        dataset_profile: str | None = None,
        max_queries: int | None = None,
        output_tag: str | None = None,
    ) -> EvaluationRunSummary:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{output_tag}" if output_tag else timestamp
        run_dir = self.settings.eval_output_path / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        profile = dataset_profile or self.settings.dataset_profile
        ragas_mode_resolved = (ragas_mode or self.settings.ragas_mode).lower().strip()
        if (
            run_ragas
            and self.settings.require_official_ragas_best_config
            and ragas_mode_resolved in {"auto", "official"}
            and not self.settings.openai_api_key
        ):
            raise RuntimeError(
                "Official RAGAs is required for best-config validation, but OPENAI_API_KEY is missing."
            )

        self.ingestion.ensure_sample_data(dataset_profile=profile)
        eval_queries = load_eval_queries(self.settings.data_path / "eval" / "eval_queries.jsonl")
        if max_queries:
            eval_queries = eval_queries[:max_queries]
        if len(eval_queries) < 50:
            self.ingestion.ensure_sample_data(dataset_profile=profile, eval_queries=80)
            eval_queries = load_eval_queries(self.settings.data_path / "eval" / "eval_queries.jsonl")
            if max_queries:
                eval_queries = eval_queries[:max_queries]

        results: list[ConfigEvaluationResult] = []
        quality_payloads: dict[str, list[dict[str, Any]]] = {}
        per_config_timings: dict[str, list[dict[str, float]]] = {}
        ingest_by_strategy: dict[str, Any] = {}

        scale_stats = None
        for config in DEFAULT_CONFIGS:
            if config.chunking_strategy not in ingest_by_strategy:
                ingest_by_strategy[config.chunking_strategy] = self.ingestion.run(
                    semantic_chunking=config.semantic_chunking,
                    chunking_strategy=config.chunking_strategy,
                    dataset_profile=profile,
                )
            scale_stats = ingest_by_strategy[config.chunking_strategy]
            system = RealEstateResearchSystem(self.settings)
            (
                retrieval_metrics,
                latency_metrics,
                quality_samples,
                timings_by_stage,
            ) = await self._evaluate_config(system, eval_queries, config)

            if run_ragas:
                quality_metrics, metric_source, notes = compute_quality_metrics(
                    quality_samples,
                    settings=self.settings,
                    ragas_mode=ragas_mode_resolved,
                )
            else:
                quality_metrics, metric_source, notes = compute_quality_metrics(
                    quality_samples,
                    settings=self.settings,
                    ragas_mode="fallback",
                )
                notes.append("run_ragas=false, using fallback quality metrics only.")

            results.append(
                ConfigEvaluationResult(
                    config_name=config.name,
                    retrieval_mode=config.retrieval_mode,
                    rerank=config.rerank,
                    semantic_chunking=config.semantic_chunking,
                    chunking_strategy=config.chunking_strategy,
                    top_k=config.top_k,
                    candidate_pool_size=config.candidate_pool_size,
                    fusion_alpha=config.fusion_alpha,
                    metric_source=metric_source,
                    ragas_metrics=quality_metrics,
                    retrieval_metrics=retrieval_metrics,
                    latency_metrics=latency_metrics,
                    notes=notes,
                )
            )
            quality_payloads[config.name] = quality_samples
            per_config_timings[config.name] = timings_by_stage

        best_configuration = self._pick_best_configuration(results)
        scale_metrics = {
            "total_indexed_documents": int(scale_stats.total_documents if scale_stats else 0),
            "total_indexed_chunks": int(scale_stats.total_chunks if scale_stats else 0),
            "total_evaluation_queries": len(eval_queries),
            "total_metadata_fields_used_for_filtering": int(
                len(scale_stats.metadata_fields_used) if scale_stats else 0
            ),
        }

        official_gate = self._validate_official_ragas_gate(
            results=results,
            best_configuration=best_configuration,
            run_ragas=run_ragas,
        )
        if (
            run_ragas
            and self.settings.require_official_ragas_best_config
            and not official_gate["best_config_official_ragas_complete"]
        ):
            raise RuntimeError(official_gate["reason"])

        latency_experiments = await self._latency_experiments(eval_queries, profile)
        corpus_stats = self._load_corpus_stats()

        self._write_json(run_dir / "corpus_stats.json", corpus_stats)
        self._write_json(run_dir / "scale_metrics.json", scale_metrics)
        self._write_json(run_dir / "quality_metrics.json", [result.model_dump() for result in results])
        self._write_json(run_dir / "latency_metrics.json", latency_experiments)
        self._write_json(run_dir / "timings_by_stage.json", per_config_timings)
        self._write_json(run_dir / "quality_samples.json", quality_payloads)
        self._write_json(run_dir / "official_ragas_gate.json", official_gate)
        self._write_comparison_table(run_dir / "comparison_table.csv", results)
        self._write_evaluation_summary(
            run_dir / "evaluation_report.md",
            scale_metrics,
            results,
            latency_experiments,
            best_configuration,
            official_gate,
        )
        self._write_evaluation_summary(
            self.settings.reports_path / "evaluation_summary.md",
            scale_metrics,
            results,
            latency_experiments,
            best_configuration,
            official_gate,
        )
        self._write_resume_claim_support(
            run_dir / "resume_claim_support.md",
            scale_metrics,
            results,
            latency_experiments,
            best_configuration,
            official_gate,
        )
        self._write_resume_claim_support(
            self.settings.reports_path / "resume_claim_support.md",
            scale_metrics,
            results,
            latency_experiments,
            best_configuration,
            official_gate,
        )

        return EvaluationRunSummary(
            run_dir=str(run_dir),
            scale_metrics=scale_metrics,
            config_results=results,
            best_configuration=best_configuration,
        )

    async def _evaluate_config(
        self,
        system: RealEstateResearchSystem,
        eval_queries,
        config: EvalConfig,
    ) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]], list[dict[str, float]]]:
        hit_flags: list[int] = []
        mrr_scores: list[float] = []
        citation_coverages: list[float] = []
        unsupported_rates: list[float] = []
        latencies: list[float] = []
        timings_by_stage: list[dict[str, float]] = []
        quality_samples: list[dict[str, Any]] = []

        for item in eval_queries:
            request = QueryRequest(
                query=item.query,
                retrieval_mode=config.retrieval_mode,
                rerank=config.rerank,
                rerank_top_n=config.rerank_top_n,
                async_tools=config.async_tools,
                top_k=config.top_k,
                candidate_pool_size=config.candidate_pool_size,
                fusion_alpha=config.fusion_alpha,
                filters=MetadataFilter() if config.name == "dense_only_baseline" else None,
                include_debug=True,
            )
            started = datetime.utcnow()
            state = await system.run_query_state(request)
            elapsed_ms = (datetime.utcnow() - started).total_seconds() * 1000.0
            latencies.append(elapsed_ms)
            timings_by_stage.append(state.get("timings_ms", {}))

            final = state["final_answer"]
            relevant_doc_ids = set(item.expected_doc_ids)
            retrieved_doc_order = [chunk.doc_id for chunk in state.get("retrieved_chunks", [])]
            citation_doc_ids = [
                c.doc_id for c in final.citations if c.doc_id and c.doc_id not in {"NO_DOC", "UNKNOWN"}
            ]

            hit_flags.append(
                int(any(doc_id in relevant_doc_ids for doc_id in retrieved_doc_order[: config.top_k]))
            )
            mrr_scores.append(self._reciprocal_rank(retrieved_doc_order, relevant_doc_ids))
            citation_coverages.append(self._citation_coverage(citation_doc_ids, relevant_doc_ids))

            claim_count = max(len(state["analyst_draft"].major_claims), 1)
            unsupported_count = len(state["fact_check_report"].unsupported_claims) + len(
                state["fact_check_report"].weak_claims
            )
            unsupported_rates.append(unsupported_count / claim_count)

            quality_samples.append(
                {
                    "question": item.query,
                    "answer": final.concise_answer,
                    "contexts": [chunk.text for chunk in state.get("retrieved_chunks", [])[:8]],
                    "ground_truth": item.reference_answer,
                }
            )

        retrieval_metrics = {
            "hit_rate_at_k": round(float(mean(hit_flags)) if hit_flags else 0.0, 4),
            "mrr_at_k": round(float(mean(mrr_scores)) if mrr_scores else 0.0, 4),
            "citation_coverage": round(float(mean(citation_coverages)) if citation_coverages else 0.0, 4),
            "unsupported_claim_rate": round(float(mean(unsupported_rates)) if unsupported_rates else 0.0, 4),
        }
        latency_metrics = self._summary_stats(latencies)
        return retrieval_metrics, latency_metrics, quality_samples, timings_by_stage

    async def _latency_experiments(self, eval_queries, dataset_profile: str) -> dict[str, Any]:
        eval_subset = eval_queries[: min(30, len(eval_queries))]
        experiments: dict[str, Any] = {"sync_vs_async_tools": {}, "rerank_on_vs_off": {}}

        self.ingestion.run(
            semantic_chunking=True,
            chunking_strategy="section_semantic",
            dataset_profile=dataset_profile,
        )
        system = RealEstateResearchSystem(self.settings)

        sync_lat = await self._run_latency_probe(
            system,
            eval_subset,
            retrieval_mode="hybrid",
            rerank=True,
            async_tools=False,
            top_k=8,
            candidate_pool_size=36,
        )
        async_lat = await self._run_latency_probe(
            system,
            eval_subset,
            retrieval_mode="hybrid",
            rerank=True,
            async_tools=True,
            top_k=8,
            candidate_pool_size=36,
        )
        experiments["sync_vs_async_tools"] = {
            "sync": sync_lat,
            "async": async_lat,
            "avg_latency_reduction_pct": round(self._delta_pct(sync_lat["avg_ms"], async_lat["avg_ms"]), 2),
        }

        rerank_off = await self._run_latency_probe(
            system,
            eval_subset,
            retrieval_mode="hybrid",
            rerank=False,
            async_tools=True,
            top_k=8,
            candidate_pool_size=36,
        )
        rerank_on = await self._run_latency_probe(
            system,
            eval_subset,
            retrieval_mode="hybrid",
            rerank=True,
            async_tools=True,
            top_k=8,
            candidate_pool_size=36,
        )
        experiments["rerank_on_vs_off"] = {
            "rerank_off": rerank_off,
            "rerank_on": rerank_on,
            "avg_latency_delta_pct": round(self._delta_pct(rerank_off["avg_ms"], rerank_on["avg_ms"]), 2),
        }
        return experiments

    async def _run_latency_probe(
        self,
        system: RealEstateResearchSystem,
        eval_queries,
        retrieval_mode: str,
        rerank: bool,
        async_tools: bool,
        top_k: int,
        candidate_pool_size: int,
    ) -> dict[str, Any]:
        total_latencies: list[float] = []
        stage_samples: dict[str, list[float]] = {}
        retrieval_ms: list[float] = []
        rerank_ms: list[float] = []

        for item in eval_queries:
            req = QueryRequest(
                query=item.query,
                retrieval_mode=retrieval_mode,
                rerank=rerank,
                async_tools=async_tools,
                top_k=top_k,
                candidate_pool_size=candidate_pool_size,
                filters=MetadataFilter(),
                include_debug=True,
            )
            t0 = datetime.utcnow()
            state = await system.run_query_state(req)
            total_latencies.append((datetime.utcnow() - t0).total_seconds() * 1000.0)
            for stage, ms in state.get("timings_ms", {}).items():
                stage_samples.setdefault(stage, []).append(ms)
            if state.get("retrieval_debug"):
                debug = state["retrieval_debug"]
                retrieval_ms.append(
                    debug.timings_ms.get("dense_ms", 0.0)
                    + debug.timings_ms.get("sparse_ms", 0.0)
                    + debug.timings_ms.get("fusion_ms", 0.0)
                )
                rerank_ms.append(debug.timings_ms.get("rerank_ms", 0.0))

        return {
            **self._summary_stats(total_latencies),
            "retrieval_time_ms": self._summary_stats(retrieval_ms),
            "reranking_time_ms": self._summary_stats(rerank_ms),
            "per_agent_timing_ms": {
                stage: self._summary_stats(values) for stage, values in stage_samples.items()
            },
        }

    @staticmethod
    def _summary_stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        return {
            "avg_ms": round(float(np.mean(values)), 2),
            "p50_ms": round(float(np.percentile(values, 50)), 2),
            "p95_ms": round(float(np.percentile(values, 95)), 2),
        }

    @staticmethod
    def _reciprocal_rank(retrieved_doc_order: list[str], relevant_doc_ids: set[str]) -> float:
        if not relevant_doc_ids:
            return 0.0
        for idx, doc_id in enumerate(retrieved_doc_order, start=1):
            if doc_id in relevant_doc_ids:
                return 1.0 / idx
        return 0.0

    @staticmethod
    def _citation_coverage(cited_doc_ids: list[str], relevant_doc_ids: set[str]) -> float:
        if not relevant_doc_ids:
            return 1.0 if cited_doc_ids else 0.0
        return len(set(cited_doc_ids).intersection(relevant_doc_ids)) / max(len(relevant_doc_ids), 1)

    @staticmethod
    def _pick_best_configuration(results: list[ConfigEvaluationResult]) -> str:
        scored: list[tuple[str, float]] = []
        for result in results:
            ragas_values = [
                value
                for key, value in result.ragas_metrics.items()
                if key in {"faithfulness", "answer_relevancy", "context_recall"} and isinstance(value, (int, float))
            ]
            quality_score = float(np.mean(ragas_values)) if ragas_values else 0.0
            retrieval_score = result.retrieval_metrics.get("hit_rate_at_k", 0.0) * 0.7 + result.retrieval_metrics.get(
                "mrr_at_k", 0.0
            ) * 0.3
            scored.append((result.config_name, quality_score * 0.6 + retrieval_score * 0.4))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[0][0] if scored else "n/a"

    def _load_corpus_stats(self) -> dict[str, Any]:
        path = self.settings.reports_path / "corpus_stats.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _validate_official_ragas_gate(
        results: list[ConfigEvaluationResult],
        best_configuration: str,
        run_ragas: bool,
    ) -> dict[str, Any]:
        required = ["faithfulness", "answer_relevancy", "context_recall"]
        best = next((item for item in results if item.config_name == best_configuration), None)
        if not run_ragas:
            return {"best_config_official_ragas_complete": False, "reason": "run_ragas=false."}
        if best is None:
            return {"best_config_official_ragas_complete": False, "reason": "Best config not found."}
        if best.metric_source != "official_ragas":
            return {
                "best_config_official_ragas_complete": False,
                "reason": f"Best config `{best_configuration}` does not use official_ragas metrics.",
            }
        missing = [m for m in required if not isinstance(best.ragas_metrics.get(m), (int, float))]
        if missing:
            return {
                "best_config_official_ragas_complete": False,
                "reason": f"Missing numeric official metrics for {missing}",
            }
        return {"best_config_official_ragas_complete": True, "reason": "Official gate passed."}

    @staticmethod
    def _delta_pct(baseline: float, value: float) -> float:
        if baseline == 0:
            return 0.0
        return ((baseline - value) / baseline) * 100.0

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _write_comparison_table(path: Path, results: list[ConfigEvaluationResult]) -> None:
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
            for row in results:
                writer.writerow(
                    {
                        "config_name": row.config_name,
                        "chunking_strategy": row.chunking_strategy,
                        "retrieval_mode": row.retrieval_mode,
                        "rerank": row.rerank,
                        "top_k": row.top_k,
                        "candidate_pool_size": row.candidate_pool_size,
                        "fusion_alpha": row.fusion_alpha,
                        "metric_source": row.metric_source,
                        "faithfulness": row.ragas_metrics.get("faithfulness"),
                        "answer_relevancy": row.ragas_metrics.get("answer_relevancy"),
                        "context_recall": row.ragas_metrics.get("context_recall"),
                        "context_precision": row.ragas_metrics.get("context_precision"),
                        "hit_rate_at_k": row.retrieval_metrics.get("hit_rate_at_k"),
                        "mrr_at_k": row.retrieval_metrics.get("mrr_at_k"),
                        "citation_coverage": row.retrieval_metrics.get("citation_coverage"),
                        "unsupported_claim_rate": row.retrieval_metrics.get("unsupported_claim_rate"),
                        "avg_latency_ms": row.latency_metrics.get("avg_ms"),
                        "p50_latency_ms": row.latency_metrics.get("p50_ms"),
                        "p95_latency_ms": row.latency_metrics.get("p95_ms"),
                    }
                )

    def _write_evaluation_summary(
        self,
        path: Path,
        scale_metrics: dict[str, Any],
        results: list[ConfigEvaluationResult],
        latency_experiments: dict[str, Any],
        best_configuration: str,
        official_gate: dict[str, Any],
    ) -> None:
        baseline = next((r for r in results if r.config_name == "dense_only_baseline"), None)
        best = next((r for r in results if r.config_name == best_configuration), None)
        lines = ["# Evaluation Summary", "", "## Scale Metrics"]
        for key, value in scale_metrics.items():
            lines.append(f"- {key}: {value}")

        lines += [
            "",
            "## Config Comparison",
            "| Config | Source | Faithfulness | Answer Relevancy | Context Recall | Hit@k | MRR@k | Unsupported Claim Rate |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for result in results:
            lines.append(
                f"| {result.config_name} | {result.metric_source} | {result.ragas_metrics.get('faithfulness')} | "
                f"{result.ragas_metrics.get('answer_relevancy')} | {result.ragas_metrics.get('context_recall')} | "
                f"{result.retrieval_metrics.get('hit_rate_at_k')} | {result.retrieval_metrics.get('mrr_at_k')} | "
                f"{result.retrieval_metrics.get('unsupported_claim_rate')} |"
            )
        lines.append("")
        lines.append(f"Best configuration: **{best_configuration}**")
        lines.append(f"Official RAGAs gate: **{official_gate.get('best_config_official_ragas_complete')}**")
        lines.append(f"Gate reason: {official_gate.get('reason')}")

        if baseline and best:
            lines += ["", "## Baseline to Best Deltas"]
            for metric in ["faithfulness", "answer_relevancy", "context_recall"]:
                b = baseline.ragas_metrics.get(metric)
                t = best.ragas_metrics.get(metric)
                if isinstance(b, (int, float)) and isinstance(t, (int, float)):
                    lines.append(f"- {metric}: {b:.4f} -> {t:.4f} (delta {t-b:+.4f})")
            lines.append(
                f"- hit_rate_at_k: {baseline.retrieval_metrics.get('hit_rate_at_k')} -> {best.retrieval_metrics.get('hit_rate_at_k')}"
            )
            lines.append(
                f"- mrr_at_k: {baseline.retrieval_metrics.get('mrr_at_k')} -> {best.retrieval_metrics.get('mrr_at_k')}"
            )
            lines.append(
                f"- unsupported_claim_rate: {baseline.retrieval_metrics.get('unsupported_claim_rate')} -> {best.retrieval_metrics.get('unsupported_claim_rate')}"
            )

        lines += [
            "",
            "## Latency Experiments",
            f"- sync_vs_async_tools avg reduction (%): {latency_experiments['sync_vs_async_tools'].get('avg_latency_reduction_pct')}",
            f"- rerank_on_vs_off avg delta (%): {latency_experiments['rerank_on_vs_off'].get('avg_latency_delta_pct')}",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_resume_claim_support(
        self,
        path: Path,
        scale_metrics: dict[str, Any],
        results: list[ConfigEvaluationResult],
        latency_experiments: dict[str, Any],
        best_configuration: str,
        official_gate: dict[str, Any],
    ) -> None:
        baseline = next((r for r in results if r.config_name == "dense_only_baseline"), None)
        best = next((r for r in results if r.config_name == best_configuration), None)
        sync_avg = latency_experiments.get("sync_vs_async_tools", {}).get("sync", {}).get("avg_ms", 0.0)
        async_avg = latency_experiments.get("sync_vs_async_tools", {}).get("async", {}).get("avg_ms", 0.0)
        async_reduction = latency_experiments.get("sync_vs_async_tools", {}).get(
            "avg_latency_reduction_pct", 0.0
        )

        claim_rows: list[tuple[str, str, str]] = []
        chunks = int(scale_metrics.get("total_indexed_chunks", 0))
        docs = int(scale_metrics.get("total_indexed_documents", 0))
        if chunks >= 15000:
            claim_rows.append(("Verified", f"Indexed {chunks:,} property-related passages/chunks.", "scale_metrics.json"))
        elif chunks >= 8000:
            claim_rows.append(("Partially supported", f"Indexed {chunks:,} passages; below 15K target.", "scale_metrics.json"))
        else:
            claim_rows.append(("Unsupported", f"Indexed only {chunks:,} passages.", "scale_metrics.json"))

        if baseline and best:
            hit_delta = best.retrieval_metrics.get("hit_rate_at_k", 0.0) - baseline.retrieval_metrics.get(
                "hit_rate_at_k", 0.0
            )
            if hit_delta >= 0.1:
                claim_rows.append(("Verified", f"Improved retrieval hit_rate@k by {hit_delta:+.3f} from dense baseline.", "comparison_table.csv"))
            elif hit_delta > 0:
                claim_rows.append(("Partially supported", f"Small retrieval improvement hit_rate@k {hit_delta:+.3f}.", "comparison_table.csv"))
            else:
                claim_rows.append(("Unsupported", "No retrieval improvement over dense baseline.", "comparison_table.csv"))

        if official_gate.get("best_config_official_ragas_complete") and baseline and best:
            ragas_delta = np.mean(
                [
                    float(best.ragas_metrics[k]) - float(baseline.ragas_metrics[k])
                    for k in ["faithfulness", "answer_relevancy", "context_recall"]
                ]
            )
            if ragas_delta > 0:
                claim_rows.append(("Verified", f"Official RAGAs composite improved by {ragas_delta:+.4f} from baseline.", "quality_metrics.json"))
            else:
                claim_rows.append(("Unsupported", "Official RAGAs did not improve over baseline.", "quality_metrics.json"))
        else:
            claim_rows.append(("Unsupported", "Best config lacks numeric official RAGAs for required metrics.", "official_ragas_gate.json"))

        if sync_avg > 0 and async_avg > 0 and async_reduction > 0:
            claim_rows.append(("Verified", f"Reduced average latency by {async_reduction:.2f}% using async tool execution.", "latency_metrics.json"))
        elif sync_avg > 0 and async_avg > 0:
            claim_rows.append(("Partially supported", "Measured sync/async latency, but no reduction observed.", "latency_metrics.json"))
        else:
            claim_rows.append(("Unsupported", "Missing sync/async latency measurements.", "latency_metrics.json"))

        lines = [
            "# Resume Claim Support",
            "",
            "## Claim Verification Matrix",
            "| Status | Claim | Evidence Source |",
            "|---|---|---|",
        ]
        for status, claim, source in claim_rows:
            lines.append(f"| {status} | {claim} | {source} |")

        lines += [
            "",
            "## Verified Metrics Snapshot",
            f"- indexed_documents: {docs}",
            f"- indexed_chunks: {chunks}",
            f"- best_configuration: {best_configuration}",
            f"- official_ragas_gate: {official_gate.get('best_config_official_ragas_complete')}",
            f"- sync_avg_ms: {sync_avg}",
            f"- async_avg_ms: {async_avg}",
            f"- async_latency_reduction_pct: {async_reduction}",
            "",
            "## Conservative Resume Bullet Candidates",
        ]
        for bullet in self._resume_bullet_candidates(
            claim_rows=claim_rows,
            scale_metrics=scale_metrics,
            baseline=baseline,
            best=best,
            async_reduction=async_reduction,
        ):
            lines.append(f"- {bullet}")

        path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _resume_bullet_candidates(
        claim_rows: list[tuple[str, str, str]],
        scale_metrics: dict[str, Any],
        baseline: ConfigEvaluationResult | None,
        best: ConfigEvaluationResult | None,
        async_reduction: float,
    ) -> list[str]:
        verified_claims = {claim for status, claim, _ in claim_rows if status == "Verified"}
        bullets: list[str] = []

        if any("Indexed" in claim for claim in verified_claims):
            bullets.append(
                f"Architected a LangGraph multi-agent real estate research system indexing {int(scale_metrics.get('total_indexed_chunks', 0)):,}+ property passages with citation-backed responses."
            )
        if baseline and best and any("retrieval hit_rate@k" in claim for claim in verified_claims):
            bullets.append(
                f"Implemented hybrid retrieval (dense + BM25 + reranking + chunking tuning), improving hit_rate@k from {baseline.retrieval_metrics.get('hit_rate_at_k')} to {best.retrieval_metrics.get('hit_rate_at_k')}."
            )
        if baseline and best and any("Official RAGAs composite improved" in claim for claim in verified_claims):
            bullets.append(
                f"Built an official RAGAs harness and improved faithfulness/relevancy/recall from dense baseline to {best.config_name}."
            )
        if any("Reduced average latency" in claim for claim in verified_claims):
            bullets.append(
                f"Instrumented per-agent latency tracing and reduced average response latency by {async_reduction:.2f}% through async execution."
            )
        if not bullets:
            bullets.append(
                "Implemented reproducible multi-agent retrieval and evaluation pipelines with conservative, artifact-backed reporting."
            )
        return bullets[:5]


def run_evaluation_sync(
    settings: Settings,
    run_ragas: bool = True,
    ragas_mode: str | None = None,
    dataset_profile: str | None = None,
    max_queries: int | None = None,
    output_tag: str | None = None,
) -> EvaluationRunSummary:
    runner = EvaluationRunner(settings)
    return asyncio.run(
        runner.run(
            run_ragas=run_ragas,
            ragas_mode=ragas_mode,
            dataset_profile=dataset_profile,
            max_queries=max_queries,
            output_tag=output_tag,
        )
    )
