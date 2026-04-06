from __future__ import annotations

import re
from typing import Any

from app.core.config import Settings


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", (text or "").lower()))


def _safe_ratio(a: int, b: int) -> float:
    if b <= 0:
        return 0.0
    return a / b


def _fallback_quality_metrics(samples: list[dict[str, Any]]) -> dict[str, float]:
    faithfulness_scores: list[float] = []
    answer_relevancy_scores: list[float] = []
    context_recall_scores: list[float] = []
    context_precision_scores: list[float] = []

    for item in samples:
        answer = str(item.get("answer", ""))
        question = str(item.get("question", ""))
        ground_truth = str(item.get("ground_truth", ""))
        contexts = " ".join(item.get("contexts", []))

        ans_tokens = _tokens(answer)
        q_tokens = _tokens(question)
        gt_tokens = _tokens(ground_truth)
        ctx_tokens = _tokens(contexts)

        # Approximate faithfulness: answer content supported by context.
        faithfulness_scores.append(_safe_ratio(len(ans_tokens.intersection(ctx_tokens)), max(len(ans_tokens), 1)))
        # Approximate answer relevancy: answer overlaps expected answer and question.
        rel_a = _safe_ratio(len(ans_tokens.intersection(gt_tokens)), max(len(gt_tokens), 1))
        rel_q = _safe_ratio(len(ans_tokens.intersection(q_tokens)), max(len(q_tokens), 1))
        answer_relevancy_scores.append((rel_a + rel_q) / 2.0)
        # Approximate context recall: expected answer tokens covered by contexts.
        context_recall_scores.append(_safe_ratio(len(gt_tokens.intersection(ctx_tokens)), max(len(gt_tokens), 1)))
        # Approximate context precision: context tokens aligned to question intent.
        context_precision_scores.append(_safe_ratio(len(ctx_tokens.intersection(q_tokens)), max(len(ctx_tokens), 1)))

    return {
        "faithfulness": round(sum(faithfulness_scores) / max(len(faithfulness_scores), 1), 4),
        "answer_relevancy": round(sum(answer_relevancy_scores) / max(len(answer_relevancy_scores), 1), 4),
        "context_recall": round(sum(context_recall_scores) / max(len(context_recall_scores), 1), 4),
        "context_precision": round(sum(context_precision_scores) / max(len(context_precision_scores), 1), 4),
    }


def compute_quality_metrics(
    samples: list[dict[str, Any]],
    settings: Settings,
    ragas_mode: str = "auto",
) -> tuple[dict[str, float], str, list[str]]:
    notes: list[str] = []
    if not samples:
        return (
            {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_recall": 0.0,
                "context_precision": 0.0,
            },
            "fallback_quality",
            ["No samples available; emitted zero fallback metrics."],
        )

    mode = ragas_mode.lower().strip()
    if mode not in {"auto", "official", "fallback"}:
        mode = "auto"

    can_run_official = bool(settings.openai_api_key)

    if mode == "fallback":
        notes.append("RAGAs mode set to fallback.")
        return _fallback_quality_metrics(samples), "fallback_quality", notes

    if not can_run_official and mode == "official":
        raise RuntimeError("RAGAS_MODE=official requires OPENAI_API_KEY.")

    if can_run_official:
        try:
            from datasets import Dataset
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas import evaluate
            from ragas import metrics as ragas_metrics
            from ragas.run_config import RunConfig

            dataset = Dataset.from_list(samples)
            llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key, temperature=0)
            embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key,
            )
            run_cfg = RunConfig(timeout=180, max_retries=8, max_wait=45, max_workers=4)

            faithfulness_metric = getattr(ragas_metrics, "faithfulness", None)
            answer_metric = getattr(ragas_metrics, "answer_relevancy", None)
            if answer_metric is None:
                answer_metric = getattr(ragas_metrics, "response_relevancy", None)
            context_recall_metric = getattr(ragas_metrics, "context_recall", None)
            context_precision_metric = getattr(ragas_metrics, "context_precision", None)

            selected_metrics = [
                metric
                for metric in [
                    faithfulness_metric,
                    answer_metric,
                    context_recall_metric,
                    context_precision_metric,
                ]
                if metric is not None
            ]
            if len(selected_metrics) < 3:
                raise RuntimeError("RAGAs metric API mismatch: required metrics unavailable.")

            result = None
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    result = evaluate(
                        dataset=dataset,
                        metrics=selected_metrics,
                        llm=llm,
                        embeddings=embeddings,
                        run_config=run_cfg,
                        raise_exceptions=True,
                        show_progress=False,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    notes.append(f"Official RAGAs retry {attempt}/3 failed: {exc}")
            if result is None:
                raise RuntimeError(f"Official RAGAs retries exhausted: {last_error}")
            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                result_map: dict[str, float] = {}
                for col in df.columns:
                    if col == "question":
                        continue
                    series = df[col]
                    try:
                        import pandas as pd

                        numeric = pd.to_numeric(series, errors="coerce")
                        if numeric.notna().any():
                            result_map[col] = float(numeric.mean())
                    except Exception:  # noqa: BLE001
                        continue
            elif isinstance(result, dict):
                result_map = {k: float(v) for k, v in result.items()}
            else:
                result_map = {}

            answer_value = result_map.get("answer_relevancy")
            if answer_value is None:
                answer_value = result_map.get("response_relevancy")
            faithfulness_value = result_map.get("faithfulness")
            context_recall_value = result_map.get("context_recall")
            if faithfulness_value is None or answer_value is None or context_recall_value is None:
                raise RuntimeError(
                    "Official RAGAs output missing required metric values "
                    f"(faithfulness={faithfulness_value}, answer={answer_value}, context_recall={context_recall_value})."
                )

            metrics = {
                "faithfulness": float(faithfulness_value),
                "answer_relevancy": float(answer_value),
                "context_recall": float(context_recall_value),
                "context_precision": float(result_map.get("context_precision"))
                if result_map.get("context_precision") is not None
                else 0.0,
            }
            return metrics, "official_ragas", notes
        except Exception as exc:  # noqa: BLE001
            if mode == "official":
                raise RuntimeError(f"Official RAGAs execution failed: {exc}") from exc
            notes.append(f"Official RAGAs failed; using fallback metrics. Error: {exc}")
            return _fallback_quality_metrics(samples), "fallback_quality", notes

    notes.append("OPENAI_API_KEY missing; using fallback quality metrics.")
    return _fallback_quality_metrics(samples), "fallback_quality", notes
