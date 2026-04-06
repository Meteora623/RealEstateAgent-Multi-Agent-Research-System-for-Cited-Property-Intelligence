from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # noqa: BLE001
    END = "END"
    START = "START"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False

from app.agents.analyst import AnalystAgent
from app.agents.fact_checker import FactCheckerAgent
from app.agents.retriever import RetrieverAgent
from app.agents.supervisor import SupervisorAgent
from app.core.config import Settings
from app.core.logging import get_logger
from app.retrieval.service import HybridRetriever
from app.schemas.agent import FinalAnswer
from app.schemas.api import QueryRequest, QueryResponse, StructuredQueryResponse
from app.schemas.state import GraphState

logger = get_logger(__name__)


@dataclass
class AgentStack:
    supervisor: SupervisorAgent
    retriever: RetrieverAgent
    analyst: AnalystAgent
    fact_checker: FactCheckerAgent


class _FallbackGraph:
    def __init__(self, system: "RealEstateResearchSystem"):
        self.system = system

    async def ainvoke(self, state: GraphState) -> GraphState:
        state = self.system._supervisor_node(state)
        while True:
            state = self.system._retriever_node(state)
            state = await self.system._analyst_node(state)
            state = self.system._fact_checker_node(state)
            route = self.system._route_after_fact_check(state)
            if route == "retriever":
                continue
            state = self.system._finalize_node(state)
            return state


class RealEstateResearchSystem:
    def __init__(self, settings: Settings):
        self.settings = settings
        retriever = HybridRetriever(settings=settings)
        self.stack = AgentStack(
            supervisor=SupervisorAgent(),
            retriever=RetrieverAgent(retriever),
            analyst=AnalystAgent(settings),
            fact_checker=FactCheckerAgent(),
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        if not LANGGRAPH_AVAILABLE:
            logger.warning("langgraph is not installed; using sequential fallback graph.")
            return _FallbackGraph(self)

        graph = StateGraph(GraphState)
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("retriever", self._retriever_node)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("fact_checker", self._fact_checker_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "supervisor")
        graph.add_edge("supervisor", "retriever")
        graph.add_edge("retriever", "analyst")
        graph.add_edge("analyst", "fact_checker")
        graph.add_conditional_edges(
            "fact_checker",
            self._route_after_fact_check,
            {
                "retriever": "retriever",
                "finalize": "finalize",
            },
        )
        graph.add_edge("finalize", END)
        return graph.compile()

    @staticmethod
    def _timed(state: GraphState, stage: str) -> float:
        state.setdefault("timings_ms", {})
        return time.perf_counter()

    @staticmethod
    def _end_timed(state: GraphState, stage: str, started_at: float) -> None:
        state["timings_ms"][stage] = (time.perf_counter() - started_at) * 1000.0

    def _supervisor_node(self, state: GraphState) -> GraphState:
        t0 = self._timed(state, "supervisor")
        query = state["query"]
        intent = self.stack.supervisor.classify_intent(query)
        inferred_filters = self.stack.supervisor.infer_filters(query)
        if state.get("filters") is None and inferred_filters is not None:
            state["filters"] = inferred_filters
        state["intent"] = intent
        state["plan"] = self.stack.supervisor.build_plan(intent)
        state.setdefault("trace_events", []).append(
            {"stage": "supervisor", "event": f"intent={intent.value}"}
        )
        self._end_timed(state, "supervisor", t0)
        return state

    def _retriever_node(self, state: GraphState) -> GraphState:
        t0 = self._timed(state, "retriever")
        state["retrieval_round"] = int(state.get("retrieval_round", 0)) + 1
        chunks, debug = self.stack.retriever.run(
            query=state["query"],
            filters=state.get("filters"),
            top_k=state["top_k"],
            rerank=state["rerank"],
            retrieval_mode=state["retrieval_mode"],
            candidate_pool_size=state.get("candidate_pool_size"),
            fusion_alpha=state.get("fusion_alpha"),
            rerank_top_n=state.get("rerank_top_n"),
        )
        state["retrieved_chunks"] = chunks
        state["retrieval_debug"] = debug
        state.setdefault("trace_events", []).append(
            {
                "stage": "retriever",
                "event": f"round={state['retrieval_round']} chunks={len(chunks)} mode={debug.retrieval_mode}",
            }
        )
        self._end_timed(state, "retriever", t0)
        return state

    async def _analyst_node(self, state: GraphState) -> GraphState:
        t0 = self._timed(state, "analyst")
        draft = await self.stack.analyst.run(
            query=state["query"],
            intent=state["intent"],
            chunks=state.get("retrieved_chunks", []),
            async_tools=state.get("async_tools", True),
        )
        state["analyst_draft"] = draft
        state.setdefault("trace_events", []).append(
            {
                "stage": "analyst",
                "event": f"claims={len(draft.major_claims)} calcs={len(draft.numeric_calculations)}",
            }
        )
        self._end_timed(state, "analyst", t0)
        return state

    def _fact_checker_node(self, state: GraphState) -> GraphState:
        t0 = self._timed(state, "fact_checker")
        report = self.stack.fact_checker.run(
            draft=state["analyst_draft"],
            chunks=state.get("retrieved_chunks", []),
            retrieval_round=state.get("retrieval_round", 1),
            max_retrieval_rounds=self.settings.max_retrieval_rounds,
        )
        state["fact_check_report"] = report
        state.setdefault("trace_events", []).append(
            {"stage": "fact_checker", "event": f"verdict={report.verdict.value}"}
        )
        self._end_timed(state, "fact_checker", t0)
        return state

    def _route_after_fact_check(self, state: GraphState) -> str:
        verdict = state["fact_check_report"].verdict.value
        if verdict == "retrieve_more" and state.get("retrieval_round", 0) < self.settings.max_retrieval_rounds:
            return "retriever"
        return "finalize"

    def _finalize_node(self, state: GraphState) -> GraphState:
        t0 = self._timed(state, "finalize")
        draft = state["analyst_draft"]
        report = state["fact_check_report"]
        chunks = state.get("retrieved_chunks", [])
        citations = HybridRetriever.map_citations(chunks[:8])
        if not citations:
            citations = HybridRetriever.map_citations(
                [
                    chunks[0]
                ]
            ) if chunks else []
        if not citations:
            from app.schemas.retrieval import Citation

            citations = [
                Citation(
                    citation_id="S1",
                    chunk_id="NO_CHUNK",
                    doc_id="NO_DOC",
                    property_id="NO_PROPERTY",
                    title="No supporting document retrieved",
                    source="system",
                    snippet="No supporting chunk was retrieved for this query.",
                )
            ]

        unsupported = list(report.unsupported_claims) + list(draft.uncertainty_notes)
        weak_count = len(report.weak_claims)
        claim_count = max(len(draft.major_claims), 1)
        confidence = max(0.1, min(0.99, 1.0 - (len(report.unsupported_claims) / claim_count) * 0.5 - (weak_count / claim_count) * 0.2))

        final = FinalAnswer(
            concise_answer=draft.answer_summary,
            reasoning=draft.supporting_reasoning,
            citations=citations,
            confidence_level=round(confidence, 2),
            unsupported_or_uncertain=unsupported[:8],
        )
        state["final_answer"] = final
        state.setdefault("trace_events", []).append({"stage": "finalize", "event": "response_compiled"})
        self._end_timed(state, "finalize", t0)
        return state

    async def run_query_structured(self, request: QueryRequest) -> StructuredQueryResponse:
        request_id = str(uuid.uuid4())
        initial_state: GraphState = {
            "request_id": request_id,
            "query": request.query,
            "filters": request.filters,
            "top_k": request.top_k or self.settings.default_top_k,
            "candidate_pool_size": request.candidate_pool_size,
            "rerank": True if request.rerank is None else request.rerank,
            "rerank_top_n": request.rerank_top_n,
            "retrieval_mode": request.retrieval_mode or self.settings.default_retrieval_mode,
            "fusion_alpha": request.fusion_alpha,
            "async_tools": request.async_tools,
            "retrieval_round": 0,
            "trace_events": [],
            "timings_ms": {},
        }
        logger.info("Processing query", extra={"request_id": request_id, "event": "query_start"})
        result = await self.graph.ainvoke(initial_state)
        return StructuredQueryResponse(
            request_id=request_id,
            intent=result["intent"],
            answer=result["final_answer"],
            retrieval_debug=result.get("retrieval_debug"),
            timings_ms=result.get("timings_ms", {}),
            trace_events=result.get("trace_events", []),
        )

    async def run_query_state(self, request: QueryRequest) -> GraphState:
        request_id = str(uuid.uuid4())
        initial_state: GraphState = {
            "request_id": request_id,
            "query": request.query,
            "filters": request.filters,
            "top_k": request.top_k or self.settings.default_top_k,
            "candidate_pool_size": request.candidate_pool_size,
            "rerank": True if request.rerank is None else request.rerank,
            "rerank_top_n": request.rerank_top_n,
            "retrieval_mode": request.retrieval_mode or self.settings.default_retrieval_mode,
            "fusion_alpha": request.fusion_alpha,
            "async_tools": request.async_tools,
            "retrieval_round": 0,
            "trace_events": [],
            "timings_ms": {},
        }
        result = await self.graph.ainvoke(initial_state)
        return result

    async def run_query(self, request: QueryRequest) -> QueryResponse:
        structured = await self.run_query_structured(request)
        return QueryResponse(
            request_id=structured.request_id,
            intent=structured.intent,
            concise_answer=structured.answer.concise_answer,
            reasoning=structured.answer.reasoning,
            citations=[c.model_dump() for c in structured.answer.citations],
            confidence_level=structured.answer.confidence_level,
            unsupported_or_uncertain=structured.answer.unsupported_or_uncertain,
            retrieval_debug=structured.retrieval_debug if request.include_debug else None,
            timings_ms=structured.timings_ms,
        )
