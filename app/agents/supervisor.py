from __future__ import annotations

import re

from app.schemas.agent import IntentType
from app.schemas.retrieval import MetadataFilter


class SupervisorAgent:
    def classify_intent(self, query: str) -> IntentType:
        q = query.lower()
        if any(k in q for k in ["compare", "vs", "versus", "better investment"]):
            return IntentType.comparison
        if any(k in q for k in ["risk", "downside", "concern"]):
            return IntentType.risk_analysis
        if any(k in q for k in ["cap-rate", "cap rate", "valuation", "undervalued", "comps"]):
            return IntentType.valuation_reasoning
        if any(k in q for k in ["investment thesis", "upside", "thesis", "value-add"]):
            return IntentType.investment_thesis
        return IntentType.lookup

    def build_plan(self, intent: IntentType) -> list[str]:
        if intent == IntentType.comparison:
            return ["retrieve", "analyze_with_comparison", "fact_check"]
        if intent == IntentType.risk_analysis:
            return ["retrieve", "analyze_risks", "fact_check"]
        if intent == IntentType.valuation_reasoning:
            return ["retrieve", "analyze_valuation", "fact_check"]
        return ["retrieve", "analyze", "fact_check"]

    def infer_filters(self, query: str) -> MetadataFilter | None:
        q = query.lower()
        property_id = None
        ids = re.findall(r"prop-\d{4,5}", q)
        if len(ids) == 1:
            property_id = ids[0].upper()

        city = None
        for c in ("brooklyn", "manhattan", "queens"):
            if c in q:
                city = c.title()
                break

        property_type = None
        for p in ("multifamily", "mixed_use", "condo", "office"):
            if p in q:
                property_type = p
                break

        min_price, max_price = self._extract_price_range(q)
        if not any([property_id, city, property_type, min_price, max_price]):
            return None
        return MetadataFilter(
            property_id=property_id,
            city=city,
            property_type=property_type,
            min_price=min_price,
            max_price=max_price,
        )

    @staticmethod
    def _extract_price_range(query: str) -> tuple[float | None, float | None]:
        # Supports rough patterns such as "$2m-$4m" or "between 2 and 4 million".
        m = re.search(r"\$?(\d+(?:\.\d+)?)\s*m?\s*[-to]+\s*\$?(\d+(?:\.\d+)?)\s*m", query)
        if m:
            return float(m.group(1)) * 1_000_000, float(m.group(2)) * 1_000_000
        return None, None
