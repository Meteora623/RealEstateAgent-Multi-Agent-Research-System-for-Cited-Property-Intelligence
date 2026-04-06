from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from app.schemas.evaluation import EvaluationQuery

BOROUGH_NEIGHBORHOODS = {
    "Brooklyn": [
        "Williamsburg",
        "Bushwick",
        "Park Slope",
        "Crown Heights",
        "Downtown Brooklyn",
        "Greenpoint",
        "Bedford-Stuyvesant",
    ],
    "Manhattan": [
        "Harlem",
        "Lower East Side",
        "Upper West Side",
        "Chelsea",
        "Financial District",
        "Hell's Kitchen",
        "Washington Heights",
    ],
    "Queens": [
        "Long Island City",
        "Astoria",
        "Flushing",
        "Jamaica",
        "Sunnyside",
        "Forest Hills",
        "Jackson Heights",
    ],
}

PROPERTY_TYPES = ["multifamily", "mixed_use", "condo", "office"]
BUILDING_CLASSES = ["A", "B", "C"]
DOC_TYPES = [
    "listing_description",
    "broker_note",
    "comp_summary",
    "rent_roll_snippet",
    "market_commentary",
    "renovation_assessment",
    "neighborhood_note",
    "property_tax_snippet",
    "financing_note",
    "building_record",
]

QUERY_CATEGORIES = [
    "comparison",
    "valuation",
    "risk",
    "investment_thesis",
    "market_trend",
    "insufficient_evidence",
]


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _compose_document_text(doc_type: str, row: dict[str, Any]) -> str:
    base = (
        f"Property {row['property_id']} is a {row['property_type']} asset in {row['neighborhood']}, {row['city']} "
        f"with {row['units']} units, built in {row['year_built']}, and classified as building class {row['building_class']}."
    )
    operating = (
        f"Current ask is {_money(row['asking_price'])}, modeled NOI is {_money(row['noi_estimate'])}, "
        f"occupancy is {row['occupancy_rate']:.1%}, and renovation budget is {_money(row['renovation_budget_estimate'])}."
    )
    comps = (
        f"Recent comps in the same micro-market traded between {_money(row['comp_price_low'])} and "
        f"{_money(row['comp_price_high'])}. Reported market cap-rate band is {row['market_cap_low']:.2f}% to "
        f"{row['market_cap_high']:.2f}%."
    )

    templates: dict[str, list[str]] = {
        "listing_description": [
            "## Asset Overview\n"
            f"{base} {operating} Unit mix has a stable share of one- and two-bedroom layouts with below-market in-place rents. "
            "Investor materials describe above-average tenant retention and strong leasing velocity near transit.",
            "## Leasing and Demand\n"
            "Leasing activity over the past three quarters showed shorter downtime between turns than the submarket average. "
            "Broker tours highlighted renewed demand from renters priced out of prime corridors, supporting absorption assumptions.",
            "## Value-Add Angle\n"
            "Potential upside is tied to turnover-driven upgrades, selective common-area improvements, and utility pass-through controls. "
            "Execution risk remains moderate due to permit timing and labor-cost variability.",
        ],
        "broker_note": [
            "## Broker Commentary\n"
            f"{base} Seller communication indicates flexibility on close timeline and transition support. "
            "Tour feedback repeatedly cited curb appeal and realistic upside assumptions versus local alternatives.",
            "## Buyer Interest\n"
            "Current buyer pool includes private syndicates and small funds seeking transitional assets with immediate cash flow. "
            "Indicative bids show sensitivity to financing spreads and expected renovation pacing.",
            "## Negotiation Signals\n"
            "Broker notes suggest pricing discipline for fully as-is bids but willingness to adjust on diligence findings. "
            "This dynamic can improve entry basis if capex risk is documented credibly.",
        ],
        "comp_summary": [
            "## Comparable Sales\n"
            f"{comps} Comparable set includes assets with similar unit count, vintage profile, and transit access. "
            "One outlier trade at premium valuation reflected superior amenity package and lower deferred maintenance.",
            "## Relative Positioning\n"
            f"{operating} On a per-unit basis, this asset screens {'below' if row['asking_price'] < row['comp_price_low'] else 'within'} "
            "the observed range once condition and occupancy are normalized.",
            "## Cap-Rate Interpretation\n"
            "Comps imply valuation support is strongest when underwriting assumes conservative rent growth and reserve buffers. "
            "Aggressive rent-lift scenarios require faster-than-baseline permit and turnover execution.",
        ],
        "rent_roll_snippet": [
            "## Rent Roll Snapshot\n"
            f"{operating} Roughly one-third of leases are set to roll in the next twelve months, creating staged repricing opportunity.",
            "## Tenant Profile\n"
            "Tenant concentration is low, with no single tenant segment accounting for outsized gross rent share. "
            "Arrears remain manageable, but concessions were used selectively for slower winter leasing periods.",
            "## Revenue Stability\n"
            "Collections performance stayed resilient during the prior two quarters. "
            "Upside assumptions depend on disciplined turn-cost control and measured lease-up timing.",
        ],
        "market_commentary": [
            "## Submarket Conditions\n"
            f"{base} Lending and transaction velocity indicate a stabilizing environment rather than an overheated one. "
            "Neighborhood-level demand is positive but uneven by block and product quality.",
            "## Pricing and Liquidity\n"
            f"{comps} Liquidity is strongest for assets with clear operational plans and transparent deferred-maintenance disclosures.",
            "## Forward View\n"
            "Near-term risk factors include financing volatility and construction-input inflation. "
            "Supportive factors include steady renter demand and incremental cap-rate compression for renovated stock.",
        ],
        "renovation_assessment": [
            "## Scope and Budget\n"
            f"Proposed renovation budget of {_money(row['renovation_budget_estimate'])} targets kitchens, baths, corridors, and curb appeal upgrades. "
            "Budget includes contingency for code-triggered scope expansion and MEP repairs.",
            "## Operational Impact\n"
            "Execution plan phases work by stack to reduce displacement and protect occupancy continuity. "
            "Cash-flow drag is front-loaded but improves as renovated units reprice to market.",
            "## Expected Outcome\n"
            "Base-case underwriting assumes moderate rent lift and limited vacancy disruption. "
            "Downside case includes delayed permits and higher labor costs, which compress near-term yield.",
        ],
        "neighborhood_note": [
            "## Neighborhood Dynamics\n"
            f"{base} Retail mix has broadened, transit reliability improved, and weekend foot traffic increased across adjacent corridors.",
            "## Demand Drivers\n"
            "Local employment centers and education nodes continue to support renter depth. "
            "Comparable lease velocity suggests resilient demand for well-maintained mid-market housing stock.",
            "## Watch Items\n"
            "Rising insurance premiums and evolving local policy can pressure operating margins. "
            "Investors with disciplined expense controls are better positioned to sustain NOI growth.",
        ],
        "property_tax_snippet": [
            "## Tax Position\n"
            f"Current estimated annual property tax is {_money(row['property_tax_annual'])}, "
            f"equivalent to approximately {_money(row['property_tax_annual'] / max(row['units'], 1))} per unit.",
            "## Assessment Trend\n"
            "Recent assessment updates imply modest upward drift in tax burden over the next cycle. "
            "Appeal options exist but success likelihood depends on valuation comparables and assessment methodology.",
            "## Underwriting Implication\n"
            "NOI sensitivity to tax escalation is material in downside scenarios. "
            "Prudent underwriting should include step-up assumptions rather than static tax lines.",
        ],
        "financing_note": [
            "## Debt Snapshot\n"
            f"Indicative financing assumes {row['ltv_ratio']:.0%} LTV at {row['interest_rate']:.2%} fixed/hedged equivalent cost. "
            f"Modeled DSCR in base case is {row['dscr']:.2f}x.",
            "## Credit Considerations\n"
            "Lender appetite remains healthy for well-documented business plans with realistic capex and lease-up pacing. "
            "Execution clarity improves pricing and covenant flexibility.",
            "## Sensitivity\n"
            "Debt service pressure increases quickly under slower lease-up or elevated capex overrun assumptions. "
            "Structured reserves can mitigate near-term volatility in coverage.",
        ],
        "building_record": [
            "## Physical Condition\n"
            f"{base} Recent inspections flagged manageable deferred maintenance with no immediate life-safety red flags.",
            "## Systems History\n"
            "Roof and boiler interventions were completed in recent years; facade and corridor work remains partially deferred. "
            "Elevator and fire systems are functional with recommended preventive maintenance cadence.",
            "## Risk Considerations\n"
            "Key operational risks center on capital timing, contractor availability, and permit cycle variability. "
            "Risk-adjusted underwriting should reserve for schedule slippage and contingency scope.",
        ],
    }

    sections = templates.get(doc_type, templates["listing_description"])
    common_section = (
        "## Financial Snapshot\n"
        f"{operating} Taxes are estimated at {_money(row['property_tax_annual'])} annually and market financing "
        f"indicates {row['ltv_ratio']:.0%} LTV near {row['interest_rate']:.2%} borrowing cost."
    )
    return "\n\n".join([common_section, *sections])


def generate_synthetic_dataset(
    data_dir: Path,
    num_properties: int = 780,
    docs_per_property: int = 6,
    eval_queries: int = 80,
    seed: int = 13,
) -> dict[str, int]:
    random.seed(seed)
    raw_dir = data_dir / "raw"
    structured_dir = data_dir / "structured"
    eval_dir = data_dir / "eval"
    raw_dir.mkdir(parents=True, exist_ok=True)
    structured_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    properties: list[dict[str, Any]] = []
    docs: list[dict[str, Any]] = []

    for i in range(num_properties):
        city = random.choice(list(BOROUGH_NEIGHBORHOODS.keys()))
        neighborhood = random.choice(BOROUGH_NEIGHBORHOODS[city])
        property_type = random.choice(PROPERTY_TYPES)
        asking_price = random.randint(1_400_000, 24_000_000)
        noi = asking_price * random.uniform(0.038, 0.078)
        occupancy = random.uniform(0.8, 0.99)
        renovation = asking_price * random.uniform(0.05, 0.18)
        comp_low = asking_price * random.uniform(0.88, 0.98)
        comp_high = asking_price * random.uniform(1.03, 1.22)
        units = random.randint(10, 220)
        year_built = random.randint(1905, 2023)
        property_tax_annual = asking_price * random.uniform(0.006, 0.02)
        ltv = random.uniform(0.52, 0.72)
        interest_rate = random.uniform(0.049, 0.079)
        dscr = random.uniform(1.12, 1.58)
        market_cap_low = random.uniform(4.2, 5.3)
        market_cap_high = market_cap_low + random.uniform(0.3, 1.0)

        row: dict[str, Any] = {
            "property_id": f"PROP-{i:05d}",
            "city": city,
            "neighborhood": neighborhood,
            "property_type": property_type,
            "building_class": random.choice(BUILDING_CLASSES),
            "year_built": year_built,
            "units": units,
            "asking_price": round(asking_price, 2),
            "noi_estimate": round(noi, 2),
            "occupancy_rate": round(occupancy, 4),
            "renovation_budget_estimate": round(renovation, 2),
            "comp_price_low": round(comp_low, 2),
            "comp_price_high": round(comp_high, 2),
            "property_tax_annual": round(property_tax_annual, 2),
            "ltv_ratio": round(ltv, 4),
            "interest_rate": round(interest_rate, 4),
            "dscr": round(dscr, 3),
            "market_cap_low": round(market_cap_low, 3),
            "market_cap_high": round(market_cap_high, 3),
            "source": "synthetic_corpus_v2",
        }
        properties.append(row)

        core_types = ["listing_description", "comp_summary", "rent_roll_snippet"]
        remaining = [dt for dt in DOC_TYPES if dt not in core_types]
        target_total = max(docs_per_property, len(core_types))
        selected_types = list(core_types)
        if target_total > len(selected_types):
            selected_types.extend(random.sample(remaining, k=min(target_total - len(selected_types), len(remaining))))

        for j, doc_type in enumerate(selected_types[:target_total]):
            doc_id = f"DOC-{i:05d}-{j:02d}"
            docs.append(
                {
                    "doc_id": doc_id,
                    "property_id": row["property_id"],
                    "doc_type": doc_type,
                    "title": f"{row['property_id']} {doc_type.replace('_', ' ').title()}",
                    "text": _compose_document_text(doc_type, row),
                    "metadata": row,
                }
            )

    docs_path = raw_dir / "property_documents.jsonl"
    with docs_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    pd.DataFrame(properties).to_csv(structured_dir / "properties.csv", index=False)
    _build_eval_set(eval_dir / "eval_queries.jsonl", properties, docs, n=eval_queries, seed=seed + 101)

    return {"properties": len(properties), "documents": len(docs), "eval_queries": eval_queries}


def _build_eval_set(
    path: Path,
    properties: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    n: int = 80,
    seed: int = 114,
) -> None:
    random.seed(seed)
    samples = random.sample(properties, k=min(max(n, 60), len(properties)))
    by_property: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        by_property.setdefault(doc["property_id"], []).append(doc)

    rows: list[EvaluationQuery] = []
    for idx, prop in enumerate(samples):
        pid = prop["property_id"]
        property_docs = by_property.get(pid, [])
        category = QUERY_CATEGORIES[idx % len(QUERY_CATEGORIES)]
        difficulty = "hard" if category in {"comparison", "insufficient_evidence"} else "medium"

        if category == "comparison":
            peer = samples[(idx + 7) % len(samples)]["property_id"]
            query = f"Compare {pid} vs {peer} for risk-adjusted investment quality and cite evidence."
            expected_types = {"comp_summary", "financing_note", "rent_roll_snippet", "market_commentary"}
            ref = (
                f"Comparison should weigh cap-rate support, financing resilience, and renovation execution risks "
                f"between {pid} and {peer}."
            )
        elif category == "valuation":
            query = f"Is {pid} undervalued relative to nearby comps and cap-rate evidence?"
            expected_types = {"comp_summary", "listing_description", "market_commentary"}
            ref = (
                f"{pid} appears {'potentially undervalued' if prop['asking_price'] < prop['comp_price_low'] else 'roughly fairly priced'} "
                f"given comp range and cap-rate context."
            )
        elif category == "risk":
            query = f"Summarize primary downside risks for {pid} with citations."
            expected_types = {"building_record", "property_tax_snippet", "financing_note"}
            ref = (
                f"Risk profile for {pid} should discuss deferred maintenance, tax drift, and debt-service sensitivity."
            )
        elif category == "investment_thesis":
            query = f"What is the investment thesis for {pid} and where is evidence weak?"
            expected_types = {"listing_description", "broker_note", "renovation_assessment"}
            ref = (
                f"Thesis for {pid} combines operational upside and phased renovation, with uncertainty tied to execution pace."
            )
        elif category == "market_trend":
            query = f"What local market signals support or weaken the outlook for {pid}?"
            expected_types = {"market_commentary", "neighborhood_note", "comp_summary"}
            ref = f"Outlook for {pid} should reference neighborhood demand, liquidity trends, and cap-rate backdrop."
        else:
            query = f"What flood zone and zoning variance evidence exists for {pid}?"
            expected_types = {"building_record"}
            ref = (
                f"Available corpus for {pid} is likely insufficient for definitive flood-zone or zoning-variance conclusions; "
                "response should explicitly state uncertainty."
            )
            difficulty = "hard"

        expected_docs = [doc["doc_id"] for doc in property_docs if doc["doc_type"] in expected_types]
        if not expected_docs:
            expected_docs = [doc["doc_id"] for doc in property_docs[:3]]

        rows.append(
            EvaluationQuery(
                query_id=f"Q-{idx:04d}",
                query=query,
                reference_answer=ref,
                expected_doc_ids=expected_docs[:4],
                expected_property_ids=[pid],
                expected_doc_types=sorted(expected_types),
                difficulty=difficulty,
            )
        )

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
