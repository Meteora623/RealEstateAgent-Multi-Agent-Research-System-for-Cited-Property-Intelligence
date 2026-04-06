from __future__ import annotations

from typing import Any


def compare_property_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    fields = ["asking_price", "noi_estimate", "occupancy_rate", "renovation_budget_estimate"]
    result: dict[str, Any] = {}
    for field in fields:
        lv = left.get(field)
        rv = right.get(field)
        if lv is None or rv is None:
            continue
        delta = lv - rv
        winner = "left" if delta > 0 else "right" if delta < 0 else "tie"
        result[field] = {"left": lv, "right": rv, "delta": delta, "winner": winner}
    return result

