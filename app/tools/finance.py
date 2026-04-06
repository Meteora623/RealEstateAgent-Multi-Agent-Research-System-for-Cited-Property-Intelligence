from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CapRateResult:
    noi: float
    price: float
    cap_rate: float


def calculate_noi(gross_income: float, vacancy_rate: float, operating_expenses: float) -> float:
    effective_income = gross_income * (1.0 - vacancy_rate)
    return max(effective_income - operating_expenses, 0.0)


def calculate_cap_rate(noi: float, price: float) -> CapRateResult:
    if price <= 0:
        raise ValueError("Price must be positive for cap-rate calculation.")
    cap_rate = (noi / price) * 100.0
    return CapRateResult(noi=noi, price=price, cap_rate=round(cap_rate, 2))


def estimate_monthly_mortgage(
    loan_amount: float, annual_rate: float, years: int, down_payment: float = 0.0
) -> float:
    principal = max(loan_amount - down_payment, 0.0)
    if principal <= 0:
        return 0.0
    monthly_rate = annual_rate / 12.0
    payments = years * 12
    if monthly_rate == 0:
        return principal / payments
    factor = (1 + monthly_rate) ** payments
    return principal * (monthly_rate * factor) / (factor - 1)

