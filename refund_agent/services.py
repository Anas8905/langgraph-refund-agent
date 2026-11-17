"""Simulated external systems (orders, payments, tickets) with injectable faults.

The three exception types are the point: each needs different handling.
  ServiceUnavailable  transient, nothing happened      -> retry with backoff
  PaymentTimeout      ambiguous, it MAY have happened  -> check before retrying
  PaymentDeclined     permanent                        -> stop, hand to a human
"""
import copy
from dataclasses import dataclass, field
from datetime import date, timedelta


class ServiceUnavailable(Exception): ...
class PaymentTimeout(Exception): ...
class PaymentDeclined(Exception): ...


def _days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).isoformat()


SEED_ORDERS = {
    "A1001": {"item": "Ceramic pour-over set", "total": 49.00, "delivered": 5, "final_sale": False},
    "A1002": {"item": "Wool overcoat", "total": 249.00, "delivered": 10, "final_sale": False},
    "A1003": {"item": "Desk lamp", "total": 39.00, "delivered": 45, "final_sale": False},
    "A1004": {"item": "Running shoes", "total": 89.00, "delivered": 3, "final_sale": False},
    "A1005": {"item": "Clearance headphones", "total": 29.00, "delivered": 7, "final_sale": True},
}


@dataclass
class OrdersAPI:
    outages: int = 0      # the next N calls fail with ServiceUnavailable
    calls: int = 0

    def get(self, order_id: str) -> dict | None:
        self.calls += 1
        if self.outages > 0:
            self.outages -= 1
            raise ServiceUnavailable("orders service returned 503")
        seed = SEED_ORDERS.get(order_id)
        if seed is None:
            return None
        return {"id": order_id, "item": seed["item"], "total": seed["total"], "status": "delivered",
                "delivered_on": _days_ago(seed["delivered"]), "final_sale": seed["final_sale"]}


@dataclass
class PaymentsAPI:
    """faults are consumed one per refund() call:
    None | "timeout_after_commit" | "timeout_before_commit" | "decline"."""
    faults: list = field(default_factory=list)
    ledger: dict = field(default_factory=dict)   # idempotency_key -> refund record
    calls: int = 0

    def refund(self, order_id: str, amount: float, idempotency_key: str) -> dict:
        self.calls += 1
        fault = self.faults.pop(0) if self.faults else None
        if fault == "decline":
            raise PaymentDeclined("card issuer declined the refund")
        if fault == "timeout_before_commit":
            raise PaymentTimeout("no response after 10s")
        if idempotency_key not in self.ledger:   # same key never refunds twice
            self.ledger[idempotency_key] = {"id": f"re_{len(self.ledger) + 1:04d}",
                                            "order_id": order_id, "amount": amount}
        if fault == "timeout_after_commit":
            raise PaymentTimeout("no response after 10s")
        return copy.deepcopy(self.ledger[idempotency_key])

    def find_refund(self, idempotency_key: str) -> dict | None:
        return copy.deepcopy(self.ledger.get(idempotency_key))


@dataclass
class TicketsAPI:
    tickets: list = field(default_factory=list)

    def create(self, summary: str) -> str:
        self.tickets.append(summary)
        return f"T-{1000 + len(self.tickets)}"
