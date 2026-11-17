"""Typed state for the refund agent."""
import operator
from typing import Annotated, Literal, TypedDict

Outcome = Literal["refunded", "denied", "escalated", "needs_info"]


class Request(TypedDict):
    is_refund: bool
    order_id: str | None   # only kept if it appears verbatim in the customer's message
    reason: str


class Order(TypedDict):
    id: str
    item: str
    total: float
    status: str
    delivered_on: str      # ISO date
    final_sale: bool


class RefundState(TypedDict):
    message: str                                                    # customer's message
    request: Request | None                                         # LLM-parsed, code-validated
    order: Order | None
    order_lookup: Literal["found", "not_found", "unavailable"] | None
    decision: Literal["auto_approve", "needs_approval", "deny"] | None
    decision_reason: str
    approval: Literal["approved", "rejected"] | None                # from the human reviewer
    refund_status: Literal["succeeded", "unknown", "not_found", "failed"] | None
    refund_attempts: int
    refund_id: str | None
    ticket_id: str | None
    tool_errors: Annotated[list[str], operator.add]                 # appended, never overwritten
    outcome: Outcome | None
    reply: str


def initial_state(message: str) -> RefundState:
    return RefundState(message=message, request=None, order=None, order_lookup=None, decision=None,
                       decision_reason="", approval=None, refund_status=None, refund_attempts=0,
                       refund_id=None, ticket_id=None, tool_errors=[], outcome=None, reply="")
