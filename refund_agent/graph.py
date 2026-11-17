"""Refund agent: tool calls that fail, a human approval gate, and money that never moves twice.

parse_request -> lookup_order -> check_policy -> [request_approval] -> issue_refund -> compose_reply
with failure branches: order API down -> escalate; payment timeout -> confirm_refund (check the
ledger by idempotency key) -> retry or escalate; payment declined -> escalate.

The LLM parses the message and writes the reply. Policy, approval and payment are code and people.
"""
import json
import re
import time
from datetime import date

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer, RetryPolicy, interrupt
from pydantic import BaseModel, Field

from .policy import evaluate
from .services import OrdersAPI, PaymentDeclined, PaymentsAPI, PaymentTimeout, ServiceUnavailable, TicketsAPI
from .state import RefundState

LOOKUP_ATTEMPTS = 3       # order API calls before escalating
MAX_REFUND_ATTEMPTS = 2   # payment attempts before escalating
ORDER_ID = re.compile(r"A\d{4}")

PARSE_PROMPT = """You triage customer messages for an online store's refund desk.
- is_refund_request: true if the customer wants their money back for an order.
- order_id: the order number exactly as the customer wrote it (A + 4 digits), or null.
  Never invent or correct an order number.
- reason: the customer's reason in a few words."""

REPLY_PROMPT = """Write the store's reply to the customer in 2-4 plain, friendly sentences.
Use only the facts in the JSON. Do not add policies, timelines or promises that are not there.
- outcome "refunded": confirm the amount and give the refund_id.
- outcome "denied": explain decision_reason.
- outcome "escalated": say a teammate will follow up and give the ticket_id. Do not say a refund was made.
- outcome "needs_info": ask for the order number (it looks like A1234)."""


class ParsedRequest(BaseModel):
    is_refund_request: bool
    order_id: str | None = Field(description="Order number exactly as written, e.g. A1001; null if absent")
    reason: str


def idempotency_key(state: RefundState) -> str:
    return f"refund-{state['order']['id']}"   # stable per order: retries can't refund twice


def build_graph(llm: BaseChatModel, orders: OrdersAPI, payments: PaymentsAPI, tickets: TicketsAPI,
                checkpointer: Checkpointer = None, backoff: float = 0.5, today: date | None = None):
    parser = llm.with_structured_output(ParsedRequest)

    def parse_request(state: RefundState) -> dict:
        parsed = parser.invoke([("system", PARSE_PROMPT), ("human", state["message"])])
        order_id = parsed.order_id
        if not (order_id and ORDER_ID.fullmatch(order_id) and order_id in state["message"]):
            order_id = None   # the model may not invent or "fix" an order number
        update = {"request": {"is_refund": parsed.is_refund_request, "order_id": order_id,
                              "reason": parsed.reason}}
        if parsed.is_refund_request and not order_id:
            update["outcome"] = "needs_info"
        return update

    def lookup_order(state: RefundState) -> dict:
        errors = []
        for attempt in range(1, LOOKUP_ATTEMPTS + 1):
            try:
                order = orders.get(state["request"]["order_id"])
            except ServiceUnavailable as exc:
                errors.append(f"orders.get attempt {attempt}/{LOOKUP_ATTEMPTS}: {exc}")
                if attempt < LOOKUP_ATTEMPTS:
                    time.sleep(backoff * 2 ** (attempt - 1))
                continue
            if order is None:
                return {"order_lookup": "not_found", "outcome": "needs_info", "tool_errors": errors}
            return {"order_lookup": "found", "order": order, "tool_errors": errors}
        return {"order_lookup": "unavailable", "tool_errors": errors}

    def check_policy(state: RefundState) -> dict:
        already = payments.find_refund(idempotency_key(state)) is not None
        decision, reason = evaluate(state["order"], today or date.today(), already)
        return {"decision": decision, "decision_reason": reason,
                **({"outcome": "denied"} if decision == "deny" else {})}

    def request_approval(state: RefundState) -> dict:
        order = state["order"]
        answer = interrupt({"order_id": order["id"], "item": order["item"], "amount": order["total"],
                            "customer_reason": state["request"]["reason"], "why_review": state["decision_reason"]})
        if answer.get("approved") is True:
            return {"approval": "approved"}
        return {"approval": "rejected", "outcome": "denied",
                "decision_reason": f"not approved by reviewer: {answer.get('note') or 'no note'}"}

    def issue_refund(state: RefundState) -> dict:
        order, attempt = state["order"], state["refund_attempts"] + 1
        try:
            refund = payments.refund(order["id"], order["total"], idempotency_key=idempotency_key(state))
        except PaymentTimeout as exc:
            return {"refund_attempts": attempt, "refund_status": "unknown",
                    "tool_errors": [f"payments.refund attempt {attempt}: {exc}; outcome unknown"]}
        except PaymentDeclined as exc:
            return {"refund_attempts": attempt, "refund_status": "failed",
                    "tool_errors": [f"payments.refund attempt {attempt}: {exc}"]}
        return {"refund_attempts": attempt, "refund_status": "succeeded", "refund_id": refund["id"],
                "outcome": "refunded"}

    def confirm_refund(state: RefundState) -> dict:
        refund = payments.find_refund(idempotency_key(state))
        if refund:
            return {"refund_status": "succeeded", "refund_id": refund["id"], "outcome": "refunded"}
        return {"refund_status": "not_found"}

    def escalate(state: RefundState) -> dict:
        summary = (f"Refund request needs a human. Message: {state['message']!r}. "
                   f"Order: {state['order'] and state['order']['id']}. Errors: {state['tool_errors']}")
        return {"ticket_id": tickets.create(summary), "outcome": "escalated"}

    def compose_reply(state: RefundState) -> dict:
        order = state["order"] or {}
        facts = {"outcome": state["outcome"], "order_id": order.get("id"), "item": order.get("item"),
                 "amount": order.get("total"), "refund_id": state["refund_id"],
                 "decision_reason": state["decision_reason"], "ticket_id": state["ticket_id"]}
        reply = llm.invoke([("system", REPLY_PROMPT), ("human", json.dumps(facts))])
        return {"reply": reply.content.strip()}

    # ── routing: every branch is a conditional edge on state ──────────────────────────
    def after_parse(s):
        if not s["request"]["is_refund"]:
            return "escalate"
        return "lookup_order" if s["request"]["order_id"] else "compose_reply"
    def after_lookup(s):   return {"found": "check_policy", "not_found": "compose_reply",
                                   "unavailable": "escalate"}[s["order_lookup"]]
    def after_policy(s):   return {"auto_approve": "issue_refund", "needs_approval": "request_approval",
                                   "deny": "compose_reply"}[s["decision"]]
    def after_approval(s): return "issue_refund" if s["approval"] == "approved" else "compose_reply"
    def after_refund(s):   return {"succeeded": "compose_reply", "unknown": "confirm_refund",
                                   "failed": "escalate"}[s["refund_status"]]
    def after_confirm(s):
        if s["refund_status"] == "succeeded":
            return "compose_reply"
        return "issue_refund" if s["refund_attempts"] < MAX_REFUND_ATTEMPTS else "escalate"

    g = StateGraph(RefundState)
    llm_retry = RetryPolicy(max_attempts=3)
    g.add_node("parse_request", parse_request, retry_policy=llm_retry)
    for name, fn in [("lookup_order", lookup_order), ("check_policy", check_policy),
                     ("request_approval", request_approval), ("issue_refund", issue_refund),
                     ("confirm_refund", confirm_refund), ("escalate", escalate)]:
        g.add_node(name, fn)
    g.add_node("compose_reply", compose_reply, retry_policy=llm_retry)

    g.add_edge(START, "parse_request")
    g.add_conditional_edges("parse_request", after_parse, ["lookup_order", "compose_reply", "escalate"])
    g.add_conditional_edges("lookup_order", after_lookup, ["check_policy", "compose_reply", "escalate"])
    g.add_conditional_edges("check_policy", after_policy, ["issue_refund", "request_approval", "compose_reply"])
    g.add_conditional_edges("request_approval", after_approval, ["issue_refund", "compose_reply"])
    g.add_conditional_edges("issue_refund", after_refund, ["compose_reply", "confirm_refund", "escalate"])
    g.add_conditional_edges("confirm_refund", after_confirm, ["compose_reply", "issue_refund", "escalate"])
    g.add_edge("escalate", "compose_reply")
    g.add_edge("compose_reply", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
