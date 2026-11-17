"""Offline tests for every branch: scripted LLM, simulated services with injected faults."""
import sqlite3
from datetime import date

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from refund_agent.graph import ParsedRequest, build_graph
from refund_agent.policy import evaluate
from refund_agent.services import OrdersAPI, PaymentsAPI, TicketsAPI
from refund_agent.state import initial_state


class ScriptedLLM:
    """Stands in for ChatOpenAI: structured parse first, then free-text replies."""
    def __init__(self, parsed: ParsedRequest):
        self.parsed, self.replies = parsed, 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        if messages[0][1].startswith("You triage"):
            return self.parsed
        self.replies += 1
        return AIMessage(content="(reply)")


def refund_request(order_id, reason="damaged"):
    return ParsedRequest(is_refund_request=True, order_id=order_id, reason=reason)


class Harness:
    def __init__(self, parsed, message, outages=0, faults=(), checkpointer=None):
        self.orders, self.payments, self.tickets = OrdersAPI(outages=outages), PaymentsAPI(faults=list(faults)), TicketsAPI()
        self.graph = build_graph(ScriptedLLM(parsed), self.orders, self.payments, self.tickets,
                                 checkpointer=checkpointer, backoff=0.0)
        self.config = {"configurable": {"thread_id": "t1"}}
        self.path = []
        self.run(initial_state(message))

    def run(self, inputs):
        for step in self.graph.stream(inputs, self.config, stream_mode="updates"):
            self.path.extend(step)
        return self

    @property
    def state(self):
        return self.graph.get_state(self.config)


# ── happy path and approval gate ──────────────────────────────────────────────────────

def test_small_refund_is_auto_approved():
    h = Harness(refund_request("A1001"), "Refund order A1001 please")
    assert h.path == ["parse_request", "lookup_order", "check_policy", "issue_refund", "compose_reply"]
    assert h.state.values["outcome"] == "refunded" and len(h.payments.ledger) == 1


def test_large_refund_pauses_then_approved():
    h = Harness(refund_request("A1002"), "Refund order A1002")
    assert h.path[-1] == "__interrupt__" and h.state.next == ("request_approval",)
    assert h.payments.calls == 0                                # no money moved while paused
    h.run(Command(resume={"approved": True}))
    assert h.state.values["outcome"] == "refunded" and len(h.payments.ledger) == 1


def test_large_refund_rejected_by_reviewer():
    h = Harness(refund_request("A1002"), "Refund order A1002")
    h.run(Command(resume={"approved": False, "note": "replacement already sent"}))
    v = h.state.values
    assert v["outcome"] == "denied" and "replacement already sent" in v["decision_reason"]
    assert h.payments.calls == 0


def test_approval_resumes_in_a_new_process_from_sqlite():
    saver = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    first = Harness(refund_request("A1002"), "Refund order A1002", checkpointer=saver)
    assert first.state.next == ("request_approval",)
    # a fresh graph and fresh service clients, sharing only the checkpoint store
    payments = PaymentsAPI()
    later = build_graph(ScriptedLLM(refund_request("A1002")), OrdersAPI(), payments, TicketsAPI(), saver)
    final = later.invoke(Command(resume={"approved": True}), first.config)
    assert final["outcome"] == "refunded" and len(payments.ledger) == 1


# ── tool failures ─────────────────────────────────────────────────────────────────────

def test_flaky_orders_api_recovers_within_retry_budget():
    h = Harness(refund_request("A1001"), "Refund A1001", outages=2)
    assert h.orders.calls == 3 and h.state.values["outcome"] == "refunded"
    assert len(h.state.values["tool_errors"]) == 2


def test_orders_api_down_escalates_to_a_ticket():
    h = Harness(refund_request("A1001"), "Refund A1001", outages=5)
    assert h.path == ["parse_request", "lookup_order", "escalate", "compose_reply"]
    assert h.orders.calls == 3 and h.state.values["outcome"] == "escalated"
    assert h.state.values["ticket_id"] == "T-1001" and h.payments.calls == 0


def test_timeout_after_commit_is_confirmed_not_retried():
    h = Harness(refund_request("A1004"), "Refund A1004", faults=["timeout_after_commit"])
    assert h.path == ["parse_request", "lookup_order", "check_policy", "issue_refund",
                      "confirm_refund", "compose_reply"]
    assert h.payments.calls == 1 and len(h.payments.ledger) == 1
    assert h.state.values["outcome"] == "refunded"


def test_timeout_before_commit_retries_with_same_idempotency_key():
    h = Harness(refund_request("A1004"), "Refund A1004", faults=["timeout_before_commit"])
    assert h.path.count("issue_refund") == 2 and h.payments.calls == 2
    assert len(h.payments.ledger) == 1 and h.state.values["outcome"] == "refunded"


def test_repeated_timeouts_escalate_without_claiming_success():
    h = Harness(refund_request("A1004"), "Refund A1004", faults=["timeout_before_commit"] * 3)
    v = h.state.values
    assert v["refund_attempts"] == 2 and v["outcome"] == "escalated" and v["refund_id"] is None
    assert h.payments.ledger == {}


def test_declined_refund_escalates():
    h = Harness(refund_request("A1004"), "Refund A1004", faults=["decline"])
    assert h.path[-2:] == ["escalate", "compose_reply"] and h.payments.calls == 1


# ── parsing guards and policy ─────────────────────────────────────────────────────────

def test_invented_order_number_is_discarded():
    h = Harness(refund_request("A1001"), "I want a refund for my lamp")   # A1001 not in the message
    assert h.path == ["parse_request", "compose_reply"]
    assert h.state.values["outcome"] == "needs_info" and h.orders.calls == 0


def test_unknown_order_asks_for_details():
    h = Harness(refund_request("A9999"), "Refund A9999")
    assert h.state.values["order_lookup"] == "not_found" and h.state.values["outcome"] == "needs_info"


def test_non_refund_message_goes_to_a_human():
    h = Harness(ParsedRequest(is_refund_request=False, order_id=None, reason="address change"), "Change my address")
    assert h.path == ["parse_request", "escalate", "compose_reply"]


def test_order_outside_window_is_denied_without_touching_payments():
    h = Harness(refund_request("A1003"), "Refund A1003")
    assert h.state.values["outcome"] == "denied" and h.payments.calls == 0
    assert "refund window is 30 days" in h.state.values["decision_reason"]


def test_already_refunded_order_is_denied():
    h = Harness(refund_request("A1001"), "Refund A1001")
    h.config = {"configurable": {"thread_id": "t2"}}
    h.path = []
    h.run(initial_state("Refund A1001 again"))
    assert h.state.values["decision"] == "deny" and len(h.payments.ledger) == 1


ORDER = {"id": "A1", "item": "x", "total": 50.0, "status": "delivered", "delivered_on": "2026-11-01", "final_sale": False}


@pytest.mark.parametrize("change,already,decision", [
    ({}, False, "auto_approve"),
    ({"total": 100.01}, False, "needs_approval"),
    ({"total": 100.00}, False, "auto_approve"),
    ({"delivered_on": "2026-10-01"}, False, "deny"),
    ({"final_sale": True}, False, "deny"),
    ({"status": "shipped"}, False, "deny"),
    ({}, True, "deny"),
])
def test_policy(change, already, decision):
    assert evaluate({**ORDER, **change}, date(2026, 11, 17), already)[0] == decision
