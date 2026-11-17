# Refund agent with tool-failure handling and human approval (LangGraph)

A customer-support agent that issues refunds against external APIs that fail the way real
ones do. The orders API goes down, the payments API times out *after* charging, and the card
issuer declines. The graph handles each failure differently, pauses for a human on large
amounts, and never refunds the same order twice.

Python · LangGraph · LangChain · OpenAI · ~230 lines of agent code (graph, policy, state),
plus 80 lines of simulated services and a 100-line CLI.

```bash
cp .env.example .env        # add OPENAI_API_KEY
uv run run.py               # all six scenarios, each printing every state transition
```

`uv run pytest` runs 22 offline tests and needs no API key.

## Graph

```mermaid
graph TD;
    __start__([start]) --> parse_request
    parse_request -. refund + order no. .-> lookup_order
    parse_request -. no order no. .-> compose_reply
    parse_request -. not a refund .-> escalate
    lookup_order -. found .-> check_policy
    lookup_order -. not found .-> compose_reply
    lookup_order -. "API down after 3 tries" .-> escalate
    check_policy -. "≤ $100" .-> issue_refund
    check_policy -. "> $100" .-> request_approval
    check_policy -. deny .-> compose_reply
    request_approval -. approved .-> issue_refund
    request_approval -. rejected .-> compose_reply
    issue_refund -. succeeded .-> compose_reply
    issue_refund -. "timeout: unknown" .-> confirm_refund
    issue_refund -. declined .-> escalate
    confirm_refund -. found in ledger .-> compose_reply
    confirm_refund -. "not found, attempts < 2" .-> issue_refund
    confirm_refund -. "not found, attempts = 2" .-> escalate
    escalate --> compose_reply
    compose_reply --> __end__([end])
```

Every branch is a conditional edge reading typed state ([refund_agent/state.py](refund_agent/state.py)).
The routers are six small functions at the bottom of [refund_agent/graph.py](refund_agent/graph.py).

## Who decides what

| Decision | Made by |
|---|---|
| Is this a refund request? Which order? | LLM (structured output), then **checked by code**: an order number the customer didn't actually write is discarded |
| Is the refund allowed? | [refund_agent/policy.py](refund_agent/policy.py): 30-day window, final sale, already refunded, $100 auto-approve limit |
| Refunds over $100 | A **human**, via `interrupt()` |
| Did the money move? | The payments ledger, looked up by idempotency key |
| The words the customer reads | LLM, from a fixed fact sheet (outcome, amount, refund ID, reason, ticket) |

The LLM never decides whether money moves.

## Failure paths

The simulated services ([refund_agent/services.py](refund_agent/services.py)) raise three different
errors because each one needs different handling:

| Failure | What it means | What the graph does |
|---|---|---|
| `ServiceUnavailable` from the orders API | Transient; nothing happened | Retries 3 times with exponential backoff, logging each attempt to `tool_errors`. Still down? Escalates to a human ticket rather than telling the customer "no". |
| `PaymentTimeout` from refund | **Ambiguous**: the refund may have gone through | Sets `refund_status = "unknown"` and goes to `confirm_refund`, which looks the refund up in the ledger by idempotency key. Found: done. Not found: retry with the *same* key. Still failing after 2 attempts: escalate. The customer is never told "refunded" until the ledger confirms it. |
| `PaymentDeclined` | Permanent | No retry; escalate |

The idempotency key is derived from the order (`refund-A1004`), not generated per attempt. So
a retry, a resumed run or a duplicate customer message can't refund twice. `check_policy`
also consults the ledger, so a second request for an already-refunded order is denied.

`issue_refund` deliberately has **no** automatic `RetryPolicy`. Blindly re-running a
money-moving call after an ambiguous error is how double refunds happen. The LLM nodes do
have one, for transient API errors.

## Human approval and checkpoints

`request_approval` calls `interrupt()`. The graph stops, and the SQLite checkpointer saves the
full state. The process can exit. A reviewer can approve hours later from a different process:

```bash
uv run run.py --scenario large-refund      # stops at the approval gate, prints the run ID
uv run run.py --approve run-1a2b3c4d       # resumes from checkpoints.sqlite and issues the refund
uv run run.py --reject  run-1a2b3c4d --note "replacement already shipped"
```

No payment call happens before approval (a test asserts it).

## Example runs

| Scenario | Path |
|---|---|
| `auto-refund`: A1001, $49 | parse → lookup → policy → issue_refund → reply |
| `large-refund`: A1002, $249 | … → policy → ⏸ approval → issue_refund → reply |
| `orders-down`: orders API returns 503 five times | lookup (3 attempts) → escalate → reply with ticket |
| `payment-timeout`: refund commits, then times out | issue_refund (unknown) → confirm_refund (found) → reply. One refund in the ledger. |
| `outside-window`: A1003, delivered 45 days ago | policy deny → reply with the reason |
| `no-order-number` | parse → reply asking for the order number |

<!-- EXAMPLE-RUNS: replace with real output from `uv run run.py` -->

```text
(paste output of: uv run run.py --scenario payment-timeout)
```

```text
(paste output of: uv run run.py --scenario large-refund, then --approve RUN_ID)
```

## Tests

22 offline tests (scripted LLM, simulated services with injected faults), covering:

- every scenario above, asserting the exact node path;
- approval approved, approval rejected, and approval resumed by a fresh graph instance sharing only the SQLite store;
- timeout after commit (confirmed, *not* retried), timeout before commit (retried with the same key), repeated timeouts (escalated, `refund_id` stays empty), decline;
- the orders API recovering inside the retry budget, and staying down;
- an invented order number being discarded, unknown order, non-refund message, already-refunded order;
- policy edge cases ($100.00 vs $100.01, the window boundary, final sale, not delivered).
