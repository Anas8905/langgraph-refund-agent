"""Run the refund agent and print every state transition.

    uv run run.py                          # all scenarios (the approval one is approved in-process)
    uv run run.py --scenario large-refund  # one scenario; pauses at the approval gate and exits
    uv run run.py --approve RUN_ID         # resume a paused run from SQLite, in a new process
    uv run run.py --reject RUN_ID --note "customer already got a replacement"
"""
import argparse
import os
import sqlite3
import sys
import textwrap
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from refund_agent.graph import build_graph
from refund_agent.services import OrdersAPI, PaymentsAPI, TicketsAPI
from refund_agent.state import initial_state

# name: (customer message, OrdersAPI outages, PaymentsAPI faults)
SCENARIOS = {
    "auto-refund":     ("Hi, order A1001 arrived with a cracked carafe. Can I get a refund?", 0, []),
    "large-refund":    ("I'd like a refund for order A1002, the coat doesn't fit.", 0, []),
    "orders-down":     ("Please refund order A1001, it arrived broken.", 5, []),
    "payment-timeout": ("Refund A1004 please, they're the wrong size.", 0, ["timeout_after_commit"]),
    "outside-window":  ("Can I get my money back for order A1003? The lamp flickers.", 0, []),
    "no-order-number": ("I want a refund for the lamp I bought last month.", 0, []),
}


def show(node: str, update) -> None:
    print(f"\n── {node} {'─' * (64 - len(node))}")
    if node == "__interrupt__":
        update = {"PAUSED": "waiting for human approval", **update[0].value}
    for key, value in (update or {}).items():
        if key == "tool_errors" and not value:
            continue
        print(textwrap.fill(str(value), width=100, initial_indent=f"   {key + ':':<17} ",
                            subsequent_indent=" " * 21))


def stream(graph, run_id: str, inputs) -> bool:
    """Stream one leg of a run. Returns True if the graph paused at the approval gate."""
    config = {"configurable": {"thread_id": run_id}}
    for step in graph.stream(inputs, config, stream_mode="updates"):
        for node, update in step.items():
            show(node, update)
    state = graph.get_state(config)
    if state.next:
        print(f"\n   resume later with: uv run run.py --approve {run_id}   (or --reject {run_id})")
        return True
    v = state.values
    print(f"\n══ outcome [{run_id}] {v['outcome']}  refund_id={v['refund_id']}  ticket={v['ticket_id']}")
    print(textwrap.fill(v["reply"], width=100) + "\n")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--approve", metavar="RUN_ID")
    parser.add_argument("--reject", metavar="RUN_ID")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0)
    checkpointer = SqliteSaver(sqlite3.connect("checkpoints.sqlite", check_same_thread=False))

    if args.approve or args.reject:
        run_id = args.approve or args.reject
        graph = build_graph(llm, OrdersAPI(), PaymentsAPI(), TicketsAPI(), checkpointer)
        if not graph.get_state({"configurable": {"thread_id": run_id}}).next:
            sys.exit(f"Run {run_id} is not waiting for approval.")
        print(f"[reviewer] {'approve' if args.approve else 'reject'} {run_id} {args.note}")
        stream(graph, run_id, Command(resume={"approved": bool(args.approve), "note": args.note}))
        return

    for name in [args.scenario] if args.scenario else SCENARIOS:
        message, outages, faults = SCENARIOS[name]
        graph = build_graph(llm, OrdersAPI(outages=outages), PaymentsAPI(faults=list(faults)),
                            TicketsAPI(), checkpointer)
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        print(f"\n{'═' * 68}\n[{name}] customer: {message}")
        paused = stream(graph, run_id, initial_state(message))
        if paused and not args.scenario:   # demo mode: a reviewer approves straight away
            print("\n[reviewer] approve: within policy, customer is a repeat buyer")
            stream(graph, run_id, Command(resume={"approved": True, "note": "repeat buyer"}))


if __name__ == "__main__":
    main()
