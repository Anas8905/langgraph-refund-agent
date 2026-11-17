"""Refund policy as plain code. The LLM never decides whether money moves."""
from datetime import date

from .state import Order

REFUND_WINDOW_DAYS = 30
AUTO_APPROVE_LIMIT = 100.00


def evaluate(order: Order, today: date, already_refunded: bool) -> tuple[str, str]:
    """Return (decision, reason). decision is auto_approve | needs_approval | deny."""
    if already_refunded:
        return "deny", f"order {order['id']} has already been refunded"
    if order["final_sale"]:
        return "deny", "final-sale items are not refundable"
    if order["status"] != "delivered":
        return "deny", f"order is {order['status']}, not delivered"
    age = (today - date.fromisoformat(order["delivered_on"])).days
    if age > REFUND_WINDOW_DAYS:
        return "deny", f"delivered {age} days ago; the refund window is {REFUND_WINDOW_DAYS} days"
    if order["total"] > AUTO_APPROVE_LIMIT:
        return "needs_approval", f"${order['total']:.2f} is over the ${AUTO_APPROVE_LIMIT:.0f} auto-approve limit"
    return "auto_approve", f"delivered {age} days ago and under ${AUTO_APPROVE_LIMIT:.0f}"
