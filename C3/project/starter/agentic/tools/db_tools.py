from langchain_core.tools import tool
from agentic.tools.mcp_server import (
    lookup_customer,
    get_user_subscription,
    cancel_subscription_action,
    get_available_experiences,
    get_user_reservations,
    create_reservation_action
)

@tool
def get_customer_profile(email: str) -> str:
    """Look up a customer's ID and status by their email address."""
    return lookup_customer(email)

@tool
def get_subscription_details(user_id: str) -> str:
    """Get subscription plan, status, and quota for a user ID."""
    return get_user_subscription(user_id)

@tool
def cancel_subscription(user_id: str) -> str:
    """Cancel a user's subscription. Use only upon explicit request."""
    return cancel_subscription_action(user_id)

@tool
def list_experiences() -> str:
    """List upcoming cultural experiences available for booking."""
    return get_available_experiences()

@tool
def list_user_reservations(user_id: str) -> str:
    """List the current reservations held by a user."""
    return get_user_reservations(user_id)

@tool
def make_reservation(user_id: str, experience_id: str) -> str:
    """Create a new reservation for a user. Requires User ID and Experience ID."""
    return create_reservation_action(user_id, experience_id)