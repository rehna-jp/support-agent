# mcp_server.py

import json
import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mock_data import CUSTOMERS, ORDERS

load_dotenv()

mcp = FastMCP("support-agent-tools")

@mcp.tool()
def get_customer(query: str) -> str:
    """
    Look up a customer record by name, email address, or customer ID.
    Use this tool when you need to verify who you are speaking with or
    retrieve account information. Returns the customer's full profile
    including account status, contact details, and a list of their order IDs.
    Do not use this tool to look up order details — use lookup_order for that.

    Args:
        query: The search term. Can be a full name (e.g. 'Sarah Chen'),
               an email address (e.g. 'sarah@email.com'),
               or a customer ID (e.g. 'CUST-4492').
    """
    query = query.strip().lower()

    for customer in CUSTOMERS.values():
        if (
            query == customer["customer_id"].lower()
            or query == customer["email"].lower()
            or query == customer["name"].lower()
        ):
            return json.dumps(customer)

    return json.dumps({
        "error": {
            "type": "validation",
            "retryable": False,
            "message": (
                f"No customer found matching '{query}'. The input may be "
                "misspelled or in the wrong format. Customer IDs follow the "
                "format CUST-XXXX. You can also search by full name or "
                "email address. Ask the customer to verify their details."
            )
        }
    })

@mcp.tool()
def lookup_order(order_id: str) -> str:
    """
    Look up a specific order by order ID. Use this tool when you need
    details about a particular order — status, items, shipping information,
    estimated delivery dates, or notes. Requires a valid order ID.
    To find a customer's order IDs, call get_customer first.

    Args:
        order_id: The order ID to look up. Must follow the format ORD-XXXX
                  (e.g. 'ORD-8821'). This must be an exact match.
    """
    order_id = order_id.strip().upper()

    if order_id in ORDERS:
        return json.dumps(ORDERS[order_id])

    return json.dumps({
        "error": {
            "type": "validation",
            "retryable": False,
            "message": (
                f"No order found with ID '{order_id}'. "
                "Please check the order ID and try again."
            )
        }
    })

if __name__ == "__main__":
    mcp.run(transport="stdio")