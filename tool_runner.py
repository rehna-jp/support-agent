import json
# Import mock datasets that simulate a database
from mock_data import CUSTOMERS, ORDERS


# Fetch a customer by matching against multiple possible identifiers
def get_customer(query: str) -> str:
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
                "format CUST-XXXX. You can also search by full name or email "
                "address. Ask the customer to verify their details and try again."
            )
        }
    })


# Fetch an order using its order ID
def lookup_order(order_id: str) -> str:
    order_id = order_id.strip().upper()

    if order_id in ORDERS:
        return json.dumps(ORDERS[order_id])

    return json.dumps(
        {
            "error": {
                "type": "validation",
                "retryable": False,
                "message": (
                    f"No order found with ID '{order_id}'. The order ID may be "
                    "incorrect or in the wrong format. Order IDs follow the format "
                    "ORD-XXXX (e.g. 'ORD-8821'). Ask the customer to double-check the "
                    "order number from their confirmation email/receipt and try again."
                ),
            }
        }
    )



# Central dispatcher that routes tool calls to the correct function
def run_tool(tool_name: str, tool_input: dict) -> str:
    # Route to the appropriate tool based on its name
    if tool_name == "get_customer":
        # Expecting "query" in tool_input
        return get_customer(tool_input["query"])

    elif tool_name == "lookup_order":
        # Expecting "order_id" in tool_input
        return lookup_order(tool_input["order_id"])

    else:
        # Handle unknown tool calls safely with a structured error
        return json.dumps({
            "error": "unknown_tool",
            "message": f"Tool '{tool_name}' is not recognised."
        })