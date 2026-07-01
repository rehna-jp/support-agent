import anthropic
import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

client = anthropic.Anthropic()

# ─────────────────────────────────────────────────────────────────
# Layer 1 — Persistent case facts (Walkthrough 1)
# ─────────────────────────────────────────────────────────────────

@dataclass
class CaseFacts:
    """
    Persistent structured record of everything important in this support case.

    This lives outside the conversation history. Every API call gets this
    injected at position zero in the system prompt so it's always in the
    well-attended beginning of the context, regardless of how much tool
    output has accumulated since turn 1.

    The case_id is used as the key for stratified sampling in the routing
    layer — same case always gets the same sampling decision.
    """
    case_id: str
    customer_id: Optional[str] = None
    customer_tier: Optional[str] = None
    primary_issues: list[str] = field(default_factory=list)
    orders: dict[str, dict] = field(default_factory=dict)
    refunds_confirmed: list[dict] = field(default_factory=list)
    refunds_pending: list[dict] = field(default_factory=list)
    identity_verified: bool = False
    escalation_requested: bool = False
    escalation_reason: Optional[str] = None
    opened_at: str = field(default_factory=lambda: datetime.now().isoformat())

def format_case_facts(facts: CaseFacts) -> str:
    """
    Format the CaseFacts as a dense text block for system prompt injection.

    Position matters. This goes at the very top of the system prompt —
    before instructions, before context, before anything else. It's the
    anchor that keeps the agent focused on what the customer actually needs
    even when the conversation runs twenty turns and fills up with tool results.
    """
    lines = ["=== ACTIVE CASE FACTS ==="]
    if facts.customer_id:
        lines.append(f"Customer: {facts.customer_id} ({facts.customer_tier or 'tier unknown'})")
    if facts.primary_issues:
        lines.append("ISSUES TO RESOLVE:")
        for i, issue in enumerate(facts.primary_issues, 1):
            lines.append(f"  {i}. {issue}")
    if facts.orders:
        lines.append("ORDERS:")
        for order_id, data in facts.orders.items():
            parts = [f"  {order_id}:"]
            if data.get("duplicate_charge"):
                parts.append(f"duplicate charge ${data['duplicate_charge']:.2f}")
            if data.get("refund_status"):
                parts.append(f"refund {data['refund_status']}")
            if data.get("sla_breach"):
                parts.append("SLA BREACHED")
            lines.append(" ".join(parts))
    if facts.refunds_confirmed:
        lines.append("CONFIRMED REFUNDS:")
        for r in facts.refunds_confirmed:
            lines.append(f"  ${r['amount']:.2f} for {r['order_id']}")
    lines.append(f"Identity verified: {'YES' if facts.identity_verified else 'NO'}")
    if facts.escalation_requested:
        lines.append(f"Escalation requested: YES — {facts.escalation_reason or 'unspecified'}")
    lines.append("=" * 30)
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────
# Layer 2 — Explicit escalation criteria (Walkthrough 2)
# ─────────────────────────────────────────────────────────────────

ESCALATION_SYSTEM = """You are evaluating whether a support case should escalate to a human agent.

ESCALATE when:
1. Customer explicitly requests a human agent (direct request, not frustration)
2. Policy has a genuine gap — situation not covered, human judgment required
3. Agent cannot make meaningful progress after thorough investigation

DO NOT escalate when:
- Customer is frustrated but the issue is solvable
- Case sounds complex but agent has a clear resolution path
- Customer asks about supervisors out of frustration, not as a direct request

Examples:
- "I want to speak with a human agent" → ESCALATE (explicit request)
- "This is absolutely ridiculous, fix it NOW" about a refundable charge → DO NOT ESCALATE
- Calm customer, policy doesn't cover their edge case → ESCALATE (policy gap)"""

def check_escalation(
    latest_message: str,
    case_facts: CaseFacts
) -> dict:
    """
    Evaluate escalation as a separate API call.

    The main agent is focused on resolving the issue. Asking the same
    instance to also evaluate whether to stop and hand off creates a
    conflict — the agent has momentum toward resolution and may
    rationalise away legitimate escalation triggers.

    A separate call with no stake in the resolution evaluates cleanly.
    """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=ESCALATION_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"""Case state:
{format_case_facts(case_facts)}

Latest message: "{latest_message}"

Should this case escalate? Reply: ESCALATE or DO_NOT_ESCALATE, one-sentence reason."""
        }]
    )

    text = response.content[0].text.strip()
    should_escalate = text.startswith("ESCALATE")
    reason = text.split("\n")[1] if "\n" in text else "escalation criteria met"

    return {"should_escalate": should_escalate, "reason": reason}

def build_handoff_summary(facts: CaseFacts, escalation_reason: str, turns: int) -> str:
    """
    Build a complete handoff summary for the human agent taking this case.

    A human agent receiving an escalated case without context has to
    reconstruct the situation from scratch. This summary gives them
    everything they need in thirty seconds: who the customer is, what
    they came in for, what's been done, why we're handing off, and what
    to do next.
    """
    lines = [
        "ESCALATION HANDOFF",
        "=" * 40,
        f"Case ID: {facts.case_id}",
        f"Customer: {facts.customer_id or 'unknown'} ({facts.customer_tier or 'unknown tier'})",
        f"Conversation: {turns} turns",
        f"Identity verified: {'Yes' if facts.identity_verified else 'No'}",
        "",
        "ISSUES REQUIRING RESOLUTION:"
    ]
    for i, issue in enumerate(facts.primary_issues, 1):
        lines.append(f"  {i}. {issue}")
    lines.extend([
        "",
        f"WHY ESCALATED: {escalation_reason}",
        "",
        "RECOMMENDED NEXT ACTION:"
    ])
    if "explicit" in escalation_reason.lower() or "human" in escalation_reason.lower():
        lines.append("  Customer explicitly requested human assistance. Introduce yourself.")
    elif "policy" in escalation_reason.lower():
        lines.append("  Policy gap — judgment call required. Review the case and decide.")
    else:
        lines.append("  Agent could not progress. Review what was attempted and determine next steps.")
    lines.append("=" * 40)
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────
# Layer 3 — Structured subagent failures (Walkthrough 3)
# ─────────────────────────────────────────────────────────────────

async def run_policy_subagent(issue_type: str) -> dict:
    """
    Policy lookup subagent with structured failure handling.

    A timeout or API error returns a structured result with failure
    information rather than propagating an exception. The coordinator
    never sees a raw exception — it always sees a dict it can reason about.

    The coverage_note is what goes into the synthesis prompt when this
    subagent fails — so the synthesis can note the gap rather than
    pretending the policy data is available.
    """
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            timeout=25.0,
            system="You are a policy specialist. Find relevant policies for customer support issues.",
            messages=[{
                "role": "user",
                "content": f"Find policies applicable to: {issue_type}"
            }]
        )
        return {
            "status": "success",
            "data": response.content[0].text,
            "coverage": "complete",
            "coverage_note": None
        }
    except anthropic.APITimeoutError:
        return {
            "status": "failed",
            "failure_type": "timeout",
            "retryable": True,
            "coverage": "none",
            "coverage_note": f"Policy lookup timed out for {issue_type}. Resolution guidance unavailable — human review recommended."
        }
    except anthropic.APIStatusError as e:
        return {
            "status": "failed",
            "failure_type": f"api_error_{e.status_code}",
            "retryable": e.status_code in (429, 500, 502, 503),
            "coverage": "none",
            "coverage_note": f"Policy lookup failed ({e.status_code}) for {issue_type}."
        }
    except Exception as e:
        return {
            "status": "failed",
            "failure_type": "unexpected",
            "retryable": False,
            "coverage": "none",
            "coverage_note": f"Policy lookup failed unexpectedly: {str(e)}"
        }

# ─────────────────────────────────────────────────────────────────
# Layer 4 — Tool result trimming (Walkthrough 1 companion)
# ─────────────────────────────────────────────────────────────────

RELEVANT_FIELDS = {
    "get_customer": ["customer_id", "name", "tier", "account_status", "email"],
    "lookup_order": ["order_id", "status", "total_amount", "created_at", "payment_records"],
    "check_refund_status": ["refund_id", "amount", "status", "initiated_at", "sla_breach"],
    "verify_identity": ["verified", "verification_token"],
    "process_refund": ["order_id", "amount", "status", "confirmation_id"]
}

def trim_tool_result(tool_name: str, result: dict) -> dict:
    """
    Filter a tool result to only the fields the agent needs for its decisions.

    The raw get_customer result might have 40 fields. The agent needs 5 to
    handle a billing dispute. The other 35 accumulate in context on every
    turn and compete for attention with what actually matters.

    This runs on every tool result before it enters the conversation history.
    The full result still goes to update_case_facts (below) — trimming only
    applies to what gets stored in the conversation.
    """
    if tool_name not in RELEVANT_FIELDS:
        return result
    relevant = RELEVANT_FIELDS[tool_name]
    return {k: v for k, v in result.items() if k in relevant}

def update_case_facts(facts: CaseFacts, tool_name: str, result: dict) -> None:
    """
    Update the CaseFacts object from a full (untrimmed) tool result.

    Uses the full result so case facts have complete information.
    The trimming only applies to what goes into conversation history.
    """
    if tool_name == "get_customer":
        facts.customer_id = result.get("customer_id")
        facts.customer_tier = result.get("tier")
    elif tool_name == "verify_identity":
        facts.identity_verified = result.get("verified", False)
    elif tool_name == "check_refund_status":
        refund_id = result.get("refund_id", "")
        order_id = refund_id.replace("R-", "") if refund_id else None
        if order_id and order_id in facts.orders:
            facts.orders[order_id]["refund_status"] = result.get("status")
            if result.get("sla_breach"):
                facts.orders[order_id]["sla_breach"] = True
    elif tool_name == "process_refund":
        order_id = result.get("order_id")
        amount = result.get("amount")
        if order_id and amount:
            facts.refunds_confirmed.append({"order_id": order_id, "amount": amount})
            facts.refunds_pending = [r for r in facts.refunds_pending
                                     if r.get("order_id") != order_id]

# ─────────────────────────────────────────────────────────────────
# The main agent loop
# ─────────────────────────────────────────────────────────────────

BASE_SYSTEM = """You are a customer support agent. You have tools to look up customers,
orders, and process refunds. Resolve every issue the customer raised.
Before closing, confirm all open issues are addressed."""

TOOLS = [
    {
        "name": "get_customer",
        "description": "Retrieve customer account information by customer ID",
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"]
        }
    },
    {
        "name": "lookup_order",
        "description": "Look up order details by order ID",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
        }
    },
    {
        "name": "verify_identity",
        "description": "Verify customer identity before processing refunds",
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"]
        }
    },
    {
        "name": "process_refund",
        "description": "Process a refund for a customer order. Requires identity verification first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"}
            },
            "required": ["order_id", "amount"]
        }
    },
    {
        "name": "check_refund_status",
        "description": "Check the current status of a refund by order ID",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
        }
    }
]

def run_agent(case_id: str, opening_message: str, mock_tool_executor=None) -> dict:
    """
    The main agent loop with all five reliability layers active.

    mock_tool_executor is used in tests to control what tools return.
    In production this would call real backend services.
    """
    facts = CaseFacts(case_id=case_id)
    conversation = [{"role": "user", "content": opening_message}]
    turn_count = 0
    result = {"case_id": case_id, "resolved": False, "routing": "in_progress"}

    # Escalation check on the opening message — some customers open with
    # "I want to speak to a manager" and there's no point starting an investigation
    initial_escalation = check_escalation(opening_message, facts)
    if initial_escalation["should_escalate"]:
        handoff = build_handoff_summary(facts, initial_escalation["reason"], 0)
        result.update({
            "routing": "escalated",
            "escalation_reason": initial_escalation["reason"],
            "handoff_summary": handoff
        })
        return result

    while turn_count < 30:  # Hard limit — prevents infinite loops
        turn_count += 1

        # Case facts at position zero — always in the well-attended beginning
        system_prompt = f"{format_case_facts(facts)}\n\n{BASE_SYSTEM}"

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            tools=TOOLS,
            messages=conversation
        )

        if response.stop_reason == "tool_use":
            tool_calls = [b for b in response.content if b.type == "tool_use"]
            conversation.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tc in tool_calls:
                # Execute the tool — use mock in tests, real service in production
                if mock_tool_executor:
                    raw_result = mock_tool_executor(tc.name, tc.input)
                else:
                    raw_result = {"error": "no tool executor configured"}

                # Update case facts from the full result before trimming
                update_case_facts(facts, tc.name, raw_result)

                # Store only trimmed result in conversation history
                trimmed = trim_tool_result(tc.name, raw_result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": json.dumps(trimmed)
                })

            conversation.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            agent_response = next(
                (b.text for b in response.content if b.type == "text"), ""
            )

            # Get the next customer message from the conversation fixture
            # In real usage this would be the actual next customer message
            result["last_agent_response"] = agent_response

            # Check whether all primary issues are resolved
            unresolved = [i for i in facts.primary_issues
                         if not any(r.get("order_id") in i for r in facts.refunds_confirmed)]

            if not unresolved or not facts.primary_issues:
                result.update({"resolved": True, "routing": "resolved", "turns": turn_count})
                return result

            # Issues still open — would continue in a real conversation
            # For the test scenarios, we treat end_turn as resolution attempt
            result.update({
                "resolved": len(facts.refunds_confirmed) > 0,
                "routing": "resolved" if facts.refunds_confirmed else "unresolved",
                "turns": turn_count,
                "case_facts": facts
            })
            return result

    result.update({"routing": "timeout", "turns": turn_count})
    return result