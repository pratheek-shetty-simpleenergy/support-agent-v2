NORMALIZE_PROMPT_VERSION = "v1"
CLASSIFY_PROMPT_VERSION = "v1"
PLAN_PROMPT_VERSION = "v1"
FINALIZE_PROMPT_VERSION = "v1"


def build_normalize_prompt(raw_message: str, conversation: str) -> str:
    return f"""
You are a support investigation assistant.
Summarize the issue clearly for internal use.

Conversation history:
{conversation}

Latest customer message:
{raw_message}

Return JSON with:
- normalized_issue_summary: concise issue summary
""".strip()


def build_classification_prompt(normalized_issue_summary: str) -> str:
    return f"""
Classify this support issue into a support-friendly category and problem type.

Issue:
{normalized_issue_summary}

Allowed broad categories include:
payment, refund, booking, delivery, mobile_app, app_sync, login, service_complaint,
service_scheduling, ownership_profile, dealership_sales, charging, vehicle, unknown

Return JSON with:
- normalized_issue_summary
- issue_category
- problem_type
- confidence (0 to 1)
""".strip()


def build_investigation_plan_prompt(
    issue_summary: str,
    issue_category: str,
    problem_type: str,
    available_tools: list[str],
    ticket_context: str,
    rag_context: str,
) -> str:
    return f"""
You are planning an investigation for a support ticket.

Issue summary:
{issue_summary}

Issue category: {issue_category}
Problem type: {problem_type}

Ticket context:
{ticket_context}

Retrieved reference context:
{rag_context}

Available tools:
{", ".join(available_tools)}

Choose only tools that help establish facts. Do not invent identifiers.
Business chain to reason over:
- mobile -> user in users DB
- user_id -> enquiry records in orders DB
- successful payment moves flow from Enquiry to Order
- after delivery, ownership records link user_id and order_id to VIN
If a payment-specific tool needs a transaction or payment identifier and none is available, prefer user-level lookup tools first such as order, enquiry, ticket, or profile tools.
For pending-order issues:
- if only user_id is available, use user/order enquiry lookups first
- active pending records may still exist in the enquiry flow before an order is created
Do not choose a tool if its required identifier is unavailable from ticket context.

Return JSON with:
- rationale
- required_tools: list of tool names
- tool_arguments: object keyed by tool name
- should_stop_after_tools: boolean
""".strip()


def build_final_response_prompt(
    issue_summary: str,
    issue_category: str,
    problem_type: str,
    rag_context: str,
    facts: str,
    tool_results: str,
) -> str:
    return f"""
You are finalizing a support investigation result.

Issue summary: {issue_summary}
Issue category: {issue_category}
Problem type: {problem_type}

Retrieved context:
{rag_context}

Facts:
{facts}

Tool results:
{tool_results}

Return JSON with:
- ticket_id
- issue_summary
- issue_category
- problem_type
- decision (resolved, needs_clarification, escalate, pending)
- customer_response: non-empty string for the customer
- internal_summary
- facts: object/dictionary, not a list
- confidence: decimal number between 0 and 1
Use the investigated facts already provided. Do not invent new IDs, statuses, or facts.
Never return null for customer_response.
""".strip()
