# Support Agent v2

Production-friendly first version of a support investigation agent built with LangGraph, Ollama, Pinecone, and read-only Postgres access.

## Architecture

The code is split by responsibility so reasoning stays inside the agent layer:

- `support_agent/agent`: LangGraph state, nodes, routing, workflow assembly
- `support_agent/llm`: local Ollama/Llama wrapper, prompts, JSON parsing
- `support_agent/retrieval`: Pinecone retriever, embedding adapter, context formatting
- `support_agent/db`: read-only business DB manager, business DB catalog, and repository queries
- `support_agent/tools`: small business DB-backed tools with structured outputs
- `support_agent/schemas`: shared Pydantic models
- `support_agent/services`: app bootstrap
- `scripts`: runnable entrypoints
- `tests`: unit and graph-level tests

## Flow

The graph follows this sequence:

1. Load ticket input
2. Normalize the issue
3. Classify category and problem type
4. Retrieve support context from Pinecone
5. Plan the investigation
6. Run focused DB tools
7. Finalize a structured result

The main intelligence lives in the LangGraph nodes. DB and retrieval layers only fetch facts and reference context.

## Setup

1. Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

2. Create `.env` from `.env.example` and fill in your real values.

Minimum required variables:

```bash
OLLAMA_MODEL=llama3
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_EMBEDDING_MODEL=llama3
PINECONE_API_KEY=...
PINECONE_INDEX=ticket-agent-index
DATABASE_SERVER_URL=postgres://user:pass@host:5432?schema=public
```

Recommended production-oriented variables:

```bash
LOG_LEVEL=INFO
REDACT_PII_IN_LOGS=true
DEPENDENCY_RETRY_ATTEMPTS=2
DEPENDENCY_RETRY_BACKOFF_SECONDS=0.25
OLLAMA_TIMEOUT_SECONDS=60
DB_CONNECT_TIMEOUT_SECONDS=10
DB_STATEMENT_TIMEOUT_MS=10000
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=5
DB_POOL_RECYCLE_SECONDS=1800
PINECONE_REQUIRED=false
```

Business database names:

```bash
BUSINESS_DB_NAMES={"dms_stage":"dms-stage","ownership_stage":"ownership-stage","orders_stage":"orders-stage","users_stage":"users-stage","unified_ticketing_stage":"unified-ticketing-stage"}
```

The agent builds one connection per business database from the shared RDS server URL.

## Business DB Catalog

The internal catalog currently binds capabilities to these databases:

- `dms-stage`: dealer details
- `ownership-stage`: ownership mappings, VIN, registration, vehicle details
- `orders-stage`: `Order`, `Enquiry`, `Transaction`, `Refund`, `PaymentLink`, and related order-side tables
- `users-stage`: user profile data
- `unified-ticketing-stage`: support tickets

The DB layer is intentionally read-only:

- repositories only run `SELECT` queries
- tools expose narrow lookup operations
- no mutation logic exists in the code path

The business DB bindings live in [support_agent/db/catalog.py](/Users/pratheekshetty/Documents/SimpleEnergy/Codebase/github/ai-agent/Support-agent-v2/support_agent/db/catalog.py:1). Once you provide the actual Prisma schema or live schema details, update that file so the approved tables and columns match reality.

## Running

Use the included example ticket or provide your own JSON payload matching `SupportTicketInput`.

```bash
python scripts/run_agent.py examples/sample_ticket.json
```

Run a dependency healthcheck:

```bash
python scripts/healthcheck.py
```

## Output Shape

The graph returns a structured `AgentResult`:

```json
{
  "ticket_id": "string",
  "issue_summary": "string",
  "issue_category": "string",
  "problem_type": "string",
  "decision": "resolved | needs_clarification | escalate | pending",
  "customer_response": "string",
  "internal_summary": "string",
  "facts": {},
  "confidence": 0.0
}
```

## Notes For Production Hardening

This repo now includes a production-hardening pass:

- typed runtime errors for dependency, tool, and model-output failures
- safe tool execution and fallback escalation when investigation tools fail
- bounded retries for Ollama requests
- DB session safety controls through read-only mode and statement timeout
- healthcheck entrypoint for Ollama, DB, and Pinecone
- redacted CLI logging for sensitive fields

Remaining work for a fuller production rollout:

- add centralized structured log sinks and metrics export
- add CI, linting, and type checking
- add deeper integration tests against live dependency contracts
- benchmark local models on your real support-ticket set

## Tests

```bash
pytest
```
