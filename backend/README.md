# Hashed RAG — sales assistant backend

A retrieval-grounded chat backend that replaces the website contact form.

The frontend hands it the details the form used to collect (name, email, company,
phone). It then holds a short qualifying conversation: what the visitor needs,
what their budget is, and — if they ask — what the work costs, quoting both a
typical market price and the company's discounted price. It answers questions
about the company from a curated knowledge base and **never speculates**:
anything outside that base gets an honest "I'm an AI and don't know that one, a
representative will follow up."

Every conversation ends in a **lead record** — contact details, stated concern,
budget, quoted prices, unanswered questions and a written summary — stored
locally and optionally forwarded to a CRM.

Backend only. It exposes a small JSON API; the widget is the frontend team's.

---

## Quick start

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

cp .env.example .env          # then set GROQ_API_KEY and the two secrets
.venv/bin/python -m app.knowledge.ingest

.venv/bin/uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> for the interactive API reference.

With Docker:

```bash
docker compose up --build      # indexes on first boot
```

Generate the secrets with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## How it works

```
POST /v1/sessions            ← contact-form payload, returns session_id + token
POST /v1/sessions/{id}/messages   ← one visitor message, one reply
POST /v1/sessions/{id}/close      ← ends it, captures the lead
        │
        ▼
┌──────────────── FastAPI ────────────────┐
│ api/        auth, rate limit, CORS      │
│ chat/       LangGraph turn + 5 tools    │
│ knowledge/  Chroma retriever + pricing  │
│ leads/      isolated capture + outbox   │
└─────────────────────────────────────────┘
     │                    │
  Chroma            SQLite/Postgres
 (public docs)   (sessions, messages, leads)
```

### The conversation is driven by slots, not by the model's mood

The bot has a job — understand the need, *then* ask about budget — and a
free-running agent drifts away from it. So the stage is **derived** from which
facts have been captured, and the current objective is injected into the prompt
each turn:

| Stage | Entered when | Objective |
|---|---|---|
| `greeting` | session start | Greet, invite them to describe the need |
| `discovery` | first message | Understand it; fill `use_case` |
| `qualify` | `use_case` filled | Ask the budget range; fill `budget` |
| `wrap_up` | `budget` filled | Confirm, set follow-up expectations |
| `handoff` | any termination | Hand over warmly and stop |

Slots are filled only through an explicit `record_detail` tool call, so every
transition is logged, replayable and testable without an LLM in the loop.

### Five tools, and why "I don't know" is one of them

| Tool | Purpose |
|---|---|
| `search_company_knowledge` | Vector search over public docs |
| `lookup_pricing` | Deterministic catalog lookup — the only source of numbers |
| `record_detail` | Fills a slot |
| `flag_unanswered` | The honest "I don't know" path |
| `escalate_to_human` | Explicit handoff |

Making the no-speculation rule a *tool call* is what turns it from a hope into a
mechanism: the honest sentence is returned by our code, and the question the bot
could not answer lands in the lead record where sales can act on it.

The set is deliberately small — tool-calling accuracy on mid-size open models
degrades quickly as the list grows. The pricing tool is also **withheld** during
`greeting`/`discovery` unless the visitor mentions money, because availability
enforces what instruction only suggests.

### Prices are not retrieved, they are looked up

Company prose goes through vector search. **Prices do not.** A model
paraphrasing a retrieved chunk will eventually transpose a digit, and a wrong
number in a sales conversation is a real-world problem. Prices live in
`content/public/pricing.yaml` as typed records; `lookup_pricing` is a dictionary
lookup. The model may only relay what it returns.

A final guard extracts every currency figure from each reply and checks it
against the catalog. A number that appears neither in the catalog nor in the
visitor's own messages never reaches them.

### Nothing internal can leak, in three independent layers

1. **Ingestion reads only `content/public`.** `content/internal/` is never
   scanned, embedded or stored — what is not indexed cannot be retrieved, by any
   prompt.
2. **Every query filters `audience == "public"`** — redundant by design, so a
   future change to ingestion cannot silently widen what the bot can see.
3. **The output guard** blocks internal-sounding phrasing on the way out.

`tests/test_leak_prevention.py` asserts a canary phrase from `content/internal`
never reaches a chunk, a retrieval result or a reply.

---

## REST API contract

Everything is JSON. `{id}` is the `session_id` from step 1.

### 1. Start a conversation

```http
POST /v1/sessions
X-Widget-Key: <WIDGET_PUBLIC_KEY>
Content-Type: application/json

{
  "name": "Dana Reyes",
  "email": "dana@brightpath.io",
  "phone": "+1 555 0100",
  "company": "Brightpath",
  "page_url": "https://site.com/contact",
  "utm": {"utm_source": "google", "utm_campaign": "spring"}
}
```

Only `name` is required. `201 Created`:

```json
{
  "session_id": "1cc645c5-…",
  "token": "1cc645c5-….1786107483.cc69f18d…",
  "expires_in": 7200,
  "message": "Hi Dana, thanks for getting in touch with Hashed Systems…",
  "stage": "greeting"
}
```

Render `message` as the first assistant bubble — it is canned, so it costs
nothing and paints instantly. Keep `token` for every later call.

### 2. Send a message

```http
POST /v1/sessions/{id}/messages
Authorization: Bearer <token>
Content-Type: application/json

{"message": "We need a customer portal for our clients"}
```

`200 OK`:

```json
{
  "session_id": "1cc645c5-…",
  "reply": "That sounds like a good fit. Roughly how many users…",
  "stage": "discovery",
  "closed": false,
  "handoff_reason": null
}
```

**When `closed` is `true`, stop sending messages** and show the reply as the
final bubble — the lead has been captured. `handoff_reason` is one of
`completed`, `user_requested`, `agent_escalated`, `knowledge_gap`, `max_turns`,
`idle_timeout`, `error`.

Expect replies to take a few seconds; show a typing indicator.

### 3. Close the conversation

```http
POST /v1/sessions/{id}/close
Authorization: Bearer <token>
```

```json
{"session_id": "…", "closed": true, "lead_captured": true, "message": "A member of our team…"}
```

Call this when the visitor closes the widget. It is **idempotent** — calling it
twice returns `lead_captured: false` the second time and still produces exactly
one lead. Not calling it is also safe: the idle sweeper closes abandoned
conversations after `SESSION_IDLE_MINUTES` and captures the lead anyway.

### Status codes

| Code | Meaning | What the frontend should do |
|---|---|---|
| `401` | Bad widget key, or missing/expired/mismatched token | Start a new session |
| `404` | Unknown session | Start a new session |
| `409` | Conversation already ended | Show the closed state |
| `422` | Empty or over-long message (max 2000 chars) | Validate before sending |
| `429` | Rate limited | Back off and show a gentle notice |
| `500` | Server error | Show a fallback "contact us" message |

### Worked example

```js
const API = "https://bot.example.com";
const WIDGET_KEY = "pub_…";           // public by design, not a secret
let session = null;

async function startChat(contact) {
  const r = await fetch(`${API}/v1/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Widget-Key": WIDGET_KEY },
    body: JSON.stringify(contact),
  });
  if (!r.ok) throw new Error(`start failed: ${r.status}`);
  session = await r.json();
  return session.message;
}

async function send(text) {
  const r = await fetch(`${API}/v1/sessions/${session.session_id}/messages`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${session.token}`,
    },
    body: JSON.stringify({ message: text }),
  });
  if (r.status === 429) return { reply: "One moment — I'm catching up.", closed: false };
  if (!r.ok) throw new Error(`send failed: ${r.status}`);
  return r.json();
}

function endChat() {
  if (!session) return;
  navigator.sendBeacon?.(
    `${API}/v1/sessions/${session.session_id}/close`,
  ) || fetch(`${API}/v1/sessions/${session.session_id}/close`, {
    method: "POST",
    headers: { Authorization: `Bearer ${session.token}` },
    keepalive: true,
  });
}
```

> `sendBeacon` cannot set an `Authorization` header — if you rely on it for
> unload, either accept the fallback `fetch(..., {keepalive: true})` or lean on
> the idle sweeper, which captures the lead regardless.

### CORS

Set `ALLOWED_ORIGINS` to the site origin(s), comma-separated. `*` is refused in
production (logged loudly at startup). The API allows `GET`, `POST`, `OPTIONS`
and the `Content-Type`, `Authorization` and `X-Widget-Key` headers.

---

## Editing content and pricing

```
content/
  public/            ← indexed
    about/ services/ projects/ faq/ process/     *.md
    pricing.yaml     ← the only source of numbers
  internal/          ← NEVER indexed
```

Markdown files take optional front matter:

```markdown
---
title: Frequently Asked Questions
doc_type: faq
audience: public
---
```

- **Prose change** → `POST /v1/admin/reindex` (or re-run the ingest command).
- **Price change** → nothing. The catalog is read per request.

`discount_pct` is always derived from the two prices, so it can never contradict
them. `tests/test_pricing.py` asserts every quote byte-matches the YAML.

---

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq`, `openai`, or `fake` for tests |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | conversation model |
| `LLM_SUMMARY_MODEL` | `llama-3.1-8b-instant` | cheaper; used only for lead summaries |
| `LLM_FALLBACK_MODEL` | `llama-3.1-8b-instant` | used when the main model is rate-limited; empty disables |
| `GROQ_API_KEY` / `OPENAI_API_KEY` | — | whichever provider is selected |
| `EMBEDDING_PROVIDER` | `fastembed` | local ONNX; **Groq has no embeddings API** |
| `DATABASE_URL` | `sqlite+aiosqlite:///./var/app.db` | Postgres works via `postgresql+asyncpg://` |
| `LEAD_SINK` | `sqlite` | `http` also forwards to `CRM_WEBHOOK_URL` |
| `CRM_WEBHOOK_URL` / `CRM_WEBHOOK_SECRET` | — | required when `LEAD_SINK=http` |
| `SESSION_TOKEN_SECRET` | — | **set this**; signs session tokens |
| `ADMIN_API_KEY` | — | **set this**; guards `/v1/admin/*` |
| `WIDGET_PUBLIC_KEY` | `pub_dev` | sent by the frontend; public by design |
| `ALLOWED_ORIGINS` | `*` | comma-separated; pin in production |
| `MAX_TURNS` | `25` | then hands off |
| `MAX_UNANSWERED_STREAK` | `2` | consecutive "I don't know"s before handing off |
| `SESSION_IDLE_MINUTES` | `30` | idle sweeper threshold |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.45` | below this a passage is treated as noise |
| `RATE_LIMIT_MESSAGES_PER_MINUTE` | `20` | per session |
| `RATE_LIMIT_SESSIONS_PER_HOUR` | `10` | per client IP |

### Switching Groq → OpenAI

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_SUMMARY_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-…
```

Nothing else changes. Embeddings stay local either way, so **the vector index
does not need rebuilding** when the chat provider changes.

---

## Lead capture

`app/leads/` is deliberately self-contained — nothing in it imports from
`app.chat`. The rest of the app touches two names, `LeadRecord` and
`LeadService`. Lifting it into its own service later is a directory move.

The local `leads` table is **always** written first and is the durable record.
With `LEAD_SINK=http` it doubles as an outbox: a background worker forwards each
lead to your CRM with exponential backoff (30s → 1h, capped, jittered), signs
the body with HMAC-SHA256, and sends a stable `Idempotency-Key`. If the CRM is
down for a day, nothing is lost.

Delivery states: `not_required` → local only · `pending` → queued · `sent` ·
`failed` → attempts exhausted, needs a human.

Lead scores are deterministic and explainable (`app/leads/scoring.py`): budget
+30, use case +25, saw a quote +15, timeline +10, engaged conversation +10,
business email +10, abandoned −20.

Read them back with `GET /v1/admin/leads?days=7&min_score=50`.

---

## Operations

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Liveness. Deliberately says nothing about internals. |
| `GET /v1/admin/health` | `X-Admin-Key` | Index readiness, provider, pending deliveries |
| `POST /v1/admin/reindex` | `X-Admin-Key` | Rebuild the index after a content edit |
| `GET /v1/admin/leads` | `X-Admin-Key` | List leads (`limit`, `offset`, `days`, `min_score`) |
| `POST /v1/admin/sweep` | `X-Admin-Key` | Close idle sessions now |

Logs are one JSON object per line, correlated by `session_id`. Email addresses
are only ever logged as a truncated SHA-256 (`hash_pii`), never in the clear.

**Only the visitor's first name is ever sent to the LLM.** Email, phone and
company stay in the database and go into the lead record. That matters on a free
tier with weak data guarantees.

---

## Testing

```bash
.venv/bin/python -m pytest          # 101 tests, no network, no API key
.venv/bin/ruff check app tests
```

The suite runs entirely offline: `tests/conftest.py` provides a
`ScriptedChatModel` that replays a fixed sequence of assistant turns — including
tool calls — through the **real** graph, tools and database. Every behavioural
test is a golden transcript rather than a mock of an internal.

Covered: pricing exactness against the YAML, the money extractor's edge cases,
each termination trigger producing exactly one correctly-attributed lead,
double-close idempotency, outbox retry and give-up behaviour, and the three
leak-prevention layers.

---

## Design decisions worth knowing

**No LangGraph checkpointer.** The graph is stateless between turns and rebuilt
from the database each time. One store means one thing to back up, and the
transcript the lead record needs is a by-product rather than a second copy.

**Prior tool traffic is not replayed to the model.** Facts that matter are
promoted into slots; replaying stale tool output wastes tokens and invites
re-quoting figures out of context.

**Summarisation is a separate call on a cheaper model.** If the chat went
sideways, the summariser still sees a clean transcript. If it fails entirely, a
deterministic fallback builds the note from slots — a lead is never lost because
summarisation failed.

**Schema via `create_all`, not migrations.** One less moving part for a handful
of tables. Introduce Alembic the first time you alter a populated table in
production.

**Rate limiting is in-process.** Correct for one instance, which is how this is
sized. More than one replica means moving those counters to Redis — otherwise
each replica enforces the limit independently.

---

## Known limits

- **Groq's free tier is metered per day, per model.** `llama-3.3-70b-versatile`
  has a 100k tokens/day cap and a few dozen real conversations will reach it.
  Two mitigations are built in: on a 429 the turn retries on
  `LLM_FALLBACK_MODEL` (a separate quota), and if that also fails the circuit
  breaker hands off cleanly with the lead still captured. Neither is a
  substitute for a paid tier before this goes public.
- **Mid-size open models are imperfect at tool calling.** They occasionally emit
  a malformed call as plain text; the graph strips it, retries without tools,
  and never shows it to a visitor. Moving to a stronger model reduces the
  frequency.
- **One replica.** See rate limiting above.
- **`fastembed` downloads its model on first run** (~130 MB) unless you use the
  Docker image, which bakes it in.
