# Hashed RAG

A retrieval-grounded sales assistant that replaces the website contact form.
The visitor's details come in from the form, the assistant qualifies the
enquiry in conversation, and every conversation ends as a **lead record** for
the sales team.

```
backend/    FastAPI + LangGraph + Chroma — the whole product
frontend/   Next.js test harness — proves the API works end to end
```

The WordPress team builds the real widget. `frontend/` exists so the API can be
exercised in a browser before that happens, and so there is a working reference
for the three calls involved — see `frontend/lib/api.ts`, which is plain `fetch`
with no framework dependency and is the file worth porting.

**Full documentation, including the REST contract, is in
[`backend/README.md`](backend/README.md).**

---

## Run it locally

Two terminals.

**Backend** (port 8000):

```bash
cd backend
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

cp .env.example .env          # set GROQ_API_KEY and the two secrets
.venv/bin/python -m app.knowledge.ingest

.venv/bin/uvicorn app.main:app --reload --port 8000
```

**Frontend** (port 3000):

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open <http://localhost:3000>. The header shows a live backend health indicator,
so a red dot means the API is not up rather than the page being broken.

Generate the backend secrets with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### With Docker (backend only)

```bash
docker compose up --build      # indexes on first boot
```

Then run the frontend against it as above.

---

## A conversation worth trying

1. *"We need a customer portal where clients can log in and see their orders"*
2. *"Roughly how much would that cost?"* — quotes real figures from
   `backend/content/public/pricing.yaml`, never invented ones
3. *"Our budget is around 20 to 30 thousand"* — should recommend the tier that
   fits rather than re-listing all of them
4. *"Who founded the company?"* — not in the knowledge base, so it should say it
   does not know rather than guess

Then read the lead back:

```bash
curl "http://localhost:8000/v1/admin/leads?limit=1" \
  -H "X-Admin-Key: $(grep '^ADMIN_API_KEY=' backend/.env | cut -d= -f2)" | python -m json.tool
```

---

## Checks

```bash
cd backend  && .venv/bin/python -m pytest && .venv/bin/ruff check app tests
cd frontend && npm run typecheck && npm run build
```

The backend suite is 102 tests and runs fully offline — no API key, no network.

---

## CORS

`ALLOWED_ORIGINS` in `backend/.env` must list the origin the browser loads the
page from — `http://localhost:3000` for the harness, and the WordPress site's
origin in production. A `*` there is refused in production and logged loudly at
startup.
