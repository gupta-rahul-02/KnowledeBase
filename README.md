# Knowledge Base RAG

A Retrieval-Augmented-Generation knowledge base with a chat UI. Each browser session gets its own set of isolated knowledge bases (KBs) backed by a Qdrant collection; users upload documents (PDF, TXT, CSV, DOCX, XLSX, JSON, MD) and chat with them via a streamed Groq LLM response.

## Architecture

```
frontend/ (Vite + React + TS)          ->     src/knowledebase/api.py  (FastAPI)
   chat UI, upload, SSE reader                    |
                                                  +--> Postgres (Neon)     -- sessions, KBs, files, chat history
                                                  +--> Qdrant Cloud        -- one collection per KB
                                                  +--> Cloudflare R2       -- uploaded document blobs
                                                  +--> Groq API            -- LLM streaming
```

- `data_loader.py` — load PDF/TXT/CSV/DOCX/XLSX/JSON/MD into LangChain documents.
- `embedding.py` — chunk (`RecursiveCharacterTextSplitter`) + embed (SentenceTransformers `all-MiniLM-L6-v2`).
- `vector_store_qdrant.py` — `QdrantKbStore`, one collection per KB (`kb_<kb_id>`), 384-dim / cosine.
- `blob_store.py` — S3-compatible wrapper over Cloudflare R2 for uploaded blobs.
- `db.py` — Postgres persistence (psycopg + pool) for sessions, KBs, files list, chat history.
- `search.py` — `RAGSearch` with history-aware prompt + `stream_answer()` using `ChatGroq.stream()`.
- `session_manager.py` — per-session registry, lazy reload from DB.
- `api.py` — FastAPI endpoints.

## Setup

Requires Python 3.12+, Node 18+, a Groq API key, and free-tier accounts at Neon (Postgres), Qdrant Cloud (vectors), and Cloudflare R2 (object storage). See [Deploy](#deploy-google-cloud-run--managed-free-tiers) for provisioning links.

Copy `.env.example` to `.env` and fill in:

```
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
QDRANT_URL=https://xxxxxxxx.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
R2_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your_r2_access_key
R2_SECRET_ACCESS_KEY=your_r2_secret_key
R2_BUCKET=kb-uploads
# Optional; required only for /api/admin/*
ADMIN_API_KEY=some-long-random-string
# Optional overrides
SESSION_TTL_SECONDS=86400
SESSION_CLEANUP_INTERVAL_SECONDS=3600
MAX_UPLOAD_MB=25
```

### Backend

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.knowledebase.api:app --reload --port 8000
```

Backend runs at `http://localhost:8000`. Swagger UI: `http://localhost:8000/docs`.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173` and proxies `/api` to the backend.

## API

**Session-level**
- `POST /api/sessions` — create a session; returns `{session_id, active_kb_id, kbs}`. A "Default" KB is auto-created.
- `GET  /api/sessions/{sid}` — session info.
- `DELETE /api/sessions/{sid}` — delete the session and all its KBs.

**Knowledge base CRUD**
- `GET  /api/sessions/{sid}/kbs` — list KBs + active id.
- `POST /api/sessions/{sid}/kbs` — body `{name?}`, create a KB.
- `PATCH /api/sessions/{sid}/kbs/{kb_id}` — body `{name}`, rename.
- `DELETE /api/sessions/{sid}/kbs/{kb_id}` — delete a KB.
- `POST /api/sessions/{sid}/kbs/{kb_id}/active` — set active KB.

**KB documents / URLs (SSE streaming progress)**
- `POST /api/sessions/{sid}/kbs/{kb_id}/documents` — multipart file upload; frames: `loading`, `loaded`, `chunking`, `chunked`, `embedding`, `indexing`, `done`, `cancelled`, `error`.
- `POST /api/sessions/{sid}/kbs/{kb_id}/urls` — body `{urls: [str]}`; same SSE shape.
- `GET  /api/sessions/{sid}/kbs/{kb_id}/documents` — list files/URLs.
- `DELETE /api/sessions/{sid}/kbs/{kb_id}/documents?name=<encoded>` — remove a single item.

**KB chat**
- `GET  /api/sessions/{sid}/kbs/{kb_id}/history` — chat history.
- `POST /api/sessions/{sid}/kbs/{kb_id}/chat` — body `{message, top_k}`; SSE frames: `citations`, `token`, `done`, `error`.

**Admin (require `X-Admin-Key`)**
- `GET  /api/admin/sessions` — list all sessions with idle time + per-KB summary.
- `POST /api/admin/sessions/cleanup?max_age_seconds=` — manual TTL sweep.

## Notes

- `session_id` + `active_kb_id` are stored in browser `localStorage`; clearing them (or the sidebar "Reset Session" button) starts fresh.
- App state is fully externalized — the container writes nothing to its local filesystem:
  - **Postgres (Neon)** — `sessions`, `kbs`, `kb_items`, `chat_history` tables.
  - **Qdrant Cloud** — one collection per KB, named `kb_<kb_id>`, 384-dim vectors, cosine distance.
  - **Cloudflare R2** — uploaded document blobs under `kbs/<kb_id>/<filename>`.
- A background sweep runs every `SESSION_CLEANUP_INTERVAL_SECONDS` and deletes sessions idle longer than `SESSION_TTL_SECONDS` (both configurable via `.env`). Any request touching a session refreshes its idle timer.

## Deploy (Google Cloud Run + managed free tiers)

Backend + frontend ship as one Docker image. Frontend is built in a `node:20-alpine` stage and served by FastAPI as static files at `/`; `/api/*` routes take precedence over the SPA catch-all. The container is stateless — all persistence lives in managed free-tier services — so Cloud Run's scale-to-zero doesn't lose data.

### 1. Provision free-tier services

| Service | Free tier | Output |
|---------|-----------|--------|
| [Neon](https://neon.tech) — Postgres | 0.5 GB storage, auto-suspend after 5 min idle | `DATABASE_URL` |
| [Qdrant Cloud](https://qdrant.tech/cloud/) — vectors | 1 GB cluster, always on | `QDRANT_URL`, `QDRANT_API_KEY` |
| [Cloudflare R2](https://developers.cloudflare.com/r2/) — object storage | 10 GB storage, $0 egress | `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` |
| [Groq](https://console.groq.com) — LLM | Free API access | `GROQ_API_KEY` |

Also generate `ADMIN_API_KEY` for the admin endpoints:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

### 2. Local prod-parity (optional)

```powershell
Copy-Item .env.example .env         # fill in all secrets above
docker compose up --build
# open http://localhost:8080
```

Requires Docker Desktop. Skippable — `uvicorn --reload` against the same `.env` gives you the same behavior for dev.

### 3. Deploy to Google Cloud Run

Cloud Run's always-free tier includes 2M requests / month, 360k GB-seconds of memory, and 200k CPU-seconds — comfortable for a personal RAG demo. Cloud Build (used to build the image) is free up to 120 build-minutes / day.

**One-time setup**

1. Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install) and sign in:
   ```powershell
   gcloud auth login
   gcloud config set project <YOUR_PROJECT_ID>
   ```
2. Enable the required APIs and set a default region:
   ```powershell
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
   gcloud config set run/region us-central1
   ```
3. Store the four sensitive secrets in Secret Manager (the non-sensitive R2/Qdrant URLs and the bucket name go as plain env vars in step 4):
   ```powershell
   $env:GROQ_API_KEY | gcloud secrets create GROQ_API_KEY --data-file=-
   $env:ADMIN_API_KEY | gcloud secrets create ADMIN_API_KEY --data-file=-
   $env:DATABASE_URL | gcloud secrets create DATABASE_URL --data-file=-
   $env:QDRANT_API_KEY | gcloud secrets create QDRANT_API_KEY --data-file=-
   $env:R2_SECRET_ACCESS_KEY | gcloud secrets create R2_SECRET_ACCESS_KEY --data-file=-
   ```
   (Populate the `$env:` variables from your local `.env` first, e.g. `$env:GROQ_API_KEY = "gsk_..."`.)

**Deploy**

```powershell
gcloud run deploy knowledebase-rag `
  --source . `
  --allow-unauthenticated `
  --port 8080 `
  --memory 1Gi `
  --cpu 1 `
  --min-instances 0 --max-instances 1 `
  --timeout 300 `
  --concurrency 40 `
  --set-env-vars "LOG_LEVEL=INFO,QDRANT_URL=https://xxxxxxxx.cloud.qdrant.io:6333,R2_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com,R2_ACCESS_KEY_ID=your_r2_access_key,R2_BUCKET=kb-uploads,ALLOWED_ORIGINS=https://knowledebase-rag-<hash>-uc.a.run.app" `
  --update-secrets "GROQ_API_KEY=GROQ_API_KEY:latest,ADMIN_API_KEY=ADMIN_API_KEY:latest,DATABASE_URL=DATABASE_URL:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest,R2_SECRET_ACCESS_KEY=R2_SECRET_ACCESS_KEY:latest"
```

Cloud Build builds the `Dockerfile` in-region, pushes the image to Artifact Registry, and Cloud Run rolls it out. Total first deploy: ~5–10 min.

**Fix ALLOWED_ORIGINS after the first deploy.** The initial deploy prints the assigned URL (`https://knowledebase-rag-<hash>-uc.a.run.app`). Copy that into the `ALLOWED_ORIGINS` env var and redeploy:

```powershell
gcloud run services update knowledebase-rag --update-env-vars "ALLOWED_ORIGINS=https://knowledebase-rag-<hash>-uc.a.run.app"
```

### Constraints

- **`--max-instances 1` is required.** `SessionManager` keeps an in-memory session cache. Multiple Cloud Run instances would each hold their own cache with no invalidation, causing stale reads after deletes. If you ever need horizontal scale, move that cache to Redis (Upstash free tier) or drop it entirely and always rehydrate from Postgres.
- **Cold starts ~5–15 s.** Cloud Run scales to zero when idle; the first request after idle pays image start + FastAPI boot + model load. The Dockerfile already bakes `all-MiniLM-L6-v2` into the image so no HuggingFace download happens on start.
- **Neon auto-suspend.** After 5 min of DB inactivity the Neon compute pauses; the next query pays ~1–2 s wake latency.
- **Uploads pass through the container.** Each file is streamed from the client into R2, then downloaded to `/tmp` for chunking, then discarded. Bump `MAX_UPLOAD_MB` cautiously — the container reads the whole upload into memory once.
- **Free-tier egress from Cloud Run is 1 GB / month in North America.** Sufficient for a personal demo; beyond that Cloud Run charges standard GCP egress rates.
