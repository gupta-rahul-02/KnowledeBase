# Knowledge Base RAG

A Retrieval-Augmented-Generation knowledge base with a chat UI. Each browser session gets its own isolated FAISS vector store; users upload documents (PDF, TXT, CSV, DOCX, XLSX, JSON, MD) and chat with them via a streamed Groq LLM response.

## Architecture

```
frontend/ (Vite + React + TS)      <-->      src/knowledebase/api.py  (FastAPI)
   chat UI, upload, SSE reader                sessions -> FaissVectorStore + RAGSearch
                                              per-session dir: faiss_store/sessions/<id>/
```

- `data_loader.py` — load PDF/TXT/CSV/DOCX/XLSX/JSON/MD into LangChain documents.
- `embedding.py` — chunk (RecursiveCharacterTextSplitter) + embed (SentenceTransformers `all-MiniLM-L6-v2`).
- `vector_store.py` — FAISS `IndexFlatL2`, append/save/load per session directory.
- `search.py` — `RAGSearch` with history-aware prompt + `stream_answer()` using `ChatGroq.stream()`.
- `session_manager.py` — per-session registry, lazy reload from disk.
- `api.py` — FastAPI endpoints.

## Setup

Requires Python 3.12+, Node 18+, and a Groq API key.

Create `.env` in the repo root:

```
GROQ_API_KEY=your_groq_api_key_here
# Required to call /api/admin/* routes (unset = admin API disabled)
ADMIN_API_KEY=some-long-random-string
# Optional: stale session cleanup (defaults shown)
SESSION_TTL_SECONDS=86400
SESSION_CLEANUP_INTERVAL_SECONDS=3600
# Optional: where the app stores its DB + vector index + uploaded blobs (default: .kb_data)
DATA_DIR=.kb_data
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
- `GET  /api/admin/sessions` — list all sessions on disk with idle time + per-KB summary.
- `POST /api/admin/sessions/cleanup?max_age_seconds=` — manual TTL sweep.

## Notes

- `session_id` + `active_kb_id` are stored in browser `localStorage`; clearing them (or the sidebar "Reset Session" button) starts fresh.
- App state lives entirely under `.kb_data/` (configurable via `DATA_DIR` env var):
  - `.kb_data/db.sqlite` — sessions, KBs, files list, chat history (SQLite).
  - `.kb_data/chroma/` — Chroma persistent vector store (one collection per KB, named `kb_<kb_id>`).
  - `.kb_data/uploads/<kb_id>/` — original uploaded file blobs.
- Legacy single-KB sessions (from earlier versions living under `faiss_store/sessions/`) are auto-migrated into SQLite + Chroma on first startup; the old dir is renamed to `faiss_store/sessions.migrated_<timestamp>/` as a backup.
- A background sweep runs every `SESSION_CLEANUP_INTERVAL_SECONDS` and deletes sessions idle longer than `SESSION_TTL_SECONDS` (both configurable via `.env`). Any request touching a session refreshes its idle timer.

## Deploy (Fly.io, single-image, Pattern A)

Backend + frontend ship as one Docker image. Frontend is built in a `node:20-alpine` stage and served by FastAPI as static files at `/`; `/api/*` routes take precedence over the SPA catch-all.

### Local prod-parity

```powershell
Copy-Item .env.example .env         # then fill in GROQ_API_KEY
docker compose up --build
# open http://localhost:8080
```

`.kb_data` is stored in a named Docker volume so restarts persist data.

### First-time Fly.io setup

```powershell
# One-time
flyctl auth login
flyctl apps create knowledebase-rag
flyctl volumes create kb_data --size 1 --region iad
flyctl secrets set `
    GROQ_API_KEY=your_groq_key `
    ADMIN_API_KEY=$(python -c "import secrets; print(secrets.token_hex(32))") `
    ALLOWED_ORIGINS=https://knowledebase-rag.fly.dev
```

Adjust `app` name and `primary_region` in `fly.toml` to match. The image URL in `fly.toml` (`ghcr.io/gupta-rahul-02/knowledebase:latest`) must be **public** on GHCR, or add `image_auth` credentials.

### CI/CD (GitHub Actions → GHCR → Fly)

The `.github/workflows/build-and-deploy.yml` workflow runs on push to `master`:

1. Builds the multi-stage Dockerfile.
2. Pushes `ghcr.io/<owner>/knowledebase:{latest,sha}` (uses the built-in `GITHUB_TOKEN`).
3. Runs `flyctl deploy --image ...:{sha}` using a `FLY_API_TOKEN` GitHub secret.

**Required GitHub secret**: `FLY_API_TOKEN` — generate with `flyctl auth token` and paste into repo Settings → Secrets → Actions.

After the first successful push, mark the GHCR package as public (GitHub → Packages → Package settings → Change visibility → Public) so Fly can pull without credentials.

### Constraints on Fly free/hobby

- **Single worker only** (`CMD ... --workers 1`) — Chroma's `PersistentClient` isn't multi-process safe.
- **512 MB RAM VM** is the minimum that reliably runs SentenceTransformer + Chroma; 256 MB will OOM.
- **1 GB volume** covers HF model cache (~200 MB), Chroma vectors, and a modest set of uploads. Extend with `flyctl volumes extend`.
- **Auto-stop when idle** is enabled in `fly.toml` (`auto_stop_machines = "stop"`) so idle demo traffic stays within Fly credit limits. First request after idle is ~5 s while the machine wakes.
- **First cold start** downloads the ~80 MB model into `/data/hf_cache`; subsequent starts reuse it from the volume.

