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
