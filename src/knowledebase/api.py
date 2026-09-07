import asyncio
import contextlib
import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.knowledebase.data_loader import SUPPORTED_EXTENSIONS, is_valid_url
from src.knowledebase.migration import run_migration_if_needed
from src.knowledebase.session_manager import (
    KnowledgeBase,
    Session,
    get_shared_embedding_model,
    session_manager,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("knowledebase.api")

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(24 * 3600)))
SESSION_CLEANUP_INTERVAL_SECONDS = int(os.getenv("SESSION_CLEANUP_INTERVAL_SECONDS", str(3600)))
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", "frontend_dist"))


class IngestCancelledError(Exception):
    """Raised inside the ingest worker when the client aborts the request."""


def require_admin_key(x_admin_key: Optional[str] = Header(default=None)) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin API is disabled (ADMIN_API_KEY not configured)")
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Key")


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(SESSION_CLEANUP_INTERVAL_SECONDS)
        session_manager.cleanup_stale_sessions(SESSION_TTL_SECONDS)


def _warmup_embedding_model() -> None:
    try:
        model = get_shared_embedding_model()
        model.encode(["warmup"])
        log.info("Embedding model warmed up")
    except Exception as e:
        log.warning("Embedding model warm-up failed: %s", e)


def _fail_fast_startup_checks() -> None:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set; refusing to start")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    _fail_fast_startup_checks()
    try:
        run_migration_if_needed()
    except Exception as e:
        log.error("Startup migration failed: %s", e)
    threading.Thread(target=_warmup_embedding_model, daemon=True).start()
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="Knowledge Base RAG API", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == "POST" and (
        "/documents" in request.url.path or "/urls" in request.url.path
    ):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
            return StreamingResponse(
                iter([json.dumps({"detail": f"Payload too large (max {MAX_UPLOAD_MB} MB)"}).encode()]),
                status_code=413,
                media_type="application/json",
            )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- request/response models ----------

class ChatRequest(BaseModel):
    message: str
    top_k: int = 5


class UrlIngestRequest(BaseModel):
    urls: List[str]


class KbCreateRequest(BaseModel):
    name: Optional[str] = None


class KbRenameRequest(BaseModel):
    name: str


class KbSummary(BaseModel):
    kb_id: str
    name: str
    files_count: int
    created_at: float


class SessionResponse(BaseModel):
    session_id: str
    active_kb_id: Optional[str]
    kbs: List[KbSummary]


# ---------- helpers ----------

def _require_session(session_id: str) -> Session:
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _require_kb(session_id: str, kb_id: str) -> Tuple[Session, KnowledgeBase]:
    session = _require_session(session_id)
    kb = session_manager.get_kb(session, kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return session, kb


def _session_response(session: Session) -> SessionResponse:
    return SessionResponse(
        session_id=session.session_id,
        active_kb_id=session.active_kb_id,
        kbs=[KbSummary(**k) for k in session_manager.list_kbs(session)],
    )


def _run_ingest_stream(
    request: Request,
    session: Session,
    accepted: List[str],
    rejected: List[dict],
    run_fn: Callable[[Callable[[dict], None]], int],
    files_getter: Callable[[], List[str]],
    on_cancel_cleanup: Optional[Callable[[], None]] = None,
) -> StreamingResponse:
    q: "queue.Queue[Any]" = queue.Queue()
    sentinel = object()
    cancel_event = threading.Event()

    def cb(evt: dict) -> None:
        if cancel_event.is_set():
            raise IngestCancelledError()
        q.put(evt)

    def worker() -> None:
        try:
            chunks_added = run_fn(cb)
            q.put(
                {
                    "type": "done",
                    "accepted": accepted,
                    "rejected": rejected,
                    "documents_indexed": chunks_added,
                    "files": files_getter(),
                }
            )
        except IngestCancelledError:
            print("[INFO] Ingest pipeline cancelled by client")
            if on_cancel_cleanup:
                try:
                    on_cancel_cleanup()
                except Exception as e:
                    print(f"[WARN] Cancel cleanup failed: {e}")
            q.put({"type": "cancelled"})
        except Exception as e:
            print(f"[ERROR] Ingest pipeline failed: {e}")
            q.put({"type": "error", "content": str(e)})
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()

    async def event_stream():
        loop = asyncio.get_event_loop()

        async def watch_disconnect() -> None:
            while not cancel_event.is_set():
                if await request.is_disconnected():
                    cancel_event.set()
                    return
                await asyncio.sleep(0.5)

        watcher = asyncio.create_task(watch_disconnect())
        try:
            while True:
                evt = await loop.run_in_executor(None, q.get)
                if evt is sentinel:
                    break
                yield f"data: {json.dumps(evt)}\n\n"
        finally:
            cancel_event.set()
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------- session endpoints ----------

@app.post("/api/sessions", response_model=SessionResponse)
def create_session() -> SessionResponse:
    session = session_manager.create_session()
    return _session_response(session)


@app.get("/api/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    session = _require_session(session_id)
    return _session_response(session)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    ok = session_manager.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


# ---------- KB CRUD ----------

@app.get("/api/sessions/{session_id}/kbs")
def list_kbs(session_id: str) -> dict:
    session = _require_session(session_id)
    return {"active_kb_id": session.active_kb_id, "kbs": session_manager.list_kbs(session)}


@app.post("/api/sessions/{session_id}/kbs")
def create_kb(session_id: str, body: KbCreateRequest) -> dict:
    session = _require_session(session_id)
    kb = session_manager.create_kb(session, body.name)
    return {
        "kb_id": kb.kb_id,
        "name": kb.name,
        "files_count": len(kb.files),
        "created_at": kb.created_at,
    }


@app.patch("/api/sessions/{session_id}/kbs/{kb_id}")
def rename_kb(session_id: str, kb_id: str, body: KbRenameRequest) -> dict:
    session = _require_session(session_id)
    kb = session_manager.rename_kb(session, kb_id, body.name)
    if kb is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return {"kb_id": kb.kb_id, "name": kb.name}


@app.delete("/api/sessions/{session_id}/kbs/{kb_id}")
def delete_kb(session_id: str, kb_id: str) -> dict:
    session = _require_session(session_id)
    ok = session_manager.delete_kb(session, kb_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return {"deleted": True, "active_kb_id": session.active_kb_id}


@app.post("/api/sessions/{session_id}/kbs/{kb_id}/active")
def set_active_kb(session_id: str, kb_id: str) -> dict:
    session = _require_session(session_id)
    ok = session_manager.set_active_kb(session, kb_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return {"active_kb_id": session.active_kb_id}


# ---------- KB documents ----------

@app.get("/api/sessions/{session_id}/kbs/{kb_id}/documents")
def list_documents(session_id: str, kb_id: str) -> dict:
    _, kb = _require_kb(session_id, kb_id)
    return {"files": kb.files}


@app.delete("/api/sessions/{session_id}/kbs/{kb_id}/documents")
def delete_document(session_id: str, kb_id: str, name: str) -> dict:
    session, kb = _require_kb(session_id, kb_id)
    removed = session_manager.delete_document(session, kb, name)
    if removed is None:
        raise HTTPException(status_code=404, detail="Item not found in this knowledge base")
    return {"deleted": name, "chunks_removed": removed, "files": kb.files}


@app.post("/api/sessions/{session_id}/kbs/{kb_id}/documents")
async def upload_documents(
    request: Request,
    session_id: str,
    kb_id: str,
    files: List[UploadFile] = File(...),
):
    session, kb = _require_kb(session_id, kb_id)
    saved_paths: List[str] = []
    original_names: List[str] = []
    rejected: List[dict] = []

    for upload in files:
        name = upload.filename or "unnamed"
        ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in SUPPORTED_EXTENSIONS:
            rejected.append({"name": name, "reason": f"Unsupported extension '{ext}'"})
            continue
        dest = kb.uploads_dir / name
        content = await upload.read()
        dest.write_bytes(content)
        saved_paths.append(str(dest))
        original_names.append(name)

    if not saved_paths:
        raise HTTPException(status_code=400, detail={"message": "No supported files uploaded", "rejected": rejected})

    def run(cb: Callable[[dict], None]) -> int:
        return session_manager.add_documents(session, kb, saved_paths, original_names, progress_callback=cb)

    def cleanup() -> None:
        for p in saved_paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    return _run_ingest_stream(request, session, original_names, rejected, run, lambda: list(kb.files), on_cancel_cleanup=cleanup)


@app.post("/api/sessions/{session_id}/kbs/{kb_id}/urls")
def ingest_urls(request: Request, session_id: str, kb_id: str, req: UrlIngestRequest):
    session, kb = _require_kb(session_id, kb_id)
    accepted: List[str] = []
    rejected: List[dict] = []
    seen: set = set()
    for raw in req.urls:
        u = (raw or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        if not is_valid_url(u):
            rejected.append({"name": u, "reason": "Not a valid http(s) URL"})
            continue
        accepted.append(u)

    if not accepted:
        raise HTTPException(status_code=400, detail={"message": "No valid URLs provided", "rejected": rejected})

    def run(cb: Callable[[dict], None]) -> int:
        return session_manager.add_urls(session, kb, accepted, progress_callback=cb)

    return _run_ingest_stream(request, session, accepted, rejected, run, lambda: list(kb.files))


# ---------- KB chat ----------

@app.get("/api/sessions/{session_id}/kbs/{kb_id}/history")
def get_history(session_id: str, kb_id: str) -> dict:
    _, kb = _require_kb(session_id, kb_id)
    return {"history": kb.history}


@app.post("/api/sessions/{session_id}/kbs/{kb_id}/chat")
def chat(session_id: str, kb_id: str, req: ChatRequest):
    session, kb = _require_kb(session_id, kb_id)

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is empty")

    history_snapshot = list(kb.history)
    session_manager.append_history(session, kb, "user", req.message)

    def event_stream():
        collected: List[str] = []
        try:
            citations, token_iter = kb.rag.stream_answer(
                query=req.message,
                top_k=req.top_k,
                history=history_snapshot,
            )
            yield f"data: {json.dumps({'type': 'citations', 'content': citations})}\n\n"
            for token in token_iter:
                collected.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            full = "".join(collected)
            session_manager.append_history(session, kb, "assistant", full, citations=citations)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            err = str(e)
            log.error("chat stream failed: %s", err)
            yield f"data: {json.dumps({'type': 'error', 'content': err})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------- admin ----------

@app.get("/api/admin/sessions", dependencies=[Depends(require_admin_key)])
def list_sessions() -> dict:
    return {"ttl_seconds": SESSION_TTL_SECONDS, "sessions": session_manager.list_sessions_info()}


@app.post("/api/admin/sessions/cleanup", dependencies=[Depends(require_admin_key)])
def cleanup_sessions(max_age_seconds: Optional[int] = None) -> dict:
    ttl = max_age_seconds if max_age_seconds is not None else SESSION_TTL_SECONDS
    deleted = session_manager.cleanup_stale_sessions(ttl)
    return {"deleted": deleted, "count": len(deleted)}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


# ---------- static frontend (Pattern A: single-image serve) ----------
# Must be registered LAST so /api/* routes above take precedence.

if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
    _assets_dir = FRONTEND_DIST / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    _index_html = FRONTEND_DIST / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_index_html)

    log.info("Serving frontend from %s", FRONTEND_DIST.resolve())
else:
    log.info("Frontend dist not found at %s; running API-only", FRONTEND_DIST)
