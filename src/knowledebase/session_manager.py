import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from src.knowledebase import db
from src.knowledebase.data_loader import load_documents_from_paths, load_documents_from_urls
from src.knowledebase.search import RAGSearch
from src.knowledebase.vector_store_chroma import ChromaKbStore, get_chroma_client

ProgressCallback = Callable[[dict], None]

DATA_DIR = Path(os.getenv("DATA_DIR", ".kb_data"))
UPLOADS_ROOT = DATA_DIR / "uploads"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_shared_model: Optional[SentenceTransformer] = None
_shared_model_lock = threading.Lock()


def get_shared_embedding_model() -> SentenceTransformer:
    global _shared_model
    if _shared_model is None:
        with _shared_model_lock:
            if _shared_model is None:
                _shared_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                print(f"[INFO] Shared embedding model loaded: {EMBEDDING_MODEL_NAME}")
    return _shared_model


@dataclass
class KnowledgeBase:
    kb_id: str
    name: str
    store: ChromaKbStore
    rag: RAGSearch
    created_at: float
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def uploads_dir(self) -> Path:
        return UPLOADS_ROOT / self.kb_id

    @property
    def files(self) -> List[str]:
        return db.list_items(self.kb_id)

    @property
    def files_count(self) -> int:
        return db.count_items(self.kb_id)

    @property
    def history(self) -> List[dict]:
        return db.get_history(self.kb_id)


@dataclass
class Session:
    session_id: str
    kbs: Dict[str, KnowledgeBase] = field(default_factory=dict)
    active_kb_id: Optional[str] = None
    last_active: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionManager:
    def __init__(self) -> None:
        UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
        db.init_schema()
        self._sessions: Dict[str, Session] = {}
        self._registry_lock = threading.Lock()

    # ---- KB construction ----

    def _build_kb(self, kb_id: str, name: str, created_at: float) -> KnowledgeBase:
        (UPLOADS_ROOT / kb_id).mkdir(parents=True, exist_ok=True)
        model = get_shared_embedding_model()
        store = ChromaKbStore(kb_id=kb_id, model=model)
        rag = RAGSearch(vectorstore=store)
        return KnowledgeBase(kb_id=kb_id, name=name, store=store, rag=rag, created_at=created_at)

    # ---- Session construction ----

    def _build_session_from_db(self, session_id: str) -> Optional[Session]:
        row = db.get_session_row(session_id)
        if row is None:
            return None
        session = Session(session_id=session_id, last_active=row["last_active"], active_kb_id=row["active_kb_id"])
        for kb_row in db.list_kbs_for_session(session_id):
            kb = self._build_kb(kb_row["kb_id"], kb_row["name"], kb_row["created_at"])
            session.kbs[kb.kb_id] = kb
        if session.active_kb_id not in session.kbs:
            session.active_kb_id = next(iter(session.kbs), None)
            if session.active_kb_id != row["active_kb_id"]:
                db.update_session_active_kb(session_id, session.active_kb_id)
        return session

    # ---- Session CRUD ----

    def create_session(self) -> Session:
        session_id = uuid.uuid4().hex
        now = time.time()
        with self._registry_lock:
            db.insert_session(session_id, None, now, now)
            session = Session(session_id=session_id, last_active=now)
            self._sessions[session_id] = session
            default_kb = self._create_kb_unlocked(session, "Default")
            session.active_kb_id = default_kb.kb_id
            db.update_session_active_kb(session_id, default_kb.kb_id)
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._registry_lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
            else:
                session = self._build_session_from_db(session_id)
                if session is None:
                    return None
                self._sessions[session_id] = session
        session.last_active = time.time()
        db.touch_session(session_id, session.last_active)
        return session

    def delete_session(self, session_id: str) -> bool:
        with self._registry_lock:
            session = self._sessions.pop(session_id, None)
            if session:
                for kb in list(session.kbs.values()):
                    kb.store.delete_collection()
            existed = db.delete_session_row(session_id)
            for kb_id in ([kb.kb_id for kb in session.kbs.values()] if session else []):
                shutil.rmtree(UPLOADS_ROOT / kb_id, ignore_errors=True)
            return existed

    # ---- KB CRUD ----

    def _create_kb_unlocked(self, session: Session, name: Optional[str]) -> KnowledgeBase:
        kb_id = uuid.uuid4().hex
        final_name = (name or "Untitled").strip() or "Untitled"
        now = time.time()
        db.insert_kb(kb_id, session.session_id, final_name, now)
        kb = self._build_kb(kb_id, final_name, now)
        session.kbs[kb_id] = kb
        return kb

    def create_kb(self, session: Session, name: Optional[str] = None) -> KnowledgeBase:
        with session.lock:
            kb = self._create_kb_unlocked(session, name)
            if session.active_kb_id is None:
                session.active_kb_id = kb.kb_id
                db.update_session_active_kb(session.session_id, kb.kb_id)
            session.last_active = time.time()
            db.touch_session(session.session_id, session.last_active)
            return kb

    def get_kb(self, session: Session, kb_id: str) -> Optional[KnowledgeBase]:
        return session.kbs.get(kb_id)

    def list_kbs(self, session: Session) -> List[dict]:
        return [
            {"kb_id": kb.kb_id, "name": kb.name, "files_count": kb.files_count, "created_at": kb.created_at}
            for kb in session.kbs.values()
        ]

    def rename_kb(self, session: Session, kb_id: str, name: str) -> Optional[KnowledgeBase]:
        kb = session.kbs.get(kb_id)
        if kb is None:
            return None
        clean = name.strip() or kb.name
        with kb.lock:
            kb.name = clean
            db.update_kb_name(kb_id, clean)
        with session.lock:
            session.last_active = time.time()
            db.touch_session(session.session_id, session.last_active)
        return kb

    def delete_kb(self, session: Session, kb_id: str) -> bool:
        with session.lock:
            kb = session.kbs.pop(kb_id, None)
            if kb is None:
                return False
            kb.store.delete_collection()
            db.delete_kb_row(kb_id)
            shutil.rmtree(UPLOADS_ROOT / kb_id, ignore_errors=True)
            if session.active_kb_id == kb_id:
                session.active_kb_id = next(iter(session.kbs), None)
                db.update_session_active_kb(session.session_id, session.active_kb_id)
            session.last_active = time.time()
            db.touch_session(session.session_id, session.last_active)
        return True

    def set_active_kb(self, session: Session, kb_id: str) -> bool:
        with session.lock:
            if kb_id not in session.kbs:
                return False
            session.active_kb_id = kb_id
            db.update_session_active_kb(session.session_id, kb_id)
            session.last_active = time.time()
            db.touch_session(session.session_id, session.last_active)
        return True

    # ---- KB content ----

    def add_documents(
        self,
        session: Session,
        kb: KnowledgeBase,
        saved_paths: List[str],
        original_names: List[str],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> int:
        with kb.lock:
            docs: List = []
            total = len(saved_paths)
            for i, (path, name) in enumerate(zip(saved_paths, original_names), start=1):
                if progress_callback:
                    progress_callback({"type": "loading", "file": name, "index": i, "total": total})
                file_docs = load_documents_from_paths([path])
                if progress_callback:
                    progress_callback({"type": "loaded", "file": name, "docs": len(file_docs), "index": i, "total": total})
                docs.extend(file_docs)
            if not docs:
                return 0
            kb.store.build_from_documents(docs, progress_callback=progress_callback)
            for name in original_names:
                db.add_item(kb.kb_id, name, "file", time.time())
        with session.lock:
            session.last_active = time.time()
            db.touch_session(session.session_id, session.last_active)
        return len(docs)

    def add_urls(
        self,
        session: Session,
        kb: KnowledgeBase,
        urls: List[str],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> int:
        with kb.lock:
            docs: List = []
            total = len(urls)
            for i, url in enumerate(urls, start=1):
                if progress_callback:
                    progress_callback({"type": "loading", "file": url, "index": i, "total": total})
                url_docs = load_documents_from_urls([url])
                if progress_callback:
                    progress_callback({"type": "loaded", "file": url, "docs": len(url_docs), "index": i, "total": total})
                docs.extend(url_docs)
            if not docs:
                return 0
            kb.store.build_from_documents(docs, progress_callback=progress_callback)
            for url in urls:
                db.add_item(kb.kb_id, url, "url", time.time())
        with session.lock:
            session.last_active = time.time()
            db.touch_session(session.session_id, session.last_active)
        return len(docs)

    def delete_document(self, session: Session, kb: KnowledgeBase, name: str) -> Optional[int]:
        with kb.lock:
            if not db.has_item(kb.kb_id, name):
                return None
            removed = kb.store.remove_document(name)
            file_path = kb.uploads_dir / name
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"[WARN] Failed to delete upload file {file_path}: {e}")
            db.remove_item(kb.kb_id, name)
        with session.lock:
            session.last_active = time.time()
            db.touch_session(session.session_id, session.last_active)
        return removed

    def append_history(
        self,
        session: Session,
        kb: KnowledgeBase,
        role: str,
        content: str,
        citations: Optional[List[dict]] = None,
    ) -> None:
        db.append_history(kb.kb_id, role, content, citations, time.time())
        with session.lock:
            session.last_active = time.time()
            db.touch_session(session.session_id, session.last_active)

    # ---- Admin ----

    def list_sessions_info(self) -> List[dict]:
        infos: List[dict] = []
        for s in db.list_all_sessions():
            kbs_info = []
            for kb_row in db.list_kbs_for_session(s["session_id"]):
                kbs_info.append(
                    {
                        "kb_id": kb_row["kb_id"],
                        "name": kb_row["name"],
                        "files_count": db.count_items(kb_row["kb_id"]),
                    }
                )
            infos.append(
                {
                    "session_id": s["session_id"],
                    "last_active": s["last_active"],
                    "idle_seconds": max(0.0, time.time() - s["last_active"]),
                    "kbs": kbs_info,
                }
            )
        return infos

    def cleanup_stale_sessions(self, max_age_seconds: float) -> List[str]:
        now = time.time()
        deleted: List[str] = []
        for s in db.list_all_sessions():
            if now - s["last_active"] > max_age_seconds:
                if self.delete_session(s["session_id"]):
                    deleted.append(s["session_id"])
        if deleted:
            print(f"[INFO] Cleaned up {len(deleted)} stale session(s): {deleted}")
        return deleted


session_manager = SessionManager()
