"""One-shot migration from the legacy filesystem layout (faiss_store/sessions/<sid>/kbs/<kid>/) to SQLite + Chroma."""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.knowledebase import db
from src.knowledebase.session_manager import UPLOADS_ROOT, get_shared_embedding_model
from src.knowledebase.vector_store import FaissVectorStore
from src.knowledebase.vector_store_chroma import ChromaKbStore, _chunk_id

LEGACY_ROOT = Path("faiss_store") / "sessions"


def _migrate_kb(kb_dir: Path, kb_id: str) -> None:
    """Move a legacy KB directory into Chroma (vectors + metadata) and update SQLite items."""
    faiss_path = kb_dir / "faiss_index.index"
    meta_path = kb_dir / "metadata.pkl"

    if faiss_path.exists() and meta_path.exists():
        legacy = FaissVectorStore(persist_directory=str(kb_dir), model=get_shared_embedding_model())
        legacy.load()
        if legacy.index is not None and legacy.index.ntotal > 0:
            store = ChromaKbStore(kb_id=kb_id, model=get_shared_embedding_model())
            n = legacy.index.ntotal
            ids: list[str] = []
            embeddings: list[list[float]] = []
            metadatas: list[dict] = []
            texts: list[str] = []
            seen: set[str] = set()
            for i in range(n):
                vec = legacy.index.reconstruct(int(i))
                meta = dict(legacy.metadata[i] or {})
                text = meta.pop("text", "")
                source = meta.get("source", "")
                page = meta.get("page")
                cid = _chunk_id(source, page, text)
                if cid in seen:
                    continue
                seen.add(cid)
                ids.append(cid)
                embeddings.append(np.asarray(vec, dtype="float32").tolist())
                metadatas.append(meta)
                texts.append(text)
            if ids:
                store.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
                print(f"[INFO] Migrated {len(ids)} chunks for KB {kb_id}")

    state_path = kb_dir / "state.json"
    if state_path.exists():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Legacy KB state read failed for {kb_id}: {e}")
            payload = {}
        for name in payload.get("files", []):
            kind = "url" if isinstance(name, str) and name.startswith(("http://", "https://")) else "file"
            db.add_item(kb_id, name, kind, time.time())
        for entry in payload.get("history", []):
            db.append_history(
                kb_id,
                entry.get("role", "user"),
                entry.get("content", ""),
                entry.get("citations"),
                time.time(),
            )

    uploads_src = kb_dir / "uploads"
    if uploads_src.exists() and uploads_src.is_dir():
        dest = UPLOADS_ROOT / kb_id
        dest.mkdir(parents=True, exist_ok=True)
        for item in uploads_src.iterdir():
            try:
                shutil.copy2(item, dest / item.name)
            except Exception as e:
                print(f"[WARN] Copy failed for {item}: {e}")


def _migrate_session(session_dir: Path) -> None:
    session_id = session_dir.name
    session_json = session_dir / "session.json"
    if session_json.exists():
        try:
            payload = json.loads(session_json.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Legacy session state read failed for {session_id}: {e}")
            payload = {}
    else:
        payload = {}

    last_active = float(payload.get("last_active", time.time()))
    created_at = last_active
    db.insert_session(session_id, None, last_active, created_at)

    kbs_dir = session_dir / "kbs"
    active_kb_id: Optional[str] = payload.get("active_kb_id")
    legacy_kb_ids: list[str] = []
    if kbs_dir.exists():
        for kb_dir in kbs_dir.iterdir():
            if not kb_dir.is_dir():
                continue
            kb_id = kb_dir.name
            legacy_kb_ids.append(kb_id)
            kb_state = kb_dir / "state.json"
            name = "Default"
            kb_created = last_active
            if kb_state.exists():
                try:
                    st = json.loads(kb_state.read_text(encoding="utf-8"))
                    name = st.get("name", name)
                    kb_created = float(st.get("created_at", kb_created))
                except Exception:
                    pass
            db.insert_kb(kb_id, session_id, name, kb_created)
            _migrate_kb(kb_dir, kb_id)

    if active_kb_id not in legacy_kb_ids:
        active_kb_id = legacy_kb_ids[0] if legacy_kb_ids else None
    if active_kb_id is not None:
        db.update_session_active_kb(session_id, active_kb_id)


def run_migration_if_needed() -> None:
    """Migrate legacy faiss_store/sessions/... into DB + Chroma if the DB is empty and legacy data exists."""
    db.init_schema()
    if not LEGACY_ROOT.exists():
        return
    existing = db.list_all_sessions()
    if existing:
        return
    session_dirs = [d for d in LEGACY_ROOT.iterdir() if d.is_dir()]
    if not session_dirs:
        return
    print(f"[INFO] Legacy layout detected; migrating {len(session_dirs)} session(s)...")
    for d in session_dirs:
        try:
            _migrate_session(d)
        except Exception as e:
            print(f"[ERROR] Migration failed for session {d.name}: {e}")
    backup = LEGACY_ROOT.parent / f"sessions.migrated_{int(time.time())}"
    try:
        LEGACY_ROOT.rename(backup)
        print(f"[INFO] Legacy layout backed up to {backup}")
    except Exception as e:
        print(f"[WARN] Could not rename legacy layout: {e}")
