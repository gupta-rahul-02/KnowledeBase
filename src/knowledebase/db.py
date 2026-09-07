"""SQLite persistence for the session/KB registry, files list, and chat history."""

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

DATA_DIR = Path(os.getenv("DATA_DIR", ".kb_data"))
DATABASE_PATH = DATA_DIR / "db.sqlite"

_write_lock = threading.Lock()


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    _ensure_dir()
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
    finally:
        conn.close()


def init_schema() -> None:
    with _write_lock, get_conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                active_kb_id TEXT,
                last_active  REAL NOT NULL,
                created_at   REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS kbs (
                kb_id       TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                name        TEXT NOT NULL,
                created_at  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_kbs_session ON kbs(session_id);

            CREATE TABLE IF NOT EXISTS kb_items (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_id    TEXT NOT NULL REFERENCES kbs(kb_id) ON DELETE CASCADE,
                name     TEXT NOT NULL,
                kind     TEXT NOT NULL CHECK (kind IN ('file','url')),
                added_at REAL NOT NULL,
                UNIQUE (kb_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_items_kb ON kb_items(kb_id);

            CREATE TABLE IF NOT EXISTS chat_history (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_id          TEXT NOT NULL REFERENCES kbs(kb_id) ON DELETE CASCADE,
                role           TEXT NOT NULL,
                content        TEXT NOT NULL,
                citations_json TEXT,
                ts             REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chat_kb ON chat_history(kb_id);
            """
        )
        c.commit()


# ---- sessions ----

def insert_session(session_id: str, active_kb_id: Optional[str], last_active: float, created_at: float) -> None:
    with _write_lock, get_conn() as c:
        c.execute(
            "INSERT INTO sessions(session_id, active_kb_id, last_active, created_at) VALUES (?,?,?,?)",
            (session_id, active_kb_id, last_active, created_at),
        )
        c.commit()


def session_exists(session_id: str) -> bool:
    with get_conn() as c:
        row = c.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return row is not None


def get_session_row(session_id: str) -> Optional[dict]:
    with get_conn() as c:
        row = c.execute(
            "SELECT session_id, active_kb_id, last_active, created_at FROM sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {"session_id": row[0], "active_kb_id": row[1], "last_active": row[2], "created_at": row[3]}


def update_session_active_kb(session_id: str, active_kb_id: Optional[str]) -> None:
    with _write_lock, get_conn() as c:
        c.execute("UPDATE sessions SET active_kb_id=? WHERE session_id=?", (active_kb_id, session_id))
        c.commit()


def touch_session(session_id: str, last_active: float) -> None:
    with _write_lock, get_conn() as c:
        c.execute("UPDATE sessions SET last_active=? WHERE session_id=?", (last_active, session_id))
        c.commit()


def delete_session_row(session_id: str) -> bool:
    with _write_lock, get_conn() as c:
        cur = c.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        c.commit()
        return cur.rowcount > 0


def list_all_sessions() -> List[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT session_id, active_kb_id, last_active, created_at FROM sessions"
        ).fetchall()
        return [
            {"session_id": r[0], "active_kb_id": r[1], "last_active": r[2], "created_at": r[3]}
            for r in rows
        ]


# ---- kbs ----

def insert_kb(kb_id: str, session_id: str, name: str, created_at: float) -> None:
    with _write_lock, get_conn() as c:
        c.execute(
            "INSERT INTO kbs(kb_id, session_id, name, created_at) VALUES (?,?,?,?)",
            (kb_id, session_id, name, created_at),
        )
        c.commit()


def list_kbs_for_session(session_id: str) -> List[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT kb_id, name, created_at FROM kbs WHERE session_id=? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [{"kb_id": r[0], "name": r[1], "created_at": r[2]} for r in rows]


def get_kb_row(kb_id: str) -> Optional[dict]:
    with get_conn() as c:
        row = c.execute(
            "SELECT kb_id, session_id, name, created_at FROM kbs WHERE kb_id=?",
            (kb_id,),
        ).fetchone()
        if row is None:
            return None
        return {"kb_id": row[0], "session_id": row[1], "name": row[2], "created_at": row[3]}


def update_kb_name(kb_id: str, name: str) -> None:
    with _write_lock, get_conn() as c:
        c.execute("UPDATE kbs SET name=? WHERE kb_id=?", (name, kb_id))
        c.commit()


def delete_kb_row(kb_id: str) -> bool:
    with _write_lock, get_conn() as c:
        cur = c.execute("DELETE FROM kbs WHERE kb_id=?", (kb_id,))
        c.commit()
        return cur.rowcount > 0


# ---- items ----

def add_item(kb_id: str, name: str, kind: str, added_at: float) -> None:
    with _write_lock, get_conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO kb_items(kb_id, name, kind, added_at) VALUES (?,?,?,?)",
            (kb_id, name, kind, added_at),
        )
        c.commit()


def list_items(kb_id: str) -> List[str]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT name FROM kb_items WHERE kb_id=? ORDER BY added_at, id",
            (kb_id,),
        ).fetchall()
        return [r[0] for r in rows]


def count_items(kb_id: str) -> int:
    with get_conn() as c:
        row = c.execute("SELECT count(*) FROM kb_items WHERE kb_id=?", (kb_id,)).fetchone()
        return int(row[0])


def has_item(kb_id: str, name: str) -> bool:
    with get_conn() as c:
        row = c.execute(
            "SELECT 1 FROM kb_items WHERE kb_id=? AND name=?", (kb_id, name)
        ).fetchone()
        return row is not None


def remove_item(kb_id: str, name: str) -> bool:
    with _write_lock, get_conn() as c:
        cur = c.execute("DELETE FROM kb_items WHERE kb_id=? AND name=?", (kb_id, name))
        c.commit()
        return cur.rowcount > 0


# ---- chat history ----

def append_history(kb_id: str, role: str, content: str, citations: Optional[List[dict]], ts: float) -> None:
    with _write_lock, get_conn() as c:
        c.execute(
            "INSERT INTO chat_history(kb_id, role, content, citations_json, ts) VALUES (?,?,?,?,?)",
            (kb_id, role, content, json.dumps(citations) if citations else None, ts),
        )
        c.commit()


def get_history(kb_id: str) -> List[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT role, content, citations_json FROM chat_history WHERE kb_id=? ORDER BY id",
            (kb_id,),
        ).fetchall()
        out: List[dict] = []
        for role, content, cj in rows:
            entry: dict = {"role": role, "content": content}
            if cj:
                try:
                    entry["citations"] = json.loads(cj)
                except Exception:
                    pass
            out.append(entry)
        return out
