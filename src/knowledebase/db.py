"""Postgres persistence for the session/KB registry, files list, and chat history."""

import json
import os
import threading
from contextlib import contextmanager
from typing import Iterator, List, Optional

from psycopg import Connection
from psycopg_pool import ConnectionPool

_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                url = os.getenv("DATABASE_URL")
                if not url:
                    raise RuntimeError("DATABASE_URL is not set")
                _pool = ConnectionPool(
                    url,
                    min_size=int(os.getenv("DB_POOL_MIN", "1")),
                    max_size=int(os.getenv("DB_POOL_MAX", "10")),
                    open=True,
                )
    return _pool


@contextmanager
def get_conn() -> Iterator[Connection]:
    pool = _get_pool()
    with pool.connection() as conn:
        yield conn


def init_schema() -> None:
    with get_conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   text PRIMARY KEY,
                active_kb_id text,
                last_active  double precision NOT NULL,
                created_at   double precision NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS kbs (
                kb_id       text PRIMARY KEY,
                session_id  text NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                name        text NOT NULL,
                created_at  double precision NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_kbs_session ON kbs(session_id)")
        c.execute("""
            CREATE TABLE IF NOT EXISTS kb_items (
                id       bigserial PRIMARY KEY,
                kb_id    text NOT NULL REFERENCES kbs(kb_id) ON DELETE CASCADE,
                name     text NOT NULL,
                kind     text NOT NULL CHECK (kind IN ('file','url')),
                added_at double precision NOT NULL,
                UNIQUE (kb_id, name)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_items_kb ON kb_items(kb_id)")
        c.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id             bigserial PRIMARY KEY,
                kb_id          text NOT NULL REFERENCES kbs(kb_id) ON DELETE CASCADE,
                role           text NOT NULL,
                content        text NOT NULL,
                citations_json text,
                ts             double precision NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_chat_kb ON chat_history(kb_id)")


# ---- sessions ----

def insert_session(session_id: str, active_kb_id: Optional[str], last_active: float, created_at: float) -> None:
    with get_conn() as c:
        c.execute(
            "INSERT INTO sessions(session_id, active_kb_id, last_active, created_at) VALUES (%s,%s,%s,%s)",
            (session_id, active_kb_id, last_active, created_at),
        )


def session_exists(session_id: str) -> bool:
    with get_conn() as c:
        row = c.execute("SELECT 1 FROM sessions WHERE session_id=%s", (session_id,)).fetchone()
        return row is not None


def get_session_row(session_id: str) -> Optional[dict]:
    with get_conn() as c:
        row = c.execute(
            "SELECT session_id, active_kb_id, last_active, created_at FROM sessions WHERE session_id=%s",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {"session_id": row[0], "active_kb_id": row[1], "last_active": row[2], "created_at": row[3]}


def update_session_active_kb(session_id: str, active_kb_id: Optional[str]) -> None:
    with get_conn() as c:
        c.execute("UPDATE sessions SET active_kb_id=%s WHERE session_id=%s", (active_kb_id, session_id))


def touch_session(session_id: str, last_active: float) -> None:
    with get_conn() as c:
        c.execute("UPDATE sessions SET last_active=%s WHERE session_id=%s", (last_active, session_id))


def delete_session_row(session_id: str) -> bool:
    with get_conn() as c:
        cur = c.execute("DELETE FROM sessions WHERE session_id=%s", (session_id,))
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
    with get_conn() as c:
        c.execute(
            "INSERT INTO kbs(kb_id, session_id, name, created_at) VALUES (%s,%s,%s,%s)",
            (kb_id, session_id, name, created_at),
        )


def list_kbs_for_session(session_id: str) -> List[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT kb_id, name, created_at FROM kbs WHERE session_id=%s ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [{"kb_id": r[0], "name": r[1], "created_at": r[2]} for r in rows]


def get_kb_row(kb_id: str) -> Optional[dict]:
    with get_conn() as c:
        row = c.execute(
            "SELECT kb_id, session_id, name, created_at FROM kbs WHERE kb_id=%s",
            (kb_id,),
        ).fetchone()
        if row is None:
            return None
        return {"kb_id": row[0], "session_id": row[1], "name": row[2], "created_at": row[3]}


def update_kb_name(kb_id: str, name: str) -> None:
    with get_conn() as c:
        c.execute("UPDATE kbs SET name=%s WHERE kb_id=%s", (name, kb_id))


def delete_kb_row(kb_id: str) -> bool:
    with get_conn() as c:
        cur = c.execute("DELETE FROM kbs WHERE kb_id=%s", (kb_id,))
        return cur.rowcount > 0


# ---- items ----

def add_item(kb_id: str, name: str, kind: str, added_at: float) -> None:
    with get_conn() as c:
        c.execute(
            "INSERT INTO kb_items(kb_id, name, kind, added_at) VALUES (%s,%s,%s,%s) "
            "ON CONFLICT (kb_id, name) DO NOTHING",
            (kb_id, name, kind, added_at),
        )


def list_items(kb_id: str) -> List[str]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT name FROM kb_items WHERE kb_id=%s ORDER BY added_at, id",
            (kb_id,),
        ).fetchall()
        return [r[0] for r in rows]


def count_items(kb_id: str) -> int:
    with get_conn() as c:
        row = c.execute("SELECT count(*) FROM kb_items WHERE kb_id=%s", (kb_id,)).fetchone()
        return int(row[0]) if row else 0


def has_item(kb_id: str, name: str) -> bool:
    with get_conn() as c:
        row = c.execute(
            "SELECT 1 FROM kb_items WHERE kb_id=%s AND name=%s", (kb_id, name)
        ).fetchone()
        return row is not None


def remove_item(kb_id: str, name: str) -> bool:
    with get_conn() as c:
        cur = c.execute("DELETE FROM kb_items WHERE kb_id=%s AND name=%s", (kb_id, name))
        return cur.rowcount > 0


# ---- chat history ----

def append_history(kb_id: str, role: str, content: str, citations: Optional[List[dict]], ts: float) -> None:
    with get_conn() as c:
        c.execute(
            "INSERT INTO chat_history(kb_id, role, content, citations_json, ts) VALUES (%s,%s,%s,%s,%s)",
            (kb_id, role, content, json.dumps(citations) if citations else None, ts),
        )


def get_history(kb_id: str) -> List[dict]:
    with get_conn() as c:
        rows = c.execute(
            "SELECT role, content, citations_json FROM chat_history WHERE kb_id=%s ORDER BY id",
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
