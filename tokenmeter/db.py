from __future__ import annotations

import datetime
import math
import os
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

SCHEMA: str = """
CREATE TABLE IF NOT EXISTS requests(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, tag TEXT, model TEXT,
  stream INTEGER, status_code INTEGER, error TEXT, prompt_tokens INTEGER,
  completion_tokens INTEGER, usage_source TEXT, ttft_ms REAL, total_ms REAL,
  cost REAL, price_version INTEGER);
CREATE TABLE IF NOT EXISTS price_versions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, loaded_at TEXT, yaml_sha TEXT UNIQUE, content TEXT);
CREATE TABLE IF NOT EXISTS tags(
  request_id INTEGER NOT NULL, key TEXT NOT NULL, value TEXT,
  FOREIGN KEY(request_id) REFERENCES requests(id));
CREATE INDEX IF NOT EXISTS ix_tags_req ON tags(request_id);
CREATE INDEX IF NOT EXISTS ix_tags_kv ON tags(key, value);
CREATE INDEX IF NOT EXISTS ix_req_price_ver ON requests(price_version);
CREATE INDEX IF NOT EXISTS ix_req_model ON requests(model);
CREATE INDEX IF NOT EXISTS ix_req_ts ON requests(ts);
"""

# Columns added in M2 (migrated with ALTER TABLE on existing DBs)
M2_COLS: dict[str, str] = {
    "project": "TEXT",
    "upstream_ms": "REAL",
    "overhead_ms": "REAL",
}

COLS: list[str] = [
    "ts",
    "tag",
    "project",
    "model",
    "stream",
    "status_code",
    "error",
    "prompt_tokens",
    "completion_tokens",
    "usage_source",
    "ttft_ms",
    "total_ms",
    "upstream_ms",
    "overhead_ms",
    "cost",
    "price_version",
]

_INT_COLS: frozenset[str] = frozenset(
    {"status_code", "prompt_tokens", "completion_tokens", "price_version"}
)
_NONNEG_INT_COLS: frozenset[str] = frozenset({"prompt_tokens", "completion_tokens"})
_FLOAT_COLS: frozenset[str] = frozenset(
    {"ttft_ms", "total_ms", "upstream_ms", "overhead_ms", "cost"}
)

TagPair = tuple[str, str | None]


class RequestRow(TypedDict, total=False):
    ts: str
    tag: str | None
    project: str | None
    model: str | None
    stream: int | bool | None
    status_code: int | None
    error: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    usage_source: str | None
    ttft_ms: float | None
    total_ms: float | None
    upstream_ms: float | None
    overhead_ms: float | None
    cost: float | None
    price_version: int | None


_lock = threading.Lock()
last_backup: str | None = None


def _is_memory_db(path_str: str) -> bool:
    return path_str in (":memory:", "") or "mode=memory" in path_str


def _configure_connection(conn: sqlite3.Connection, path_str: str) -> None:
    if not _is_memory_db(path_str):
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")


def _create_online_backup(conn: sqlite3.Connection, path_str: str) -> str:
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    backup_path = f"{path_str}.bak-{stamp}"
    conn.commit()
    bk = sqlite3.connect(backup_path)
    try:
        conn.backup(bk)
    finally:
        bk.close()
    return backup_path


def _migrate(c: sqlite3.Connection, path: str | os.PathLike[str]) -> str | None:
    path_str = os.fspath(path)
    have = {str(r[1]) for r in c.execute("PRAGMA table_info(requests)")}
    missing = {k: v for k, v in M2_COLS.items() if k not in have}
    if not have or not missing:
        return None
    backup: str | None = None
    if not _is_memory_db(path_str) and os.path.exists(path_str):
        backup = _create_online_backup(c, path_str)
    with c:
        for col_name, col_type in missing.items():
            c.execute(f"ALTER TABLE requests ADD COLUMN {col_name} {col_type}")
    return backup


def connect(path: str | os.PathLike[str]) -> sqlite3.Connection:
    global last_backup
    path_str = os.fspath(path)
    c = sqlite3.connect(path_str, check_same_thread=False)
    _configure_connection(c, path_str)
    exists = c.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='requests'"
    ).fetchone()
    backup = _migrate(c, path_str) if exists else None
    c.executescript(SCHEMA)
    if not exists:
        _migrate(c, ":memory:")
    last_backup = backup
    connect.__dict__["last_backup"] = backup
    return c


connect.__dict__["last_backup"] = None


def normalize_request_row(row: Mapping[str, Any]) -> list[Any]:
    """Validate and normalize request row values prior to SQLite persistence."""
    values: list[Any] = []
    for col in COLS:
        val = row.get(col)
        if val is None:
            values.append(None)
            continue
        if col == "stream":
            if isinstance(val, bool):
                values.append(int(val))
            elif isinstance(val, int):
                values.append(val)
            else:
                raise TypeError(f"stream must be bool or int, got {type(val).__name__}")
            continue
        if col in _INT_COLS:
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"{col} must be int, got {type(val).__name__}")
            if col in _NONNEG_INT_COLS and val < 0:
                raise ValueError(f"{col} must be >= 0, got {val}")
            values.append(val)
            continue
        if col in _FLOAT_COLS:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(f"{col} must be numeric, got {type(val).__name__}")
            fval = float(val)
            if not math.isfinite(fval):
                raise ValueError(f"{col} must be finite, got {val!r}")
            values.append(fval)
            continue
        values.append(str(val))
    return values


def _normalize_tags(tags: Sequence[TagPair] | None) -> list[tuple[str, str | None]]:
    if not tags:
        return []
    out: list[tuple[str, str | None]] = []
    for key, val in tags:
        out.append((str(key), None if val is None else str(val)))
    return out


_INSERT_REQ_SQL: str = (
    f"INSERT INTO requests({','.join(COLS)}) VALUES({','.join('?' * len(COLS))})"
)
_INSERT_TAG_SQL: str = "INSERT INTO tags(request_id,key,value) VALUES(?,?,?)"


def insert(
    conn: sqlite3.Connection,
    row: Mapping[str, Any],
    tags: Sequence[TagPair] | None = None,
) -> int:
    values = normalize_request_row(row)
    clean_tags = _normalize_tags(tags)
    with _lock:
        cur = conn.execute(_INSERT_REQ_SQL, values)
        rid = int(cur.lastrowid or 0)
        if clean_tags:
            conn.executemany(
                _INSERT_TAG_SQL,
                [(rid, k, v) for k, v in clean_tags],
            )
        conn.commit()
        return rid
