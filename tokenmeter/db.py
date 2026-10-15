import sqlite3, threading, shutil, datetime, os

SCHEMA = """
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
"""
# columns added in M2 (migrated with ALTER TABLE on existing DBs)
M2_COLS = {"project": "TEXT", "upstream_ms": "REAL", "overhead_ms": "REAL"}
COLS = ["ts","tag","project","model","stream","status_code","error","prompt_tokens",
        "completion_tokens","usage_source","ttft_ms","total_ms","upstream_ms","overhead_ms",
        "cost","price_version"]
_lock = threading.Lock()

def _migrate(c, path):
    have = {r[1] for r in c.execute("PRAGMA table_info(requests)")}
    missing = {k: v for k, v in M2_COLS.items() if k not in have}
    if not have or not missing:
        return None
    backup = None
    if path not in (":memory:", "") and os.path.exists(path):
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = f"{path}.bak-{stamp}"
        c.commit()
        bk = sqlite3.connect(backup); c.backup(bk); bk.close()   # consistent online backup
    with c:
        for k, v in missing.items():
            c.execute(f"ALTER TABLE requests ADD COLUMN {k} {v}")
    return backup

def connect(path):
    c = sqlite3.connect(path, check_same_thread=False)
    exists = c.execute("SELECT 1 FROM sqlite_master WHERE name='requests'").fetchone()
    backup = _migrate(c, path) if exists else None   # existing DB: back up, then ALTER
    c.executescript(SCHEMA)
    if not exists:
        _migrate(c, ":memory:")   # fresh DB: just add M2 columns, nothing to back up
    connect.last_backup = backup
    return c

def insert(conn, row: dict, tags=None):
    with _lock:
        cur = conn.execute(f"INSERT INTO requests({','.join(COLS)}) VALUES({','.join('?'*len(COLS))})",
                           [row.get(k) for k in COLS])
        rid = cur.lastrowid
        if tags:
            conn.executemany("INSERT INTO tags(request_id,key,value) VALUES(?,?,?)",
                             [(rid, k, v) for k, v in tags])
        conn.commit()
        return rid
