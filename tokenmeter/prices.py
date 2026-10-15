import hashlib, datetime, yaml
from . import db as dbm

def compute_cost(data, model, pt, ct):
    """Cost or None if unpriced / tokens unknown. Never guesses prices."""
    # keyed by the REQUESTED model id; upstream-reported names (e.g. grok-build-0.1) are ignored
    m = ((data or {}).get("models") or {}).get(model) or {}
    if pt is None or ct is None: return None
    lc = m.get("long_context") or {}
    if lc.get("threshold") is not None and pt >= lc["threshold"] and lc.get("input") is not None:
        m = lc
    i, o = m.get("input"), m.get("output")
    if i is None or o is None: return None
    per = float(data.get("per") or 1_000_000)
    return (pt * float(i) + ct * float(o)) / per

class Prices:
    def __init__(self, path, conn=None):
        self.path, self.conn = path, conn
        self.version = None; self.data = {}; self._sha = None
        self.reload()

    def reload(self):
        """Re-read YAML; on change store a new price_versions row (history kept)
        and recompute cost of all existing rows under the new prices."""
        raw = open(self.path, "rb").read()
        sha = hashlib.sha256(raw).hexdigest()
        if sha == self._sha: return False
        self._sha = sha
        self.data = yaml.safe_load(raw) or {}
        if self.conn is None: return True
        with dbm._lock:
            c = self.conn
            row = c.execute("SELECT id FROM price_versions WHERE yaml_sha=?", (sha,)).fetchone()
            if row:  # same content seen before (e.g. revert) => reuse that version id
                self.version = row[0]
            else:  # new content => new version row; older rows stay as history
                c.execute("INSERT INTO price_versions(loaded_at,yaml_sha,content) VALUES(?,?,?)",
                          (datetime.datetime.now(datetime.timezone.utc).isoformat(), sha, raw.decode()))
                self.version = c.execute("SELECT last_insert_rowid()").fetchone()[0]
            stale = c.execute("SELECT id, model, prompt_tokens, completion_tokens FROM requests "
                              "WHERE price_version IS NULL OR price_version != ?", (self.version,)).fetchall()
            c.executemany("UPDATE requests SET cost=?, price_version=? WHERE id=?",
                          [(compute_cost(self.data, m, pt, ct), self.version, i) for i, m, pt, ct in stale])
            c.commit()
        return True

    def cost(self, model, pt, ct):
        try: self.reload()
        except Exception: pass
        return compute_cost(self.data, model, pt, ct)
