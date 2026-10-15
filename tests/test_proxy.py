import asyncio, json, sqlite3, os
import httpx, pytest
from tokenmeter.proxy import create_app

SECRET = "sk-testsecret-DO-NOT-STORE-123456"
CALLS = []

def fake_upstream(request: httpx.Request):
    CALLS.append(request)
    if request.url.path.endswith("/models"):
        return httpx.Response(200, json={"object": "list", "data": [{"id": "xai/grok-code-fast-1"}]})
    body = json.loads(request.content)
    if body["model"] == "nousage/m" and body.get("stream"):
        async def g2():
            yield b'data: {"choices":[{"delta":{"content":"abcdefgh"}}]}\n\n'
            yield b"data: [DONE]\n\n"
        return httpx.Response(200, content=g2(), headers={"content-type": "text/event-stream"})
    if body["model"] == "bad/model":
        return httpx.Response(404, json={"error": {"message": f"model not found (key {SECRET})"}})
    usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    if not body.get("stream"):
        return httpx.Response(200, json={"id": "x", "choices": [{"message": {"role": "assistant", "content": "hi"}}], "usage": usage})
    assert body["stream_options"]["include_usage"] is True
    chunks = [{"choices": [{"delta": {"role": "assistant"}}]},
              {"choices": [{"delta": {"content": "he"}}]},
              {"choices": [{"delta": {"content": "llo"}}]},
              {"choices": [], "usage": usage}]
    async def gen():
        for c in chunks:
            await asyncio.sleep(0.01)
            yield f"data: {json.dumps(c)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    return httpx.Response(200, content=gen(), headers={"content-type": "text/event-stream"})

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("NINEROUTER_API_KEY", SECRET)
    prices = tmp_path / "prices.yaml"
    prices.write_text("currency: USD\nper: 1000000\nmodels:\n  priced/m: {input: 1.0, output: 2.0}\n  xai/grok-code-fast-1: {input: null, output: null}\n")
    db = tmp_path / "t.db"
    app = create_app("http://fake/v1", str(db), str(prices), transport=httpx.MockTransport(fake_upstream))
    CALLS.clear()
    return app, db, prices

def run(app, fn):
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tm") as c:
            return await fn(c)
    return asyncio.run(go())

def rows(db):
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    return [dict(r) for r in c.execute("SELECT * FROM requests ORDER BY id")]

def test_non_stream_unpriced(env):
    app, db, _ = env
    r = run(app, lambda c: c.post("/v1/chat/completions", json={"model": "xai/grok-code-fast-1", "messages": []}, headers={"X-TokenMeter-Tag": "t1"}))
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "hi"
    [row] = rows(db)
    assert row["tag"] == "t1" and row["prompt_tokens"] == 11 and row["completion_tokens"] == 7
    assert row["cost"] is None and row["status_code"] == 200 and row["total_ms"] > 0 and row["stream"] == 0
    assert CALLS[0].headers["authorization"] == f"Bearer {SECRET}"  # env key forwarded

def test_priced_cost_and_client_auth(env):
    app, db, _ = env
    run(app, lambda c: c.post("/v1/chat/completions", json={"model": "priced/m", "messages": []}, headers={"Authorization": "Bearer client-key"}))
    assert CALLS[0].headers["authorization"] == "Bearer client-key"
    assert rows(db)[0]["cost"] == pytest.approx((11 * 1 + 7 * 2) / 1e6)
    assert rows(db)[0]["price_version"] is not None

def test_stream(env):
    app, db, _ = env
    async def f(c):
        async with c.stream("POST", "/v1/chat/completions", json={"model": "priced/m", "stream": True, "messages": []}, headers={"X-Client": "rag"}) as r:
            return r.status_code, b"".join([b async for b in r.aiter_raw()])
    code, body = run(app, f)
    assert code == 200 and b"[DONE]" in body and b'"llo"' in body
    [row] = rows(db)
    assert row["stream"] == 1 and row["tag"] == "rag" and row["prompt_tokens"] == 11 and row["completion_tokens"] == 7
    assert row["usage_source"] == "reported" and row["ttft_ms"] is not None and row["total_ms"] >= row["ttft_ms"]
    assert row["cost"] == pytest.approx(25 / 1e6)

@pytest.mark.parametrize("stream", [False, True])
def test_error_redacted(env, stream):
    app, db, _ = env
    r = run(app, lambda c: c.post("/v1/chat/completions", json={"model": "bad/model", "stream": stream, "messages": []}))
    assert r.status_code == 404
    [row] = rows(db)
    assert row["status_code"] == 404 and "model not found" in row["error"] and SECRET not in row["error"]
    dump = "\n".join(sqlite3.connect(db).iterdump())
    assert SECRET not in dump

def test_models_passthrough(env):
    app, _, _ = env
    r = run(app, lambda c: c.get("/v1/models"))
    assert r.json()["data"][0]["id"] == "xai/grok-code-fast-1"

def test_upstream_down(tmp_path):
    def boom(req): raise httpx.ConnectError("refused")
    p = tmp_path / "p.yaml"; p.write_text("models: {}\n")
    app = create_app("http://fake/v1", str(tmp_path / "d.db"), str(p), transport=httpx.MockTransport(boom))
    r = run(app, lambda c: c.post("/v1/chat/completions", json={"model": "m", "messages": []}))
    assert r.status_code == 502
    assert rows(tmp_path / "d.db")[0]["status_code"] == 502

def test_shipped_prices_are_valid_and_sourced():
    import yaml
    path = os.path.join(os.path.dirname(__file__), "..", "config", "prices.yaml")
    text = open(path, encoding="utf-8").read()
    d = yaml.safe_load(text)
    for k, v in d["models"].items():
        assert all(v.get(f) is None or float(v[f]) >= 0 for f in ("input", "output"))
    assert text.count("# Source: http") >= len(d["models"])

def test_parse_body_lenient():
    from tokenmeter.proxy import parse_body
    assert parse_body(b'{"usage":{"prompt_tokens":1}}\n{"x":1}')["usage"]["prompt_tokens"] == 1
    assert parse_body(b'data: {"choices":[]}\n\ndata: {"usage":{"prompt_tokens":3}}\n\ndata: [DONE]\n')["usage"]["prompt_tokens"] == 3


def test_tags_project_overhead(env):
    app, db, _ = env
    run(app, lambda c: c.post("/v1/chat/completions", json={"model": "priced/m", "messages": []},
        headers={"X-TokenMeter-Tag": "rag", "X-TokenMeter-Project": "redflag", "X-TokenMeter-Tags": "pattern_id=p7; agent=diagnosis, flag"}))
    [row] = rows(db)
    assert row["project"] == "redflag" and row["tag"] == "rag"
    assert row["upstream_ms"] is not None and row["overhead_ms"] >= 0
    assert row["total_ms"] == pytest.approx(row["upstream_ms"] + row["overhead_ms"], abs=0.2)
    t = sqlite3.connect(db).execute("SELECT key,value FROM tags WHERE request_id=? ORDER BY key", (row["id"],)).fetchall()
    assert t == [("agent", "diagnosis"), ("flag", None), ("pattern_id", "p7")]
    assert "x-tokenmeter-tags" not in {k.lower() for k in CALLS[0].headers}  # not forwarded upstream

def test_stream_overhead_and_upstream(env):
    app, db, _ = env
    async def f(c):
        async with c.stream("POST", "/v1/chat/completions", json={"model": "priced/m", "stream": True, "messages": []}) as r:
            return [b async for b in r.aiter_raw()]
    run(app, f)
    [row] = rows(db)
    assert row["upstream_ms"] >= 30  # 4 chunks x 10ms sleep upstream
    assert 0 <= row["overhead_ms"] <= row["total_ms"]

def test_stream_without_usage_is_estimated(env):
    app, db, _ = env
    async def f(c):
        async with c.stream("POST", "/v1/chat/completions", json={"model": "nousage/m", "stream": True,
                            "messages": [{"role": "user", "content": "x" * 40}]}) as r:
            return [b async for b in r.aiter_raw()]
    run(app, f)
    [row] = rows(db)
    assert row["usage_source"] == "estimated"
    assert row["prompt_tokens"] == 10 and row["completion_tokens"] == 2

def test_price_change_recomputes_and_keeps_history(env):
    app, db, prices = env
    run(app, lambda c: c.post("/v1/chat/completions", json={"model": "xai/grok-code-fast-1", "messages": []}))
    assert rows(db)[0]["cost"] is None
    prices.write_text("per: 1000000\nmodels:\n  xai/grok-code-fast-1: {input: 2.0, output: 4.0}\n")
    app.state.prices.reload()
    r = rows(db)[0]
    assert r["cost"] == pytest.approx((11 * 2 + 7 * 4) / 1e6)
    pv = sqlite3.connect(db).execute("SELECT id FROM price_versions ORDER BY id").fetchall()
    assert len(pv) == 2 and r["price_version"] == pv[-1][0]

def test_migrate_old_db_with_backup(tmp_path):
    from tokenmeter import db as dbm
    p = tmp_path / "old.db"
    c = sqlite3.connect(p)
    c.executescript("""CREATE TABLE requests(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, tag TEXT, model TEXT,
      stream INTEGER, status_code INTEGER, error TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, usage_source TEXT,
      ttft_ms REAL, total_ms REAL, cost REAL, price_version INTEGER);
      CREATE TABLE price_versions(id INTEGER PRIMARY KEY AUTOINCREMENT, loaded_at TEXT, yaml_sha TEXT UNIQUE, content TEXT);
      INSERT INTO requests(ts,model,total_ms) VALUES('t','m',5.0);""")
    c.commit(); c.close()
    conn = dbm.connect(str(p))
    bk = dbm.connect.last_backup
    assert bk and os.path.exists(bk)
    assert sqlite3.connect(bk).execute("SELECT count(*) FROM requests").fetchone()[0] == 1
    cols = {r[1] for r in conn.execute("PRAGMA table_info(requests)")}
    assert {"project", "upstream_ms", "overhead_ms"} <= cols
    assert conn.execute("SELECT model,total_ms FROM requests").fetchall() == [("m", 5.0)]
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='tags'").fetchone()
    conn.close()
    dbm.connect(str(p)); assert dbm.connect.last_backup is None  # idempotent, no second backup

def test_dashboard_renders(env, monkeypatch):
    app, db, _ = env
    run(app, lambda c: c.post("/v1/chat/completions", json={"model": "priced/m", "messages": []}, headers={"X-TokenMeter-Tag": "a", "X-TokenMeter-Tags": "k=v"}))
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("TOKENMETER_DB", str(db))
    monkeypatch.setenv("TOKENMETER_PRICES", str(env[2]))
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "dashboard.py")).run(timeout=60)
    assert not at.exception, at.exception
