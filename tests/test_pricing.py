import asyncio
import sqlite3
from contextlib import closing

import httpx

from tests.test_proxy import run
from tokenmeter.prices import compute_cost
from tokenmeter.proxy import create_app


def up(req):
    # upstream reports a different model name than requested (9Router alias)
    return httpx.Response(
        200,
        json={
            "model": "grok-build-0.1",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        },
    )


def test_prices_by_requested_model(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("per: 1000000\nmodels:\n  xai/grok-code-fast-1: {input: 1.0, output: 2.0}\n", encoding="utf-8")
    db = tmp_path / "t.db"
    app = create_app("http://fake/v1", str(db), str(p), transport=httpx.MockTransport(up))

    async def fn(c):
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "xai/grok-code-fast-1", "messages": [{"role": "user", "content": "x"}]},
        )
        assert r.status_code == 200

    try:
        run(app, fn)
    finally:
        asyncio.run(app.state.client.aclose())
        app.state.conn.close()

    with closing(sqlite3.connect(db)) as conn:
        model, cost = conn.execute("SELECT model, cost FROM requests").fetchone()
    assert model == "xai/grok-code-fast-1"
    assert abs(cost - (1000 * 1.0 + 500 * 2.0) / 1e6) < 1e-12


def test_long_context_tier():
    d = {
        "per": 1000000,
        "models": {"m": {"input": 1, "output": 2, "long_context": {"threshold": 200000, "input": 2, "output": 4}}},
    }
    assert compute_cost(d, "m", 199999, 0) == 199999 / 1e6
    assert compute_cost(d, "m", 200000, 10) == (200000 * 2 + 10 * 4) / 1e6
    assert compute_cost(d, "grok-build-0.1", 10, 10) is None
