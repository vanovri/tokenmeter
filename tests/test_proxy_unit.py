from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from tokenmeter.proxy import (
    SSEStreamScanner,
    create_app,
    estimate_tokens,
    extract_usage,
    get_app,
    parse_body,
    parse_tags,
    redact,
)


def _run_async(app: Any, fn: Any) -> Any:
    async def _go() -> Any:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://tokenmeter.local",
        ) as client:
            return await fn(client)

    return asyncio.run(_go())


def _fetch_rows(db_path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        res = [dict(r) for r in conn.execute("SELECT * FROM requests ORDER BY id")]
    conn.close()
    return res


def test_redact_parse_tags_and_estimate_tokens_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "oa-secret-998877")
    monkeypatch.setenv("NINEROUTER_API_KEY", "nr-secret-112233")

    assert redact(None) is None
    raw_err = (
        "Auth failed: bEaReR eyJhbGciOiJIUzI1NiJ9.payload.sig | "
        "openai=oa-secret-998877 | router=nr-secret-112233 | "
        "raw=sk-proj_AbCdEf123456 | " + ("x" * 2500)
    )
    scrubbed = redact(raw_err)
    assert scrubbed is not None
    assert len(scrubbed) == 2000
    assert "oa-secret-998877" not in scrubbed
    assert "nr-secret-112233" not in scrubbed
    assert "eyJhbGciOiJIUzI1NiJ9" not in scrubbed
    assert "sk-proj_AbCdEf123456" not in scrubbed
    assert "bEaReR [REDACTED]" in scrubbed

    long_key = "k" * 150
    long_val = "v" * 600
    many = "; ".join(
        [f"{long_key}={long_val}", "empty_val=", "bare_flag"] + [f"t{i}=v{i}" for i in range(60)]
    )
    parsed = parse_tags(many)
    assert len(parsed) == 50
    assert parsed[0] == ("k" * 100, "v" * 500)
    assert parsed[1] == ("empty_val", None)
    assert parsed[2] == ("bare_flag", None)
    assert parse_tags(None) == []
    assert parse_tags(" ; , ") == []

    assert estimate_tokens(None) == 0
    assert estimate_tokens("") == 0
    assert estimate_tokens([]) == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens([{"role": "user", "content": "12345"}, "67"]) == 2


def test_extract_usage_openai_anthropic_gemini_and_invalid_tokens() -> None:
    # OpenAI with details
    oa = extract_usage(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "prompt_tokens_details": {"cached_tokens": 25, "audio_tokens": 10},
                "completion_tokens_details": {"reasoning_tokens": 15, "audio_tokens": 5},
            }
        }
    )
    assert oa["prompt_tokens"] == 100
    assert oa["completion_tokens"] == 40
    assert oa["cached_tokens"] == 25
    assert oa["audio_input_tokens"] == 10
    assert oa["reasoning_tokens"] == 15
    assert oa["audio_output_tokens"] == 5

    # Anthropic usage payload (with cache_read_input_tokens + cache_creation_input_tokens)
    anth = extract_usage(
        {
            "usage": {
                "input_tokens": 80,
                "cache_read_input_tokens": 20,
                "output_tokens": 35,
            }
        }
    )
    assert anth["prompt_tokens"] == 100
    assert anth["completion_tokens"] == 35
    assert anth["cached_tokens"] == 20

    # Gemini usageMetadata payload
    gem = extract_usage(
        {
            "usageMetadata": {
                "promptTokenCount": 60,
                "candidatesTokenCount": 22,
                "cachedContentTokenCount": 12,
                "thoughtsTokenCount": 8,
            }
        }
    )
    assert gem["prompt_tokens"] == 60
    assert gem["completion_tokens"] == 30
    assert gem["cached_tokens"] == 12
    assert gem["reasoning_tokens"] == 8

    # Reject bool / negative / NaN token counts cleanly (returns None rather than True/-5)
    bad = extract_usage({"usage": {"prompt_tokens": True, "completion_tokens": -5}})
    assert bad["prompt_tokens"] is None
    assert bad["completion_tokens"] is None


def test_sse_scanner_split_utf8_boundaries_and_multi_provider_frames() -> None:
    scanner = SSEStreamScanner(clock_ms=lambda: 12.5)
    # 4-byte emoji 🚀 (\xf0\x9f\x9a\x80) and 3-byte CJK 你 (\xe4\xbd\xa0) split across chunks!
    frame = 'data: {"choices":[{"delta":{"content":"🚀你"}}]}\r\n\r\n'.encode()
    rocket_pos = frame.index(b"\xf0\x9f\x9a\x80")
    c1 = frame[: rocket_pos + 2]  # split right in the middle of 🚀
    c2 = frame[rocket_pos + 2 : rocket_pos + 5]  # split right in the middle of 你
    c3 = frame[rocket_pos + 5 :]

    scanner.feed(c1)
    assert scanner.ttft_ms is None
    scanner.feed(c2)
    scanner.feed(c3)
    assert scanner.ttft_ms == 12.5
    assert "".join(scanner.text_parts) == "🚀你"

    # Malformed JSON, comments, non-dict JSON values do not crash scanner
    scanner.feed(b": keep-alive\ndata: 123\ndata: [null]\ndata: {malformed\n")
    # Anthropic message_start + message_delta usage frames
    scanner.feed(b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":14}}}\n\n')
    scanner.feed(b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":9}}\n\n')
    scanner.flush()

    assert scanner.prompt_tokens == 14
    assert scanner.completion_tokens == 9


def test_parse_body_handles_non_dict_and_malformed_lines() -> None:
    assert parse_body(b"123") == {}
    assert parse_body(b"data: {broken\ndata: 42\ndata: [DONE]\n") == {}
    gem = parse_body(b'{"usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4}}')
    assert gem["usageMetadata"]["promptTokenCount"] == 9


def test_proxy_rejects_non_object_json_and_handles_lifespan_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prices_file = tmp_path / "prices.yaml"
    prices_file.write_text("per: 1000000\nmodels:\n  m: {input: 1.0, output: 2.0}\n", encoding="utf-8")
    db_file = tmp_path / "lifespan.db"

    def mock_up(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("model") == "gemini-pro":
            return httpx.Response(
                200,
                json={
                    "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                    "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 5},
                },
            )
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 12, "output_tokens": 6},
            },
        )

    app = create_app(
        upstream="http://127.0.0.1:20128/v1/",
        db_path=str(db_file),
        prices_path=str(prices_file),
        transport=httpx.MockTransport(mock_up),
    )

    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"ok": True, "upstream": "http://127.0.0.1:20128/v1"}

        # Invalid JSON syntax -> 400
        r_bad = client.post(
            "/v1/chat/completions",
            content=b"{bad",
            headers={"Content-Type": "application/json"},
        )
        assert r_bad.status_code == 400

        # Valid JSON but not an object (e.g., list or int) -> 400, NOT 500 AttributeError!
        r_list = client.post(
            "/v1/chat/completions",
            content=b"[1, 2, 3]",
            headers={"Content-Type": "application/json"},
        )
        assert r_list.status_code == 400

        # Anthropic & Gemini usage extraction on non-streaming responses
        r_anth = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
        assert r_anth.status_code == 200
        r_gem = client.post("/v1/chat/completions", json={"model": "gemini-pro", "messages": []})
        assert r_gem.status_code == 200

    # After exiting TestClient lifespan context, httpx client and SQLite conn must be closed!
    assert app.state.client.is_closed
    with pytest.raises(sqlite3.ProgrammingError):
        app.state.conn.execute("SELECT 1")

    recorded = _fetch_rows(db_file)
    assert len(recorded) == 2
    assert recorded[0]["prompt_tokens"] == 12 and recorded[0]["completion_tokens"] == 6
    assert recorded[1]["prompt_tokens"] == 15 and recorded[1]["completion_tokens"] == 5

    # Verify get_app() uses env paths cleanly without touching default tokenmeter.db
    monkeypatch.setenv("TOKENMETER_DB", str(tmp_path / "get_app.db"))
    monkeypatch.setenv("TOKENMETER_PRICES", str(prices_file))
    app2 = get_app()
    with TestClient(app2):
        pass
    assert app2.state.client.is_closed


def test_proxy_streaming_split_utf8_and_midstream_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NINEROUTER_API_KEY", "sk-streamsecret-999999")
    prices_file = tmp_path / "prices.yaml"
    prices_file.write_text(
        "per: 1000000\nmodels:\n  frag/model: {input: 2.0, output: 6.0, cached_input: 0.5}\n",
        encoding="utf-8",
    )
    db_file = tmp_path / "stream_unit.db"

    def stream_upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "frag/model":
            # Invalid stream_options from client should have been normalized to dict with include_usage=True
            assert body["stream_options"]["include_usage"] is True
            emoji = "🚀".encode()

            async def fragmented() -> Any:
                yield b'data: {"choices":[{"delta":{"reasoning_content":"think","content":"' + emoji[:2]
                yield emoji[2:] + b'","tool_calls":[{"id":"call_1"}]}}]}\n\n'
                yield (
                    b'data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":10,'
                    b'"prompt_tokens_details":{"cached_tokens":10}}}'
                )

            return httpx.Response(200, content=fragmented(), headers={"content-type": "text/event-stream"})

        if body["model"] == "connect/timeout":
            raise httpx.ReadTimeout("timed out reading stream")

        async def broken_midstream() -> Any:
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadError("peer reset with key sk-streamsecret-999999")

        return httpx.Response(200, content=broken_midstream(), headers={"content-type": "text/event-stream"})

    app = create_app(
        upstream="http://fake/v1",
        db_path=str(db_file),
        prices_path=str(prices_file),
        transport=httpx.MockTransport(stream_upstream),
    )
    try:
        async def consume_ok(c: httpx.AsyncClient) -> tuple[int, bytes]:
            async with c.stream(
                "POST",
                "/p/frag-proj/v1/chat/completions",
                json={
                    "model": "frag/model",
                    "stream": True,
                    "stream_options": "malformed-string",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            ) as resp:
                return resp.status_code, b"".join([chunk async for chunk in resp.aiter_raw()])

        status_ok, raw_ok = _run_async(app, consume_ok)
        assert status_ok == 200
        assert "🚀".encode() in raw_ok

        # Stream connection timeout before headers -> 502
        r_timeout = _run_async(
            app,
            lambda c: c.post(
                "/v1/chat/completions",
                json={"model": "connect/timeout", "stream": True, "messages": []},
            ),
        )
        assert r_timeout.status_code == 502

        # Mid-stream ReadError -> records estimated tokens & redacts secret
        async def consume_broken(c: httpx.AsyncClient) -> None:
            try:
                async with c.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={
                        "model": "broken/stream",
                        "stream": True,
                        "messages": [{"role": "user", "content": "abcd"}],
                    },
                ) as resp:
                    async for _ in resp.aiter_raw():
                        pass
            except httpx.HTTPError:
                return

        _run_async(app, consume_broken)
    finally:
        asyncio.run(app.state.client.aclose())
        app.state.conn.close()

    recorded = _fetch_rows(db_file)
    assert len(recorded) == 3

    first, second, third = recorded
    assert first["project"] == "frag-proj"
    assert first["usage_source"] == "reported"
    assert first["prompt_tokens"] == 20 and first["completion_tokens"] == 10
    # 10 cached (@0.5) + 10 regular (@2.0) + 10 output (@6.0) = 85 / 1e6
    assert first["cost"] == pytest.approx((10 * 0.5 + 10 * 2.0 + 10 * 6.0) / 1_000_000)

    assert second["status_code"] == 502
    assert "ReadTimeout" in second["error"]

    assert third["usage_source"] == "estimated"
    assert third["prompt_tokens"] == 1 and third["completion_tokens"] == 2
    assert "sk-streamsecret-999999" not in third["error"]
    assert "[REDACTED]" in third["error"]
