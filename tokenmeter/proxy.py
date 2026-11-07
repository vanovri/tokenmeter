"""TokenMeter: OpenAI-compatible logging proxy. Never logs/stores API keys."""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import math
import os
import re
import sqlite3
import time
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, TypedDict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import db as dbm
from . import otel as otelm
from .prices import Prices

log = logging.getLogger(__name__)

HOP: frozenset[str] = frozenset(
    {
        "x-tokenmeter-project",
        "x-tokenmeter-tags",
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "accept-encoding",
        "x-tokenmeter-tag",
        "x-client",
    }
)

_RE_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+")
_RE_SK_KEY = re.compile(r"sk-[A-Za-z0-9._\-]{6,}")
_RE_TAG_SPLIT = re.compile(r"[;,]")

TagPair = tuple[str, str | None]


class UsageInfo(TypedDict):
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    audio_input_tokens: int | None
    audio_output_tokens: int | None


def _empty_usage() -> UsageInfo:
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
        "reasoning_tokens": None,
        "audio_input_tokens": None,
        "audio_output_tokens": None,
    }


def _valid_nonneg_int(val: Any) -> int | None:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, float):
        if not math.isfinite(val) or not val.is_integer():
            return None
        val = int(val)
    elif not isinstance(val, int):
        return None
    return val if val >= 0 else None


def redact(s: Any) -> str | None:
    if s is None:
        return None
    text = str(s)
    for env_key in ("NINEROUTER_API_KEY", "OPENAI_API_KEY"):
        secret = os.environ.get(env_key)
        if secret and len(secret) > 4:
            text = text.replace(secret, "[REDACTED]")
    text = _RE_BEARER.sub(r"\1[REDACTED]", text)
    text = _RE_SK_KEY.sub("[REDACTED]", text)
    return text[:2000]


def parse_tags(h: str | None) -> list[TagPair]:
    """'k=v; k2=v2' or 'k=v,k2=v2' -> [(k, v), ...]. Bare words become (word, None)."""
    if not h:
        return []
    out: list[TagPair] = []
    for raw_part in _RE_TAG_SPLIT.split(h):
        part = raw_part.strip()
        if not part:
            continue
        k, sep, v = part.partition("=")
        key_clean = k.strip()[:100]
        if not key_clean:
            continue
        val_clean = (v.strip()[:500] or None) if sep else None
        out.append((key_clean, val_clean))
        if len(out) >= 50:
            break
    return out


def _message_text(msg: Any) -> str:
    if isinstance(msg, Mapping):
        content = msg.get("content")
        if isinstance(content, list):
            parts = [
                str(item.get("text") or "") if isinstance(item, Mapping) else str(item)
                for item in content
            ]
            return " ".join(p for p in parts if p)
        return str(content or "")
    return str(msg)


def estimate_tokens(x: Any) -> int:
    """Rough local estimate (~4 chars/token). Used only when upstream omits usage."""
    if x is None:
        return 0
    if isinstance(x, str):
        text = x
    elif isinstance(x, Iterable):
        text = " ".join(_message_text(m) for m in x)
    else:
        text = str(x)
    return max(1, -(-len(text) // 4)) if text else 0


def _extract_openai_or_anthropic_usage(u: Mapping[str, Any]) -> UsageInfo:
    info = _empty_usage()
    pt = _valid_nonneg_int(u.get("prompt_tokens"))
    ct = _valid_nonneg_int(u.get("completion_tokens"))

    if pt is not None or ct is not None:
        info["prompt_tokens"] = pt
        info["completion_tokens"] = ct
        pt_details = u.get("prompt_tokens_details")
        if isinstance(pt_details, Mapping):
            info["cached_tokens"] = _valid_nonneg_int(pt_details.get("cached_tokens"))
            info["audio_input_tokens"] = _valid_nonneg_int(pt_details.get("audio_tokens"))
        ct_details = u.get("completion_tokens_details")
        if isinstance(ct_details, Mapping):
            info["reasoning_tokens"] = _valid_nonneg_int(ct_details.get("reasoning_tokens"))
            info["audio_output_tokens"] = _valid_nonneg_int(ct_details.get("audio_tokens"))
        return info

    # Anthropic usage schema: input_tokens (+ optional cache_read_input_tokens / cache_creation_input_tokens), output_tokens
    anth_in = _valid_nonneg_int(u.get("input_tokens"))
    cache_read = _valid_nonneg_int(u.get("cache_read_input_tokens"))
    cache_create = _valid_nonneg_int(u.get("cache_creation_input_tokens"))
    anth_out = _valid_nonneg_int(u.get("output_tokens"))

    if anth_in is not None or cache_read is not None or cache_create is not None:
        info["prompt_tokens"] = (anth_in or 0) + (cache_read or 0) + (cache_create or 0)
        info["cached_tokens"] = cache_read
    if anth_out is not None:
        info["completion_tokens"] = anth_out
    return info


def _extract_gemini_usage(meta: Mapping[str, Any]) -> UsageInfo:
    info = _empty_usage()
    pt = _valid_nonneg_int(meta.get("promptTokenCount"))
    cand = _valid_nonneg_int(meta.get("candidatesTokenCount"))
    thoughts = _valid_nonneg_int(meta.get("thoughtsTokenCount"))
    cached = _valid_nonneg_int(meta.get("cachedContentTokenCount"))

    info["prompt_tokens"] = pt
    if cand is not None or thoughts is not None:
        info["completion_tokens"] = (cand or 0) + (thoughts or 0)
    info["cached_tokens"] = cached
    info["reasoning_tokens"] = thoughts
    return info


def extract_usage(payload: Mapping[str, Any]) -> UsageInfo:
    """Extract normalized token counts from OpenAI, Anthropic, or Gemini response payloads."""
    if not isinstance(payload, Mapping):
        return _empty_usage()

    u = payload.get("usage")
    if isinstance(u, Mapping):
        extracted = _extract_openai_or_anthropic_usage(u)
        if extracted["prompt_tokens"] is not None or extracted["completion_tokens"] is not None:
            return extracted

    msg = payload.get("message")
    if isinstance(msg, Mapping) and isinstance(msg.get("usage"), Mapping):
        extracted = _extract_openai_or_anthropic_usage(msg["usage"])
        if extracted["prompt_tokens"] is not None or extracted["completion_tokens"] is not None:
            return extracted

    meta = payload.get("usageMetadata")
    if isinstance(meta, Mapping):
        return _extract_gemini_usage(meta)

    return _empty_usage()


def _merge_usage(target: UsageInfo, incoming: UsageInfo) -> None:
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "audio_input_tokens",
        "audio_output_tokens",
    ):
        val = incoming[key]  # type: ignore[literal-required]
        if val is not None:
            target[key] = val  # type: ignore[literal-required]


def parse_body(content: bytes) -> dict[str, Any]:
    """Parse a non-stream body leniently: plain JSON, JSON with trailing data, or SSE frames."""
    text = content.decode("utf-8", "replace").strip()
    if not text:
        return {}
    dec = json.JSONDecoder()
    if not text.startswith("data:") and not text.startswith("event:"):
        try:
            parsed, _ = dec.raw_decode(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    out: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]" or line.startswith(("event:", ":")):
            continue
        try:
            frame, _ = dec.raw_decode(line)
        except json.JSONDecodeError:
            continue
        if isinstance(frame, dict):
            if frame.get("usage"):
                out["usage"] = frame["usage"]
            if frame.get("usageMetadata"):
                out["usageMetadata"] = frame["usageMetadata"]
            if frame.get("error"):
                out["error"] = frame["error"]
    return out


def _extract_delta_pieces(frame: Mapping[str, Any]) -> list[str]:
    pieces: list[str] = []
    # OpenAI choices[].delta
    choices = frame.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue
            piece = str(delta.get("content") or "") + str(delta.get("reasoning_content") or "")
            if delta.get("tool_calls"):
                piece += json.dumps(delta["tool_calls"])
            if piece:
                pieces.append(piece)

    # Anthropic content_block_delta
    anth_delta = frame.get("delta")
    if isinstance(anth_delta, Mapping):
        anth_piece = str(anth_delta.get("text") or "") + str(anth_delta.get("thinking") or "")
        if anth_piece:
            pieces.append(anth_piece)

    # Gemini candidates[].content.parts[].text
    candidates = frame.get("candidates")
    if isinstance(candidates, list):
        for cand in candidates:
            if not isinstance(cand, Mapping):
                continue
            content = cand.get("content")
            if not isinstance(content, Mapping):
                continue
            parts = content.get("parts")
            if isinstance(parts, list):
                for part in parts:
                    if isinstance(part, Mapping) and part.get("text"):
                        pieces.append(str(part["text"]))
    return pieces


@dataclass(slots=True)
class SSEStreamScanner:
    """Stateful byte-level SSE frame scanner safe across arbitrary multi-byte UTF-8 chunk splits."""

    clock_ms: Callable[[], float]
    ttft_ms: float | None = None
    usage: UsageInfo = field(default_factory=_empty_usage)
    error: str | None = None
    buffer: bytes = b""
    text_parts: list[str] = field(default_factory=list)

    @property
    def prompt_tokens(self) -> int | None:
        return self.usage["prompt_tokens"]

    @property
    def completion_tokens(self) -> int | None:
        return self.usage["completion_tokens"]

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.buffer += chunk
        while b"\n" in self.buffer:
            raw_line, self.buffer = self.buffer.split(b"\n", 1)
            self._process_line(raw_line)

    def flush(self) -> None:
        if self.buffer:
            self.feed(b"\n")

    def _process_line(self, raw_line: bytes) -> None:
        line = raw_line.strip()
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            return
        try:
            frame = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(frame, Mapping):
            return

        for piece in _extract_delta_pieces(frame):
            self.text_parts.append(piece)
            if self.ttft_ms is None:
                self.ttft_ms = self.clock_ms()

        _merge_usage(self.usage, extract_usage(frame))
        if frame.get("error"):
            self.error = json.dumps(frame["error"])


def build_forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    h = {k: v for k, v in headers.items() if k.lower() not in HOP}
    placeholder = os.environ.get("TOKENMETER_REPLACE_AUTH")
    if placeholder:
        h = {
            k: v
            for k, v in h.items()
            if not (k.lower() == "authorization" and v.strip() == f"Bearer {placeholder}")
        }
    if not any(k.lower() == "authorization" for k in h):
        key = os.environ.get("NINEROUTER_API_KEY")
        if key:
            h["Authorization"] = f"Bearer {key}"
    return h


def _prepare_chat_body(raw: bytes) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    if bool(parsed.get("stream")):
        raw_so = parsed.get("stream_options")
        so = dict(raw_so) if isinstance(raw_so, Mapping) else {}
        so["include_usage"] = True
        parsed["stream_options"] = so
    return parsed


@dataclass(slots=True)
class _ProxyContext:
    upstream: str
    conn: sqlite3.Connection
    prices: Prices
    client: httpx.AsyncClient
    otel: otelm.OTelEmitter | None

    def record_sync(
        self,
        *,
        total: float | None = None,
        upstream_ms: float | None = None,
        tags: list[TagPair] | None = None,
        usage_details: UsageInfo | None = None,
        **r: Any,
    ) -> None:
        r["total_ms"] = total
        r["upstream_ms"] = round(upstream_ms, 1) if upstream_ms is not None else None
        r["overhead_ms"] = (
            round(max(total - upstream_ms, 0.0), 2)
            if total is not None and upstream_ms is not None
            else None
        )
        r["ts"] = datetime.datetime.now(datetime.UTC).isoformat()
        r["error"] = redact(r.get("error"))
        details = usage_details or _empty_usage()
        r["cost"] = self.prices.cost(
            r.get("model"),
            r.get("prompt_tokens"),
            r.get("completion_tokens"),
            cached_tokens=details["cached_tokens"],
            reasoning_tokens=details["reasoning_tokens"],
            audio_input_tokens=details["audio_input_tokens"],
            audio_output_tokens=details["audio_output_tokens"],
        )
        r["price_version"] = self.prices.version
        try:
            dbm.insert(self.conn, r, tags)
        except (sqlite3.Error, ValueError, TypeError, OSError) as exc:
            log.error("db insert failed: %s: %s", type(exc).__name__, exc)
        if self.otel is not None:
            try:
                self.otel.emit(r, tags)
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                log.error("otel emit failed: %s: %s", type(exc).__name__, exc)

    async def record_async(
        self,
        *,
        total: float | None = None,
        upstream_ms: float | None = None,
        tags: list[TagPair] | None = None,
        usage_details: UsageInfo | None = None,
        **r: Any,
    ) -> None:
        try:
            await asyncio.to_thread(
                self.record_sync,
                total=total,
                upstream_ms=upstream_ms,
                tags=tags,
                usage_details=usage_details,
                **r,
            )
        except RuntimeError:
            self.record_sync(
                total=total,
                upstream_ms=upstream_ms,
                tags=tags,
                usage_details=usage_details,
                **r,
            )


async def _handle_non_stream_chat(
    ctx: _ProxyContext,
    body: dict[str, Any],
    headers: dict[str, str],
    base: dict[str, Any],
    tags: list[TagPair],
    clock_ms: Callable[[], float],
) -> Response:
    tu = time.perf_counter()
    try:
        r = await ctx.client.post(
            f"{ctx.upstream}/chat/completions", json=body, headers=headers
        )
    except httpx.HTTPError as exc:
        up_ms = (time.perf_counter() - tu) * 1000
        await ctx.record_async(
            **base,
            tags=tags,
            status_code=502,
            error=f"{type(exc).__name__}: {exc}",
            total=clock_ms(),
            upstream_ms=up_ms,
        )
        return JSONResponse(
            {"error": {"message": f"upstream error: {type(exc).__name__}"}},
            status_code=502,
        )

    up_ms = (time.perf_counter() - tu) * 1000
    parsed = parse_body(r.content)
    usage = extract_usage(parsed)
    pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
    src = "reported" if pt is not None else None
    err: str | None = None
    if r.status_code >= 400:
        err = json.dumps(parsed["error"]) if "error" in parsed else (
            json.dumps(parsed) if parsed else r.text
        )

    resp = Response(
        r.content,
        r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )
    await ctx.record_async(
        **base,
        tags=tags,
        status_code=r.status_code,
        error=err,
        prompt_tokens=pt,
        completion_tokens=ct,
        usage_source=src,
        ttft_ms=None,
        total=clock_ms(),
        upstream_ms=up_ms,
        usage_details=usage,
    )
    return resp


async def _handle_stream_chat(
    ctx: _ProxyContext,
    body: dict[str, Any],
    headers: dict[str, str],
    base: dict[str, Any],
    tags: list[TagPair],
    clock_ms: Callable[[], float],
) -> Response:
    tu = time.perf_counter()
    try:
        ureq = ctx.client.build_request(
            "POST", f"{ctx.upstream}/chat/completions", json=body, headers=headers
        )
        r = await ctx.client.send(ureq, stream=True)
    except httpx.HTTPError as exc:
        up_ms = (time.perf_counter() - tu) * 1000
        await ctx.record_async(
            **base,
            tags=tags,
            status_code=502,
            error=f"{type(exc).__name__}: {exc}",
            total=clock_ms(),
            upstream_ms=up_ms,
        )
        return JSONResponse(
            {"error": {"message": f"upstream error: {type(exc).__name__}"}},
            status_code=502,
        )

    if r.status_code >= 400:
        content = await r.aread()
        await r.aclose()
        up_ms = (time.perf_counter() - tu) * 1000
        await ctx.record_async(
            **base,
            tags=tags,
            status_code=r.status_code,
            error=content.decode("utf-8", "replace"),
            total=clock_ms(),
            upstream_ms=up_ms,
        )
        return Response(
            content,
            r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    scanner = SSEStreamScanner(clock_ms=clock_ms)
    upstream_wait_ms = (time.perf_counter() - tu) * 1000

    async def gen() -> AsyncIterator[bytes]:
        nonlocal upstream_wait_ms
        it = r.aiter_raw().__aiter__()
        try:
            while True:
                tw = time.perf_counter()
                try:
                    chunk = await it.__anext__()
                except StopAsyncIteration:
                    upstream_wait_ms += (time.perf_counter() - tw) * 1000
                    break
                upstream_wait_ms += (time.perf_counter() - tw) * 1000
                scanner.feed(chunk)
                yield chunk
        except (httpx.HTTPError, OSError) as exc:
            scanner.error = f"{type(exc).__name__}: {exc}"
        finally:
            await r.aclose()
            scanner.flush()
            pt, ct = scanner.prompt_tokens, scanner.completion_tokens
            src = "reported"
            if pt is None:
                pt = estimate_tokens(body.get("messages"))
                ct = estimate_tokens("".join(scanner.text_parts))
                src = "estimated"
            await ctx.record_async(
                **base,
                tags=tags,
                status_code=r.status_code,
                error=scanner.error,
                prompt_tokens=pt,
                completion_tokens=ct,
                usage_source=src,
                ttft_ms=scanner.ttft_ms,
                total=clock_ms(),
                upstream_ms=upstream_wait_ms,
                usage_details=scanner.usage,
            )

    return StreamingResponse(
        gen(),
        r.status_code,
        media_type=r.headers.get("content-type", "text/event-stream"),
    )


def create_app(
    upstream: str | None = None,
    db_path: str | os.PathLike[str] | None = None,
    prices_path: str | os.PathLike[str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    resolved_upstream = (
        upstream or os.environ.get("TOKENMETER_UPSTREAM", "http://127.0.0.1:20128/v1")
    ).rstrip("/")
    resolved_db = os.fspath(db_path) if db_path is not None else os.environ.get(
        "TOKENMETER_DB", "tokenmeter.db"
    )
    default_prices = os.path.join(os.path.dirname(__file__), "..", "config", "prices.yaml")
    resolved_prices = (
        os.fspath(prices_path)
        if prices_path is not None
        else os.environ.get("TOKENMETER_PRICES", default_prices)
    )

    conn = dbm.connect(resolved_db)
    prices = Prices(resolved_prices, conn)
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(300, connect=10),
        transport=transport,
    )
    otel = otelm.from_env()
    ctx = _ProxyContext(
        upstream=resolved_upstream,
        conn=conn,
        prices=prices,
        client=client,
        otel=otel,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await client.aclose()
            if otel is not None:
                otel.shutdown()
            conn.close()

    fastapi_app = FastAPI(title="TokenMeter", lifespan=lifespan)
    fastapi_app.state.conn = conn
    fastapi_app.state.prices = prices
    fastapi_app.state.otel = otel
    fastapi_app.state.client = client

    @fastapi_app.get("/healthz")
    async def health() -> dict[str, Any]:
        return {"ok": True, "upstream": resolved_upstream}

    async def models(req: Request, project: str | None = None) -> Response:
        _ = project
        try:
            r = await client.get(
                f"{resolved_upstream}/models",
                headers=build_forward_headers(req.headers),
            )
        except httpx.HTTPError as exc:
            return JSONResponse(
                {"error": {"message": f"upstream error: {type(exc).__name__}"}},
                status_code=502,
            )
        return Response(
            r.content,
            r.status_code,
            media_type=r.headers.get("content-type", "application/json"),
        )

    async def chat(req: Request, project: str | None = None) -> Response:
        t0 = time.perf_counter()

        def clock_ms() -> float:
            return round((time.perf_counter() - t0) * 1000, 1)

        tag = req.headers.get("x-tokenmeter-tag") or req.headers.get("x-client")
        resolved_project = project or req.headers.get("x-tokenmeter-project")
        kv_tags = parse_tags(req.headers.get("x-tokenmeter-tags"))

        body = _prepare_chat_body(await req.body())
        if body is None:
            return JSONResponse({"error": {"message": "invalid JSON"}}, status_code=400)

        raw_model = body.get("model")
        model = str(raw_model) if raw_model is not None else None
        stream = bool(body.get("stream"))
        base = {
            "tag": tag,
            "project": resolved_project,
            "model": model,
            "stream": int(stream),
        }
        fwd = build_forward_headers(req.headers)

        if not stream:
            return await _handle_non_stream_chat(ctx, body, fwd, base, kv_tags, clock_ms)
        return await _handle_stream_chat(ctx, body, fwd, base, kv_tags, clock_ms)

    fastapi_app.add_api_route("/v1/models", models, methods=["GET"])
    fastapi_app.add_api_route("/p/{project}/v1/models", models, methods=["GET"])
    fastapi_app.add_api_route("/v1/chat/completions", chat, methods=["POST"])
    fastapi_app.add_api_route("/p/{project}/v1/chat/completions", chat, methods=["POST"])
    return fastapi_app


app: FastAPI | None = None


def get_app() -> FastAPI:
    return create_app()
