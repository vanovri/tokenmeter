"""TokenMeter: OpenAI-compatible logging proxy. Never logs/stores API keys."""
import os, re, json, time, datetime, logging
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
from . import db as dbm
from .prices import Prices

log = logging.getLogger("tokenmeter")
HOP = {"x-tokenmeter-project","x-tokenmeter-tags","host","content-length","connection","transfer-encoding","accept-encoding","x-tokenmeter-tag","x-client"}

def redact(s):
    if s is None: return None
    s = str(s)
    for k in ("NINEROUTER_API_KEY","OPENAI_API_KEY"):
        v = os.environ.get(k)
        if v and len(v) > 4: s = s.replace(v, "[REDACTED]")
    import re
    s = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1[REDACTED]", s)
    s = re.sub(r"sk-[A-Za-z0-9._\-]{6,}", "[REDACTED]", s)
    return s[:2000]

def parse_tags(h):
    """'k=v; k2=v2' or 'k=v,k2=v2' -> [(k, v), ...]. Bare words become (word, None)."""
    out = []
    for part in re.split(r"[;,]", h or ""):
        part = part.strip()
        if not part: continue
        k, _, v = part.partition("=")
        out.append((k.strip()[:100], (v.strip()[:500] or None) if _ else None))
    return out[:50]

def estimate_tokens(x) -> int:
    """Rough local estimate (~4 chars/token, OpenAI rule of thumb). Only used when the
    upstream sent no usage; rows are flagged usage_source='estimated'."""
    if x is None: return 0
    if not isinstance(x, str):
        x = " ".join(str(m.get("content") or "") if isinstance(m, dict) else str(m) for m in x)
    return max(1, -(-len(x) // 4)) if x else 0

def parse_body(content: bytes) -> dict:
    """Parse a non-stream body leniently: plain JSON, JSON followed by extra data,
    or SSE-framed chunks (some routers send these). Returns the merged usage/error info."""
    text = content.decode("utf-8", "replace").strip()
    dec = json.JSONDecoder()
    if not text.startswith("data:"):
        try: return dec.raw_decode(text)[0]
        except Exception: pass
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"): line = line[5:].strip()
        if not line or line == "[DONE]": continue
        try: j = dec.raw_decode(line)[0]
        except Exception: continue
        if isinstance(j, dict):
            if j.get("usage"): out["usage"] = j["usage"]
            if j.get("error"): out["error"] = j["error"]
    return out

def create_app(upstream=None, db_path=None, prices_path=None, transport=None):
    upstream = (upstream or os.environ.get("TOKENMETER_UPSTREAM", "http://127.0.0.1:20128/v1")).rstrip("/")
    db_path = db_path or os.environ.get("TOKENMETER_DB", "tokenmeter.db")
    prices_path = prices_path or os.environ.get("TOKENMETER_PRICES", os.path.join(os.path.dirname(__file__), "..", "config", "prices.yaml"))
    conn = dbm.connect(db_path)
    prices = Prices(prices_path, conn)
    client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=10), transport=transport)
    app = FastAPI(title="TokenMeter")
    from . import otel as otelm
    otel = otelm.from_env()
    app.state.conn, app.state.prices, app.state.otel = conn, prices, otel

    def fwd_headers(req: Request):
        h = {k: v for k, v in req.headers.items() if k.lower() not in HOP}
        ph = os.environ.get("TOKENMETER_REPLACE_AUTH")
        if ph:  # client sent the agreed local placeholder -> substitute the real key below
            h = {k: v for k, v in h.items() if not (k.lower() == "authorization" and v.strip() == f"Bearer {ph}")}
        if "authorization" not in {k.lower() for k in h}:
            key = os.environ.get("NINEROUTER_API_KEY")
            if key: h["Authorization"] = f"Bearer {key}"
        return h

    def record(total=None, upstream=None, tags=None, **r):
        r["total_ms"] = total
        r["upstream_ms"] = round(upstream, 1) if upstream is not None else None
        r["overhead_ms"] = round(max(total - upstream, 0.0), 2) if total is not None and upstream is not None else None
        r["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        r["error"] = redact(r.get("error"))
        r["cost"] = prices.cost(r.get("model"), r.get("prompt_tokens"), r.get("completion_tokens"))
        r["price_version"] = prices.version
        try: dbm.insert(conn, r, tags)
        except Exception as e: log.error("db insert failed: %s", type(e).__name__)
        if otel is not None:
            try: otel.emit(r, tags)
            except Exception as e: log.error("otel emit failed: %s", type(e).__name__)

    @app.get("/healthz")
    async def health(): return {"ok": True, "upstream": upstream}

    async def models(req: Request, project: str | None = None):
        try:
            r = await client.get(upstream + "/models", headers=fwd_headers(req))
        except httpx.HTTPError as e:
            return JSONResponse({"error": {"message": f"upstream error: {type(e).__name__}"}}, 502)
        return Response(r.content, r.status_code, media_type=r.headers.get("content-type", "application/json"))

    async def chat(req: Request, project: str | None = None):
        t0 = time.perf_counter()
        ms = lambda: round((time.perf_counter() - t0) * 1000, 1)
        tag = req.headers.get("x-tokenmeter-tag") or req.headers.get("x-client")
        project = project or req.headers.get("x-tokenmeter-project")
        kv = parse_tags(req.headers.get("x-tokenmeter-tags"))
        raw = await req.body()
        try: body = json.loads(raw)
        except Exception: return JSONResponse({"error": {"message": "invalid JSON"}}, 400)
        model, stream = body.get("model"), bool(body.get("stream"))
        base = dict(tag=tag, project=project, model=model, stream=int(stream), tags=kv)
        if stream:
            so = body.get("stream_options") or {}
            so["include_usage"] = True
            body["stream_options"] = so
        headers = fwd_headers(req)

        if not stream:
            tu = time.perf_counter()
            try:
                r = await client.post(upstream + "/chat/completions", json=body, headers=headers)
            except httpx.HTTPError as e:
                up = (time.perf_counter() - tu) * 1000
                record(**base, status_code=502, error=f"{type(e).__name__}: {e}", total=ms(), upstream=up)
                return JSONResponse({"error": {"message": f"upstream error: {type(e).__name__}"}}, 502)
            up = (time.perf_counter() - tu) * 1000
            pt = ct = None; err = None; src = None
            try:
                j = parse_body(r.content); u = j.get("usage") or {}
                pt, ct = u.get("prompt_tokens"), u.get("completion_tokens")
                src = "reported" if pt is not None else None
                if r.status_code >= 400: err = json.dumps(j.get("error", j))
            except Exception:
                if r.status_code >= 400: err = r.text
            resp = Response(r.content, r.status_code, media_type=r.headers.get("content-type", "application/json"))
            record(**base, status_code=r.status_code, error=err, prompt_tokens=pt, completion_tokens=ct,
                   usage_source=src, ttft_ms=None, total=ms(), upstream=up)
            return resp

        tu = time.perf_counter()
        try:
            ureq = client.build_request("POST", upstream + "/chat/completions", json=body, headers=headers)
            r = await client.send(ureq, stream=True)
        except httpx.HTTPError as e:
            up = (time.perf_counter() - tu) * 1000
            record(**base, status_code=502, error=f"{type(e).__name__}: {e}", total=ms(), upstream=up)
            return JSONResponse({"error": {"message": f"upstream error: {type(e).__name__}"}}, 502)
        if r.status_code >= 400:
            content = await r.aread(); await r.aclose()
            up = (time.perf_counter() - tu) * 1000
            record(**base, status_code=r.status_code, error=content.decode("utf-8", "replace"), total=ms(), upstream=up)
            return Response(content, r.status_code, media_type=r.headers.get("content-type", "application/json"))

        st = {"ttft": None, "pt": None, "ct": None, "err": None, "buf": b"", "text": [],
              "up": (time.perf_counter() - tu) * 1000}
        def scan(chunk: bytes):
            st["buf"] += chunk
            while b"\n" in st["buf"]:
                line, st["buf"] = st["buf"].split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"): continue
                data = line[5:].strip()
                if data == b"[DONE]": continue
                try: j = json.loads(data)
                except Exception: continue
                for c in j.get("choices") or []:
                    d = c.get("delta") or {}
                    piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                    if d.get("tool_calls"): piece += json.dumps(d["tool_calls"])
                    if piece:
                        st["text"].append(piece)
                        if st["ttft"] is None: st["ttft"] = ms()
                u = j.get("usage")
                if u:
                    st["pt"], st["ct"] = u.get("prompt_tokens"), u.get("completion_tokens")
                if j.get("error"): st["err"] = json.dumps(j["error"])

        async def gen():
            it = r.aiter_raw().__aiter__()
            try:
                while True:
                    tw = time.perf_counter()
                    try: chunk = await it.__anext__()
                    except StopAsyncIteration:
                        st["up"] += (time.perf_counter() - tw) * 1000; break
                    st["up"] += (time.perf_counter() - tw) * 1000   # time waiting on upstream
                    scan(chunk); yield chunk
            except Exception as e:
                st["err"] = f"{type(e).__name__}: {e}"
            finally:
                await r.aclose()
                scan(b"\n")
                pt, ct, src = st["pt"], st["ct"], "reported"
                if pt is None:   # upstream sent no usage: local estimate, flagged
                    pt, ct, src = estimate_tokens(body.get("messages")), estimate_tokens("".join(st["text"])), "estimated"
                record(**base, status_code=r.status_code, error=st["err"], prompt_tokens=pt,
                       completion_tokens=ct, usage_source=src, ttft_ms=st["ttft"], total=ms(), upstream=st["up"])
        return StreamingResponse(gen(), r.status_code, media_type=r.headers.get("content-type", "text/event-stream"))

    # /v1/... plus /p/{project}/v1/... so clients that can't set headers can still be tagged
    app.add_api_route("/v1/models", models, methods=["GET"])
    app.add_api_route("/p/{project}/v1/models", models, methods=["GET"])
    app.add_api_route("/v1/chat/completions", chat, methods=["POST"])
    app.add_api_route("/p/{project}/v1/chat/completions", chat, methods=["POST"])

    @app.on_event("shutdown")
    async def _close():
        await client.aclose()
        if otel is not None: otel.shutdown()
    return app

app = None
def get_app():
    return create_app()
