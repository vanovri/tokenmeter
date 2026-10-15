"""Tiny live smoke test through a running TokenMeter proxy. Key comes from NINEROUTER_API_KEY (never printed)."""
import os, sys, time, json, httpx
base = os.environ.get("TOKENMETER_URL", "http://127.0.0.1:8787/v1")
model = sys.argv[1] if len(sys.argv) > 1 else "xai/grok-code-fast-1"
h = {"X-TokenMeter-Tag": "smoke", "X-TokenMeter-Project": "tokenmeter", "X-TokenMeter-Tags": "milestone=m2"}
key = os.environ.get("NINEROUTER_API_KEY")
if key: h["Authorization"] = f"Bearer {key}"
msgs = [{"role": "user", "content": "Reply with just: ok"}]
with httpx.Client(timeout=120) as c:
    r = c.get(f"{base}/models", headers=h); print("models:", r.status_code)
    t = time.perf_counter()
    r = c.post(f"{base}/chat/completions", headers=h, json={"model": model, "messages": msgs, "max_tokens": 20})
    print("non-stream:", r.status_code, round((time.perf_counter()-t)*1000), "ms client-side; usage:", r.headers.get("content-type"), len(r.content), "bytes")
    if "--no-stream" in sys.argv: sys.exit(0)
    t = time.perf_counter(); n = 0
    with c.stream("POST", f"{base}/chat/completions", headers=h, json={"model": model, "messages": msgs, "max_tokens": 20, "stream": True}) as r:
        for line in r.iter_lines():
            if line.startswith("data:"): n += 1
    print("stream:", r.status_code, n, "data lines,", round((time.perf_counter()-t)*1000), "ms client-side")
