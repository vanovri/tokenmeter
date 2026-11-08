# TokenMeter (M0–M3)

A small OpenAI-compatible proxy that sits in front of 9Router. It logs every
`/v1/chat/completions` call to SQLite: timestamp, model, tag, tokens, TTFT, total latency, status, error and cost.
A Streamlit dashboard shows per-model calls, tokens, cost, p50/p95 latency, error rate and recent requests, and has a CSV download.

## Setup (PowerShell)
```powershell
cd .
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

## Run
```powershell
$env:TOKENMETER_UPSTREAM = "http://localhost:20128/v1"   # your 9Router base URL
$env:TOKENMETER_DB = "tokenmeter.db"
.\.venv\Scripts\python -m uvicorn tokenmeter.proxy:get_app --factory --host 127.0.0.1 --port 8787
# second window:
.\.venv\Scripts\python -m streamlit run dashboard.py --server.port 8501 --server.headless true
```
Point clients at `http://127.0.0.1:8787/v1`. Optional: tag calls with the header `X-TokenMeter-Tag: <name>` (`X-Client` also works).

## Tagging (M2)
| Header | Stored as |
|---|---|
| `X-TokenMeter-Tag: rag` (or `X-Client`) | `requests.tag`, which is the client tag |
| `X-TokenMeter-Project: redflag` | `requests.project` |
| `X-TokenMeter-Tags: pattern_id=p7; agent=diagnosis` | rows in the `tags` table (key, value) |

TokenMeter headers are not forwarded upstream.

## Proxy overhead (M2)
- `upstream_ms` is the time spent waiting on 9Router: the whole response for non-streaming calls, and the sum of the waits for each chunk for streaming calls.
- `overhead_ms = total_ms - upstream_ms` is the time spent inside the proxy (parsing, header handling, SSE scanning, and handing chunks to the client).

## Migration
Opening an older database adds the M2 columns and the `tags` table automatically.
First it writes a full backup next to the database as `tokenmeter.db.bak-YYYYmmdd-HHMMSS`.

## Auth
The client's `Authorization` header is forwarded unchanged. If a request has none, the proxy uses `NINEROUTER_API_KEY` from the environment.
Keys are never written to SQLite or logs, and bearer tokens and `sk-...` strings are redacted from stored error text.

## Prices
`config/prices.yaml` ships with **null placeholders only**. While a model's price is null, its cost is stored as NULL ("unpriced"); it is never estimated.
When you edit the file, the change is picked up on the next proxy request or dashboard refresh.
A new `price_versions` row is added (older versions are kept), and the cost of **all existing rows** is recomputed with the new prices.
The dashboard's *Price versions* tab shows every version.

## Streaming
The proxy forces `stream_options.include_usage=true` and passes the SSE bytes through unchanged.
TTFT is the time until the first chunk that carries content. If the upstream sends no usage data, tokens are estimated at about 4 characters per token and the row is marked `usage_source=estimated`. The dashboard shows a warning for these rows.

## Tests
```powershell
.\.venv\Scripts\python -m pytest -q
```
The tests use a fake OpenAI-compatible upstream (streaming and non-streaming, errors, upstream down) and check that no key is stored.

## Smoke test
```powershell
.\.venv\Scripts\python smoke.py xai/grok-code-fast-1
```

## Auto-start (M3)
```powershell
.\scripts\start.ps1           # starts proxy + dashboard hidden (skips anything already running)
.\scripts\stop.ps1            # stops them (only processes from this folder)
.\scripts\register-task.ps1   # per-user Scheduled Task "TokenMeter" at logon (no admin)
Start-ScheduledTask TokenMeter
Unregister-ScheduledTask TokenMeter -Confirm:$false   # remove
```
Logs go to `logs\`. The proxy listens on 127.0.0.1. If WSL's NAT-mode adapter (`vEthernet (WSL...)`) exists, it also listens on that adapter's IP. It never listens on 0.0.0.0.

## Clients that can't set headers: project in the path
Use `http://127.0.0.1:8787/p/<project>/v1` as the base URL. Requests are forwarded to the normal upstream `/v1/...`, with `project=<project>`.

### Agentic RAG (WSL)
No code change is needed:
```bash
export LLM_PROVIDER=openai OPENAI_BASE_URL=http://$(ip route | awk '/default/{print $3}'):8787/p/agentic-rag/v1
export OPENAI_API_KEY=tokenmeter-local   # any non-empty value; see note below
export MODEL=xai/grok-code-fast-1
.venv/bin/python -m agentic_rag.cli ask "..."
```
About the key: agentic-rag sends `Authorization: Bearer <OPENAI_API_KEY>`, and the proxy forwards a client's Authorization header as-is.
A placeholder therefore only works if 9Router doesn't need a key. Otherwise set `TOKENMETER_REPLACE_AUTH=tokenmeter-local`.
The proxy then swaps that exact placeholder for `NINEROUTER_API_KEY`, so the real key never has to live in WSL.

## OpenTelemetry (optional, off by default)
```powershell
$env:TOKENMETER_OTEL = "1"
$env:OTEL_EXPORTER_OTLP_ENDPOINT = "http://127.0.0.1:4318"   # e.g. self-hosted Langfuse/Jaeger OTLP HTTP
```
TokenMeter exports one span per request, named `chat {model}`. It carries the attributes `gen_ai.operation.name`, `gen_ai.request.model` and `gen_ai.usage.input_tokens`/`output_tokens`.
It also carries custom `tokenmeter.*` attributes: cost (only when priced), ttft, upstream and overhead time, project, client tag and key/value tags.
Spans never contain prompts, outputs or keys.
