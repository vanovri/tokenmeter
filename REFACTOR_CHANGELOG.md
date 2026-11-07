# Refactor & QA Verification Changelog (`tokenmeter`)

## 1. Branch & Git Safety Summary

| Property | Value |
| --- | --- |
| Original Branch | `master` (`1790e0a`) |
| Pre-Refactor Backup Stash | `stash@{0}: On master: pre-refactor-backup-snapshot` |
| Refactoring Branch | `codex/auto-refactor-and-tests` |
| Backward Compatibility | 100% preserved across FastAPI proxy routes, SQLite schema/migrations, `PriceBook`, OTLP exporter, CLI server, and Streamlit dashboard (live `tokenmeter.db` and `*.log` files untouched) |

---

## 2. Modified & Added Files

| File | Status | Summary of Changes |
| --- | --- | --- |
| `tokenmeter/__init__.py` | Modified | Added `from __future__ import annotations`, explicit `__all__`, and version metadata. |
| `tokenmeter/db.py` | Modified | Enabled `PRAGMA journal_mode=WAL;` and `PRAGMA synchronous=NORMAL;` on file-backed SQLite connections; added missing indexes (`ix_req_price_ver`, `ix_req_model`, `ix_req_ts`); added `normalize_request_row()` rejecting `bool`-as-`int` and non-finite floats (`NaN`/`Inf`); decomposed schema migration and online backup helpers (`98%` coverage). |
| `tokenmeter/prices.py` | Modified | Added `(st_mtime_ns, st_size)` stat-caching in `PriceBook.reload()` so unchanged `prices.yaml` files are not re-read and re-hashed on every request; wrapped file opens in `with open(..., encoding="utf-8")`; chunked historical DB repricing via `cursor.fetchmany(2000)` batches; added strict numeric tier validation (`95%` coverage). |
| `tokenmeter/proxy.py` | Modified | Replaced deprecated `@app.on_event("shutdown")` with FastAPI `lifespan` `@asynccontextmanager` closing `httpx.AsyncClient`, OTLP exporter, and SQLite `conn` deterministically (cutting deprecation warnings from `52` to `1`); decomposed SSE stream reassembly, usage extraction, and request recording into modular single-responsibility helpers (`91%` coverage). |
| `tokenmeter/otel.py` | Modified | Added full type annotations, decomposed OTLP protobuf span attribute builders, and replaced broad silent catches with structured `logging.getLogger(__name__)` diagnostics (`82%` coverage). |
| `tokenmeter/serve.py` | Modified | Added CLI argument parser, port/host validation, and testable `main()` entrypoint (`0%` -> `85%` coverage). |
| `dashboard.py` | Modified | Wrapped SQLite connection lifecycle in `try / finally: conn.close()`; vectorized `is_estimated` and `is_unpriced` groupby aggregations; extracted pure data-transformation helpers (`load_dashboard_frames`, `summarize_by_dimension`, `build_clean_export_df`) so dashboard analytics can be unit-tested without a live browser (`96%` coverage). |
| `tests/test_prices_unit.py` | Added | Unit tests for `PriceBook` stat-caching, long-context tiers, malformed YAML/rates, `bool`/`NaN`/`Inf` rejection, and batched `reprice_all()`. |
| `tests/test_proxy_unit.py` | Added | Unit tests for FastAPI `lifespan` teardown, split-chunk SSE usage extraction, auth header replacement, and upstream error resilience. |
| `tests/test_db_otel_serve_dashboard.py` | Added | Unit tests for SQLite WAL/index creation, row validation, OTLP span building, `serve.py` CLI parsing, and `dashboard.py` vectorized aggregations. |

---

## 3. Architectural, Concurrency & Resource Safety Improvements

1. **FastAPI Lifespan & Deterministic Resource Cleanup (`tokenmeter/proxy.py`, `dashboard.py`):**
   - Replaced deprecated `@app.on_event("shutdown")` with an `@asynccontextmanager` `lifespan` handler that closes `httpx.AsyncClient`, flushes OTLP spans, and closes the SQLite connection on shutdown.
   - Wrapped `dashboard.py` SQLite reads in `try / finally: conn.close()` so every Streamlit rerun releases its DB handle immediately.
2. **Stat-Cached Pricing & Batched Repricing (`tokenmeter/prices.py`, `tokenmeter/db.py`):**
   - Cached `(st_mtime_ns, st_size)` in `PriceBook.reload()` and opened `prices.yaml` via context manager (`with open(...)`), eliminating per-request file descriptor leaks and redundant SHA-256/YAML parsing.
   - Added `ix_req_price_ver`, `ix_req_model`, and `ix_req_ts` SQLite indexes and chunked `reprice` queries with `fetchmany(2000)`.
3. **Vectorized Dashboard Analytics (`dashboard.py`):** Replaced row-wise `.apply(lambda s: ...)` closures on `GroupBy` objects with precomputed boolean columns (`is_estimated`, `is_unpriced`) and vectorized `.sum().astype(int)`.

---

## 4. Before vs. After Test Suite & Coverage Metrics

| Module | Before Stmts | Before Miss | Before Cover | After Stmts | After Miss | After Cover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dashboard.py` | 72 | 6 | 92% | 158 | 6 | **96%** |
| `tokenmeter/__init__.py` | 1 | 0 | 100% | 3 | 0 | **100%** |
| `tokenmeter/db.py` | 37 | 0 | 100% | 126 | 2 | **98%** |
| `tokenmeter/otel.py` | 48 | 0 | 100% | 190 | 35 | **82%** |
| `tokenmeter/prices.py` | 39 | 1 | 97% | 196 | 10 | **95%** |
| `tokenmeter/proxy.py` | 191 | 19 | 90% | 432 | 41 | **91%** |
| `tokenmeter/serve.py` | 17 | 17 | 0% | 54 | 8 | **85%** |
| **TOTAL** | **405** | **43** | **89%** | **1,159** | **102** | **91%** |

- **Tests:** `20 passed (52 warnings)` -> **`57 passed (1 third-party Starlette warning, 0 FastAPI lifespan warnings)`**
- **Ruff Lint Errors:** `41 errors` -> **`0 errors` (`exit 0`)**
- **Mypy Type Errors:** `12 errors` -> **`0 errors` (`exit 0`)**

### Verification Commands & Exit Codes

```powershell
# 1. Pytest + Coverage -> Exit Code: 0 (57 passed, 91% coverage)
.\.venv\Scripts\python.exe -m pytest --cov=tokenmeter --cov=dashboard --cov-report=term-missing

# 2. Ruff Linter -> Exit Code: 0 (All checks passed!)
C:\Users\ardhy\.gemini\antigravity\brain\7b19fb5e-cc47-425f-9713-ee64d56c8136\scratch\qa-tools\qa-venv\Scripts\ruff.exe check tokenmeter dashboard.py tests

# 3. Mypy Type Checker -> Exit Code: 0 (Success: no issues found in 7 source files)
C:\Users\ardhy\.gemini\antigravity\brain\7b19fb5e-cc47-425f-9713-ee64d56c8136\scratch\qa-tools\qa-venv\Scripts\mypy.exe tokenmeter dashboard.py --ignore-missing-imports
```
