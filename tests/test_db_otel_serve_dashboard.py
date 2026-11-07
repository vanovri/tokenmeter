from __future__ import annotations

import json
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

import dashboard
from tokenmeter import db as dbm
from tokenmeter import otel as otelm
from tokenmeter import serve as servem


def test_db_wal_pragma_indexes_and_concurrent_read_write(tmp_path: Path) -> None:
    db_file = tmp_path / "wal_concurrent.db"
    conn = dbm.connect(str(db_file))
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(mode).lower() == "wal"

        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'ix_%'"
            ).fetchall()
        }
        assert {
            "ix_tags_req",
            "ix_tags_kv",
            "ix_req_price_ver",
            "ix_req_model",
            "ix_req_ts",
        } <= indexes

        errors: list[BaseException] = []

        def writer(tid: int) -> None:
            try:
                for i in range(25):
                    dbm.insert(
                        conn,
                        {
                            "ts": f"2026-11-07T00:00:{i:02d}+00:00",
                            "model": f"m-{tid % 2}",
                            "stream": 0,
                            "status_code": 200,
                            "prompt_tokens": 10 + i,
                            "completion_tokens": 5,
                            "total_ms": 12.5,
                        },
                        tags=[("worker", str(tid))],
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def reader() -> None:
            rconn = sqlite3.connect(str(db_file))
            try:
                for _ in range(30):
                    rconn.execute("SELECT COUNT(*) FROM requests").fetchone()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                rconn.close()

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)] + [
            threading.Thread(target=reader) for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 100
        assert conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0] == 100
    finally:
        conn.close()


@pytest.mark.parametrize(
    "bad_row",
    [
        {"ts": "t", "prompt_tokens": True},
        {"ts": "t", "completion_tokens": False},
        {"ts": "t", "prompt_tokens": -1},
        {"ts": "t", "status_code": True},
        {"ts": "t", "cost": math.nan},
        {"ts": "t", "total_ms": math.inf},
        {"ts": "t", "ttft_ms": -math.inf},
        {"ts": "t", "cost": True},
    ],
)
def test_db_insert_rejects_bool_nan_inf_and_negative_tokens(bad_row: dict[str, Any]) -> None:
    conn = dbm.connect(":memory:")
    try:
        with pytest.raises((TypeError, ValueError)):
            dbm.insert(conn, bad_row)
    finally:
        conn.close()


def test_otel_span_attributes_and_malformed_payload_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENMETER_OTEL", "off")
    assert otelm.from_env() is None

    monkeypatch.setenv("TOKENMETER_OTEL", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318/")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://127.0.0.1:9999/custom/traces")
    em_env = otelm.from_env()
    assert em_env is not None
    try:
        assert em_env.endpoint == "http://127.0.0.1:9999/custom/traces"
    finally:
        em_env.shutdown()

    exported: list[Any] = []

    class _MemoryExporter(SpanExporter):
        def export(self, spans: Any) -> SpanExportResult:
            exported.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None

    em = otelm.OTelEmitter("http://unused", exporter=_MemoryExporter())
    # Emit row containing zero, negative, bool, and NaN/Inf values -> should NOT crash,
    # preserving valid 0 while scrubbing bool/negative/NaN/Inf!
    em.emit(
        {
            "model": None,
            "stream": 1,
            "status_code": 200,
            "error": "ReadError: peer reset",
            "prompt_tokens": 0,
            "completion_tokens": True,
            "cost": math.nan,
            "ttft_ms": -5.0,
            "total_ms": math.nan,
        },
        tags=[("bare_flag", None), ("env", "local")],
    )
    em.flush()
    em.shutdown()

    assert len(exported) == 1
    span = exported[0]
    assert span.name == "chat unknown"
    assert span.attributes["gen_ai.usage.input_tokens"] == 0
    assert "gen_ai.usage.output_tokens" not in span.attributes
    assert "tokenmeter.cost" not in span.attributes
    assert "tokenmeter.ttft_ms" not in span.attributes
    assert span.attributes["tokenmeter.stream"] is True
    assert span.attributes["tokenmeter.tag.bare_flag"] == ""
    assert span.attributes["tokenmeter.tag.env"] == "local"
    assert span.status.status_code.name == "ERROR"

    # Valid JSON & Protobuf OTLP parsing + malformed payload rejection
    json_payload = json.dumps(
        {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "name": "chat gpt-4o",
                                    "attributes": [
                                        {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "12"}}
                                    ],
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    )
    parsed_spans = otelm.parse_otlp_trace_payload(json_payload)
    assert len(parsed_spans) == 1
    assert parsed_spans[0]["name"] == "chat gpt-4o"
    assert parsed_spans[0]["attributes"]["gen_ai.usage.input_tokens"] == 12

    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    pb_req = ExportTraceServiceRequest()
    sp = pb_req.resource_spans.add().scope_spans.add().spans.add()
    sp.name = "chat claude"
    kv = sp.attributes.add()
    kv.key = "gen_ai.usage.output_tokens"
    kv.value.int_value = 9
    pb_parsed = otelm.parse_otlp_trace_payload(pb_req.SerializeToString())
    assert len(pb_parsed) == 1
    assert pb_parsed[0]["name"] == "chat claude"
    assert pb_parsed[0]["attributes"]["gen_ai.usage.output_tokens"] == 9

    with pytest.raises(ValueError):
        otelm.parse_otlp_trace_payload("{invalid-json")
    with pytest.raises(ValueError):
        otelm.parse_otlp_trace_payload(b"\xff\xff\xff\xffnot-valid-protobuf")


def test_serve_cli_parsing_dedup_socket_cleanup_and_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert servem.resolve_hosts("127.0.0.1", ["127.0.0.1", " 127.0.0.1 ", "", "10.0.0.1", "10.0.0.1"]) == [
        "127.0.0.1",
        "10.0.0.1",
    ]
    with pytest.raises(ValueError):
        servem.bind_sockets(["127.0.0.1"], -1)
    with pytest.raises(ValueError):
        servem.bind_sockets(["127.0.0.1"], 70000)

    # Mock uvicorn.Server and create_app so main() runs 100% offline & closes ephemeral sockets
    captured_sockets: list[Any] = []

    class _FakeServer:
        def __init__(self, config: Any) -> None:
            self.config = config

        def run(self, sockets: list[Any] | None = None) -> None:
            assert sockets is not None
            captured_sockets.extend(sockets)

    prices_file = tmp_path / "p.yaml"
    prices_file.write_text("models: {}\n", encoding="utf-8")
    monkeypatch.setenv("TOKENMETER_DB", str(tmp_path / "serve.db"))
    monkeypatch.setenv("TOKENMETER_PRICES", str(prices_file))
    monkeypatch.setattr(servem.uvicorn, "Server", _FakeServer)

    servem.main(["--host", "127.0.0.1", "--port", "0", "--extra-host", "127.0.0.1"])
    assert len(captured_sockets) == 1
    assert captured_sockets[0].fileno() == -1  # closed cleanly in finally!


def test_dashboard_helpers_vectorized_breakdown_and_mocked_ui(tmp_path: Path) -> None:
    db_file = tmp_path / "dash.db"
    prices_file = tmp_path / "prices.yaml"
    prices_file.write_text("per: 1000000\nmodels:\n  m1: {input: 1.0, output: 2.0}\n", encoding="utf-8")

    conn = dbm.connect(str(db_file))
    try:
        dbm.insert(
            conn,
            {
                "ts": "2026-11-07T01:00:00+00:00",
                "tag": "cli",
                "project": "proj-a",
                "model": "m1",
                "stream": 1,
                "status_code": 200,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "usage_source": "reported",
                "ttft_ms": 15.0,
                "total_ms": 40.0,
                "upstream_ms": 38.0,
                "overhead_ms": 2.0,
                "cost": 0.0002,
                "price_version": 1,
            },
            tags=[("env", "dev")],
        )
        dbm.insert(
            conn,
            {
                "ts": "2026-11-07T01:05:00+00:00",
                "tag": None,
                "project": None,
                "model": "unpriced-m",
                "stream": 1,
                "status_code": 502,
                "error": "ConnectError",
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "usage_source": "estimated",
                "ttft_ms": None,
                "total_ms": 5.0,
                "upstream_ms": 4.0,
                "overhead_ms": 1.0,
                "cost": None,
                "price_version": 1,
            },
        )
    finally:
        conn.close()

    raw_df, pv_df, tags_df, warn = dashboard.load_dashboard_data(str(db_file), str(prices_file))
    assert warn is None
    assert len(raw_df) == 2
    assert len(pv_df) == 1
    assert len(tags_df) == 1

    # Broken YAML returns warning string without leaking connection or failing read
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("models: [unclosed", encoding="utf-8")
    _, _, _, warn_bad = dashboard.load_dashboard_data(str(db_file), str(bad_yaml))
    assert warn_bad is not None and "Could not load prices file" in warn_bad

    prep = dashboard.prepare_requests_df(raw_df)
    filtered = dashboard.filter_requests_df(prep, ["m1"], ["proj-a"])
    assert len(filtered) == 1

    summary = dashboard.compute_summary_metrics(prep)
    assert summary["calls"] == 2
    assert summary["error_rate"] == pytest.approx(0.5)
    assert summary["total_tokens"] == 180
    assert summary["estimated_rows"] == 1
    assert summary["cost_total"] == pytest.approx(0.0002)

    by_model = dashboard.compute_breakdown(prep, "model").set_index("model")
    assert by_model.loc["m1", "estimated_rows"] == 0
    assert by_model.loc["m1", "unpriced_calls"] == 0
    assert by_model.loc["unpriced-m", "estimated_rows"] == 1
    assert by_model.loc["unpriced-m", "unpriced_calls"] == 1

    tag_bd = dashboard.compute_tag_breakdown(tags_df, prep)
    assert len(tag_bd) == 1 and tag_bd.iloc[0]["calls"] == 1

    clean = dashboard.export_clean_df(prep)
    assert {"is_error", "is_estimated", "is_unpriced"}.isdisjoint(clean.columns)

    # Test isolated UI branches with mocked st (missing DB, empty DB, and full render)
    class _StopError(Exception):
        pass

    mock_st = MagicMock()
    mock_st.stop.side_effect = _StopError
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
    mock_st.tabs.side_effect = lambda names: [MagicMock() for _ in names]
    mock_st.multiselect.return_value = []
    mock_st.selectbox.side_effect = lambda label, opts, **kw: opts[kw.get("index", 0)] if isinstance(opts, list) else opts.iloc[0]

    # 1. Missing DB stops cleanly
    with pytest.raises(_StopError):
        dashboard.render_dashboard(str(tmp_path / "missing.db"), str(prices_file), st_mod=mock_st)
    mock_st.info.assert_called()

    # 2. Empty DB stops cleanly
    empty_db = tmp_path / "empty.db"
    dbm.connect(str(empty_db)).close()
    with pytest.raises(_StopError):
        dashboard.render_dashboard(str(empty_db), str(prices_file), st_mod=mock_st)

    # 3. Full render with broken prices warning + estimated rows warning
    dashboard.render_dashboard(str(db_file), str(bad_yaml), st_mod=mock_st)
    assert mock_st.dataframe.call_count >= 5
    assert mock_st.download_button.call_count == 1
