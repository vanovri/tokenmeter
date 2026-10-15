import asyncio, json, sqlite3, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import httpx, pytest
from tests.test_proxy import fake_upstream, run, rows, CALLS, SECRET
from tokenmeter.proxy import create_app

def mkapp(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("NINEROUTER_API_KEY", SECRET)
    for k, v in env.items(): monkeypatch.setenv(k, v)
    p = tmp_path / "p.yaml"; p.write_text("models: {}\n")
    return create_app("http://fake/v1", str(tmp_path / "d.db"), str(p), transport=httpx.MockTransport(fake_upstream)), tmp_path / "d.db"

def test_project_path_route(tmp_path, monkeypatch):
    monkeypatch.delenv("TOKENMETER_OTEL", raising=False)
    app, db = mkapp(tmp_path, monkeypatch)
    assert app.state.otel is None                          # off by default
    r = run(app, lambda c: c.post("/p/agentic-rag/v1/chat/completions", json={"model": "m", "messages": []}))
    assert r.status_code == 200 and rows(db)[0]["project"] == "agentic-rag"
    assert CALLS[-1].url.path == "/v1/chat/completions"     # upstream path unchanged
    assert run(app, lambda c: c.get("/p/x/v1/models")).status_code == 200

def test_otel_export_to_fake_collector(tmp_path, monkeypatch):
    got = []
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200); self.send_header("Content-Type", "application/x-protobuf"); self.end_headers()
        def log_message(self, *a): pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    app, db = mkapp(tmp_path, monkeypatch, TOKENMETER_OTEL="1",
                    OTEL_EXPORTER_OTLP_ENDPOINT=f"http://127.0.0.1:{srv.server_port}")
    run(app, lambda c: c.post("/v1/chat/completions", json={"model": "xai/grok-code-fast-1", "messages": []},
                              headers={"X-TokenMeter-Project": "p1", "X-TokenMeter-Tags": "pattern_id=p7"}))
    run(app, lambda c: c.post("/v1/chat/completions", json={"model": "bad/model", "messages": []}))
    app.state.otel.flush(); srv.shutdown()
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    spans = []
    for path, body in got:
        assert path == "/v1/traces"
        req = ExportTraceServiceRequest(); req.ParseFromString(body)
        for rs in req.resource_spans:
            for ss in rs.scope_spans: spans += list(ss.spans)
    assert len(spans) == 2
    by = {s.name: s for s in spans}
    a = {kv.key: kv.value for kv in by["chat xai/grok-code-fast-1"].attributes}
    assert a["gen_ai.operation.name"].string_value == "chat"
    assert a["gen_ai.request.model"].string_value == "xai/grok-code-fast-1"
    assert a["gen_ai.usage.input_tokens"].int_value == 11 and a["gen_ai.usage.output_tokens"].int_value == 7
    assert a["tokenmeter.project"].string_value == "p1" and a["tokenmeter.tag.pattern_id"].string_value == "p7"
    assert "tokenmeter.cost" not in a                      # unpriced -> attribute omitted, not guessed
    assert by["chat bad/model"].status.code == 2           # ERROR
    assert all(SECRET.encode() not in b for _, b in got)

def test_placeholder_auth_replaced(tmp_path, monkeypatch):
    app, db = mkapp(tmp_path, monkeypatch, TOKENMETER_REPLACE_AUTH="tokenmeter-local")
    run(app, lambda c: c.post("/v1/chat/completions", json={"model": "m", "messages": []}, headers={"Authorization": "Bearer tokenmeter-local"}))
    assert CALLS[-1].headers["authorization"] == f"Bearer {SECRET}"
    run(app, lambda c: c.post("/v1/chat/completions", json={"model": "m", "messages": []}, headers={"Authorization": "Bearer other"}))
    assert CALLS[-1].headers["authorization"] == "Bearer other"
