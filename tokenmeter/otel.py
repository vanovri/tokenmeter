"""Optional OpenTelemetry GenAI span export (OTLP/HTTP). Off unless TOKENMETER_OTEL=1.

Endpoint: OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, or OTEL_EXPORTER_OTLP_ENDPOINT + /v1/traces
(default http://127.0.0.1:4318/v1/traces). Spans never carry prompts, outputs or keys.
"""
import os, time, datetime

class OTelEmitter:
    def __init__(self, endpoint, service="tokenmeter", exporter=None):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        if exporter is None:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=endpoint)
        self.endpoint = endpoint
        self.provider = TracerProvider(resource=Resource.create({"service.name": service}))
        self.provider.add_span_processor(BatchSpanProcessor(exporter, schedule_delay_millis=1000))
        self.tracer = self.provider.get_tracer("tokenmeter")

    def emit(self, r: dict, tags=None):
        from opentelemetry.trace import Status, StatusCode
        end_ns = time.time_ns()
        start_ns = end_ns - int((r.get("total_ms") or 0) * 1e6)
        model = r.get("model") or "unknown"
        attrs = {"gen_ai.operation.name": "chat", "gen_ai.request.model": model, "gen_ai.system": "openai"}
        def put(k, v):
            if v is not None: attrs[k] = v
        put("gen_ai.usage.input_tokens", r.get("prompt_tokens"))
        put("gen_ai.usage.output_tokens", r.get("completion_tokens"))
        put("http.response.status_code", r.get("status_code"))
        put("tokenmeter.cost", r.get("cost"))                 # custom: semconv has no cost
        put("tokenmeter.price_version", r.get("price_version"))
        put("tokenmeter.usage_source", r.get("usage_source"))
        put("tokenmeter.ttft_ms", r.get("ttft_ms"))
        put("tokenmeter.upstream_ms", r.get("upstream_ms"))
        put("tokenmeter.overhead_ms", r.get("overhead_ms"))
        put("tokenmeter.client_tag", r.get("tag"))
        put("tokenmeter.project", r.get("project"))
        attrs["tokenmeter.stream"] = bool(r.get("stream"))
        for k, v in tags or []:
            attrs[f"tokenmeter.tag.{k}"] = v if v is not None else ""
        span = self.tracer.start_span(f"chat {model}", start_time=start_ns, attributes=attrs)
        if (r.get("status_code") or 0) >= 400 or r.get("error"):
            span.set_status(Status(StatusCode.ERROR, (r.get("error") or "")[:200]))  # already redacted
        span.end(end_time=end_ns)

    def flush(self): self.provider.force_flush()
    def shutdown(self): self.provider.shutdown()

def from_env():
    if os.environ.get("TOKENMETER_OTEL", "").lower() not in ("1", "true", "yes", "on"):
        return None
    ep = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if not ep:
        ep = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318").rstrip("/") + "/v1/traces"
    return OTelEmitter(ep)
