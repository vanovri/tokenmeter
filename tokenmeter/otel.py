"""Optional OpenTelemetry GenAI span export (OTLP/HTTP). Off unless TOKENMETER_OTEL=1.

Endpoint: OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, or OTEL_EXPORTER_OTLP_ENDPOINT + /v1/traces
(default http://127.0.0.1:4318/v1/traces). Spans never carry prompts, outputs or keys.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

log = logging.getLogger(__name__)

SpanAttrValue = str | bool | int | float
TagPair = tuple[str, str | None]


def _clean_int(val: Any, *, allow_negative: bool = False) -> int | None:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, float):
        if not math.isfinite(val) or not val.is_integer():
            return None
        val = int(val)
    elif not isinstance(val, int):
        return None
    if not allow_negative and val < 0:
        return None
    return val


def _clean_float(val: Any, *, allow_negative: bool = False) -> float | None:
    if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    fval = float(val)
    if not math.isfinite(fval):
        return None
    if not allow_negative and fval < 0.0:
        return None
    return fval


def _clean_str(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val)
    return s if s else None


def build_span_attributes(
    r: Mapping[str, Any],
    tags: Sequence[TagPair] | None = None,
) -> dict[str, SpanAttrValue]:
    """Construct validated OpenTelemetry GenAI and TokenMeter span attributes."""
    model = _clean_str(r.get("model")) or "unknown"
    attrs: dict[str, SpanAttrValue] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": model,
        "gen_ai.system": "openai",
    }

    def put(key: str, val: SpanAttrValue | None) -> None:
        if val is not None:
            attrs[key] = val

    put("gen_ai.usage.input_tokens", _clean_int(r.get("prompt_tokens")))
    put("gen_ai.usage.output_tokens", _clean_int(r.get("completion_tokens")))
    put("http.response.status_code", _clean_int(r.get("status_code")))
    put("tokenmeter.cost", _clean_float(r.get("cost")))
    put("tokenmeter.price_version", _clean_int(r.get("price_version")))
    put("tokenmeter.usage_source", _clean_str(r.get("usage_source")))
    put("tokenmeter.ttft_ms", _clean_float(r.get("ttft_ms")))
    put("tokenmeter.upstream_ms", _clean_float(r.get("upstream_ms")))
    put("tokenmeter.overhead_ms", _clean_float(r.get("overhead_ms")))
    put("tokenmeter.client_tag", _clean_str(r.get("tag")))
    put("tokenmeter.project", _clean_str(r.get("project")))
    attrs["tokenmeter.stream"] = bool(r.get("stream"))

    for key, val in tags or []:
        if key:
            attrs[f"tokenmeter.tag.{key}"] = str(val) if val is not None else ""
    return attrs


def _decode_otlp_json_anyvalue(val_obj: Mapping[str, Any]) -> SpanAttrValue | None:
    if "stringValue" in val_obj:
        return str(val_obj["stringValue"])
    if "intValue" in val_obj:
        raw_i = val_obj["intValue"]
        if isinstance(raw_i, bool):
            return None
        return int(raw_i)
    if "doubleValue" in val_obj:
        raw_f = val_obj["doubleValue"]
        if isinstance(raw_f, bool):
            return None
        fval = float(raw_f)
        return fval if math.isfinite(fval) else None
    if "boolValue" in val_obj and isinstance(val_obj["boolValue"], bool):
        return val_obj["boolValue"]
    return None


def _parse_otlp_json(text: str) -> list[dict[str, Any]]:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid OTLP JSON payload: {exc}") from exc
    if not isinstance(doc, Mapping):
        raise TypeError("OTLP JSON root must be an object")
    r_spans = doc.get("resourceSpans") if "resourceSpans" in doc else doc.get("resource_spans")
    if not isinstance(r_spans, list):
        raise TypeError("OTLP JSON missing 'resourceSpans' list")

    spans_out: list[dict[str, Any]] = []
    for rs in r_spans:
        if not isinstance(rs, Mapping):
            continue
        s_spans = rs.get("scopeSpans") if "scopeSpans" in rs else rs.get("scope_spans", [])
        if not isinstance(s_spans, list):
            continue
        for ss in s_spans:
            if not isinstance(ss, Mapping):
                continue
            raw_spans = ss.get("spans", [])
            if not isinstance(raw_spans, list):
                continue
            for sp in raw_spans:
                if not isinstance(sp, Mapping):
                    continue
                attrs: dict[str, SpanAttrValue] = {}
                for kv in sp.get("attributes", []):
                    if isinstance(kv, Mapping) and isinstance(kv.get("key"), str):
                        v_map = kv.get("value")
                        if isinstance(v_map, Mapping):
                            decoded = _decode_otlp_json_anyvalue(v_map)
                            if decoded is not None:
                                attrs[kv["key"]] = decoded
                spans_out.append({"name": str(sp.get("name", "")), "attributes": attrs})
    return spans_out


def parse_otlp_trace_payload(payload: bytes | str) -> list[dict[str, Any]]:
    """Parse OTLP trace export payload (JSON string/bytes or Protobuf bytes) into normalized spans."""
    if isinstance(payload, str):
        return _parse_otlp_json(payload)
    if not isinstance(payload, bytes):
        raise TypeError(f"Expected bytes or str, got {type(payload).__name__}")
    if not payload:
        return []

    stripped = payload.lstrip()
    if stripped.startswith(b"{"):
        try:
            return _parse_otlp_json(payload.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError(f"Invalid UTF-8 in OTLP JSON payload: {exc}") from exc

    from google.protobuf.message import DecodeError
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    req = ExportTraceServiceRequest()
    try:
        req.ParseFromString(payload)
    except DecodeError as exc:
        raise ValueError(f"Malformed OTLP protobuf payload: {exc}") from exc

    if not req.resource_spans or req.DiscardUnknownFields() is None and not req.resource_spans:
        raise ValueError("Unrecognized or empty OTLP protobuf message")

    spans_out: list[dict[str, Any]] = []
    for rs in req.resource_spans:
        for ss in rs.scope_spans:
            for sp in ss.spans:
                attrs: dict[str, SpanAttrValue] = {}
                for kv in sp.attributes:
                    which = kv.value.WhichOneof("value")
                    if which == "string_value":
                        attrs[kv.key] = kv.value.string_value
                    elif which == "int_value":
                        attrs[kv.key] = int(kv.value.int_value)
                    elif which == "double_value" and math.isfinite(kv.value.double_value):
                        attrs[kv.key] = float(kv.value.double_value)
                    elif which == "bool_value":
                        attrs[kv.key] = bool(kv.value.bool_value)
                spans_out.append({"name": sp.name, "attributes": attrs})
    return spans_out


class OTelEmitter:
    def __init__(
        self,
        endpoint: str,
        service: str = "tokenmeter",
        exporter: Any | None = None,
    ) -> None:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        if exporter is None:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=endpoint)
        self.endpoint: str = endpoint
        self.provider = TracerProvider(resource=Resource.create({"service.name": service}))
        self.provider.add_span_processor(
            BatchSpanProcessor(exporter, schedule_delay_millis=1000)
        )
        self.tracer = self.provider.get_tracer("tokenmeter")

    def emit(
        self,
        r: Mapping[str, Any],
        tags: Sequence[TagPair] | None = None,
    ) -> None:
        from opentelemetry.trace import Status, StatusCode

        end_ns = time.time_ns()
        duration_ms = _clean_float(r.get("total_ms")) or 0.0
        start_ns = end_ns - int(duration_ms * 1_000_000)
        attrs = build_span_attributes(r, tags)
        model = str(attrs["gen_ai.request.model"])

        span = self.tracer.start_span(
            f"chat {model}",
            start_time=start_ns,
            attributes=attrs,
        )
        status_code = _clean_int(r.get("status_code")) or 0
        err_msg = _clean_str(r.get("error"))
        if status_code >= 400 or err_msg:
            span.set_status(Status(StatusCode.ERROR, (err_msg or "")[:200]))
        span.end(end_time=end_ns)

    def flush(self) -> None:
        self.provider.force_flush()

    def shutdown(self) -> None:
        self.provider.shutdown()


def from_env() -> OTelEmitter | None:
    flag = os.environ.get("TOKENMETER_OTEL", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return None
    ep = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
    if not ep:
        base = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318").strip()
        ep = base.rstrip("/") + "/v1/traces"
    service = os.environ.get("OTEL_SERVICE_NAME", "tokenmeter").strip() or "tokenmeter"
    return OTelEmitter(ep, service=service)
