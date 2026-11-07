from __future__ import annotations

import datetime
import fnmatch
import hashlib
import logging
import math
import os
import sqlite3
import threading
from collections.abc import Mapping
from typing import Any, TypedDict

import yaml

from . import db as dbm

log = logging.getLogger(__name__)

REPRICE_BATCH_SIZE: int = 2000


class LongContextTier(TypedDict, total=False):
    threshold: int | float
    input: float | None
    output: float | None
    cached_input: float | None
    reasoning: float | None
    audio_input: float | None
    audio_output: float | None


class ModelPriceSpec(TypedDict, total=False):
    input: float | None
    output: float | None
    cached_input: float | None
    reasoning: float | None
    audio_input: float | None
    audio_output: float | None
    long_context: LongContextTier


class PricesCatalog(TypedDict, total=False):
    currency: str
    per: int | float
    models: Mapping[str, ModelPriceSpec]


def _validate_token_count(val: Any, name: str) -> int | None:
    if val is None:
        return None
    if isinstance(val, bool):
        raise TypeError(f"{name} cannot be bool")
    if isinstance(val, float):
        if not math.isfinite(val) or not val.is_integer():
            raise ValueError(f"{name} must be a finite integer, got {val!r}")
        val = int(val)
    elif not isinstance(val, int):
        raise TypeError(f"{name} must be int, got {type(val).__name__}")
    if val < 0:
        raise ValueError(f"{name} must be >= 0, got {val}")
    return val


def _validate_rate(val: Any, name: str) -> float | None:
    if val is None:
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(f"{name} rate must be numeric, got {type(val).__name__}")
    fval = float(val)
    if not math.isfinite(fval) or fval < 0.0:
        raise ValueError(f"{name} rate must be finite and >= 0, got {val!r}")
    return fval


def _validate_per(data: Mapping[str, Any]) -> float:
    raw_per = data.get("per")
    if raw_per is None:
        return 1_000_000.0
    if isinstance(raw_per, bool) or not isinstance(raw_per, (int, float)):
        raise TypeError(f"per must be numeric, got {type(raw_per).__name__}")
    per = float(raw_per)
    if not math.isfinite(per) or per <= 0.0:
        raise ValueError(f"per must be finite and > 0, got {raw_per!r}")
    return per


def resolve_model_pricing(
    models: Mapping[str, Any] | None,
    model: str | None,
) -> Mapping[str, Any] | None:
    """Resolve model pricing by exact requested ID first, then explicit '*' wildcard patterns."""
    if not models or not isinstance(models, Mapping) or not model or not isinstance(model, str):
        return None
    exact = models.get(model)
    if isinstance(exact, Mapping):
        return exact

    wildcard_candidates: list[tuple[int, int, str, Mapping[str, Any]]] = []
    for pattern, spec in models.items():
        if isinstance(pattern, str) and "*" in pattern and isinstance(spec, Mapping):
            literal_len = len(pattern.replace("*", ""))
            wildcard_candidates.append((literal_len, -pattern.count("*"), pattern, spec))

    wildcard_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for _, _, pattern, spec in wildcard_candidates:
        if fnmatch.fnmatchcase(model, pattern):
            return spec
    return None


def _select_pricing_tier(spec: Mapping[str, Any], prompt_tokens: int) -> Mapping[str, Any]:
    lc = spec.get("long_context")
    if not isinstance(lc, Mapping):
        return spec
    raw_thresh = lc.get("threshold")
    if raw_thresh is None:
        return spec
    thresh = _validate_rate(raw_thresh, "long_context.threshold")
    if thresh is not None and prompt_tokens >= thresh and lc.get("input") is not None:
        return lc
    return spec


def _partitioned_side_cost(
    total_tokens: int,
    base_rate: float,
    subsets: list[tuple[int | None, float | None]],
) -> float:
    remaining = total_tokens
    cost = 0.0
    for sub_tokens, sub_rate in subsets:
        if sub_tokens is not None and sub_tokens > 0 and sub_rate is not None:
            applied = min(sub_tokens, remaining)
            remaining -= applied
            cost += applied * sub_rate
    cost += remaining * base_rate
    return cost


def compute_cost(
    data: Mapping[str, Any] | None,
    model: str | None,
    pt: int | None,
    ct: int | None,
    *,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    audio_input_tokens: int | None = None,
    audio_output_tokens: int | None = None,
) -> float | None:
    """Compute list-price USD cost, or None if unpriced / token counts unknown. Never guesses."""
    clean_pt = _validate_token_count(pt, "prompt_tokens")
    clean_ct = _validate_token_count(ct, "completion_tokens")
    clean_cached = _validate_token_count(cached_tokens, "cached_tokens")
    clean_reasoning = _validate_token_count(reasoning_tokens, "reasoning_tokens")
    clean_audio_in = _validate_token_count(audio_input_tokens, "audio_input_tokens")
    clean_audio_out = _validate_token_count(audio_output_tokens, "audio_output_tokens")

    if not data or not isinstance(data, Mapping):
        return None
    per = _validate_per(data)
    models = data.get("models")
    if not isinstance(models, Mapping):
        return None

    spec = resolve_model_pricing(models, model)
    if not spec or clean_pt is None or clean_ct is None:
        return None

    tier = _select_pricing_tier(spec, clean_pt)
    in_rate = _validate_rate(tier.get("input"), "input")
    out_rate = _validate_rate(tier.get("output"), "output")
    if in_rate is None or out_rate is None:
        return None

    cached_rate = _validate_rate(tier.get("cached_input"), "cached_input")
    audio_in_rate = _validate_rate(tier.get("audio_input"), "audio_input")
    reasoning_rate = _validate_rate(tier.get("reasoning"), "reasoning")
    audio_out_rate = _validate_rate(tier.get("audio_output"), "audio_output")

    in_cost = _partitioned_side_cost(
        clean_pt,
        in_rate,
        [(clean_cached, cached_rate), (clean_audio_in, audio_in_rate)],
    )
    out_cost = _partitioned_side_cost(
        clean_ct,
        out_rate,
        [(clean_reasoning, reasoning_rate), (clean_audio_out, audio_out_rate)],
    )
    return (in_cost + out_cost) / per


def _parse_prices_yaml(content: str) -> dict[str, Any]:
    loaded = yaml.safe_load(content)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"prices YAML root must be a mapping, got {type(loaded).__name__}")
    models = loaded.get("models")
    if models is not None and not isinstance(models, Mapping):
        raise TypeError("prices YAML 'models' field must be a mapping")
    _validate_per(loaded)
    return loaded


class Prices:
    def __init__(
        self,
        path: str | os.PathLike[str],
        conn: sqlite3.Connection | None = None,
        *,
        batch_size: int = REPRICE_BATCH_SIZE,
    ) -> None:
        if isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.path: str = os.fspath(path)
        self.conn: sqlite3.Connection | None = conn
        self.batch_size: int = batch_size
        self.version: int | None = None
        self.data: dict[str, Any] = {}
        self._sha: str | None = None
        self._stat: tuple[int, int] | None = None
        self._reload_lock = threading.RLock()
        self.reload()

    def _sync_version_and_reprice(self, sha: str, content: str) -> None:
        if self.conn is None:
            return
        with dbm._lock:
            c = self.conn
            row = c.execute(
                "SELECT id FROM price_versions WHERE yaml_sha=?", (sha,)
            ).fetchone()
            if row:
                self.version = int(row[0])
            else:
                now_iso = datetime.datetime.now(datetime.UTC).isoformat()
                cur_ins = c.execute(
                    "INSERT INTO price_versions(loaded_at,yaml_sha,content) VALUES(?,?,?)",
                    (now_iso, sha, content),
                )
                self.version = int(cur_ins.lastrowid or 0)

            cur = c.execute(
                "SELECT id, model, prompt_tokens, completion_tokens FROM requests "
                "WHERE price_version IS NULL OR price_version != ? ORDER BY id",
                (self.version,),
            )
            while batch := cur.fetchmany(self.batch_size):
                updates = [
                    (compute_cost(self.data, m, pt, ct), self.version, row_id)
                    for row_id, m, pt, ct in batch
                ]
                c.executemany(
                    "UPDATE requests SET cost=?, price_version=? WHERE id=?",
                    updates,
                )
            c.commit()

    def reload(self) -> bool:
        """Re-read YAML when file stat changes; record distinct price_versions and reprice in batches."""
        with self._reload_lock:
            st = os.stat(self.path)
            sig = (st.st_mtime_ns, st.st_size)
            if sig == self._stat:
                return False
            with open(self.path, encoding="utf-8", newline="") as fh:
                content = fh.read()
            parsed = _parse_prices_yaml(content)
            sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            self._stat = sig
            if sha == self._sha:
                return False
            self._sha = sha
            self.data = parsed
            self._sync_version_and_reprice(sha, content)
            return True

    def cost(
        self,
        model: str | None,
        pt: int | None,
        ct: int | None,
        *,
        cached_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        audio_input_tokens: int | None = None,
        audio_output_tokens: int | None = None,
    ) -> float | None:
        try:
            self.reload()
        except (
            OSError,
            UnicodeDecodeError,
            yaml.YAMLError,
            sqlite3.Error,
            ValueError,
            TypeError,
        ) as exc:
            log.warning("Failed to reload prices from %s: %s: %s", self.path, type(exc).__name__, exc)
        return compute_cost(
            self.data,
            model,
            pt,
            ct,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            audio_input_tokens=audio_input_tokens,
            audio_output_tokens=audio_output_tokens,
        )
