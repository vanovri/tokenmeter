from __future__ import annotations

import builtins
import math
from pathlib import Path
from typing import Any

import pytest

from tokenmeter import db as dbm
from tokenmeter.prices import Prices, compute_cost, resolve_model_pricing


def test_compute_cost_basic_and_long_context() -> None:
    catalog: dict[str, Any] = {
        "currency": "USD",
        "per": 1_000_000,
        "models": {
            "xai/grok-4.6": {
                "input": 2.0,
                "output": 6.0,
                "cached_input": 0.5,
                "long_context": {
                    "threshold": 200_000,
                    "input": 4.0,
                    "output": 12.0,
                    "cached_input": 1.0,
                },
            },
            "unpriced/model": {"input": None, "output": None},
        },
    }

    assert compute_cost(catalog, "xai/grok-4.6", 1_000, 500) == pytest.approx(
        (1_000 * 2.0 + 500 * 6.0) / 1_000_000
    )
    assert compute_cost(catalog, "xai/grok-4.6", 199_999, 100) == pytest.approx(
        (199_999 * 2.0 + 100 * 6.0) / 1_000_000
    )
    assert compute_cost(catalog, "xai/grok-4.6", 200_000, 100) == pytest.approx(
        (200_000 * 4.0 + 100 * 12.0) / 1_000_000
    )
    assert compute_cost(catalog, "unpriced/model", 100, 100) is None
    assert compute_cost(catalog, "unknown/model", 100, 100) is None
    assert compute_cost(catalog, None, 100, 100) is None
    assert compute_cost(catalog, "xai/grok-4.6", None, 100) is None
    assert compute_cost(catalog, "xai/grok-4.6", 100, None) is None
    assert compute_cost(None, "xai/grok-4.6", 100, 100) is None


def test_wildcard_and_prefix_model_matching() -> None:
    models: dict[str, Any] = {
        "*": {"input": 0.1, "output": 0.2},
        "openai/*": {"input": 1.0, "output": 3.0},
        "openai/gpt-4o*": {"input": 2.5, "output": 10.0},
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }
    catalog = {"per": 1_000_000, "models": models}

    # Exact match beats wildcard
    assert resolve_model_pricing(models, "openai/gpt-4o-mini") == {"input": 0.15, "output": 0.60}
    assert compute_cost(catalog, "openai/gpt-4o-mini", 1_000, 1_000) == pytest.approx(0.75 / 1_000)

    # More specific wildcard beats broader prefix wildcard
    assert compute_cost(catalog, "openai/gpt-4o-2026-08-06", 1_000, 1_000) == pytest.approx(12.5 / 1_000)
    assert compute_cost(catalog, "openai/o3-mini", 1_000, 1_000) == pytest.approx(4.0 / 1_000)

    # Catch-all "*" matches only when explicitly configured
    assert compute_cost(catalog, "other/custom", 1_000, 1_000) == pytest.approx(0.3 / 1_000)


def test_cached_reasoning_and_audio_token_cost_calculation() -> None:
    catalog: dict[str, Any] = {
        "per": 1_000_000,
        "models": {
            "omni/pro": {
                "input": 2.0,
                "output": 8.0,
                "cached_input": 0.5,
                "reasoning": 12.0,
                "audio_input": 20.0,
                "audio_output": 40.0,
            },
            "text/only": {
                "input": 2.0,
                "output": 8.0,
            },
        },
    }

    # 1000 prompt tokens: 300 cached (@0.5), 100 audio_in (@20.0), remaining 600 text (@2.0)
    # 500 completion tokens: 200 reasoning (@12.0), 50 audio_out (@40.0), remaining 250 text (@8.0)
    cost = compute_cost(
        catalog,
        "omni/pro",
        1_000,
        500,
        cached_tokens=300,
        reasoning_tokens=200,
        audio_input_tokens=100,
        audio_output_tokens=50,
    )
    expected_in = 600 * 2.0 + 300 * 0.5 + 100 * 20.0
    expected_out = 250 * 8.0 + 200 * 12.0 + 50 * 40.0
    assert cost == pytest.approx((expected_in + expected_out) / 1_000_000)

    # Subset tokens exceeding total pt/ct are safely clamped to available tokens
    clamped = compute_cost(catalog, "omni/pro", 100, 50, cached_tokens=500, reasoning_tokens=500)
    assert clamped == pytest.approx((100 * 0.5 + 50 * 12.0) / 1_000_000)

    # Model without specialized rates falls back to standard input/output rates cleanly
    fallback = compute_cost(
        catalog,
        "text/only",
        1_000,
        500,
        cached_tokens=300,
        reasoning_tokens=200,
    )
    assert fallback == pytest.approx((1_000 * 2.0 + 500 * 8.0) / 1_000_000)


@pytest.mark.parametrize(
    ("pt", "ct"),
    [
        (True, 10),
        (10, False),
        (-1, 10),
        (10, -5),
        (math.nan, 10),
        (10, math.inf),
        (1.5, 10),
    ],
)
def test_compute_cost_rejects_invalid_token_types_and_values(pt: Any, ct: Any) -> None:
    catalog = {"per": 1_000_000, "models": {"m": {"input": 1.0, "output": 2.0}}}
    with pytest.raises((TypeError, ValueError)):
        compute_cost(catalog, "m", pt, ct)


@pytest.mark.parametrize(
    "bad_catalog",
    [
        {"per": 0, "models": {"m": {"input": 1.0, "output": 2.0}}},
        {"per": -1000, "models": {"m": {"input": 1.0, "output": 2.0}}},
        {"per": True, "models": {"m": {"input": 1.0, "output": 2.0}}},
        {"per": math.nan, "models": {"m": {"input": 1.0, "output": 2.0}}},
        {"per": 1_000_000, "models": {"m": {"input": True, "output": 2.0}}},
        {"per": 1_000_000, "models": {"m": {"input": -1.0, "output": 2.0}}},
        {"per": 1_000_000, "models": {"m": {"input": 1.0, "output": math.inf}}},
    ],
)
def test_compute_cost_rejects_invalid_rates_and_per(bad_catalog: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        compute_cost(bad_catalog, "m", 10, 10)


def test_prices_stat_caching_and_corrupted_reload_resilience(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    yaml_file = tmp_path / "prices.yaml"
    yaml_file.write_text("per: 1000000\nmodels:\n  m1: {input: 1.0, output: 2.0}\n", encoding="utf-8")

    conn = dbm.connect(str(tmp_path / "stat.db"))
    try:
        prices = Prices(str(yaml_file), conn)
        open_calls = 0
        real_open = builtins.open

        def counting_open(file: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal open_calls
            if str(file) == str(yaml_file):
                open_calls += 1
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", counting_open)

        # Unchanged mtime_ns and size -> zero file opens on repeated reload() / cost() calls
        assert prices.reload() is False
        assert prices.cost("m1", 100, 50) == pytest.approx(200.0 / 1_000_000)
        assert open_calls == 0

        # Corrupt YAML on disk -> reload() raises YAMLError directly, but cost() preserves last good prices
        yaml_file.write_text("models: [unclosed: {yaml\n", encoding="utf-8")
        assert prices.cost("m1", 100, 50) == pytest.approx(200.0 / 1_000_000)

        # Fixing YAML updates version and cost cleanly
        yaml_file.write_text("per: 1000000\nmodels:\n  m1: {input: 3.0, output: 4.0}\n", encoding="utf-8")
        assert prices.cost("m1", 100, 50) == pytest.approx(500.0 / 1_000_000)
    finally:
        conn.close()


def test_prices_batched_repricing_across_multiple_chunks(tmp_path: Path) -> None:
    yaml_file = tmp_path / "prices.yaml"
    yaml_file.write_text("per: 1000000\nmodels:\n  m1: {input: null, output: null}\n", encoding="utf-8")
    conn = dbm.connect(str(tmp_path / "batch.db"))
    try:
        prices = Prices(str(yaml_file), conn)
        v1 = prices.version

        # Insert 4,500 rows (exceeding the 2,000 fetchmany batch size across 3 batches)
        rows_data = [
            ("2026-11-07T00:00:00+00:00", "m1" if i % 5 != 0 else "unknown", 100, 50, None, v1)
            for i in range(4500)
        ]
        with conn:
            conn.executemany(
                "INSERT INTO requests(ts, model, prompt_tokens, completion_tokens, cost, price_version) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows_data,
            )

        yaml_file.write_text("per: 1000000\nmodels:\n  m1: {input: 2.0, output: 6.0}\n", encoding="utf-8")
        assert prices.reload() is True
        v2 = prices.version
        assert v2 != v1

        total_v2 = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE price_version = ?", (v2,)
        ).fetchone()[0]
        assert total_v2 == 4500

        priced_count, sample_cost = conn.execute(
            "SELECT COUNT(*), AVG(cost) FROM requests WHERE model = 'm1'"
        ).fetchone()
        assert priced_count == 3600
        assert sample_cost == pytest.approx((100 * 2.0 + 50 * 6.0) / 1_000_000)

        unpriced_non_null = conn.execute(
            "SELECT COUNT(*) FROM requests WHERE model = 'unknown' AND cost IS NOT NULL"
        ).fetchone()[0]
        assert unpriced_non_null == 0
    finally:
        conn.close()
