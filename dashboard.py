from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Sequence
from typing import Any, TypedDict

import pandas as pd
import streamlit as st
import yaml

from tokenmeter import db as dbm
from tokenmeter.prices import Prices

log = logging.getLogger(__name__)

NUMERIC_COLS: tuple[str, ...] = (
    "total_ms",
    "ttft_ms",
    "upstream_ms",
    "overhead_ms",
    "cost",
    "prompt_tokens",
    "completion_tokens",
)
INTERNAL_FLAG_COLS: list[str] = ["is_error", "is_estimated", "is_unpriced"]


class DashboardSummary(TypedDict):
    calls: int
    error_rate: float
    total_tokens: int
    cost_total: float | None
    overhead_p50: float | None
    estimated_rows: int


def default_paths() -> tuple[str, str]:
    db_path = os.environ.get("TOKENMETER_DB", "tokenmeter.db")
    prices_default = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config", "prices.yaml"
    )
    prices_path = os.environ.get("TOKENMETER_PRICES", prices_default)
    return db_path, prices_path


def load_dashboard_data(
    db_path: str,
    prices_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    """Connect to SQLite, sync prices.yaml safely, read DataFrames, and close connection."""
    conn = dbm.connect(db_path)
    warning_msg: str | None = None
    try:
        try:
            Prices(prices_path, conn)
        except (
            OSError,
            UnicodeDecodeError,
            yaml.YAMLError,
            sqlite3.Error,
            ValueError,
            TypeError,
        ) as exc:
            log.warning("Dashboard could not load prices file %s: %s", prices_path, exc)
            warning_msg = f"Could not load prices file: {type(exc).__name__}"

        df = pd.read_sql("SELECT * FROM requests ORDER BY id DESC", conn)
        pv = pd.read_sql(
            "SELECT id, loaded_at, yaml_sha, content FROM price_versions ORDER BY id DESC",
            conn,
        )
        tags = pd.read_sql("SELECT request_id, key, value FROM tags", conn)
    finally:
        conn.close()
    return df, pv, tags, warning_msg


def prepare_requests_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["status_code"] = pd.to_numeric(df["status_code"], errors="coerce")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df["is_error"] = (df["status_code"] >= 400) | df["error"].notna()
    df["is_estimated"] = df["usage_source"] == "estimated"
    df["is_unpriced"] = df["cost"].isna()
    df["tag"] = df["tag"].fillna("(none)")
    df["project"] = df["project"].fillna("(none)")
    return df


def filter_requests_df(
    df: pd.DataFrame,
    models: Sequence[str],
    projects: Sequence[str],
) -> pd.DataFrame:
    out = df
    if models:
        out = out[out["model"].isin(models)]
    if projects:
        out = out[out["project"].isin(projects)]
    return out


def compute_summary_metrics(df: pd.DataFrame) -> DashboardSummary:
    calls = len(df)
    error_rate = float(df["is_error"].mean()) if calls else 0.0
    total_tokens = int(
        df["prompt_tokens"].fillna(0).sum() + df["completion_tokens"].fillna(0).sum()
    )
    raw_cost = df["cost"].sum(min_count=1)
    cost_total = None if pd.isna(raw_cost) else float(raw_cost)
    overhead_p50 = (
        None if df["overhead_ms"].isna().all() else float(df["overhead_ms"].median())
    )
    estimated_rows = int(df["is_estimated"].sum())
    return {
        "calls": calls,
        "error_rate": error_rate,
        "total_tokens": total_tokens,
        "cost_total": cost_total,
        "overhead_p50": overhead_p50,
        "estimated_rows": estimated_rows,
    }


def compute_breakdown(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Compute vectorized per-dimension latency, token, error, and cost breakdown."""
    g = df.groupby(by, dropna=False)
    return pd.DataFrame(
        {
            "calls": g.size(),
            "prompt_tokens": g["prompt_tokens"].sum(min_count=1),
            "completion_tokens": g["completion_tokens"].sum(min_count=1),
            "estimated_rows": g["is_estimated"].sum().astype(int),
            "cost": g["cost"].sum(min_count=1),
            "unpriced_calls": g["is_unpriced"].sum().astype(int),
            "p50_total_ms": g["total_ms"].quantile(0.5),
            "p95_total_ms": g["total_ms"].quantile(0.95),
            "p50_ttft_ms": g["ttft_ms"].quantile(0.5),
            "p95_ttft_ms": g["ttft_ms"].quantile(0.95),
            "p50_overhead_ms": g["overhead_ms"].quantile(0.5),
            "p95_overhead_ms": g["overhead_ms"].quantile(0.95),
            "error_rate": g["is_error"].mean(),
        }
    ).reset_index()


def compute_time_series(
    df: pd.DataFrame,
    freq: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = df.set_index("ts")
    calls_ts = ts.groupby([pd.Grouper(freq=freq), "model"]).size().unstack(fill_value=0)
    cost_ts = ts.groupby([pd.Grouper(freq=freq), "model"])["cost"].sum(min_count=1).unstack()
    mg = df.groupby("model")
    latency = pd.DataFrame(
        {
            "p50_total": mg["total_ms"].median(),
            "p95_total": mg["total_ms"].quantile(0.95),
            "p50_ttft": mg["ttft_ms"].median(),
            "p50_overhead": mg["overhead_ms"].median(),
        }
    )
    return calls_ts, cost_ts, latency


def compute_tag_breakdown(tags: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    merged = tags.merge(
        df[["id", "cost", "total_ms", "prompt_tokens", "completion_tokens"]],
        left_on="request_id",
        right_on="id",
    )
    tg = merged.groupby(["key", "value"], dropna=False)
    return pd.DataFrame(
        {
            "calls": tg["id"].count(),
            "cost": tg["cost"].sum(min_count=1),
            "tokens": tg["completion_tokens"].sum(),
            "p50_total_ms": tg["total_ms"].median(),
        }
    ).reset_index()


def export_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in INTERNAL_FLAG_COLS if c in df.columns]
    return df.drop(columns=drop_cols)


def render_dashboard(
    db_path: str | None = None,
    prices_path: str | None = None,
    *,
    st_mod: Any = None,
) -> None:
    ui = st if st_mod is None else st_mod
    default_db, default_prices = default_paths()
    resolved_db = db_path if db_path is not None else default_db
    resolved_prices = prices_path if prices_path is not None else default_prices

    ui.set_page_config(page_title="TokenMeter", layout="wide")
    ui.title("TokenMeter")
    if not os.path.exists(resolved_db):
        ui.info(f"No database yet at {resolved_db}")
        ui.stop()
        return

    raw_df, pv, tags, warn = load_dashboard_data(resolved_db, resolved_prices)
    if warn:
        ui.warning(warn)
    if raw_df.empty:
        ui.info("No requests recorded yet.")
        ui.stop()
        return

    df = prepare_requests_df(raw_df)
    with ui.sidebar:
        models = ui.multiselect("Models", sorted(df["model"].dropna().unique()))
        projects = ui.multiselect("Projects", sorted(df["project"].unique()))
        df = filter_requests_df(df, models, projects)

    summary = compute_summary_metrics(df)
    cols = ui.columns(5)
    cols[0].metric("Calls", summary["calls"])
    cols[1].metric("Error rate", f"{summary['error_rate']:.1%}")
    cols[2].metric("Tokens", summary["total_tokens"])
    cols[3].metric(
        "List-price equiv. cost (USD)",
        "unpriced" if summary["cost_total"] is None else f"{summary['cost_total']:.6f}",
    )
    cols[4].metric(
        "Proxy overhead p50",
        "n/a" if summary["overhead_p50"] is None else f"{summary['overhead_p50']:.2f} ms",
    )
    ui.caption(
        "Cost = public list-price equivalent from config/prices.yaml. "
        "9Router/Antigravity routes may be free or subscription-billed, so this is not your actual bill."
    )
    if summary["estimated_rows"]:
        ui.warning(
            f"{summary['estimated_rows']} row(s) use ESTIMATED tokens "
            "(upstream sent no usage; ~4 chars/token heuristic)."
        )

    t1, t2, t3, t4, t5, t6 = ui.tabs(
        ["By model", "By client tag", "By project", "Charts", "Key/value tags", "Price versions"]
    )
    with t1:
        ui.dataframe(compute_breakdown(df, "model"), width="stretch")
    with t2:
        ui.dataframe(compute_breakdown(df, "tag"), width="stretch")
    with t3:
        ui.dataframe(compute_breakdown(df, "project"), width="stretch")
    with t4:
        freq = ui.selectbox("Bucket", ["1min", "15min", "1h", "1D"], index=2)
        calls_ts, cost_ts, latency = compute_time_series(df, freq)
        ui.caption("Calls over time")
        ui.bar_chart(calls_ts)
        ui.caption("Cost over time (priced rows only)")
        if cost_ts.notna().any().any():
            ui.line_chart(cost_ts)
        else:
            ui.info("No priced rows yet: fill config/prices.yaml to see cost.")
        ui.caption("Latency by model (p50/p95 ms)")
        ui.bar_chart(latency)
    with t5:
        if tags.empty:
            ui.info("No key/value tags yet (send header X-TokenMeter-Tags: k=v; k2=v2).")
        else:
            ui.dataframe(compute_tag_breakdown(tags, df), width="stretch")
    with t6:
        ui.caption(
            "Every distinct prices.yaml seen. On change, all rows are recomputed with the newest version."
        )
        ui.dataframe(pv[["id", "loaded_at", "yaml_sha"]], width="stretch")
        if not pv.empty:
            pick = ui.selectbox("Show content of version", pv["id"])
            ui.code(pv.set_index("id")["content"][pick], language="yaml")

    ui.subheader("Recent requests")
    clean_df = export_clean_df(df)
    ui.dataframe(clean_df.head(200), width="stretch")
    ui.download_button(
        "Download CSV",
        clean_df.to_csv(index=False),
        "tokenmeter_requests.csv",
    )


def _has_streamlit_context() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except (ImportError, TypeError):
        return False


if __name__ == "__main__" or _has_streamlit_context():
    render_dashboard()
