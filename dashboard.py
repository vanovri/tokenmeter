import os, sqlite3
import pandas as pd
import streamlit as st
from tokenmeter import db as dbm
from tokenmeter.prices import Prices

DB = os.environ.get("TOKENMETER_DB", "tokenmeter.db")
PRICES = os.environ.get("TOKENMETER_PRICES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "prices.yaml"))
st.set_page_config(page_title="TokenMeter", layout="wide")
st.title("TokenMeter")
if not os.path.exists(DB):
    st.info(f"No database yet at {DB}"); st.stop()

conn = dbm.connect(DB)            # also runs safe migration (with backup) if needed
try:
    Prices(PRICES, conn)          # picks up prices.yaml edits: new version + recompute
except Exception as e:
    st.warning(f"Could not load prices file: {type(e).__name__}")

df = pd.read_sql("SELECT * FROM requests ORDER BY id DESC", conn)
pv = pd.read_sql("SELECT id, loaded_at, yaml_sha, content FROM price_versions ORDER BY id DESC", conn)
tags = pd.read_sql("SELECT request_id, key, value FROM tags", conn)
if df.empty:
    st.info("No requests recorded yet."); st.stop()

for col in ["total_ms","ttft_ms","upstream_ms","overhead_ms","cost","prompt_tokens","completion_tokens"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["ts"] = pd.to_datetime(df.ts, utc=True, format="ISO8601")
df["is_error"] = (df.status_code >= 400) | df.error.notna()
df["tag"] = df.tag.fillna("(none)"); df["project"] = df.project.fillna("(none)")

with st.sidebar:
    models = st.multiselect("Models", sorted(df.model.dropna().unique()))
    projects = st.multiselect("Projects", sorted(df.project.unique()))
    if models: df = df[df.model.isin(models)]
    if projects: df = df[df.project.isin(projects)]

c = st.columns(5)
c[0].metric("Calls", len(df)); c[1].metric("Error rate", f"{df.is_error.mean():.1%}")
c[2].metric("Tokens", int(df.prompt_tokens.fillna(0).sum() + df.completion_tokens.fillna(0).sum()))
cost_total = df.cost.sum(min_count=1)
c[3].metric("List-price equiv. cost (USD)", "unpriced" if pd.isna(cost_total) else f"{cost_total:.6f}")
c[4].metric("Proxy overhead p50", "n/a" if df.overhead_ms.isna().all() else f"{df.overhead_ms.median():.2f} ms")
st.caption("Cost = public list-price equivalent from config/prices.yaml. 9Router/Antigravity routes may be free or subscription-billed, so this is not your actual bill.")
n_est = int((df.usage_source == "estimated").sum())
if n_est:
    st.warning(f"{n_est} row(s) use ESTIMATED tokens (upstream sent no usage; ~4 chars/token heuristic).")

def breakdown(by):
    g = df.groupby(by, dropna=False)
    return pd.DataFrame({
        "calls": g.size(),
        "prompt_tokens": g.prompt_tokens.sum(min_count=1),
        "completion_tokens": g.completion_tokens.sum(min_count=1),
        "estimated_rows": g.usage_source.apply(lambda s: int((s == "estimated").sum())),
        "cost": g.cost.sum(min_count=1),
        "unpriced_calls": g.cost.apply(lambda s: int(s.isna().sum())),
        "p50_total_ms": g.total_ms.quantile(0.5), "p95_total_ms": g.total_ms.quantile(0.95),
        "p50_ttft_ms": g.ttft_ms.quantile(0.5), "p95_ttft_ms": g.ttft_ms.quantile(0.95),
        "p50_overhead_ms": g.overhead_ms.quantile(0.5), "p95_overhead_ms": g.overhead_ms.quantile(0.95),
        "error_rate": g.is_error.mean(),
    }).reset_index()

t1, t2, t3, t4, t5, t6 = st.tabs(["By model", "By client tag", "By project", "Charts", "Key/value tags", "Price versions"])
with t1: st.dataframe(breakdown("model"), width="stretch")
with t2: st.dataframe(breakdown("tag"), width="stretch")
with t3: st.dataframe(breakdown("project"), width="stretch")
with t4:
    freq = st.selectbox("Bucket", ["1min", "15min", "1h", "1D"], index=2)
    ts = df.set_index("ts")
    st.caption("Calls over time"); st.bar_chart(ts.groupby([pd.Grouper(freq=freq), "model"]).size().unstack(fill_value=0))
    st.caption("Cost over time (priced rows only)")
    cost_ts = ts.groupby([pd.Grouper(freq=freq), "model"]).cost.sum(min_count=1).unstack()
    if cost_ts.notna().any().any(): st.line_chart(cost_ts)
    else: st.info("No priced rows yet: fill config/prices.yaml to see cost.")
    st.caption("Latency by model (p50/p95 ms)")
    lat = df.groupby("model").agg(p50_total=("total_ms", "median"), p95_total=("total_ms", lambda s: s.quantile(.95)),
                                  p50_ttft=("ttft_ms", "median"), p50_overhead=("overhead_ms", "median"))
    st.bar_chart(lat)
with t5:
    if tags.empty: st.info("No key/value tags yet (send header X-TokenMeter-Tags: k=v; k2=v2).")
    else:
        m = tags.merge(df[["id", "cost", "total_ms", "prompt_tokens", "completion_tokens"]], left_on="request_id", right_on="id")
        st.dataframe(m.groupby(["key", "value"], dropna=False).agg(calls=("id", "count"), cost=("cost", lambda s: s.sum(min_count=1)),
                     tokens=("completion_tokens", "sum"), p50_total_ms=("total_ms", "median")).reset_index(), width="stretch")
with t6:
    st.caption("Every distinct prices.yaml seen. On change, all rows are recomputed with the newest version.")
    st.dataframe(pv[["id", "loaded_at", "yaml_sha"]], width="stretch")
    if not pv.empty:
        pick = st.selectbox("Show content of version", pv.id)
        st.code(pv.set_index("id").content[pick], language="yaml")

st.subheader("Recent requests")
st.dataframe(df.drop(columns=["is_error"]).head(200), width="stretch")
st.download_button("Download CSV", df.drop(columns=["is_error"]).to_csv(index=False), "tokenmeter_requests.csv")
