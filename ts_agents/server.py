"""
server.py  –  TemporalMind FastAPI backend
==========================================
Run:  python server.py          (serves on http://localhost:8000)
Docs: http://localhost:8000/docs

Endpoints
---------
POST /api/upload            – ingest file, detect schema, return column info
POST /api/confirm-schema    – accept user column mapping
POST /api/analyze           – run full decomposition / outlier / intermittency per series
POST /api/interval-advice   – get interval recommendations
POST /api/accumulate        – resample to chosen interval
POST /api/hierarchy         – build hierarchy and aggregate
POST /api/missing-values    – impute missing values
POST /api/outliers          – detect & treat outliers
POST /api/prepare           – full data-prep pipeline → forecast-ready CSV/Excel
GET  /api/download/{token}  – stream the prepared file
GET  /                      – serve the SPA (index.html)
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# ── agent imports ─────────────────────────────────────────────────────────────
from agents.ingestion_agent import IngestionAgent
from agents.interval_advisor_agent import IntervalAdvisorAgent
from agents.accumulation_agent import AccumulationAgent
from agents.hierarchy_aggregation_agent import HierarchyAggregationAgent
from agents.decomposition_agent import DecompositionAgent
from agents.outlier_detection_agent import OutlierDetectionAgent
from agents.missing_values_agent import MissingValuesAgent
from agents.intermittency_agent import IntermittencyAgent
from agents.data_preparation_agent import DataPreparationAgent
from agents.multi_variable_agent import (
    MultiVariableAgent, suggest_roles, detect_event_columns,
    ROLE_DEPENDENT, ROLE_INDEPENDENT, ROLE_EVENT, ROLE_HIERARCHY,
    ROLE_TIMESTAMP, ROLE_IGNORE,
)
from core.context_store import ContextStore

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("server")

# ── in-memory session store (keyed by upload token) ─────────────────────────
_sessions: Dict[str, Dict[str, Any]] = {}
_downloads: Dict[str, bytes] = {}   # token → file bytes

# ── lazy FastAPI import ───────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import (HTMLResponse, JSONResponse,
                                    Response, StreamingResponse)
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    log.warning("FastAPI not installed – run: pip install fastapi uvicorn python-multipart")


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_records(df: pd.DataFrame, max_rows: int = 200) -> List[dict]:
    """Convert a DataFrame to JSON-serialisable records."""
    sub = df.head(max_rows).copy()
    # make index a column
    sub = sub.reset_index()
    # convert everything to native Python types
    return json.loads(sub.to_json(orient="records", date_format="iso"))


def _series_stats(s: pd.Series) -> dict:
    s_clean = s.dropna()
    return {
        "n": int(len(s)),
        "n_missing": int(s.isna().sum()),
        "mean": round(float(s_clean.mean()), 4) if len(s_clean) else None,
        "std":  round(float(s_clean.std()),  4) if len(s_clean) > 1 else None,
        "min":  round(float(s_clean.min()),  4) if len(s_clean) else None,
        "max":  round(float(s_clean.max()),  4) if len(s_clean) else None,
        "pct_zero": round(float((s_clean == 0).mean() * 100), 2) if len(s_clean) else 0,
    }


def _safe(val):
    """Recursively convert numpy types to Python natives."""
    if isinstance(val, dict):
        return {k: _safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_safe(v) for v in val]
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, pd.Series):
        return val.tolist()
    if isinstance(val, pd.DataFrame):
        return val.to_dict(orient="records")
    if isinstance(val, (pd.Timestamp, pd.Timedelta)):
        return str(val)
    if val is None or val != val:   # NaN check
        return None
    return val


# ─────────────────────────────────────────────────────────────────────────────
# MODEL RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────

def _upgrade_model_for_exog(base_rec: str, has_exog: bool) -> str:
    """Upgrade model suggestion when exogenous variables are present."""
    if not has_exog:
        return base_rec
    mapping = {
        "ETS / ARIMA":          "ARIMAX / LightGBM (with exog)",
        "ARIMA(p,d,q)":         "ARIMAX (with exog)",
        "ARIMA(p,1,q)":         "ARIMAX(p,1,q) + exog",
        "Holt-Winters / SARIMA":"SARIMAX (with exog)",
        "ETS(A,N,A) / SARIMA":  "SARIMAX (with exog)",
        "Holt / ARIMA(p,1,0)":  "ARIMAX + exog",
        "Croston / SBA":        "Croston + event regressors",
        "TSB / Zero-Inflated":  "TSB + event regressors",
    }
    for k, v in mapping.items():
        if k in base_rec:
            return v
    return base_rec + " + exog"


def _recommend_model(
    classification: str,
    trend_strong: bool,
    seasonal_strong: bool,
    d_order: int,
) -> str:
    if classification == "Lumpy":
        return "TSB / Zero-Inflated"
    if classification == "Intermittent":
        return "Croston / SBA"
    if classification == "Erratic":
        return "ETS(M,N,N) / TBATS"
    # Smooth series
    if d_order >= 2:
        return "ARIMA(p,2,q)"
    if trend_strong and seasonal_strong:
        return "Holt-Winters / SARIMA"
    if trend_strong:
        return "Holt / ARIMA(p,1,0)"
    if seasonal_strong:
        return "ETS(A,N,A) / SARIMA"
    if d_order == 1:
        return "ARIMA(p,1,q)"
    return "ETS(A,N,N) / Naive"


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

if _HAS_FASTAPI:
    app = FastAPI(title="TemporalMind API", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    # ── Upload ────────────────────────────────────────────────────────────────

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        try:
            raw = await file.read()
            fname = file.filename or "upload"
            suffix = Path(fname).suffix.lower()

            if suffix in (".xlsx", ".xls"):
                df_raw = pd.read_excel(io.BytesIO(raw))
            elif suffix == ".csv":
                df_raw = pd.read_csv(io.BytesIO(raw))
            else:
                raise HTTPException(400, "Only CSV and Excel files are supported.")

            token = str(uuid.uuid4())[:8]
            _sessions[token] = {"raw_df": df_raw, "filename": fname}

            # heuristic column classification
            cols = list(df_raw.columns)
            dtypes = {c: str(df_raw[c].dtype) for c in cols}
            nunique = {c: int(df_raw[c].nunique()) for c in cols}
            n_rows = len(df_raw)

            ts_candidates, val_candidates, hier_candidates = [], [], []
            for c in cols:
                col_lower = c.lower()
                # try datetime parse
                try:
                    pd.to_datetime(df_raw[c].dropna().iloc[:10])
                    ts_candidates.append(c)
                    continue
                except Exception:
                    pass
                if any(h in col_lower for h in ["date","time","ts","period","month","week","year"]):
                    ts_candidates.append(c)
                elif pd.api.types.is_numeric_dtype(df_raw[c]):
                    val_candidates.append(c)
                elif df_raw[c].dtype == object and nunique[c] < n_rows * 0.3:
                    hier_candidates.append(c)

            # auto-decide: if hier_candidates exist it's likely a hierarchy
            is_hierarchy = len(hier_candidates) > 0
            needs_confirm = (len(ts_candidates) != 1 or
                             len(val_candidates) == 0 or
                             (is_hierarchy and len(hier_candidates) > 4))

            # ── detect event candidates (binary 0/1 columns) ─────────────────
            event_candidates = detect_event_columns(df_raw, val_candidates)
            # suggested roles for ALL columns
            suggested_roles = {}
            for c in cols:
                if c in ts_candidates:
                    suggested_roles[c] = ROLE_TIMESTAMP
                elif c in hier_candidates:
                    suggested_roles[c] = ROLE_HIERARCHY
                elif c in event_candidates:
                    suggested_roles[c] = ROLE_EVENT
                elif c in val_candidates:
                    # first non-event numeric = dependent, rest = independent
                    non_event_vals = [v for v in val_candidates if v not in event_candidates]
                    idx = non_event_vals.index(c) if c in non_event_vals else -1
                    suggested_roles[c] = ROLE_DEPENDENT if idx == 0 else ROLE_INDEPENDENT
                else:
                    suggested_roles[c] = ROLE_IGNORE

            return {
                "token": token,
                "filename": fname,
                "n_rows": n_rows,
                "columns": cols,
                "dtypes": dtypes,
                "nunique": nunique,
                "ts_candidates": ts_candidates,
                "value_candidates": val_candidates,
                "hierarchy_candidates": hier_candidates,
                "event_candidates": event_candidates,
                "suggested_roles": suggested_roles,
                "is_hierarchy": is_hierarchy,
                "needs_confirm": needs_confirm,
                "preview": _df_to_records(df_raw, 8),
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, str(e))

    # ── Confirm schema ────────────────────────────────────────────────────────

    class SchemaConfirm(BaseModel):
        token: str
        timestamp_col: str
        value_cols: List[str]
        hierarchy_cols: List[str] = []
        # extended variable role mapping (col -> role string)
        variable_roles: Dict[str, str] = {}

    @app.post("/api/confirm-schema")
    async def confirm_schema(body: SchemaConfirm):
        sess = _sessions.get(body.token)
        if not sess:
            raise HTTPException(404, "Session not found")
        df_raw = sess["raw_df"]

        # Derive value_cols from roles if provided
        if body.variable_roles:
            all_val = [c for c, r in body.variable_roles.items()
                       if r in (ROLE_DEPENDENT, ROLE_INDEPENDENT, ROLE_EVENT)]
            if all_val:
                body = body.model_copy(update={"value_cols": all_val})

        agent = IngestionAgent()
        result = agent.execute(
            source=df_raw,
            timestamp_col=body.timestamp_col,
            value_cols=body.value_cols,
            hierarchy_cols=body.hierarchy_cols,
        )
        if not result.ok:
            raise HTTPException(400, result.errors[0] if result.errors else "Ingestion failed")

        df: pd.DataFrame = result.data
        roles = body.variable_roles or {}

        # Classify by role
        dep_cols   = [c for c, r in roles.items() if r == ROLE_DEPENDENT]
        indep_cols = [c for c, r in roles.items() if r == ROLE_INDEPENDENT]
        event_cols = [c for c, r in roles.items() if r == ROLE_EVENT]

        # Fallback: if no roles given, auto-suggest
        if not roles:
            roles = suggest_roles(df, ts_col=body.timestamp_col)
            dep_cols   = [c for c, r in roles.items() if r == ROLE_DEPENDENT]
            indep_cols = [c for c, r in roles.items() if r == ROLE_INDEPENDENT]
            event_cols = [c for c, r in roles.items() if r == ROLE_EVENT]

        sess["df"] = df
        sess["schema"] = result.metadata
        sess["value_cols"] = body.value_cols
        sess["hierarchy_cols"] = body.hierarchy_cols
        sess["detected_freq"] = result.metadata["detected_freq"]
        sess["variable_roles"] = roles
        sess["dependent_cols"] = dep_cols or (body.value_cols[:1] if body.value_cols else [])
        sess["independent_cols"] = indep_cols
        sess["event_cols"] = event_cols

        return {
            "ok": True,
            "schema": _safe(result.metadata["schema"]),
            "warnings": result.warnings,
            "preview": _df_to_records(df.reset_index(), 10),
            "variable_roles": roles,
            "dependent_cols": dep_cols,
            "independent_cols": indep_cols,
            "event_cols": event_cols,
        }

    # ── Interval advice ───────────────────────────────────────────────────────

    class IntervalReq(BaseModel):
        token: str
        top_n: int = 3

    @app.post("/api/interval-advice")
    async def interval_advice(body: IntervalReq):
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404, "No ingested data for this token")
        df: pd.DataFrame = sess["df"]
        val_cols = sess.get("value_cols", [])
        col = val_cols[0] if val_cols else df.select_dtypes("number").columns[0]

        series = df[col].dropna()
        agent = IntervalAdvisorAgent()
        result = agent.execute(series=series, native_freq=sess.get("detected_freq"), top_n=body.top_n)
        if not result.ok:
            raise HTTPException(500, result.errors[0])

        recs = result.data
        return {
            "recommendations": [
                {
                    "alias": r["alias"],
                    "freq": r["freq"],
                    "score": r["score"],
                    "info_loss_pct": r["info_loss_pct"],
                    "rationale": r["rationale"],
                }
                for r in recs
            ],
            "best_freq": result.metadata["best_interval"],
            "best_alias": result.metadata["best_alias"],
            "summary": result.metadata["summary"],
            "native_step_days": result.metadata["native_step_days"],
        }

    # ── Accumulate ────────────────────────────────────────────────────────────

    class AccumReq(BaseModel):
        token: str
        target_freq: str
        method: str = "auto"
        quantity_type: str = "flow"

    @app.post("/api/accumulate")
    async def accumulate(body: AccumReq):
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404)
        df = sess["df"]
        val_cols = sess.get("value_cols")

        agent = AccumulationAgent()
        result = agent.execute(
            df=df, target_freq=body.target_freq,
            method=body.method, quantity_type=body.quantity_type,
            compare_freqs=None, value_cols=val_cols,
        )
        if not result.ok:
            raise HTTPException(500, result.errors[0])

        sess["accumulated_df"] = result.data
        sess["target_freq"] = body.target_freq

        comparison = {}
        for freq, stats in result.metadata.get("comparison", {}).items():
            for col, s in stats.items():
                comparison.setdefault(col, {})[freq] = _safe(s)

        return {
            "n_input": result.metadata["n_input"],
            "n_output": result.metadata["n_output"],
            "compression_ratio": result.metadata["compression_ratio"],
            "information_retention": _safe(result.metadata["information_retention"]),
            "comparison": comparison,
            "preview": _df_to_records(result.data.reset_index(), 12),
            "report": result.metadata.get("report", ""),
        }

    # ── Hierarchy ─────────────────────────────────────────────────────────────

    class HierarchyReq(BaseModel):
        token: str
        method: str = "sum"

    @app.post("/api/hierarchy")
    async def hierarchy(body: HierarchyReq):
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404)
        df = sess["df"]
        hier_cols = sess.get("hierarchy_cols", [])
        val_cols  = sess.get("value_cols", [])
        if not hier_cols:
            raise HTTPException(400, "No hierarchy columns in this session")

        agent = HierarchyAggregationAgent()
        result = agent.execute(df=df, hierarchy_cols=hier_cols,
                               value_cols=val_cols, method=body.method)
        if not result.ok:
            raise HTTPException(500, result.errors[0])

        sess["hierarchy_result"] = result

        level_info = {}
        for lk, series_dict_or_df in result.data.items():
            prefix = lk.split("__")[0]
            level_info.setdefault(prefix, []).append(lk)

        return {
            "n_total_series": result.metadata["n_total_series"],
            "level_meta": _safe(result.metadata["level_meta"]),
            "coherence": _safe(result.metadata["coherence"]),
            "level_effects": _safe(result.metadata["level_effects"]),
            "all_series_keys": list(result.data.keys()),
            "level_groups": {k: v for k, v in level_info.items()},
            "report": result.metadata.get("report", ""),
        }



    # ── Variable roles endpoint ───────────────────────────────────────────────

    class VariableRolesReq(BaseModel):
        token: str
        roles: Dict[str, str]   # col -> role

    @app.post("/api/variable-roles")
    async def set_variable_roles(body: VariableRolesReq):
        """Update variable roles in the session and return stats per variable."""
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404)
        df: pd.DataFrame = sess["df"]

        roles = body.roles
        dep_cols   = [c for c, r in roles.items() if r == ROLE_DEPENDENT   and c in df.columns]
        indep_cols = [c for c, r in roles.items() if r == ROLE_INDEPENDENT and c in df.columns]
        event_cols = [c for c, r in roles.items() if r == ROLE_EVENT       and c in df.columns]

        sess["variable_roles"]  = roles
        sess["dependent_cols"]  = dep_cols
        sess["independent_cols"]= indep_cols
        sess["event_cols"]      = event_cols
        # Also keep value_cols as union of all numeric roles
        sess["value_cols"] = dep_cols + indep_cols + event_cols

        # Per-variable quick stats
        stats = {}
        for col in (dep_cols + indep_cols + event_cols):
            if col in df.columns:
                s = df[col].dropna()
                stats[col] = {
                    "role": roles[col],
                    "n": int(len(s)),
                    "n_missing": int(df[col].isna().sum()),
                    "mean": round(float(s.mean()), 4) if len(s) else None,
                    "std":  round(float(s.std()),  4) if len(s) > 1 else None,
                    "min":  round(float(s.min()),  4) if len(s) else None,
                    "max":  round(float(s.max()),  4) if len(s) else None,
                    "pct_zero": round(float((s == 0).mean() * 100), 2) if len(s) else 0,
                }

        return {
            "ok": True,
            "dependent_cols": dep_cols,
            "independent_cols": indep_cols,
            "event_cols": event_cols,
            "stats": _safe(stats),
        }

    # ── Cross-correlation analysis ────────────────────────────────────────────

    class CrossCorrReq(BaseModel):
        token: str
        dependent_col: Optional[str] = None
        max_lags: int = 20
        event_window: int = 5

    @app.post("/api/cross-correlation")
    async def cross_correlation(body: CrossCorrReq):
        """Run full multi-variable analysis: CCF, event impact, Granger proxy."""
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404)

        df: pd.DataFrame = sess.get("accumulated_df", sess["df"])
        roles = sess.get("variable_roles", {})

        # Build roles dict from session
        dep_cols   = sess.get("dependent_cols", [])
        indep_cols = sess.get("independent_cols", [])
        event_cols = sess.get("event_cols", [])

        if not dep_cols:
            # fallback: first value col
            vc = sess.get("value_cols", [])
            if vc:
                dep_cols = [vc[0]]
                indep_cols = vc[1:]
            else:
                raise HTTPException(400, "No dependent column found in session")

        dep_col = body.dependent_col or dep_cols[0]

        # Rebuild roles for agent
        agent_roles = {}
        for c in dep_cols:   agent_roles[c] = ROLE_DEPENDENT
        for c in indep_cols: agent_roles[c] = ROLE_INDEPENDENT
        for c in event_cols: agent_roles[c] = ROLE_EVENT

        # Use accumulated df if available, only with relevant cols
        all_cols = [c for c in (dep_cols + indep_cols + event_cols) if c in df.columns]
        if not all_cols:
            raise HTTPException(400, "No matching columns in current dataframe")

        agent = MultiVariableAgent()
        result = agent.execute(
            df=df[all_cols].copy(),
            roles=agent_roles,
            dependent_col=dep_col,
            max_ccf_lags=body.max_lags,
            event_window=body.event_window,
        )
        if not result.ok:
            raise HTTPException(500, result.errors[0] if result.errors else "Analysis failed")

        # Build correlation heatmap data
        corr = result.data["corr_matrix"]
        var_stats = result.data["var_stats"]
        ccf_data = {}
        for col, res in result.data["ccf_results"].items():
            ccf_data[col] = {
                "lags": res["lags"],
                "ccf": res["ccf"],
                "best_lag": res["best_lag"],
                "max_ccf": res["max_ccf"],
                "sig_threshold": res["sig_threshold"],
                "significant": res["significant"],
                "interpretation": res["interpretation"],
            }

        return {
            "dependent_col": dep_col,
            "independent_cols": indep_cols,
            "event_cols": event_cols,
            "var_stats": _safe(var_stats),
            "corr_matrix": _safe(corr),
            "corr_cols": all_cols,
            "ccf_data": _safe(ccf_data),
            "event_impacts": _safe(result.data["event_impacts"]),
            "granger_results": _safe(result.data["granger_results"]),
            "feature_recommendations": _safe(result.data["feature_recommendations"]),
            "report": result.metadata.get("report", ""),
            "warnings": result.warnings,
        }

    # ── Hierarchy tree structure ──────────────────────────────────────────────

    class HierarchyTreeReq(BaseModel):
        token: str

    @app.post("/api/hierarchy-tree")
    async def hierarchy_tree(body: HierarchyTreeReq):
        """
        Returns hierarchy tree with unique values per level for cascading dropdowns.
        """
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404, "No ingested data")
        df: pd.DataFrame = sess["df"]
        hier_cols = sess.get("hierarchy_cols", [])
        if not hier_cols:
            raise HTTPException(400, "No hierarchy columns in this session")

        level_values: Dict[str, List] = {}
        for col in hier_cols:
            vals = sorted(df[col].dropna().unique().tolist())
            level_values[col] = [str(v) for v in vals]

        def build_tree(sub_df: pd.DataFrame, remaining: List[str]) -> Any:
            if not remaining:
                return []
            col = remaining[0]
            rest = remaining[1:]
            result = {}
            for val in sorted(sub_df[col].dropna().unique()):
                child_df = sub_df[sub_df[col] == val]
                result[str(val)] = build_tree(child_df, rest)
            return result

        tree = build_tree(df, hier_cols)
        total_leaves = len(df.groupby(hier_cols))

        return {
            "levels": hier_cols,
            "tree": tree,
            "level_values": level_values,
            "total_leaves": total_leaves,
        }

    # ── Analyze a specific hierarchy node ─────────────────────────────────────

    class AnalyzeNodeReq(BaseModel):
        token: str
        node_path: Dict[str, str]
        value_col: Optional[str] = None
        period: Optional[int] = None
        agg_method: str = "sum"

    @app.post("/api/analyze-node")
    async def analyze_node(body: AnalyzeNodeReq):
        """
        Filter df to node_path, aggregate remaining hier levels, run full analysis.
        """
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404)

        df: pd.DataFrame = sess["df"]
        hier_cols = sess.get("hierarchy_cols", [])
        val_cols = sess.get("value_cols", [])
        val_col = body.value_col or (val_cols[0] if val_cols else None)
        if not val_col:
            raise HTTPException(400, "No value column")

        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in body.node_path.items():
            if col in df.columns:
                mask &= (df[col].astype(str) == str(val))

        filtered = df[mask]
        if len(filtered) == 0:
            raise HTTPException(404, f"No data found for path: {body.node_path}")

        path_depth = len(body.node_path)
        total_depth = len(hier_cols)
        is_leaf = (path_depth == total_depth)
        node_label = " › ".join(f"{k}={v}" for k, v in body.node_path.items())

        remaining_hier = [c for c in hier_cols if c not in body.node_path]
        agg_fn = body.agg_method

        if remaining_hier:
            series = filtered.groupby(level=0)[val_col].agg(agg_fn)
        else:
            series = filtered[val_col]

        series = series.sort_index().dropna()
        series.name = val_col
        if len(series) < 4:
            raise HTTPException(400, f"Too few data points ({len(series)}) for: {node_label}")

        ctx = ContextStore()
        mv_res = MissingValuesAgent(ctx).execute(series=series)
        clean = mv_res.data["imputed"] if mv_res.ok else series

        decomp_res = DecompositionAgent(ctx).execute(series=clean, period=body.period)
        residual = decomp_res.data.get("residual") if decomp_res.ok else None
        out_res = OutlierDetectionAgent(ctx).execute(
            series=clean, residual=residual, methods=["iqr", "zscore", "isof"]
        )
        interm_res = IntermittencyAgent(ctx).execute(series=clean)

        ts_labels = [str(d)[:10] for d in clean.index]
        chart_data: Dict[str, Any] = {
            "labels": ts_labels,
            "original": [round(float(v), 4) if not np.isnan(v) else None for v in clean.values],
        }
        if decomp_res.ok:
            for comp in ("trend", "seasonal", "cycle", "residual"):
                arr = decomp_res.data.get(comp)
                if arr is not None:
                    chart_data[comp] = [
                        round(float(v), 4) if not np.isnan(v) else None for v in arr.values
                    ]

        outlier_timestamps = []
        if out_res.ok and len(out_res.data["outlier_table"]) > 0:
            outlier_timestamps = [
                str(r["timestamp"])[:10] for _, r in out_res.data["outlier_table"].iterrows()
            ]

        Ft = decomp_res.metadata.get("trend_strength_Ft", 0) if decomp_res.ok else 0
        Fs = decomp_res.metadata.get("seasonal_strength_Fs", 0) if decomp_res.ok else 0
        d  = decomp_res.metadata.get("differencing_order", 0) if decomp_res.ok else 0
        interm_cls = interm_res.metadata.get("summary", {}).get("classification", "Smooth") if interm_res.ok else "Smooth"
        model_rec = _recommend_model(interm_cls, Ft > 0.5, Fs > 0.5, d)

        n_agg = 1
        if remaining_hier:
            try:
                n_agg = int(len(filtered.groupby(remaining_hier)))
            except Exception:
                n_agg = 1

        summary = {
            "node_label": node_label,
            "node_path": body.node_path,
            "is_leaf": is_leaf,
            "path_depth": path_depth,
            "n_rows_filtered": int(len(filtered)),
            "n_series_aggregated": n_agg,
            "agg_method": agg_fn if remaining_hier else "none (leaf)",
            "series_stats": _safe(_series_stats(clean)),
            "decomp": _safe({
                "trend_strength_Ft": decomp_res.metadata.get("trend_strength_Ft"),
                "seasonal_strength_Fs": decomp_res.metadata.get("seasonal_strength_Fs"),
                "period_used": decomp_res.metadata.get("period_used"),
                "differencing_order": decomp_res.metadata.get("differencing_order"),
                "stationarity": decomp_res.metadata.get("stationarity"),
                "trend_stats": decomp_res.metadata.get("trend_stats"),
                "interpretation": decomp_res.metadata.get("interpretation"),
                "acf_significant_lags": decomp_res.metadata.get("acf_significant_lags"),
            }) if decomp_res.ok else {},
            "outliers": _safe(out_res.metadata.get("summary", {})) if out_res.ok else {},
            "intermittency": _safe(interm_res.metadata.get("summary", {})) if interm_res.ok else {},
            "missing_values": _safe(mv_res.metadata.get("completeness", {})) if mv_res.ok else {},
            "model_recommendation": model_rec,
            "intermittency_class": interm_cls,
        }

        return {
            "chart_data": chart_data,
            "outlier_timestamps": outlier_timestamps,
            "summary": summary,
            "warnings": (
                (decomp_res.warnings if decomp_res.ok else []) +
                (out_res.warnings if out_res.ok else []) +
                (interm_res.warnings if interm_res.ok else [])
            ),
        }

    # ── Analyze ───────────────────────────────────────────────────────────────

    class AnalyzeReq(BaseModel):
        token: str
        series_key: Optional[str] = None   # None → first value col of accumulated df
        period: Optional[int] = None

    @app.post("/api/analyze")
    async def analyze(body: AnalyzeReq):
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404)

        # pick series
        work_df = sess.get("accumulated_df", sess["df"])
        val_cols = sess.get("value_cols", work_df.select_dtypes("number").columns.tolist())

        if body.series_key and body.series_key in work_df.columns:
            series = work_df[body.series_key].dropna()
        else:
            col = val_cols[0] if val_cols else work_df.columns[0]
            series = work_df[col].dropna()

        ctx = ContextStore()

        # missing values
        mv_res = MissingValuesAgent(ctx).execute(series=series)
        clean = mv_res.data["imputed"] if mv_res.ok else series

        # decomposition
        decomp_res = DecompositionAgent(ctx).execute(series=clean, period=body.period)

        # outlier
        residual = decomp_res.data.get("residual") if decomp_res.ok else None
        out_res = OutlierDetectionAgent(ctx).execute(
            series=clean, residual=residual,
            methods=["iqr", "zscore", "isof"]
        )

        # intermittency
        interm_res = IntermittencyAgent(ctx).execute(series=clean)

        # build time series chart data (original + components)
        chart_data: Dict[str, Any] = {}
        ts_labels = [str(d)[:10] for d in clean.index]
        chart_data["labels"] = ts_labels
        chart_data["original"] = [round(float(v), 4) if not np.isnan(v) else None
                                   for v in clean.values]
        if decomp_res.ok:
            for comp in ("trend", "seasonal", "cycle", "residual"):
                arr = decomp_res.data[comp]
                if arr is not None:
                    chart_data[comp] = [round(float(v), 4) if not np.isnan(v) else None
                                        for v in arr.values]

        outlier_timestamps = []
        if out_res.ok and len(out_res.data["outlier_table"]) > 0:
            ot = out_res.data["outlier_table"]
            outlier_timestamps = [str(r["timestamp"])[:10] for _, r in ot.iterrows()]

        # stats summary
        summary = {
            "series_stats": _safe(_series_stats(clean)),
            "decomp": _safe({
                "trend_strength_Ft": decomp_res.metadata.get("trend_strength_Ft"),
                "seasonal_strength_Fs": decomp_res.metadata.get("seasonal_strength_Fs"),
                "period_used": decomp_res.metadata.get("period_used"),
                "differencing_order": decomp_res.metadata.get("differencing_order"),
                "stationarity": decomp_res.metadata.get("stationarity"),
                "trend_stats": decomp_res.metadata.get("trend_stats"),
                "interpretation": decomp_res.metadata.get("interpretation"),
                "acf_significant_lags": decomp_res.metadata.get("acf_significant_lags"),
            }) if decomp_res.ok else {},
            "outliers": _safe(out_res.metadata.get("summary", {})) if out_res.ok else {},
            "intermittency": _safe(interm_res.metadata.get("summary", {})) if interm_res.ok else {},
            "missing_values": _safe(mv_res.metadata.get("completeness", {})) if mv_res.ok else {},
        }

        # store clean series back
        col = series.name or val_cols[0]
        sess["clean_series"] = {col: clean}
        sess["decomp_result"] = decomp_res

        return {
            "chart_data": chart_data,
            "outlier_timestamps": outlier_timestamps,
            "summary": summary,
            "warnings": (decomp_res.warnings + out_res.warnings + interm_res.warnings),
        }

    # ── Missing values treatment ──────────────────────────────────────────────

    class MissingReq(BaseModel):
        token: str
        method: str = "auto"
        period: int = 7
        zero_as_missing: bool = False

    @app.post("/api/missing-values")
    async def missing_values(body: MissingReq):
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404)
        work_df = sess.get("accumulated_df", sess["df"])
        val_cols = sess.get("value_cols", work_df.select_dtypes("number").columns.tolist())

        results = {}
        imputed_cols = {}
        for col in val_cols:
            series = work_df[col]
            res = MissingValuesAgent().execute(
                series=series, method=body.method,
                period=body.period, zero_as_missing=body.zero_as_missing,
            )
            if res.ok:
                imputed_cols[col] = res.data["imputed"]
                results[col] = _safe(res.metadata["completeness"])

        # update session df
        imp_df = work_df.copy()
        for col, s in imputed_cols.items():
            imp_df[col] = s
        sess["imputed_df"] = imp_df

        return {"results": results, "preview": _df_to_records(imp_df.reset_index(), 10)}

    # ── Outlier treatment ─────────────────────────────────────────────────────

    class OutlierReq(BaseModel):
        token: str
        methods: List[str] = ["iqr", "zscore", "isof"]
        treatment: str = "cap"   # cap | remove | keep

    @app.post("/api/outliers")
    async def outliers(body: OutlierReq):
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404)
        work_df = sess.get("imputed_df", sess.get("accumulated_df", sess["df"]))
        val_cols = sess.get("value_cols", work_df.select_dtypes("number").columns.tolist())

        summary_all = {}
        treated_cols = {}
        for col in val_cols:
            series = work_df[col].dropna()
            res = OutlierDetectionAgent().execute(
                series=series, methods=body.methods, contamination=0.05
            )
            if not res.ok:
                continue
            summary_all[col] = _safe(res.metadata["summary"])

            treated = series.copy()
            if body.treatment == "cap" and res.ok:
                fences = res.metadata["summary"]["fences"]
                lo, hi = fences["lower_1.5iqr"], fences["upper_1.5iqr"]
                treated = treated.clip(lower=lo, upper=hi)
            elif body.treatment == "remove" and res.ok:
                is_out = res.data["is_outlier"]
                treated[is_out] = np.nan
            treated_cols[col] = treated

        treated_df = work_df.copy()
        for col, s in treated_cols.items():
            treated_df[col] = s
        sess["treated_df"] = treated_df

        return {
            "summary": summary_all,
            "treatment_applied": body.treatment,
            "preview": _df_to_records(treated_df.reset_index(), 10),
        }

    # ── Prepare data ──────────────────────────────────────────────────────────

    class PrepareReq(BaseModel):
        token: str
        transform: str = "auto"
        scale_method: str = "minmax"
        rolling_windows: List[int] = [7, 14, 28]
        add_calendar: bool = True
        train_ratio: float = 0.70
        val_ratio: float = 0.15
        horizon: int = 1
        output_format: str = "csv"   # csv | excel
        hier_level: Optional[str] = None


    def _first_session_frame(sess: Dict[str, Any]) -> Optional[pd.DataFrame]:
        for key in ("treated_df", "imputed_df", "accumulated_df", "df"):
            if key in sess:
                candidate = sess[key]
                if isinstance(candidate, pd.DataFrame):
                    return candidate
        return None
    @app.post("/api/prepare")
    async def prepare(body: PrepareReq):
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404)
        work_df = _first_session_frame(sess)
        # work_df = (sess.get("treated_df") or
        #            sess.get("imputed_df") or
        #            sess.get("accumulated_df") or
        #            sess["df"])
        val_cols = sess.get("value_cols", work_df.select_dtypes("number").columns.tolist())
        hier_cols = sess.get("hierarchy_cols", [])

        dep_cols   = sess.get("dependent_cols", val_cols[:1])
        indep_cols = sess.get("independent_cols", [])
        event_cols = sess.get("event_cols", [])

        # Only prepare the dependent column(s); attach indep/event as extra features
        target_cols = dep_cols if dep_cols else val_cols[:1]
        all_prepared: List[pd.DataFrame] = []

        for col in target_cols:
            if col not in work_df.columns:
                continue
            series = work_df[col].dropna()

            # get ACF lags from earlier decomp if available
            acf_lags = None
            dr = sess.get("decomp_result")
            if dr and dr.ok:
                acf_lags = dr.metadata.get("acf_significant_lags")

            # intermittency classification
            interm_res = IntermittencyAgent().execute(series=series)
            interm_cls = interm_res.metadata.get("summary", {}).get("classification", "Smooth") if interm_res.ok else "Smooth"

            # decomp for model rec
            decomp_res = DecompositionAgent().execute(series=series)
            Ft = decomp_res.metadata.get("trend_strength_Ft", 0.0) if decomp_res.ok else 0.0
            Fs = decomp_res.metadata.get("seasonal_strength_Fs", 0.0) if decomp_res.ok else 0.0
            d  = decomp_res.metadata.get("differencing_order", 0) if decomp_res.ok else 0

            # Check if we have significant exogenous variables → upgrade model rec
            has_exog = bool(indep_cols or event_cols)
            base_rec = _recommend_model(interm_cls, Ft > 0.5, Fs > 0.5, d)
            model_rec = _upgrade_model_for_exog(base_rec, has_exog)

            # data prep (target series)
            prep_res = DataPreparationAgent().execute(
                series=series,
                target_col=col,
                transform=body.transform,
                scale_method=body.scale_method,
                rolling_windows=body.rolling_windows,
                add_calendar=body.add_calendar,
                train_ratio=body.train_ratio,
                val_ratio=body.val_ratio,
                horizon=body.horizon,
                acf_lags=acf_lags,
            )
            if not prep_res.ok:
                continue

            feat_df = prep_res.data["feature_matrix"].copy()

            # ── Attach independent variable features ──────────────────────────
            for ic in indep_cols:
                if ic not in work_df.columns:
                    continue
                indep_s = work_df[ic].reindex(feat_df.index)
                feat_df[f"indep__{ic}"] = indep_s.values
                # lagged version at lag-1
                feat_df[f"indep_lag1__{ic}"] = indep_s.shift(1).values

            # ── Attach event features ────────────────────────────────────────
            for ec in event_cols:
                if ec not in work_df.columns:
                    continue
                ev_s = work_df[ec].reindex(feat_df.index).fillna(0)
                feat_df[f"event__{ec}"] = ev_s.values
                # rolling event count over 7 periods
                feat_df[f"event_roll7__{ec}"] = ev_s.rolling(7, min_periods=1).sum().values

            feat_df["series_name"] = col
            feat_df["variable_role"] = ROLE_DEPENDENT
            feat_df["model_type_recommendation"] = model_rec
            feat_df["intermittency_class"] = interm_cls
            feat_df["trend_strength_Ft"] = round(Ft, 4)
            feat_df["seasonal_strength_Fs"] = round(Fs, 4)
            feat_df["has_exogenous"] = has_exog

            # split label
            n = len(feat_df)
            n_train = prep_res.data["X_train"].shape[0]
            n_val   = prep_res.data["X_val"].shape[0]
            split_labels = (
                ["train"] * n_train +
                ["validation"] * n_val +
                ["test"] * (n - n_train - n_val)
            )
            feat_df["split"] = split_labels[:n]
            all_prepared.append(feat_df)

        if not all_prepared:
            raise HTTPException(500, "Data preparation failed for all columns")

        final_df = pd.concat(all_prepared, axis=0)
        final_df = final_df.reset_index()

        # serialise
        token = str(uuid.uuid4())[:8]
        if body.output_format == "excel":
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                final_df.to_excel(writer, index=False, sheet_name="PreparedData")
            _downloads[token] = buf.getvalue()
            ext = "xlsx"
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            csv_bytes = final_df.to_csv(index=False).encode()
            _downloads[token] = csv_bytes
            ext = "csv"
            mime = "text/csv"

        # summary info
        series_summaries = []
        for col in target_cols:
            sub = final_df[final_df["series_name"] == col] if "series_name" in final_df.columns else final_df
            if len(sub):
                series_summaries.append({
                    "series": col,
                    "n_rows": len(sub),
                    "n_features": len(sub.columns),
                    "model_type": sub["model_type_recommendation"].iloc[0] if "model_type_recommendation" in sub.columns else "—",
                    "intermittency": sub["intermittency_class"].iloc[0] if "intermittency_class" in sub.columns else "—",
                    "Ft": float(sub["trend_strength_Ft"].iloc[0]) if "trend_strength_Ft" in sub.columns else 0,
                    "Fs": float(sub["seasonal_strength_Fs"].iloc[0]) if "seasonal_strength_Fs" in sub.columns else 0,
                    "has_exogenous": bool(sub["has_exogenous"].iloc[0]) if "has_exogenous" in sub.columns else False,
                    "n_indep": len(indep_cols),
                    "n_events": len(event_cols),
                })

        return {
            "download_token": token,
            "ext": ext,
            "mime": mime,
            "n_rows": len(final_df),
            "n_features": len(final_df.columns),
            "series_summaries": series_summaries,
            "preview": _df_to_records(final_df, 15),
            "columns": list(final_df.columns),
        }

    # ── Download ──────────────────────────────────────────────────────────────

    @app.get("/api/download/{token}")
    async def download(token: str, ext: str = "csv"):
        data = _downloads.get(token)
        if not data:
            raise HTTPException(404, "Download token expired or not found")
        mime = ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                if ext == "xlsx" else "text/csv")
        return Response(
            content=data,
            media_type=mime,
            headers={"Content-Disposition": f"attachment; filename=prepared_data.{ext}"},
        )

    # ── Serve SPA ─────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def root():
        html_path = Path(__file__).parent / "ui" / "index.html"
        if html_path.exists():
            return HTMLResponse(html_path.read_text())
        return HTMLResponse("<h1>TemporalMind – UI not found. Place index.html in ./ui/</h1>")

    @app.get("/health")
    async def health():
        return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import uvicorn
        log.info("Starting TemporalMind server on http://localhost:8000")
        uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        log.error("uvicorn not installed. Run: pip install uvicorn")
