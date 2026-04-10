"""
routes/analysis.py  –  Time-series analysis endpoints
======================================================
Routes
------
POST /api/analyze               – full decomposition / outlier / intermittency per series
POST /api/variable-roles        – update variable role mapping for a session
POST /api/cross-correlation     – multi-variable CCF, event impact, Granger proxy
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.decomposition_agent import DecompositionAgent
from agents.intermittency_agent import IntermittencyAgent
from agents.missing_values_agent import MissingValuesAgent
from agents.multi_variable_agent import (
    ROLE_DEPENDENT, ROLE_EVENT, ROLE_INDEPENDENT,
    MultiVariableAgent,
)
from agents.outlier_detection_agent import OutlierDetectionAgent
from core.context_store import ContextStore
from core.runtime import SESSIONS as _sessions
from utils.api_helper import _safe, _series_stats

router = APIRouter()


# ── Analyze ────────────────────────────────────────────────────────────────────

class AnalyzeReq(BaseModel):
    token:      str
    series_key: Optional[str] = None   # None → first value col of accumulated df
    period:     Optional[int] = None


@router.post("/api/analyze")
async def analyze(body: AnalyzeReq):
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404)

    work_df  = sess.get("accumulated_df", sess["df"])
    val_cols = sess.get("value_cols", work_df.select_dtypes("number").columns.tolist())

    if body.series_key and body.series_key in work_df.columns:
        series = work_df[body.series_key].dropna()
    else:
        col    = val_cols[0] if val_cols else work_df.columns[0]
        series = work_df[col].dropna()

    ctx = ContextStore()

    mv_res    = MissingValuesAgent(ctx).execute(series=series)
    clean     = mv_res.data["imputed"] if mv_res.ok else series

    decomp_res = DecompositionAgent(ctx).execute(series=clean, period=body.period)

    residual = decomp_res.data.get("residual") if decomp_res.ok else None
    out_res  = OutlierDetectionAgent(ctx).execute(
        series=clean, residual=residual, methods=["iqr", "zscore", "isof"]
    )

    interm_res = IntermittencyAgent(ctx).execute(series=clean)

    chart_data: Dict[str, Any] = {}
    ts_labels = [str(d)[:10] for d in clean.index]
    chart_data["labels"]   = ts_labels
    chart_data["original"] = [
        round(float(v), 4) if not np.isnan(v) else None for v in clean.values
    ]
    if decomp_res.ok:
        for comp in ("trend", "seasonal", "cycle", "residual"):
            arr = decomp_res.data[comp]
            if arr is not None:
                chart_data[comp] = [
                    round(float(v), 4) if not np.isnan(v) else None for v in arr.values
                ]

    outlier_timestamps = []
    if out_res.ok and len(out_res.data["outlier_table"]) > 0:
        ot = out_res.data["outlier_table"]
        outlier_timestamps = [str(r["timestamp"])[:10] for _, r in ot.iterrows()]

    summary = {
        "series_stats": _safe(_series_stats(clean)),
        "decomp": _safe({
            "trend_strength_Ft":    decomp_res.metadata.get("trend_strength_Ft"),
            "seasonal_strength_Fs": decomp_res.metadata.get("seasonal_strength_Fs"),
            "period_used":          decomp_res.metadata.get("period_used"),
            "differencing_order":   decomp_res.metadata.get("differencing_order"),
            "stationarity":         decomp_res.metadata.get("stationarity"),
            "trend_stats":          decomp_res.metadata.get("trend_stats"),
            "interpretation":       decomp_res.metadata.get("interpretation"),
            "acf_significant_lags": decomp_res.metadata.get("acf_significant_lags"),
        }) if decomp_res.ok else {},
        "outliers":      _safe(out_res.metadata.get("summary", {})) if out_res.ok else {},
        "intermittency": _safe(interm_res.metadata.get("summary", {})) if interm_res.ok else {},
        "missing_values":_safe(mv_res.metadata.get("completeness", {})) if mv_res.ok else {},
    }

    col = series.name or val_cols[0]
    sess["clean_series"]  = {col: clean}
    sess["decomp_result"] = decomp_res

    return {
        "chart_data":         chart_data,
        "outlier_timestamps": outlier_timestamps,
        "summary":            summary,
        "warnings":           (decomp_res.warnings + out_res.warnings + interm_res.warnings),
    }


# ── Variable roles ─────────────────────────────────────────────────────────────

class VariableRolesReq(BaseModel):
    token: str
    roles: Dict[str, str]   # col -> role


@router.post("/api/variable-roles")
async def set_variable_roles(body: VariableRolesReq):
    """Update variable roles in the session and return quick stats per variable."""
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404)
    df: pd.DataFrame = sess["df"]

    roles      = body.roles
    dep_cols   = [c for c, r in roles.items() if r == ROLE_DEPENDENT   and c in df.columns]
    indep_cols = [c for c, r in roles.items() if r == ROLE_INDEPENDENT and c in df.columns]
    event_cols = [c for c, r in roles.items() if r == ROLE_EVENT       and c in df.columns]

    sess["variable_roles"]   = roles
    sess["dependent_cols"]   = dep_cols
    sess["independent_cols"] = indep_cols
    sess["event_cols"]       = event_cols
    sess["value_cols"]       = dep_cols + indep_cols + event_cols

    stats: Dict[str, Any] = {}
    for col in (dep_cols + indep_cols + event_cols):
        if col in df.columns:
            s = df[col].dropna()
            stats[col] = {
                "role":     roles[col],
                "n":        int(len(s)),
                "n_missing":int(df[col].isna().sum()),
                "mean":     round(float(s.mean()), 4) if len(s) else None,
                "std":      round(float(s.std()),  4) if len(s) > 1 else None,
                "min":      round(float(s.min()),  4) if len(s) else None,
                "max":      round(float(s.max()),  4) if len(s) else None,
                "pct_zero": round(float((s == 0).mean() * 100), 2) if len(s) else 0,
            }

    return {
        "ok":               True,
        "dependent_cols":   dep_cols,
        "independent_cols": indep_cols,
        "event_cols":       event_cols,
        "stats":            _safe(stats),
    }


# ── Cross-correlation / multi-variable analysis ────────────────────────────────

class CrossCorrReq(BaseModel):
    token:          str
    dependent_col:  Optional[str] = None
    max_lags:       int = 20
    event_window:   int = 5


@router.post("/api/cross-correlation")
async def cross_correlation(body: CrossCorrReq):
    """Run full multi-variable analysis: CCF, event impact, Granger proxy."""
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404)

    df: pd.DataFrame = sess.get("accumulated_df", sess["df"])

    dep_cols   = sess.get("dependent_cols", [])
    indep_cols = sess.get("independent_cols", [])
    event_cols = sess.get("event_cols", [])

    if not dep_cols:
        vc = sess.get("value_cols", [])
        if vc:
            dep_cols   = [vc[0]]
            indep_cols = vc[1:]
        else:
            raise HTTPException(400, "No dependent column found in session")

    dep_col = body.dependent_col or dep_cols[0]

    agent_roles: Dict[str, str] = {}
    for c in dep_cols:   agent_roles[c] = ROLE_DEPENDENT
    for c in indep_cols: agent_roles[c] = ROLE_INDEPENDENT
    for c in event_cols: agent_roles[c] = ROLE_EVENT

    all_cols = [c for c in (dep_cols + indep_cols + event_cols) if c in df.columns]
    if not all_cols:
        raise HTTPException(400, "No matching columns in current dataframe")

    agent  = MultiVariableAgent()
    result = agent.execute(
        df=df[all_cols].copy(),
        roles=agent_roles,
        dependent_col=dep_col,
        max_ccf_lags=body.max_lags,
        event_window=body.event_window,
    )
    if not result.ok:
        raise HTTPException(500, result.errors[0] if result.errors else "Analysis failed")

    corr     = result.data["corr_matrix"]
    var_stats= result.data["var_stats"]
    ccf_data: Dict[str, Any] = {}
    for col, res in result.data["ccf_results"].items():
        ccf_data[col] = {
            "lags":          res["lags"],
            "ccf":           res["ccf"],
            "best_lag":      res["best_lag"],
            "max_ccf":       res["max_ccf"],
            "sig_threshold": res["sig_threshold"],
            "significant":   res["significant"],
            "interpretation":res["interpretation"],
        }

    return {
        "dependent_col":          dep_col,
        "independent_cols":       indep_cols,
        "event_cols":             event_cols,
        "var_stats":              _safe(var_stats),
        "corr_matrix":            _safe(corr),
        "corr_cols":              all_cols,
        "ccf_data":               _safe(ccf_data),
        "event_impacts":          _safe(result.data["event_impacts"]),
        "granger_results":        _safe(result.data["granger_results"]),
        "feature_recommendations":_safe(result.data["feature_recommendations"]),
        "report":                 result.metadata.get("report", ""),
        "warnings":               result.warnings,
    }
