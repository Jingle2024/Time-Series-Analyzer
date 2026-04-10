"""
routes/hierarchy.py  –  Hierarchy management & node-analysis endpoints
======================================================================
Routes
------
POST /api/hierarchy          – build hierarchy and aggregate
POST /api/hierarchy-tree     – tree structure + level values for cascading dropdowns
POST /api/hierarchy-children – on-demand child values for large datasets
POST /api/analyze-node       – full analysis for a single hierarchy node
POST /api/level-stability    – cross-series stability analysis at one hierarchy level
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.decomposition_agent import DecompositionAgent
from agents.hierarchy_aggregation_agent import HierarchyAggregationAgent
from agents.intermittency_agent import IntermittencyAgent
from agents.missing_values_agent import MissingValuesAgent
from agents.outlier_detection_agent import OutlierDetectionAgent
from core.context_store import ContextStore
from core.runtime import SESSIONS as _sessions
from utils.api_helper import (
    DEFAULT_MAX_CHART_POINTS,
    LARGE_DATASET_ROW_THRESHOLD,
    LARGE_GROUP_THRESHOLD,
    LARGE_SERIES_THRESHOLD,
    _recommend_model,
    _safe,
    _series_stats,
    _series_to_chart_payload,
)

router = APIRouter()


# ── Hierarchy aggregation ──────────────────────────────────────────────────────

class HierarchyReq(BaseModel):
    token:  str
    method: str = "sum"


@router.post("/api/hierarchy")
async def hierarchy(body: HierarchyReq):
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404)
    df        = sess["df"]
    hier_cols = sess.get("hierarchy_cols", [])
    val_cols  = sess.get("value_cols", [])
    if not hier_cols:
        raise HTTPException(400, "No hierarchy columns in this session")

    agent  = HierarchyAggregationAgent()
    result = agent.execute(
        df=df, hierarchy_cols=hier_cols, value_cols=val_cols, method=body.method
    )
    if not result.ok:
        raise HTTPException(500, result.errors[0])

    sess["hierarchy_result"] = result

    level_info: Dict[str, List] = {}
    for lk in result.data.keys():
        prefix = lk.split("__")[0]
        level_info.setdefault(prefix, []).append(lk)

    return {
        "n_total_series": result.metadata["n_total_series"],
        "level_meta":     _safe(result.metadata["level_meta"]),
        "coherence":      _safe(result.metadata["coherence"]),
        "level_effects":  _safe(result.metadata["level_effects"]),
        "all_series_keys":list(result.data.keys()),
        "level_groups":   {k: v for k, v in level_info.items()},
        "report":         result.metadata.get("report", ""),
    }


# ── Hierarchy tree ─────────────────────────────────────────────────────────────

class HierarchyTreeReq(BaseModel):
    token:               str
    safe_mode:           bool = True
    include_tree:        bool = True
    max_values_per_level:int  = 500


@router.post("/api/hierarchy-tree")
async def hierarchy_tree(body: HierarchyTreeReq):
    """Return hierarchy tree with unique values per level for cascading dropdowns."""
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404, "No ingested data")
    df: pd.DataFrame = sess["df"]
    hier_cols = sess.get("hierarchy_cols", [])
    if not hier_cols:
        raise HTTPException(400, "No hierarchy columns in this session")

    use_safe_mode = body.safe_mode and len(df) >= LARGE_DATASET_ROW_THRESHOLD
    level_values: Dict[str, List] = {}
    for col in hier_cols:
        vals = sorted(df[col].dropna().unique().tolist())
        if use_safe_mode:
            vals = vals[:body.max_values_per_level]
        level_values[col] = [str(v) for v in vals]

    def build_tree(sub_df: pd.DataFrame, remaining: List[str]) -> Any:
        if not remaining:
            return []
        col  = remaining[0]
        rest = remaining[1:]
        result: Dict[str, Any] = {}
        for val in sorted(sub_df[col].dropna().unique()):
            child_df = sub_df[sub_df[col] == val]
            result[str(val)] = build_tree(child_df, rest)
        return result

    tree         = None
    total_leaves = None
    if not use_safe_mode:
        if body.include_tree:
            tree = build_tree(df, hier_cols)
        total_leaves = len(df.groupby(hier_cols))

    return {
        "levels":        hier_cols,
        "tree":          tree,
        "level_values":  level_values,
        "total_leaves":  total_leaves,
        "safe_mode_used":use_safe_mode,
        "tree_deferred": bool(use_safe_mode and body.include_tree),
        "message": (
            "Large dataset safe mode enabled. Returning level values only; load deeper nodes on demand."
            if use_safe_mode else ""
        ),
    }


# ── Hierarchy children (on-demand) ─────────────────────────────────────────────

class HierarchyChildrenReq(BaseModel):
    token:      str
    path:       Dict[str, str] = {}
    next_level: Optional[str]  = None
    max_values: int = 500


@router.post("/api/hierarchy-children")
async def hierarchy_children(body: HierarchyChildrenReq):
    """Return valid child values for the next level given the current path."""
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404, "No ingested data")

    df: pd.DataFrame = sess["df"]
    hier_cols = sess.get("hierarchy_cols", [])
    if not hier_cols:
        raise HTTPException(400, "No hierarchy columns in this session")

    path = body.path or {}
    if body.next_level:
        if body.next_level not in hier_cols:
            raise HTTPException(400, f"Unknown hierarchy level: {body.next_level}")
        next_level = body.next_level
    else:
        next_idx = len(path)
        if next_idx >= len(hier_cols):
            return {"level": None, "values": [], "path": path}
        next_level = hier_cols[next_idx]

    filtered = df
    for col in hier_cols:
        if col in path and path[col] != "":
            filtered = filtered[filtered[col].astype(str) == str(path[col])]

    if next_level not in filtered.columns:
        raise HTTPException(400, f"Level not found in dataframe: {next_level}")

    values = sorted(filtered[next_level].dropna().astype(str).unique().tolist())
    if body.max_values and body.max_values > 0:
        values = values[:body.max_values]

    return {
        "level":  next_level,
        "values": values,
        "path":   path,
        "count":  len(values),
    }


# ── Analyze node ───────────────────────────────────────────────────────────────

class AnalyzeNodeReq(BaseModel):
    token:            str
    node_path:        Dict[str, str]
    value_col:        Optional[str] = None
    period:           Optional[int] = None
    agg_method:       str  = "sum"
    safe_mode:        bool = True
    max_chart_points: int  = DEFAULT_MAX_CHART_POINTS


@router.post("/api/analyze-node")
async def analyze_node(body: AnalyzeNodeReq):
    """Filter df to node_path, aggregate remaining hier levels, run full analysis."""
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404)

    df: pd.DataFrame = sess["df"]
    hier_cols = sess.get("hierarchy_cols", [])
    val_cols  = sess.get("value_cols", [])
    val_col   = body.value_col or (val_cols[0] if val_cols else None)
    if not val_col:
        raise HTTPException(400, "No value column")

    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in body.node_path.items():
        if col in df.columns:
            mask &= (df[col].astype(str) == str(val))

    filtered = df[mask]
    if len(filtered) == 0:
        raise HTTPException(404, f"No data found for path: {body.node_path}")

    path_depth  = len(body.node_path)
    total_depth = len(hier_cols)
    is_leaf     = (path_depth == total_depth)
    node_label  = " › ".join(f"{k}={v}" for k, v in body.node_path.items())

    remaining_hier = [c for c in hier_cols if c not in body.node_path]
    agg_fn = body.agg_method

    series = (
        filtered.groupby(level=0)[val_col].agg(agg_fn) if remaining_hier
        else filtered[val_col]
    )
    series = series.sort_index().dropna()
    series.name = val_col
    if len(series) < 4:
        raise HTTPException(400, f"Too few data points ({len(series)}) for: {node_label}")

    n_agg = 1
    if remaining_hier:
        try:
            n_agg = int(len(filtered.groupby(remaining_hier)))
        except Exception:
            n_agg = 1

    use_safe_mode = body.safe_mode and (
        len(filtered) >= LARGE_DATASET_ROW_THRESHOLD
        or len(series)  >= LARGE_SERIES_THRESHOLD
        or n_agg        >= LARGE_GROUP_THRESHOLD
    )

    ctx       = ContextStore()
    mv_res    = MissingValuesAgent(ctx).execute(series=series)
    clean     = mv_res.data["imputed"] if mv_res.ok else series

    decomp_res = out_res = interm_res = None
    outlier_timestamps: List[str] = []
    Ft = Fs = d = 0
    interm_cls = "Deferred"
    model_rec  = "Select a smaller node for full model recommendation"

    if use_safe_mode:
        payload = _series_to_chart_payload(clean, max_points=body.max_chart_points)
        chart_data: Dict[str, Any] = {
            "labels":           payload["labels"],
            "original":         payload["values"],
            "downsampled":      payload["downsampled"],
            "n_points_original":payload["n_points_original"],
            "n_points_returned":payload["n_points_returned"],
        }
    else:
        decomp_res = DecompositionAgent(ctx).execute(series=clean, period=body.period)
        residual   = decomp_res.data.get("residual") if decomp_res.ok else None
        out_res    = OutlierDetectionAgent(ctx).execute(
            series=clean, residual=residual, methods=["iqr", "zscore", "isof"]
        )
        interm_res = IntermittencyAgent(ctx).execute(series=clean)

        payload = _series_to_chart_payload(clean, max_points=body.max_chart_points)
        chart_data = {
            "labels":           payload["labels"],
            "original":         payload["values"],
            "downsampled":      payload["downsampled"],
            "n_points_original":payload["n_points_original"],
            "n_points_returned":payload["n_points_returned"],
        }
        if decomp_res.ok:
            for comp in ("trend", "seasonal", "cycle", "residual"):
                arr = decomp_res.data.get(comp)
                if arr is not None:
                    cp = _series_to_chart_payload(arr, max_points=body.max_chart_points)
                    chart_data[comp] = cp["values"]

        if out_res.ok and len(out_res.data["outlier_table"]) > 0:
            outlier_timestamps = [
                str(r["timestamp"])[:10] for _, r in out_res.data["outlier_table"].iterrows()
            ]

        Ft         = decomp_res.metadata.get("trend_strength_Ft", 0)      if decomp_res.ok else 0
        Fs         = decomp_res.metadata.get("seasonal_strength_Fs", 0)   if decomp_res.ok else 0
        d          = decomp_res.metadata.get("differencing_order", 0)     if decomp_res.ok else 0
        interm_cls = interm_res.metadata.get("summary", {}).get("classification", "Smooth") if interm_res.ok else "Smooth"
        model_rec  = _recommend_model(interm_cls, Ft > 0.5, Fs > 0.5, d)

    summary = {
        "node_label":         node_label,
        "node_path":          body.node_path,
        "is_leaf":            is_leaf,
        "path_depth":         path_depth,
        "n_rows_filtered":    int(len(filtered)),
        "n_series_aggregated":n_agg,
        "agg_method":         agg_fn if remaining_hier else "none (leaf)",
        "safe_mode_used":     use_safe_mode,
        "analysis_mode":      "lightweight" if use_safe_mode else "full",
        "series_stats":       _safe(_series_stats(clean)),
        "decomp": _safe({
            "trend_strength_Ft":    decomp_res.metadata.get("trend_strength_Ft"),
            "seasonal_strength_Fs": decomp_res.metadata.get("seasonal_strength_Fs"),
            "period_used":          decomp_res.metadata.get("period_used"),
            "differencing_order":   decomp_res.metadata.get("differencing_order"),
            "stationarity":         decomp_res.metadata.get("stationarity"),
            "trend_stats":          decomp_res.metadata.get("trend_stats"),
            "interpretation":       decomp_res.metadata.get("interpretation"),
            "acf_significant_lags": decomp_res.metadata.get("acf_significant_lags"),
        }) if decomp_res and decomp_res.ok else {},
        "outliers":            _safe(out_res.metadata.get("summary", {}))    if out_res    and out_res.ok    else {},
        "intermittency":       _safe(interm_res.metadata.get("summary", {})) if interm_res and interm_res.ok else {},
        "missing_values":      _safe(mv_res.metadata.get("completeness", {}))if mv_res.ok else {},
        "model_recommendation":model_rec,
        "intermittency_class": interm_cls,
        "message": (
            "Large dataset safe mode enabled. Returned summary and downsampled chart only; drill into a smaller node for full decomposition."
            if use_safe_mode else ""
        ),
    }

    return {
        "chart_data":         chart_data,
        "outlier_timestamps": outlier_timestamps,
        "summary":            summary,
        "warnings": (
            (decomp_res.warnings if decomp_res and decomp_res.ok else []) +
            (out_res.warnings    if out_res    and out_res.ok    else []) +
            (interm_res.warnings if interm_res and interm_res.ok else [])
        ),
    }


# ── Level stability ────────────────────────────────────────────────────────────

class LevelStabilityReq(BaseModel):
    token:       str
    level_col:   str
    parent_path: Dict[str, str] = {}
    value_col:   Optional[str]  = None
    agg_method:  str = "sum"
    interval:    str = "M"    # D | W | M | Q | 6M | Y | native
    max_series:  int = 30


@router.post("/api/level-stability")
async def level_stability(body: LevelStabilityReq):
    """
    Return all child series at level_col (optionally filtered by parent_path),
    each resampled to interval. Includes superimposed plot data, seasonal profiles,
    stability stats, and cross-series envelope stats.
    """
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404)
    df: pd.DataFrame = sess["df"]
    hier_cols = sess.get("hierarchy_cols", [])
    dep_cols  = sess.get("dependent_cols") or sess.get("value_cols", [])
    val_col   = body.value_col or (dep_cols[0] if dep_cols else None)
    if not val_col:
        raise HTTPException(400, "No value column")
    if body.level_col not in df.columns:
        raise HTTPException(400, f"level_col '{body.level_col}' not found in data")

    # 1. Filter to parent path
    filtered = df.copy()
    for col, val in body.parent_path.items():
        if col in filtered.columns:
            filtered = filtered[filtered[col].astype(str) == str(val)]
    if len(filtered) == 0:
        raise HTTPException(404, "No data for parent path")

    # 2. Group by level_col → one series per child
    level_col_idx = hier_cols.index(body.level_col) if body.level_col in hier_cols else 0
    child_series: Dict[str, pd.Series] = {}
    for child_val, grp in filtered.groupby(body.level_col):
        child_val_str = str(child_val)
        s = grp.groupby(level=0)[val_col].agg(body.agg_method)
        child_series[child_val_str] = s.sort_index().dropna()

    series_names = sorted(child_series.keys())[:body.max_series]

    # 3. Resample each series
    INTERVAL_MAP = {
        "D":      ("D",    "Day"),
        "W":      ("W",    "Week"),
        "M":      ("MS",   "Month"),
        "Q":      ("QS",   "Quarter"),
        "6M":     ("6MS",  "Semi-year"),
        "Y":      ("YS",   "Year"),
        "native": (None,   "Native"),
    }
    freq_alias, freq_label = INTERVAL_MAP.get(body.interval, ("MS", "Month"))

    resampled: Dict[str, pd.Series] = {}
    for name in series_names:
        s = child_series[name]
        if freq_alias and len(s) >= 2:
            try:
                r = s.resample(freq_alias).sum() if body.agg_method == "sum" else s.resample(freq_alias).mean()
                resampled[name] = r.dropna()
            except Exception:
                resampled[name] = s
        else:
            resampled[name] = s

    # 4. Build unified time axis
    all_dates: set = set()
    for s in resampled.values():
        all_dates.update(s.index.tolist())
    sorted_dates = sorted(all_dates)
    date_labels  = [str(d)[:10] for d in sorted_dates]

    # 5. Series data for superimposed plot
    series_data: Dict[str, List] = {}
    for name, s in resampled.items():
        s_reindexed = s.reindex(sorted_dates)
        series_data[name] = [
            round(float(v), 4) if pd.notna(v) else None for v in s_reindexed.values
        ]

    # 6. Cross-series envelope
    matrix = np.full((len(sorted_dates), len(series_names)), np.nan)
    for j, name in enumerate(series_names):
        matrix[:, j] = resampled[name].reindex(sorted_dates).values

    env_mean = [round(float(np.nanmean(matrix[i])), 4) if not np.all(np.isnan(matrix[i])) else None for i in range(len(sorted_dates))]
    env_std  = [round(float(np.nanstd(matrix[i])),  4) if not np.all(np.isnan(matrix[i])) else None for i in range(len(sorted_dates))]
    env_min  = [round(float(np.nanmin(matrix[i])),  4) if not np.all(np.isnan(matrix[i])) else None for i in range(len(sorted_dates))]
    env_max  = [round(float(np.nanmax(matrix[i])),  4) if not np.all(np.isnan(matrix[i])) else None for i in range(len(sorted_dates))]
    env_cv   = [
        round(float(np.nanstd(matrix[i]) / (abs(np.nanmean(matrix[i])) + 1e-12)), 4)
        if env_mean[i] else None
        for i in range(len(sorted_dates))
    ]

    # 7. Seasonal profiles
    def get_season_label(dt: pd.Timestamp, interval: str) -> str:
        if interval == "D":   return dt.strftime("%a")
        elif interval == "W": return f"W{dt.isocalendar().week:02d}"
        elif interval == "M": return dt.strftime("%b")
        elif interval == "Q": return f"Q{dt.quarter}"
        elif interval == "6M":return "H1" if dt.month <= 6 else "H2"
        elif interval == "Y": return str(dt.year)
        else:                 return str(dt)[:7]

    season_order_map = {
        "D":  ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        "W":  [f"W{i:02d}" for i in range(1, 54)],
        "M":  ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        "Q":  ["Q1","Q2","Q3","Q4"],
        "6M": ["H1","H2"],
        "Y":  [],
        "native": [],
    }
    ordered_seasons = season_order_map.get(body.interval, [])

    seasonal_profiles: Dict[str, Dict] = {}
    for name, s in resampled.items():
        season_vals: Dict[str, List[float]] = {}
        for ts, val in s.items():
            if pd.isna(val):
                continue
            sl = get_season_label(ts, body.interval)
            season_vals.setdefault(sl, []).append(float(val))
        season_avg = {sl: round(float(np.mean(vals)), 4) for sl, vals in season_vals.items()}
        season_cv  = {
            sl: round(float(np.std(vals) / (abs(np.mean(vals)) + 1e-12)), 4)
            for sl, vals in season_vals.items() if len(vals) > 1
        }
        seasonal_profiles[name] = {
            "season_avg":      season_avg,
            "season_cv":       season_cv,
            "seasons_present": list(season_avg.keys()),
        }

    all_seasons: set = set()
    for p in seasonal_profiles.values():
        all_seasons.update(p["season_avg"].keys())
    if ordered_seasons:
        combined_season_labels = [s for s in ordered_seasons if s in all_seasons]
        combined_season_labels += sorted(all_seasons - set(ordered_seasons))
    else:
        combined_season_labels = sorted(all_seasons)

    combined_season_avg: Dict[str, float] = {}
    combined_season_cv:  Dict[str, float] = {}
    for sl in combined_season_labels:
        vals = [p["season_avg"][sl] for p in seasonal_profiles.values() if sl in p["season_avg"]]
        if vals:
            combined_season_avg[sl] = round(float(np.mean(vals)), 4)
            combined_season_cv[sl]  = round(float(np.std(vals) / (abs(np.mean(vals)) + 1e-12)), 4)

    # 8. Per-series stability stats
    stability_stats: Dict[str, Dict] = {}
    for name, s in resampled.items():
        vals   = s.dropna().values.astype(float)
        if len(vals) < 2:
            continue
        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals))
        cv     = std_v / (abs(mean_v) + 1e-12)
        yoy: Optional[float] = None
        if len(s) >= 2:
            try:
                yoy_s = s.pct_change(periods=max(1, len(s) // 2)).dropna()
                if len(yoy_s):
                    yoy = round(float(yoy_s.mean() * 100), 2)
            except Exception:
                pass
        stability_stats[name] = {
            "mean":  round(mean_v, 4),
            "std":   round(std_v,  4),
            "cv":    round(cv, 4),
            "min":   round(float(np.min(vals)), 4),
            "max":   round(float(np.max(vals)), 4),
            "range": round(float(np.max(vals) - np.min(vals)), 4),
            "n_obs": int(len(vals)),
            "avg_yoy_pct_change": yoy,
            "stability": "stable" if cv < 0.2 else "moderate" if cv < 0.5 else "volatile",
        }

    # 9. YoY matrix (series × year)
    yoy_matrix: Dict[str, Dict[str, float]] = {}
    all_years: set = set()
    for name, s in resampled.items():
        yearly = s.resample("YS").sum() if body.agg_method == "sum" else s.resample("YS").mean()
        yearly = yearly.dropna()
        row    = {str(ts.year): round(float(v), 2) for ts, v in yearly.items()}
        yoy_matrix[name] = row
        all_years.update(row.keys())
    yoy_years = sorted(all_years)

    return {
        "level_col":              body.level_col,
        "series_names":           series_names,
        "n_series":               len(series_names),
        "interval":               body.interval,
        "interval_label":         freq_label,
        "val_col":                val_col,
        "date_labels":            date_labels,
        "series_data":            series_data,
        "envelope": {
            "mean": env_mean, "std": env_std,
            "min":  env_min,  "max": env_max, "cv": env_cv,
        },
        "seasonal_profiles":      _safe(seasonal_profiles),
        "combined_season_labels": combined_season_labels,
        "combined_season_avg":    _safe(combined_season_avg),
        "combined_season_cv":     _safe(combined_season_cv),
        "stability_stats":        _safe(stability_stats),
        "yoy_matrix":             _safe(yoy_matrix),
        "yoy_years":              yoy_years,
        "parent_path":            body.parent_path,
    }
