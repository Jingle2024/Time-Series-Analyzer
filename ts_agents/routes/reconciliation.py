from __future__ import annotations

import io
import uuid
import zipfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_STATSMODELS = True
except Exception:
    ExponentialSmoothing = None
    _HAS_STATSMODELS = False

from agents.intermittency_agent import IntermittencyAgent
from core.runtime import DOWNLOADS as _downloads, SESSIONS as _sessions
from utils.api_helper import _croston_family_forecast, _hierarchy_session_frame, _safe

router = APIRouter()

TOTAL_NODE_ID = "total"


class ReconciledForecastReq(BaseModel):
    token: str
    dep_col: Optional[str] = None
    horizon: int = 13
    strategy: str = "bottom_up"  # bottom_up | top_down | middle_out
    middle_level: Optional[str] = None
    interval_mode: str = "session"  # session | manual
    target_freq: Optional[str] = None
    quantity_type: str = "flow"
    accumulation_method: str = "auto"
    allow_negative_forecast: bool = False
    output_format: str = "excel"  # excel | csv
    max_nodes_preview: int = 200


def _series_weight(series: pd.Series) -> float:
    vals = pd.Series(series).dropna().astype(float)
    if vals.empty:
        return 1.0
    total = float(vals.sum())
    if abs(total) > 1e-9:
        return abs(total)
    mean_abs = float(np.abs(vals).mean())
    return mean_abs if mean_abs > 1e-9 else 1.0


def _seasonal_period_for_freq(freq: Optional[str]) -> int:
    f = (freq or "").upper()
    if f.startswith("H"):
        return 24
    if f.startswith("D"):
        return 7
    if f.endswith("W") or f == "W":
        return 52
    if "Q" in f:
        return 4
    if "M" in f:
        return 12
    if "Y" in f or "A" in f:
        return 1
    return 1


def _apply_accumulation(series: pd.Series, freq: Optional[str], method: str) -> pd.Series:
    s = pd.Series(series).dropna().sort_index().astype(float)
    if s.empty or not freq or not isinstance(s.index, pd.DatetimeIndex):
        return s
    inferred = None
    try:
        inferred = pd.infer_freq(s.index)
    except Exception:
        inferred = None
    if inferred == freq:
        return s

    resampler = s.resample(freq)
    agg_map = {
        "sum": resampler.sum,
        "mean": resampler.mean,
        "median": resampler.median,
        "last": resampler.last,
        "first": resampler.first,
        "max": resampler.max,
        "min": resampler.min,
    }
    fn = agg_map.get(method, resampler.sum)
    return fn().dropna()


def _make_future_index(index: pd.DatetimeIndex, freq: Optional[str], horizon: int) -> pd.DatetimeIndex:
    if not len(index):
        raise ValueError("Cannot build future index from an empty history.")
    if horizon <= 0:
        return pd.DatetimeIndex([])
    freq_use = freq
    if not freq_use:
        try:
            freq_use = pd.infer_freq(index)
        except Exception:
            freq_use = None
    if not freq_use:
        if len(index) >= 2:
            step = index[-1] - index[-2]
            return pd.DatetimeIndex([index[-1] + step * i for i in range(1, horizon + 1)])
        return pd.date_range(index[-1], periods=horizon + 1, freq="D")[1:]
    return pd.date_range(index[-1], periods=horizon + 1, freq=freq_use)[1:]


def _forecast_series_auto(
    series: pd.Series,
    horizon: int,
    freq: Optional[str],
    allow_negative: bool,
) -> np.ndarray:
    s = pd.Series(series).dropna().sort_index().astype(float)
    if horizon <= 0:
        return np.array([], dtype=float)
    if s.empty:
        return np.zeros(horizon, dtype=float)
    if len(s) == 1:
        fc = np.repeat(float(s.iloc[-1]), horizon)
        return fc if allow_negative else np.maximum(fc, 0.0)

    season_period = _seasonal_period_for_freq(freq)

    try:
        interm_res = IntermittencyAgent().execute(series=s)
        classification = interm_res.metadata.get("summary", {}).get("classification", "Smooth") if interm_res.ok else "Smooth"
    except Exception:
        classification = "Smooth"

    if classification in {"Intermittent", "Lumpy"}:
        variant = "sba" if classification == "Intermittent" else "tsb"
        croston = _croston_family_forecast(s, horizon, variant=variant)
        if croston is not None:
            return croston if allow_negative else np.maximum(croston, 0.0)

    if _HAS_STATSMODELS and ExponentialSmoothing is not None and len(s) >= 6:
        try:
            seasonal = "add" if season_period > 1 and len(s) >= max(2 * season_period, 8) else None
            trend = "add" if len(s) >= 8 else None
            model = ExponentialSmoothing(
                s,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=season_period if seasonal else None,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True)
            fc = np.asarray(fit.forecast(horizon), dtype=float)
            return fc if allow_negative else np.maximum(fc, 0.0)
        except Exception:
            pass

    if season_period > 1 and len(s) >= season_period:
        last_season = s.iloc[-season_period:].to_numpy(dtype=float)
        reps = int(np.ceil(horizon / len(last_season)))
        fc = np.tile(last_season, reps)[:horizon]
        return fc if allow_negative else np.maximum(fc, 0.0)

    drift = (float(s.iloc[-1]) - float(s.iloc[0])) / max(1, len(s) - 1)
    fc = np.array([float(s.iloc[-1]) + drift * (i + 1) for i in range(horizon)], dtype=float)
    return fc if allow_negative else np.maximum(fc, 0.0)


def _node_id(depth: int, values: Tuple[str, ...]) -> str:
    return TOTAL_NODE_ID if depth == 0 else f"d{depth}__" + "__".join(values)


def _build_hierarchy_histories(
    df: pd.DataFrame,
    hier_cols: List[str],
    dep_col: str,
    freq: Optional[str],
    accumulation_method: str,
) -> Tuple[Dict[str, pd.Series], Dict[str, Dict[str, Any]], Dict[str, List[str]], Dict[str, List[str]], List[str]]:
    histories: Dict[str, pd.Series] = {}
    node_meta: Dict[str, Dict[str, Any]] = {}
    children_by_node: Dict[str, List[str]] = defaultdict(list)
    level_nodes: Dict[str, List[str]] = defaultdict(list)
    level_names = ["total"] + list(hier_cols)

    total_series = df.groupby(level=0)[dep_col].sum().sort_index()
    total_series = _apply_accumulation(total_series, freq, accumulation_method)
    histories[TOTAL_NODE_ID] = total_series
    node_meta[TOTAL_NODE_ID] = {
        "node_id": TOTAL_NODE_ID,
        "parent_id": None,
        "depth": 0,
        "level_name": "total",
        "node_value": "All",
        "path": {},
        "label": "Total",
        "display_path": "Total",
    }
    level_nodes["total"].append(TOTAL_NODE_ID)

    for depth, level_name in enumerate(hier_cols, start=1):
        group_keys = hier_cols[:depth]
        for keys, group in df.groupby(group_keys):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            values = tuple(str(v) for v in key_tuple)
            path = {col: values[i] for i, col in enumerate(group_keys)}
            nid = _node_id(depth, values)
            series = group.groupby(level=0)[dep_col].sum().sort_index()
            series = _apply_accumulation(series, freq, accumulation_method)
            histories[nid] = series
            parent_id = TOTAL_NODE_ID if depth == 1 else _node_id(depth - 1, values[:-1])
            node_meta[nid] = {
                "node_id": nid,
                "parent_id": parent_id,
                "depth": depth,
                "level_name": level_name,
                "node_value": values[-1],
                "path": path,
                "label": " > ".join(values),
                "display_path": " > ".join(f"{k}={v}" for k, v in path.items()),
            }
            children_by_node[parent_id].append(nid)
            level_nodes[level_name].append(nid)

    for parent_id, child_ids in children_by_node.items():
        child_ids.sort(key=lambda cid: node_meta[cid]["label"])

    return histories, node_meta, children_by_node, level_nodes, level_names


def _child_weight_map(
    histories: Dict[str, pd.Series],
    children_by_node: Dict[str, List[str]],
) -> Dict[Tuple[str, str], float]:
    weights: Dict[Tuple[str, str], float] = {}
    for parent_id, child_ids in children_by_node.items():
        raw = [_series_weight(histories.get(child_id, pd.Series(dtype=float))) for child_id in child_ids]
        total = float(sum(raw))
        if total <= 0:
            raw = [1.0] * len(child_ids)
            total = float(len(child_ids))
        for child_id, weight in zip(child_ids, raw):
            weights[(parent_id, child_id)] = float(weight / total)
    return weights


def _aggregate_up(
    forecasts: Dict[str, np.ndarray],
    children_by_node: Dict[str, List[str]],
    level_nodes: Dict[str, List[str]],
    ordered_levels: List[str],
) -> None:
    for level_name in reversed(ordered_levels[:-1]):
        for node_id in level_nodes.get(level_name, []):
            child_ids = children_by_node.get(node_id, [])
            if child_ids and all(child_id in forecasts for child_id in child_ids):
                forecasts[node_id] = np.sum([forecasts[child_id] for child_id in child_ids], axis=0)


def _allocate_down(
    root_ids: List[str],
    forecasts: Dict[str, np.ndarray],
    children_by_node: Dict[str, List[str]],
    weight_map: Dict[Tuple[str, str], float],
) -> None:
    stack = list(root_ids)
    while stack:
        parent_id = stack.pop()
        parent_fc = forecasts.get(parent_id)
        if parent_fc is None:
            continue
        for child_id in children_by_node.get(parent_id, []):
            w = weight_map.get((parent_id, child_id), 0.0)
            forecasts[child_id] = parent_fc * float(w)
            stack.append(child_id)


def _render_reconciliation_report(
    dep_col: str,
    hier_cols: List[str],
    strategy: str,
    middle_level: Optional[str],
    chosen_freq: Optional[str],
    horizon: int,
    level_counts: Dict[str, int],
) -> str:
    lines = [
        "=== Reconciled Forecast Report ===",
        f"Target column        : {dep_col}",
        f"Hierarchy order      : {' -> '.join(hier_cols)}",
        f"Strategy             : {strategy}",
        f"Middle level         : {middle_level or '-'}",
        f"Forecast frequency   : {chosen_freq or 'session/native'}",
        f"Horizon              : {horizon}",
        "",
        "Forecast nodes by level:",
    ]
    for level_name, count in level_counts.items():
        lines.append(f"  {level_name:15s}: {count}")
    return "\n".join(lines)


@router.post("/api/reconciled-forecast")
async def reconciled_forecast(body: ReconciledForecastReq):
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404, "Session not found")

    work_df = _hierarchy_session_frame(sess)
    if work_df is None or not isinstance(work_df, pd.DataFrame):
        raise HTTPException(400, "No prepared frame available in session")

    hier_cols = [c for c in sess.get("hierarchy_cols", []) if c in work_df.columns]
    if not hier_cols:
        raise HTTPException(400, "Reconciled forecasting requires an ordered hierarchy")

    dep_cols = sess.get("dependent_cols") or sess.get("value_cols", [])
    dep_col = body.dep_col or (dep_cols[0] if dep_cols else None)
    if not dep_col or dep_col not in work_df.columns:
        raise HTTPException(400, f"Dependent column not found: {dep_col}")

    strategy = (body.strategy or "bottom_up").lower().strip()
    if strategy not in {"bottom_up", "top_down", "middle_out"}:
        raise HTTPException(400, "strategy must be one of: bottom_up, top_down, middle_out")

    middle_level = body.middle_level
    if strategy == "middle_out":
        if not middle_level:
            raise HTTPException(400, "middle_level is required for middle_out reconciliation")
        if middle_level not in hier_cols:
            raise HTTPException(400, f"Unknown middle_level: {middle_level}")

    source_df = work_df[[dep_col] + hier_cols].copy()
    source_df[dep_col] = pd.to_numeric(source_df[dep_col], errors="coerce").fillna(0.0)
    source_df = source_df.dropna(subset=hier_cols).sort_index()
    if source_df.empty:
        raise HTTPException(400, "No data available after filtering hierarchy rows")

    chosen_freq = sess.get("target_freq") or sess.get("detected_freq")
    if (body.interval_mode or "session").lower().strip() == "manual" and body.target_freq:
        chosen_freq = str(body.target_freq).strip()

    accumulation_method = (body.accumulation_method or "auto").lower().strip()
    if accumulation_method == "auto":
        qty = (body.quantity_type or "flow").lower().strip()
        accumulation_method = {"flow": "sum", "stock": "last", "rate": "mean"}.get(qty, "sum")

    histories, node_meta, children_by_node, level_nodes, ordered_levels = _build_hierarchy_histories(
        source_df, hier_cols, dep_col, chosen_freq, accumulation_method
    )
    total_series = histories.get(TOTAL_NODE_ID)
    if total_series is None or total_series.empty:
        raise HTTPException(400, "Unable to build the top-level history for reconciliation")

    horizon = max(1, int(body.horizon))
    future_idx = _make_future_index(pd.DatetimeIndex(total_series.index), chosen_freq, horizon)
    weight_map = _child_weight_map(histories, children_by_node)
    forecasts: Dict[str, np.ndarray] = {}

    leaf_level = hier_cols[-1]
    leaf_ids = level_nodes.get(leaf_level, [])

    if strategy == "bottom_up":
        for leaf_id in leaf_ids:
            forecasts[leaf_id] = _forecast_series_auto(
                histories[leaf_id], horizon, chosen_freq, bool(body.allow_negative_forecast)
            )
        _aggregate_up(forecasts, children_by_node, level_nodes, ordered_levels)
    elif strategy == "top_down":
        forecasts[TOTAL_NODE_ID] = _forecast_series_auto(
            total_series, horizon, chosen_freq, bool(body.allow_negative_forecast)
        )
        _allocate_down([TOTAL_NODE_ID], forecasts, children_by_node, weight_map)
    else:
        anchor_ids = level_nodes.get(middle_level or "", [])
        for node_id in anchor_ids:
            forecasts[node_id] = _forecast_series_auto(
                histories[node_id], horizon, chosen_freq, bool(body.allow_negative_forecast)
            )
        anchor_depth = hier_cols.index(middle_level) + 1  # type: ignore[arg-type]
        _aggregate_up(forecasts, children_by_node, level_nodes, ordered_levels[:anchor_depth + 1])
        _allocate_down(anchor_ids, forecasts, children_by_node, weight_map)

    missing_ids = [node_id for node_id in node_meta if node_id not in forecasts]
    if missing_ids:
        raise HTTPException(500, f"Reconciliation incomplete. Missing forecasts for {len(missing_ids)} nodes.")

    # ── Base (pre-reconciliation) forecast for every node ─────────────────────
    # Run _forecast_series_auto independently on every node's history so we can
    # show "Model Forecast vs Reconciled Forecast" in the UI.  This is cheap
    # because the same function was already called for the anchor nodes during the
    # reconciliation pass; we just need the remaining nodes.
    base_forecasts: Dict[str, np.ndarray] = {}
    for node_id, hist in histories.items():
        base_forecasts[node_id] = _forecast_series_auto(
            hist, horizon, chosen_freq, bool(body.allow_negative_forecast)
        )

    level_counts = {level_name: len(level_nodes.get(level_name, [])) for level_name in ordered_levels}
    all_node_ids = sorted(node_meta.keys(), key=lambda nid: (node_meta[nid]["depth"], node_meta[nid]["label"]))
    max_preview = max(1, int(body.max_nodes_preview))
    preview_node_ids = all_node_ids[:max_preview]

    preview_nodes = []
    node_previews: Dict[str, Dict[str, Any]] = {}
    preview_level_nodes: Dict[str, List[str]] = defaultdict(list)
    forecast_rows: List[Dict[str, Any]] = []
    for node_id in all_node_ids:
        meta = node_meta[node_id]
        hist = histories[node_id]
        fc = forecasts[node_id]
        future_vals = [round(float(v), 4) for v in fc.tolist()]
        if node_id in preview_node_ids:
            labels = [str(d)[:10] for d in hist.index] + [str(d)[:10] for d in future_idx]
            history_vals = [round(float(v), 4) for v in hist.tolist()]
            future_path = [None] * len(history_vals) + future_vals
            observed_path = history_vals + [None] * len(future_vals)
            # Base (pre-reconciliation) model forecast path — nulls in history slots
            base_fc_vals = [round(float(v), 4) for v in base_forecasts[node_id].tolist()]
            base_forecast_path = [None] * len(history_vals) + base_fc_vals
            node_previews[node_id] = {
                "labels": labels,
                "history": observed_path,
                "future": future_path,
                "base_forecast": base_forecast_path,
                "history_n": int(len(hist)),
                "future_n": int(len(future_vals)),
            }
            preview_nodes.append(meta)
            preview_level_nodes[meta["level_name"]].append(node_id)
        for ts, val in zip(future_idx, future_vals):
            forecast_rows.append({
                "timestamp": str(ts)[:10],
                "level_name": meta["level_name"],
                "node_id": node_id,
                "node_label": meta["label"],
                "display_path": meta["display_path"],
                "forecast": val,
            })

    summary_rows = []
    for node_id in all_node_ids:
        meta = node_meta[node_id]
        hist = histories[node_id]
        future_vals = forecasts[node_id]
        summary_rows.append({
            "level_name": meta["level_name"],
            "depth": meta["depth"],
            "node_id": node_id,
            "node_label": meta["label"],
            "display_path": meta["display_path"],
            "history_points": int(len(hist)),
            "history_total": round(float(hist.sum()), 4) if len(hist) else 0.0,
            "future_total": round(float(np.sum(future_vals)), 4),
            "parent_id": meta["parent_id"],
        })

    report = _render_reconciliation_report(
        dep_col, hier_cols, strategy, middle_level, chosen_freq, horizon, level_counts
    )

    export_token = str(uuid.uuid4())[:8]
    output_format = (body.output_format or "excel").lower().strip()
    summary_df = pd.DataFrame(summary_rows)
    forecast_df = pd.DataFrame(forecast_rows)
    meta_df = pd.DataFrame([{
        "dep_col": dep_col,
        "strategy": strategy,
        "middle_level": middle_level,
        "chosen_freq": chosen_freq,
        "horizon": horizon,
        "hierarchy_order": " > ".join(hier_cols),
    }])
    if output_format in {"excel", "xlsx"}:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            meta_df.to_excel(writer, index=False, sheet_name="Summary")
            summary_df.to_excel(writer, index=False, sheet_name="NodeSummary")
            forecast_df.to_excel(writer, index=False, sheet_name="Forecasts")
        _downloads[export_token] = buf.getvalue()
        download_ext = "xlsx"
    else:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("summary.csv", meta_df.to_csv(index=False))
            zf.writestr("node_summary.csv", summary_df.to_csv(index=False))
            zf.writestr("forecasts.csv", forecast_df.to_csv(index=False))
            zf.writestr("report.txt", report)
        _downloads[export_token] = zbuf.getvalue()
        download_ext = "zip"

    default_node_id = TOTAL_NODE_ID if TOTAL_NODE_ID in node_previews else (preview_node_ids[0] if preview_node_ids else None)

    return {
        "ok": True,
        "dep_col": dep_col,
        "strategy": strategy,
        "middle_level": middle_level,
        "hierarchy_order": hier_cols,
        "ordered_levels": ordered_levels,
        "chosen_freq": chosen_freq,
        "horizon": horizon,
        "level_counts": level_counts,
        "n_total_nodes": len(all_node_ids),
        "n_preview_nodes": len(preview_node_ids),
        "preview_truncated": len(preview_node_ids) < len(all_node_ids),
        "default_node_id": default_node_id,
        "preview_nodes": _safe(preview_nodes),
        "preview_level_nodes": _safe(dict(preview_level_nodes)),
        "node_previews": _safe(node_previews),
        "forecast_preview": _safe(forecast_rows[:500]),
        "node_summary_preview": _safe(summary_rows[:300]),
        "download_token": export_token,
        "download_ext": download_ext,
        "report": report,
    }
