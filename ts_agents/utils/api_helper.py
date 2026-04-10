"""
util/api-helper.py  –  TemporalMind shared utilities
=====================================================
Pure helper functions used across route modules.
No FastAPI dependencies; no side effects on import.
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from agents.flann_family import run_flann_family_forecast
except ImportError:
    run_flann_family_forecast = None  # type: ignore

# ── Constants ──────────────────────────────────────────────────────────────────

LARGE_DATASET_ROW_THRESHOLD = 100_000
LARGE_SERIES_THRESHOLD      = 10_000
LARGE_GROUP_THRESHOLD       = 2_000
DEFAULT_MAX_CHART_POINTS    = 1_500

_TIMESTAMP_NAME_TOKENS = {
    "date", "time", "timestamp", "datetime", "period",
    "month", "week", "year", "day", "ds", "dt", "ts",
}

# ── Column heuristics ──────────────────────────────────────────────────────────

def _looks_like_timestamp_name(name: str) -> bool:
    tokens = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
    return any(t in _TIMESTAMP_NAME_TOKENS for t in tokens)


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _df_to_records(df: pd.DataFrame, max_rows: int = 200) -> List[dict]:
    """Convert a DataFrame to JSON-serialisable records."""
    sub = df.head(max_rows).copy()
    sub = sub.reset_index()
    return json.loads(sub.to_json(orient="records", date_format="iso"))


def _safe(val: Any) -> Any:
    """Recursively convert numpy types to Python natives."""
    if isinstance(val, dict):
        return {k: _safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_safe(v) for v in val]
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        f = float(val)
        return f if np.isfinite(f) else None
    if isinstance(val, np.ndarray):
        return [_safe(v) for v in val.tolist()]
    if isinstance(val, pd.Series):
        return [_safe(v) for v in val.tolist()]
    if isinstance(val, pd.DataFrame):
        return [{k: _safe(v) for k, v in row.items()} for row in val.to_dict(orient="records")]
    if isinstance(val, (pd.Timestamp, pd.Timedelta)):
        return str(val)
    if val is None or val != val:   # NaN check
        return None
    return val


# ── Series statistics ──────────────────────────────────────────────────────────

def _series_stats(s: pd.Series) -> dict:
    s_clean = s.dropna()
    return {
        "n":         int(len(s)),
        "n_missing": int(s.isna().sum()),
        "mean":      round(float(s_clean.mean()), 4) if len(s_clean) else None,
        "std":       round(float(s_clean.std()),  4) if len(s_clean) > 1 else None,
        "min":       round(float(s_clean.min()),  4) if len(s_clean) else None,
        "max":       round(float(s_clean.max()),  4) if len(s_clean) else None,
        "pct_zero":  round(float((s_clean == 0).mean() * 100), 2) if len(s_clean) else 0,
    }


# ── Forecast metrics ───────────────────────────────────────────────────────────

def _forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Optional[float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)
    if y_pred.ndim != 1:
        y_pred = y_pred.reshape(-1)
    if y_true.size == 0 or y_pred.size == 0:
        return {"mae": None, "rmse": None, "mape": None}
    if y_true.size != y_pred.size:
        n = min(y_true.size, y_pred.size)
        y_true = y_true[-n:]
        y_pred = y_pred[-n:]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"mae": None, "rmse": None, "mape": None}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    nz   = np.abs(yt) > 1e-8
    mape = float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100) if nz.any() else None
    return {
        "mae":  round(mae,  4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4) if mape is not None else None,
    }


# ── Croston-family intermittent forecast ───────────────────────────────────────

def _croston_family_forecast(
    series: pd.Series,
    horizon: int,
    variant: str = "classic",
    alpha: float = 0.1,
    beta: float = 0.1,
    zero_threshold: float = 0.0,
) -> Optional[np.ndarray]:
    """Forecast intermittent demand with Croston/SBA/TSB family methods."""
    y = pd.Series(series).dropna().astype(float).values
    if horizon <= 0:
        return np.array([], dtype=float)
    if len(y) == 0:
        return None

    variant = (variant or "classic").lower().strip()

    if variant == "tsb":
        occ = (y > zero_threshold).astype(float)
        nz  = y[y > zero_threshold]
        if len(nz) == 0:
            return np.zeros(horizon, dtype=float)
        z = float(nz[0])
        p = float(np.clip(np.mean(occ), 1e-6, 1.0))
        for val, is_occ in zip(y, occ):
            p = p + beta * (float(is_occ) - p)
            if is_occ:
                z = z + alpha * (float(val) - z)
        fcst = max(0.0, z * p)
        return np.repeat(fcst, horizon).astype(float)

    nonzero_idx = np.where(y > zero_threshold)[0]
    if len(nonzero_idx) == 0:
        return np.zeros(horizon, dtype=float)

    z = float(y[nonzero_idx[0]])
    p = float(max(1, nonzero_idx[0] + 1))
    last_nonzero = int(nonzero_idx[0])
    for idx in nonzero_idx[1:]:
        interval = float(idx - last_nonzero)
        z = alpha * float(y[idx]) + (1 - alpha) * z
        p = alpha * interval       + (1 - alpha) * p
        last_nonzero = int(idx)

    fcst = z / max(p, 1e-12)
    if variant == "sba":
        fcst *= (1 - alpha / 2.0)
    return np.repeat(max(0.0, fcst), horizon).astype(float)


# ── FLANN forecast wrapper ─────────────────────────────────────────────────────

def _run_flann_forecast(
    y_full,
    y_train,
    y_test,
    future_idx,
    exog_all,
    period_used,
    variant: str = "flann",
    basis_family: str = "mixed",
    order: int = 3,
    ridge_lambda: float = 1e-2,
    recurrent_depth: int = 1,
    rvfl_n_hidden: int = 64,
    rvfl_activation: str = "sigmoid",
):
    """
    Thin wrapper — delegates to agents.flann_family.run_flann_family_forecast.
    Kept for backwards-compatibility with existing call-sites.
    """
    return run_flann_family_forecast(
        y_full=y_full,
        y_train=y_train,
        y_test=y_test,
        future_idx=future_idx,
        exog_all=exog_all,
        period_used=period_used,
        variant=variant,
        basis_family=basis_family,
        order=order,
        ridge_lambda=ridge_lambda,
        recurrent_depth=recurrent_depth,
        rvfl_n_hidden=rvfl_n_hidden,
        rvfl_activation=rvfl_activation,
    )


# ── Chart helpers ──────────────────────────────────────────────────────────────

def _downsample_series(
    series: pd.Series,
    max_points: int = DEFAULT_MAX_CHART_POINTS,
) -> pd.Series:
    """Return an evenly-spaced subset for large chart payloads."""
    if len(series) <= max_points:
        return series
    idx = np.linspace(0, len(series) - 1, num=max_points, dtype=int)
    return series.iloc[idx]


def _series_to_chart_payload(
    series: pd.Series,
    max_points: int = DEFAULT_MAX_CHART_POINTS,
) -> Dict[str, Any]:
    sampled = _downsample_series(series, max_points=max_points)
    return {
        "labels":            [str(d)[:10] for d in sampled.index],
        "values":            [round(float(v), 4) if not np.isnan(v) else None for v in sampled.values],
        "n_points_original": int(len(series)),
        "n_points_returned": int(len(sampled)),
        "downsampled":       bool(len(sampled) < len(series)),
    }


# ── Model recommendation helpers ───────────────────────────────────────────────

def _upgrade_model_for_exog(base_rec: str, has_exog: bool) -> str:
    """Upgrade model suggestion when exogenous variables are present."""
    if not has_exog:
        return base_rec
    mapping = {
        "ETS / ARIMA":           "ARIMAX / LightGBM (with exog)",
        "ARIMA(p,d,q)":          "ARIMAX (with exog)",
        "ARIMA(p,1,q)":          "ARIMAX(p,1,q) + exog",
        "Holt-Winters / SARIMA": "SARIMAX (with exog)",
        "ETS(A,N,A) / SARIMA":   "SARIMAX (with exog)",
        "Holt / ARIMA(p,1,0)":   "ARIMAX + exog",
        "FLANN":                 "FLANN + exog",
        "RecurrentFLANN":        "RecurrentFLANN + exog",
        "RVFL":                  "RVFL + exog",
        "Croston / SBA":         "Croston + event regressors",
        "TSB / Zero-Inflated":   "TSB + event regressors",
    }
    for k, v in mapping.items():
        if k in base_rec:
            return v
    return base_rec + " + exog"


def _resolve_recommended_candidate(
    model_rec: str,
    available_models: List[str],
) -> Optional[str]:
    """Map a descriptive recommendation string onto an evaluated candidate model."""
    available = set(available_models or [])
    rec = (model_rec or "").lower()

    preference_map = [
        (("tsb", "zero-inflated"),          ["TSB"]),
        (("croston", "sba"),                ["Croston", "SBA"]),
        (("arimax", "sarimax"),             ["ARIMAX", "ARIMA", "ETS"]),
        (("tbats",),                        ["ETS", "ARIMA"]),
        (("holt-winters",),                 ["ETS", "ARIMA"]),
        (("ets", "holt"),                   ["ETS"]),
        (("arima", "sarima"),               ["ARIMA", "ARIMAX"]),
        (("recurrentflann", "recurrent_flann"), ["RecurrentFLANN", "FLANN"]),
        (("rvfl",),                         ["RVFL", "FLANN"]),
        (("flann",),                        ["FLANN", "RecurrentFLANN", "RVFL"]),
        (("lightgbm", "linear regression", "linear"), ["LinearRegression"]),
    ]
    for markers, candidates in preference_map:
        if any(marker in rec for marker in markers):
            for candidate in candidates:
                if candidate in available:
                    return candidate
    for candidate in (available_models or []):
        if candidate.lower() in rec:
            return candidate
    return None


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


# ── Session frame helpers ──────────────────────────────────────────────────────

def _first_session_frame(sess: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Return the most-processed non-hierarchy DataFrame in the session."""
    for key in ("treated_df", "imputed_df", "accumulated_df", "df"):
        if key in sess:
            candidate = sess[key]
            if isinstance(candidate, pd.DataFrame):
                return candidate
    return None


def _hierarchy_session_frame(sess: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Return a session DataFrame that still carries hierarchy columns."""
    raw_df = sess.get("df")
    if not isinstance(raw_df, pd.DataFrame):
        return None

    hier_cols = [c for c in sess.get("hierarchy_cols", []) if c in raw_df.columns]
    if not hier_cols:
        return _first_session_frame(sess) or raw_df

    for key in ("treated_df", "imputed_df", "accumulated_df", "df"):
        candidate = sess.get(key)
        if isinstance(candidate, pd.DataFrame) and all(col in candidate.columns for col in hier_cols):
            return candidate

    # Treated/imputed frames may have dropped hierarchy columns.
    # When they still align 1-to-1 with the raw frame, stitch numeric values back.
    for key in ("treated_df", "imputed_df"):
        candidate = sess.get(key)
        if not isinstance(candidate, pd.DataFrame):
            continue
        if len(candidate) != len(raw_df) or not candidate.index.equals(raw_df.index):
            continue
        merged = raw_df.copy()
        overlay_cols = [c for c in candidate.columns if c in merged.columns and c not in hier_cols]
        if not overlay_cols:
            continue
        merged.loc[:, overlay_cols] = candidate.loc[:, overlay_cols].to_numpy()
        return merged

    return raw_df
