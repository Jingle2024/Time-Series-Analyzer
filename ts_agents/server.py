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
POST /api/forecast-prepare  – per-series forecast preparation (model params + future frame)
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
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_STATSMODELS = True
except Exception:
    ExponentialSmoothing = None
    SARIMAX = None
    _HAS_STATSMODELS = False

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
from agents.forecast_preparation_agent import ForecastPreparationAgent
from agents.flann_family import (
    FLANNFamily, run_flann_family_forecast,
    SUPPORTED_VARIANTS as FLANN_VARIANTS,
    SUPPORTED_BASIS    as FLANN_BASIS,
)
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

LARGE_DATASET_ROW_THRESHOLD = 100_000
LARGE_SERIES_THRESHOLD = 10_000
LARGE_GROUP_THRESHOLD = 2_000
DEFAULT_MAX_CHART_POINTS = 1_500

# ── lazy FastAPI import ───────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import (HTMLResponse, JSONResponse,
                                    Response, StreamingResponse)
    from pydantic import BaseModel, Field
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


def _forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)
    if y_pred.ndim != 1:
        y_pred = y_pred.reshape(-1)
    if y_true.size == 0 or y_pred.size == 0:
        return {"mae": None, "rmse": None, "mape": None}
    if y_true.size != y_pred.size:
        # Forecast fallbacks can be evaluated against an implicit holdout even when
        # the UI timeline has no explicit holdout labels. Align on the overlapping
        # tail instead of raising on shape mismatch.
        n = min(y_true.size, y_pred.size)
        y_true = y_true[-n:]
        y_pred = y_pred[-n:]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"mae": None, "rmse": None, "mape": None}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    nz = np.abs(yt) > 1e-8
    mape = float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100) if nz.any() else None
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4) if mape is not None else None}


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
        nz = y[y > zero_threshold]
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
        p = alpha * interval + (1 - alpha) * p
        last_nonzero = int(idx)

    fcst = z / max(p, 1e-12)
    if variant == "sba":
        fcst *= (1 - alpha / 2.0)
    return np.repeat(max(0.0, fcst), horizon).astype(float)

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
    Kept for backwards-compatibility with the existing call-site.
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
# def _flann_expand_matrix(
#     X: np.ndarray,
#     mean_: np.ndarray,
#     std_: np.ndarray,
#     order: int = 3,
# ) -> np.ndarray:
#     """Expand tabular inputs into a compact FLANN-style basis space."""
#     X_arr = np.asarray(X, dtype=float)
#     if X_arr.ndim == 1:
#         X_arr = X_arr.reshape(1, -1)
#     scale = np.where(np.abs(std_) < 1e-8, 1.0, std_)
#     Xn = (X_arr - mean_) / scale
#     Xn = np.clip(Xn, -4.0, 4.0)

#     parts = [Xn]
#     for p in range(2, max(2, int(order)) + 1):
#         parts.append(np.power(Xn, p))
#     parts.append(np.sin(np.pi * Xn))
#     parts.append(np.cos(np.pi * Xn))
#     Z = np.hstack(parts)
#     return np.hstack([np.ones((Z.shape[0], 1)), Z])


# def _fit_flann_model(
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     order: int = 3,
#     ridge_lambda: float = 1e-2,
# ) -> Optional[Dict[str, Any]]:
#     X_arr = np.asarray(X_train, dtype=float)
#     y_arr = np.asarray(y_train, dtype=float).reshape(-1)
#     if X_arr.ndim != 2 or len(X_arr) < 5 or X_arr.shape[1] == 0 or len(y_arr) != len(X_arr):
#         return None

#     mean_ = np.nanmean(X_arr, axis=0)
#     std_ = np.nanstd(X_arr, axis=0)
#     Z = _flann_expand_matrix(X_arr, mean_, std_, order=order)

#     reg = float(max(1e-8, ridge_lambda))
#     eye = np.eye(Z.shape[1], dtype=float)
#     eye[0, 0] = 0.0  # keep intercept unregularized
#     try:
#         weights = np.linalg.solve(Z.T @ Z + reg * eye, Z.T @ y_arr)
#     except np.linalg.LinAlgError:
#         weights = np.linalg.pinv(Z.T @ Z + reg * eye) @ Z.T @ y_arr

#     return {
#         "weights": weights,
#         "mean": mean_,
#         "std": std_,
#         "order": int(order),
#     }


# def _predict_flann_model(model: Dict[str, Any], X: np.ndarray) -> np.ndarray:
#     Z = _flann_expand_matrix(
#         X=np.asarray(X, dtype=float),
#         mean_=np.asarray(model["mean"], dtype=float),
#         std_=np.asarray(model["std"], dtype=float),
#         order=int(model.get("order", 3)),
#     )
#     return np.asarray(Z @ np.asarray(model["weights"], dtype=float), dtype=float).reshape(-1)


# def _run_flann_forecast(
#     y_full: pd.Series,
#     y_train: pd.Series,
#     y_test: pd.Series,
#     future_idx: pd.Index,
#     exog_all: Optional[pd.DataFrame],
#     period_used: int,
# ) -> Optional[Dict[str, np.ndarray]]:
#     """Train a compact FLANN competitor on lag/exog features and forecast holdout + future."""
#     y_full = pd.Series(y_full).dropna().astype(float)
#     y_train = pd.Series(y_train).dropna().astype(float)
#     y_test = pd.Series(y_test).dropna().astype(float)
#     if len(y_train) < 8 or len(y_test) == 0:
#         return None

#     p = int(period_used) if period_used and int(period_used) > 1 else 0
#     lag_candidates = [1, 2, 3]
#     if p > 1:
#         lag_candidates.extend([p, p + 1])
#     lag_candidates = sorted({lag for lag in lag_candidates if lag < len(y_full)})

#     fl_df = pd.DataFrame(index=y_full.index)
#     fl_df["y"] = y_full.values
#     for lag in lag_candidates:
#         fl_df[f"lag{lag}"] = fl_df["y"].shift(lag)
#     if len(y_full) >= 4:
#         fl_df["roll_mean_3"] = fl_df["y"].shift(1).rolling(3).mean()
#         fl_df["roll_std_3"] = fl_df["y"].shift(1).rolling(3).std()
#     if len(y_full) >= 8:
#         fl_df["roll_mean_7"] = fl_df["y"].shift(1).rolling(7).mean()
#         fl_df["roll_std_7"] = fl_df["y"].shift(1).rolling(7).std()

#     ex_cols: List[str] = []
#     if exog_all is not None and len(exog_all.columns):
#         for c in exog_all.columns:
#             safe_name = f"exog__{c}"
#             fl_df[safe_name] = exog_all[c].reindex(fl_df.index).astype(float).values
#             ex_cols.append(safe_name)

#     fl_df = fl_df.dropna()
#     if len(fl_df) < 8:
#         return None

#     train_cut = y_train.index[-1]
#     fl_train = fl_df[fl_df.index <= train_cut]
#     fl_test = fl_df[fl_df.index.isin(y_test.index)]
#     if len(fl_train) < 5 or len(fl_test) == 0:
#         return None

#     x_cols = [c for c in fl_df.columns if c != "y"]
#     model = _fit_flann_model(
#         X_train=fl_train[x_cols].values,
#         y_train=fl_train["y"].values,
#         order=3,
#         ridge_lambda=1e-2,
#     )
#     if model is None:
#         return None

#     yhat_test = _predict_flann_model(model, fl_test[x_cols].values)

#     hist = y_full.copy()
#     fut_preds: List[float] = []
#     ex_future = pd.DataFrame(0.0, index=future_idx, columns=ex_cols) if ex_cols else None
#     for dt in future_idx:
#         row: Dict[str, float] = {}
#         for lag in lag_candidates:
#             row[f"lag{lag}"] = float(hist.iloc[-lag]) if len(hist) >= lag else float(hist.iloc[-1])
#         if "roll_mean_3" in x_cols:
#             row["roll_mean_3"] = float(hist.iloc[-3:].mean()) if len(hist) >= 3 else float(hist.mean())
#         if "roll_std_3" in x_cols:
#             row["roll_std_3"] = float(hist.iloc[-3:].std()) if len(hist) >= 3 else 0.0
#         if "roll_mean_7" in x_cols:
#             row["roll_mean_7"] = float(hist.iloc[-7:].mean()) if len(hist) >= 7 else float(hist.mean())
#         if "roll_std_7" in x_cols:
#             row["roll_std_7"] = float(hist.iloc[-7:].std()) if len(hist) >= 7 else 0.0
#         for c in ex_cols:
#             row[c] = float(ex_future.loc[dt, c]) if ex_future is not None else 0.0
#         x_row = np.array([row[col] for col in x_cols], dtype=float).reshape(1, -1)
#         pred = float(_predict_flann_model(model, x_row)[0])
#         fut_preds.append(pred)
#         hist = pd.concat([hist, pd.Series([pred], index=[dt])])

#     return {
#         "holdout_pred": np.asarray(yhat_test, dtype=float),
#         "future_pred": np.asarray(fut_preds, dtype=float),
#     }


# def _safe(val):
#     """Recursively convert numpy types to Python natives."""
#     if isinstance(val, dict):
#         return {k: _safe(v) for k, v in val.items()}
#     if isinstance(val, (list, tuple, set)):
#         return [_safe(v) for v in val]
#     if isinstance(val, (np.bool_, bool)):
#         return bool(val)
#     if isinstance(val, (np.integer,)):
#         return int(val)
#     if isinstance(val, (np.floating,)):
#         return float(val)
#     if isinstance(val, np.ndarray):
#         return val.tolist()
#     if isinstance(val, pd.Series):
#         return val.tolist()
#     if isinstance(val, pd.DataFrame):
#         return val.to_dict(orient="records")
#     if isinstance(val, (pd.Timestamp, pd.Timedelta)):
#         return str(val)
#     if val is None or val != val:   # NaN check
#         return None
#     return val


def _downsample_series(series: pd.Series, max_points: int = DEFAULT_MAX_CHART_POINTS) -> pd.Series:
    """Return an evenly spaced subset for large chart payloads."""
    if len(series) <= max_points:
        return series
    idx = np.linspace(0, len(series) - 1, num=max_points, dtype=int)
    return series.iloc[idx]


def _series_to_chart_payload(series: pd.Series, max_points: int = DEFAULT_MAX_CHART_POINTS) -> Dict[str, Any]:
    sampled = _downsample_series(series, max_points=max_points)
    return {
        "labels": [str(d)[:10] for d in sampled.index],
        "values": [round(float(v), 4) if not np.isnan(v) else None for v in sampled.values],
        "n_points_original": int(len(series)),
        "n_points_returned": int(len(sampled)),
        "downsampled": bool(len(sampled) < len(series)),
    }


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
        "FLANN":           "FLANN + exog",
        "RecurrentFLANN":  "RecurrentFLANN + exog",
        "RVFL":            "RVFL + exog",
        "Croston / SBA":        "Croston + event regressors",
        "TSB / Zero-Inflated":  "TSB + event regressors",
    }
    for k, v in mapping.items():
        if k in base_rec:
            return v
    return base_rec + " + exog"


def _resolve_recommended_candidate(model_rec: str, available_models: List[str]) -> Optional[str]:
    """Map a descriptive recommendation string onto an evaluated candidate model."""
    available = set(available_models or [])
    rec = (model_rec or "").lower()

    preference_map = [
        (("tsb", "zero-inflated"), ["TSB"]),
        (("croston", "sba"), ["Croston", "SBA"]),
        (("arimax", "sarimax"), ["ARIMAX", "ARIMA", "ETS"]),
        (("tbats",), ["ETS", "ARIMA"]),
        (("holt-winters",), ["ETS", "ARIMA"]),
        (("ets", "holt"), ["ETS"]),
        (("arima", "sarima"), ["ARIMA", "ARIMAX"]),
        (("recurrentflann", "recurrent_flann"), ["RecurrentFLANN", "FLANN"]),
        (("rvfl",),          ["RVFL", "FLANN"]),
        (("flann",),         ["FLANN", "RecurrentFLANN", "RVFL"]),
        (("lightgbm", "linear regression", "linear"), ["LinearRegression"]),
    ]
    for markers, candidates in preference_map:
        if any(marker in rec for marker in markers):
            for candidate in candidates:
                if candidate in available:
                    return candidate
    for candidate in available_models or []:
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
                # try datetime parse without emitting noisy per-value parse warnings
                try:
                    sample = df_raw[c].dropna().head(10)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        parsed = pd.to_datetime(sample, errors="coerce")
                    if len(parsed) and float(parsed.notna().mean()) >= 0.8:
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
        safe_mode: bool = True
        include_tree: bool = True
        max_values_per_level: int = 500

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
            col = remaining[0]
            rest = remaining[1:]
            result = {}
            for val in sorted(sub_df[col].dropna().unique()):
                child_df = sub_df[sub_df[col] == val]
                result[str(val)] = build_tree(child_df, rest)
            return result

        tree = None
        if body.include_tree and not use_safe_mode:
            tree = build_tree(df, hier_cols)

        total_leaves = None
        if not use_safe_mode:
            total_leaves = len(df.groupby(hier_cols))

        return {
            "levels": hier_cols,
            "tree": tree,
            "level_values": level_values,
            "total_leaves": total_leaves,
            "safe_mode_used": use_safe_mode,
            "tree_deferred": bool(use_safe_mode and body.include_tree),
            "message": (
                "Large dataset safe mode enabled. Returning level values only; load deeper nodes on demand."
                if use_safe_mode else ""
            ),
        }

    class HierarchyChildrenReq(BaseModel):
        token: str
        path: Dict[str, str] = {}
        next_level: Optional[str] = None
        max_values: int = 500

    @app.post("/api/hierarchy-children")
    async def hierarchy_children(body: HierarchyChildrenReq):
        """
        Returns valid child values for the next hierarchy level given the current path.
        Used by cascading dropdowns so large datasets do not need the full tree in memory.
        """
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
            "level": next_level,
            "values": values,
            "path": path,
            "count": len(values),
        }

    # ── Analyze a specific hierarchy node ─────────────────────────────────────

    class AnalyzeNodeReq(BaseModel):
        token: str
        node_path: Dict[str, str]
        value_col: Optional[str] = None
        period: Optional[int] = None
        agg_method: str = "sum"
        safe_mode: bool = True
        max_chart_points: int = DEFAULT_MAX_CHART_POINTS

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

        n_agg = 1
        if remaining_hier:
            try:
                n_agg = int(len(filtered.groupby(remaining_hier)))
            except Exception:
                n_agg = 1

        use_safe_mode = body.safe_mode and (
            len(filtered) >= LARGE_DATASET_ROW_THRESHOLD or
            len(series) >= LARGE_SERIES_THRESHOLD or
            n_agg >= LARGE_GROUP_THRESHOLD
        )

        ctx = ContextStore()
        mv_res = MissingValuesAgent(ctx).execute(series=series)
        clean = mv_res.data["imputed"] if mv_res.ok else series

        decomp_res = None
        out_res = None
        interm_res = None
        outlier_timestamps = []
        Ft = 0
        Fs = 0
        d = 0
        interm_cls = "Deferred"
        model_rec = "Select a smaller node for full model recommendation"

        if use_safe_mode:
            payload = _series_to_chart_payload(clean, max_points=body.max_chart_points)
            chart_data: Dict[str, Any] = {
                "labels": payload["labels"],
                "original": payload["values"],
                "downsampled": payload["downsampled"],
                "n_points_original": payload["n_points_original"],
                "n_points_returned": payload["n_points_returned"],
            }
        else:
            decomp_res = DecompositionAgent(ctx).execute(series=clean, period=body.period)
            residual = decomp_res.data.get("residual") if decomp_res.ok else None
            out_res = OutlierDetectionAgent(ctx).execute(
                series=clean, residual=residual, methods=["iqr", "zscore", "isof"]
            )
            interm_res = IntermittencyAgent(ctx).execute(series=clean)

            payload = _series_to_chart_payload(clean, max_points=body.max_chart_points)
            chart_data = {
                "labels": payload["labels"],
                "original": payload["values"],
                "downsampled": payload["downsampled"],
                "n_points_original": payload["n_points_original"],
                "n_points_returned": payload["n_points_returned"],
            }
            if decomp_res.ok:
                for comp in ("trend", "seasonal", "cycle", "residual"):
                    arr = decomp_res.data.get(comp)
                    if arr is not None:
                        comp_payload = _series_to_chart_payload(arr, max_points=body.max_chart_points)
                        chart_data[comp] = comp_payload["values"]

            if out_res.ok and len(out_res.data["outlier_table"]) > 0:
                outlier_timestamps = [
                    str(r["timestamp"])[:10] for _, r in out_res.data["outlier_table"].iterrows()
                ]

            Ft = decomp_res.metadata.get("trend_strength_Ft", 0) if decomp_res.ok else 0
            Fs = decomp_res.metadata.get("seasonal_strength_Fs", 0) if decomp_res.ok else 0
            d = decomp_res.metadata.get("differencing_order", 0) if decomp_res.ok else 0
            interm_cls = interm_res.metadata.get("summary", {}).get("classification", "Smooth") if interm_res.ok else "Smooth"
            model_rec = _recommend_model(interm_cls, Ft > 0.5, Fs > 0.5, d)

        summary = {
            "node_label": node_label,
            "node_path": body.node_path,
            "is_leaf": is_leaf,
            "path_depth": path_depth,
            "n_rows_filtered": int(len(filtered)),
            "n_series_aggregated": n_agg,
            "agg_method": agg_fn if remaining_hier else "none (leaf)",
            "safe_mode_used": use_safe_mode,
            "analysis_mode": "lightweight" if use_safe_mode else "full",
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
            }) if decomp_res and decomp_res.ok else {},
            "outliers": _safe(out_res.metadata.get("summary", {})) if out_res and out_res.ok else {},
            "intermittency": _safe(interm_res.metadata.get("summary", {})) if interm_res and interm_res.ok else {},
            "missing_values": _safe(mv_res.metadata.get("completeness", {})) if mv_res.ok else {},
            "model_recommendation": model_rec,
            "intermittency_class": interm_cls,
            "message": (
                "Large dataset safe mode enabled. Returned summary and downsampled chart only; drill into a smaller node for full decomposition."
                if use_safe_mode else ""
            ),
        }

        return {
            "chart_data": chart_data,
            "outlier_timestamps": outlier_timestamps,
            "summary": summary,
            "warnings": (
                (decomp_res.warnings if decomp_res and decomp_res.ok else []) +
                (out_res.warnings if out_res and out_res.ok else []) +
                (interm_res.warnings if interm_res and interm_res.ok else [])
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

    def _hierarchy_session_frame(sess: Dict[str, Any]) -> Optional[pd.DataFrame]:
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

        # Treated/imputed frames currently drop hierarchy columns. When they still
        # align 1:1 with the raw frame, stitch the numeric values back onto the
        # raw hierarchy keys so hierarchy mode can still respect session cleanup.
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

    class ForecastPrepareReq(BaseModel):
        token: str
        mode: str = "single"  # single | hierarchy
        dep_col: Optional[str] = None
        node_path: Dict[str, str] = Field(default_factory=dict)
        interval_mode: str = "session"  # session | advisor | manual
        target_freq: Optional[str] = None
        quantity_type: str = "flow"  # flow | stock | rate
        accumulation_method: str = "auto"  # auto | sum | mean | median | last | first | max | min
        transform: str = "auto"
        scale_method: str = "minmax"
        apply_missing_treatment: bool = True
        missing_method: str = "auto"
        missing_period: int = 7
        missing_zero_as_missing: bool = False
        apply_outlier_treatment: bool = True
        outlier_methods: List[str] = Field(default_factory=lambda: ["iqr", "zscore", "isof"])
        outlier_treatment: str = "cap"  # cap | remove | keep
        rolling_windows: List[int] = Field(default_factory=lambda: [7, 14, 28])
        add_calendar: bool = True
        train_ratio: float = 0.70
        val_ratio: float = 0.15
        horizon: int = 13
        n_holdout: int = 0
        output_format: str = "excel"  # excel | csv (zip bundle)
        flann_variant:        str   = "flann"      # flann | recurrent_flann | rvfl
        flann_basis:          str   = "mixed"      # polynomial | trigonometric |
                                                # chebyshev | legendre | mixed
        flann_order:          int   = 3            # expansion order / polynomial degree
        flann_ridge_lambda:   float = 1e-2         # L2 regularisation for output weights
        flann_recurrent_depth: int  = 1            # RecurrentFLANN: feedback window depth
        rvfl_n_hidden:        int   = 64           # RVFL: number of random hidden nodes
        rvfl_activation:      str   = "sigmoid"    # RVFL: sigmoid | relu | tanh | sin
        run_all_flann_variants: bool = False        # run all 3 variants and compare
        enable_combination_models: bool = False
        allow_negative_forecast: bool = False

    @app.post("/api/forecast-prepare")
    async def forecast_prepare(body: ForecastPrepareReq):
        sess = _sessions.get(body.token)
        if not sess or "df" not in sess:
            raise HTTPException(404, "Session not found")

        hier_cols = sess.get("hierarchy_cols", [])
        mode = body.mode if body.mode in ("single", "hierarchy") else "single"
        node_path = {k: str(v) for k, v in (body.node_path or {}).items() if v is not None and str(v) != ""}

        if mode == "hierarchy":
            work_df = _hierarchy_session_frame(sess)
        else:
            work_df = _first_session_frame(sess)

        if work_df is None or not isinstance(work_df, pd.DataFrame):
            raise HTTPException(400, "No prepared frame available in session")

        dep_cols = sess.get("dependent_cols") or sess.get("value_cols", [])
        dep_col = body.dep_col or (dep_cols[0] if dep_cols else None)
        if not dep_col or dep_col not in work_df.columns:
            raise HTTPException(400, f"Dependent column not found: {dep_col}")

        indep_cols = [c for c in sess.get("independent_cols", []) if c in work_df.columns]
        event_cols = [c for c in sess.get("event_cols", []) if c in work_df.columns]

        source_df = work_df
        if mode == "hierarchy":
            if not hier_cols:
                raise HTTPException(400, "No hierarchy columns available in this session")
            if not node_path:
                raise HTTPException(400, "node_path is required in hierarchy mode")

            mask = pd.Series([True] * len(source_df), index=source_df.index)
            for col, val in node_path.items():
                if col in source_df.columns:
                    mask &= (source_df[col].astype(str) == str(val))
            source_df = source_df[mask]
            if source_df.empty:
                raise HTTPException(404, f"No data found for node path: {node_path}")

        source_df = source_df.sort_index()

        # Aggregate to one time series if the selected hierarchy node is not a leaf.
        remaining_hier = [c for c in hier_cols if c not in node_path] if mode == "hierarchy" else []
        if remaining_hier:
            agg_spec: Dict[str, str] = {dep_col: "sum"}
            for c in indep_cols:
                agg_spec[c] = "mean"
            for c in event_cols:
                agg_spec[c] = "max"
            grouped = source_df.groupby(level=0).agg(agg_spec).sort_index()
            series = grouped[dep_col].dropna()
            indep_df = grouped[indep_cols] if indep_cols else None
            event_df = grouped[event_cols] if event_cols else None
        else:
            series = source_df[dep_col].dropna().sort_index()
            indep_df = source_df[indep_cols].reindex(series.index) if indep_cols else None
            event_df = source_df[event_cols].reindex(series.index).fillna(0) if event_cols else None

        prep_stage_warnings: List[str] = []
        preprocessing: Dict[str, Any] = {}

        # Optional interval selection + accumulation before forecast preparation.
        chosen_freq = sess.get("target_freq") or sess.get("detected_freq")
        interval_mode = (body.interval_mode or "session").lower().strip()
        if interval_mode == "manual" and body.target_freq:
            chosen_freq = str(body.target_freq).strip()
        elif interval_mode == "advisor":
            advisor_res = IntervalAdvisorAgent().execute(
                series=series,
                native_freq=sess.get("detected_freq"),
                top_n=3,
            )
            if advisor_res.ok:
                chosen_freq = advisor_res.metadata.get("best_interval") or chosen_freq
                preprocessing["interval_advisor"] = {
                    "best_interval": advisor_res.metadata.get("best_interval"),
                    "best_alias": advisor_res.metadata.get("best_alias"),
                }
                prep_stage_warnings.extend(advisor_res.warnings or [])
            else:
                prep_stage_warnings.append("Interval advisor failed; falling back to session frequency.")

        if chosen_freq and isinstance(series.index, pd.DatetimeIndex):
            # Skip no-op accumulation when frequency already matches.
            inferred = None
            try:
                inferred = pd.infer_freq(series.index)
            except Exception:
                inferred = None
            if inferred != chosen_freq:
                method = (body.accumulation_method or "auto").lower()
                qty = (body.quantity_type or "flow").lower()
                if method == "auto":
                    method = {"flow": "sum", "stock": "last", "rate": "mean"}.get(qty, "sum")

                try:
                    # Dependent series accumulation.
                    dep_resampler = series.sort_index().resample(chosen_freq)
                    if method == "sum":
                        series = dep_resampler.sum()
                    elif method == "mean":
                        series = dep_resampler.mean()
                    elif method == "median":
                        series = dep_resampler.median()
                    elif method == "last":
                        series = dep_resampler.last()
                    elif method == "first":
                        series = dep_resampler.first()
                    elif method == "max":
                        series = dep_resampler.max()
                    elif method == "min":
                        series = dep_resampler.min()
                    else:
                        prep_stage_warnings.append(f"Unknown accumulation method '{method}', using sum.")
                        series = dep_resampler.sum()
                    series = series.dropna()

                    # Exogenous columns: keep conservative aggregations.
                    if indep_df is not None and len(indep_df.columns):
                        indep_df = indep_df.reindex(series.index.union(indep_df.index)).sort_index().resample(chosen_freq).mean()
                        indep_df = indep_df.reindex(series.index)
                    if event_df is not None and len(event_df.columns):
                        event_df = event_df.reindex(series.index.union(event_df.index)).sort_index().resample(chosen_freq).max()
                        event_df = event_df.reindex(series.index).fillna(0.0)

                    preprocessing["accumulation"] = {
                        "applied": True,
                        "target_freq": chosen_freq,
                        "method": method,
                        "quantity_type": qty,
                    }
                except Exception as e:
                    prep_stage_warnings.append(f"Accumulation failed ({e}); continuing with original frequency.")
                    preprocessing["accumulation"] = {"applied": False, "target_freq": chosen_freq}
            else:
                preprocessing["accumulation"] = {"applied": False, "target_freq": chosen_freq, "reason": "already_at_frequency"}

        # Optional missing-value treatment.
        if bool(body.apply_missing_treatment):
            mv_res = MissingValuesAgent().execute(
                series=series,
                method=body.missing_method,
                period=max(2, int(body.missing_period)),
                zero_as_missing=bool(body.missing_zero_as_missing),
            )
            if mv_res.ok:
                series = mv_res.data["imputed"].dropna().sort_index()
                preprocessing["missing"] = _safe(mv_res.metadata.get("completeness", {}))
                prep_stage_warnings.extend(mv_res.warnings or [])
            else:
                prep_stage_warnings.append("Missing value treatment failed; continuing without imputation.")

        # Optional outlier treatment.
        if bool(body.apply_outlier_treatment):
            out_res = OutlierDetectionAgent().execute(
                series=series,
                methods=(body.outlier_methods or ["iqr", "zscore", "isof"]),
                contamination=0.05,
            )
            if out_res.ok:
                treated = series.copy()
                out_treatment = (body.outlier_treatment or "cap").lower().strip()
                if out_treatment == "cap":
                    fences = out_res.metadata.get("summary", {}).get("fences", {})
                    lo = fences.get("lower_1.5iqr")
                    hi = fences.get("upper_1.5iqr")
                    if lo is not None and hi is not None:
                        treated = treated.clip(lower=float(lo), upper=float(hi))
                elif out_treatment == "remove":
                    is_out = out_res.data.get("is_outlier")
                    if is_out is not None:
                        treated.loc[is_out[is_out].index] = np.nan
                        # Re-impute removed points to keep a complete modeling series.
                        refill = MissingValuesAgent().execute(
                            series=treated,
                            method=(body.missing_method or "linear"),
                            period=max(2, int(body.missing_period)),
                            zero_as_missing=False,
                        )
                        if refill.ok:
                            treated = refill.data["imputed"]
                            prep_stage_warnings.extend(refill.warnings or [])
                # keep -> no change

                series = treated.dropna().sort_index()
                preprocessing["outliers"] = {
                    "treatment": out_treatment,
                    "summary": _safe(out_res.metadata.get("summary", {})),
                }
                prep_stage_warnings.extend(out_res.warnings or [])
            else:
                prep_stage_warnings.append("Outlier treatment failed; continuing with original series.")

        if len(series) < 10:
            raise HTTPException(400, f"Series too short for forecast preparation ({len(series)} obs)")

        # Recompute decomposition + intermittency for this exact selected series.
        decomp_res = DecompositionAgent().execute(series=series)
        interm_res = IntermittencyAgent().execute(series=series)

        Ft = decomp_res.metadata.get("trend_strength_Ft", 0.0) if decomp_res.ok else 0.0
        Fs = decomp_res.metadata.get("seasonal_strength_Fs", 0.0) if decomp_res.ok else 0.0
        d_order = decomp_res.metadata.get("differencing_order", 0) if decomp_res.ok else 0
        period_used = decomp_res.metadata.get("period_used", 12) if decomp_res.ok else 12
        acf_lags = decomp_res.metadata.get("acf_significant_lags") if decomp_res.ok else None
        interm_cls = (
            interm_res.metadata.get("summary", {}).get("classification", "Smooth")
            if interm_res.ok else "Smooth"
        )
        base_rec = _recommend_model(interm_cls, Ft > 0.5, Fs > 0.5, d_order)
        model_rec = _upgrade_model_for_exog(base_rec, bool(indep_cols or event_cols))

        if indep_df is not None and len(indep_df.columns):
            indep_df = indep_df.reindex(series.index).ffill().bfill().fillna(0.0)
        if event_df is not None and len(event_df.columns):
            event_df = event_df.reindex(series.index).fillna(0.0)

        prep_res = ForecastPreparationAgent().execute(
            series=series,
            node_path=node_path if mode == "hierarchy" else None,
            dep_col=dep_col,
            indep_df=indep_df,
            event_df=event_df,
            transform=body.transform,
            scale_method=body.scale_method,
            freq=chosen_freq or sess.get("target_freq") or sess.get("detected_freq"),
            n_holdout=max(0, int(body.n_holdout)),
            horizon=max(1, int(body.horizon)),
            train_ratio=body.train_ratio,
            val_ratio=body.val_ratio,
            rolling_windows=body.rolling_windows,
            add_calendar=body.add_calendar,
            acf_lags=acf_lags,
            model_rec=model_rec,
            interm_cls=interm_cls,
            Ft=Ft,
            Fs=Fs,
            d=d_order,
            period=period_used,
        )
        if not prep_res.ok:
            msg = prep_res.errors[0] if prep_res.errors else "Forecast preparation failed"
            raise HTTPException(500, msg)

        feature_df = prep_res.data["feature_matrix"]
        future_df = prep_res.data["future_frame"]
        split_counts = (
            feature_df["split"].value_counts().to_dict()
            if "split" in feature_df.columns else {}
        )
        split_counts["future"] = int(len(future_df))

        # Candidate model comparison on holdout by MAPE.
        model_comparison: List[Dict[str, Any]] = []
        model_predictions: Dict[str, Dict[str, Any]] = {}
        best_model_name = "Baseline"
        best_test_pred: Optional[np.ndarray] = None
        best_future_pred: Optional[np.ndarray] = None
        allow_negative_forecast = bool(body.allow_negative_forecast)

        def _contains_negative(vals: Optional[np.ndarray]) -> bool:
            if vals is None:
                return False
            arr = np.asarray(vals, dtype=float)
            return bool(np.any(np.isfinite(arr) & (arr < 0)))

        def _clip_non_negative(vals: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if vals is None:
                return None
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
                return arr
            arr = np.where(np.isfinite(arr), np.maximum(arr, 0.0), arr)
            return arr

        sess["forecast_prep_result"] = prep_res
        sess["forecast_feature_df"] = feature_df
        sess["forecast_future_df"] = future_df

        series_profile_safe = _safe(prep_res.metadata.get("series_profile", {}))
        model_params_safe = _safe(prep_res.data.get("model_params", {}))

        export_token = str(uuid.uuid4())[:8]
        output_format = (body.output_format or "excel").lower()
        if output_format in ("excel", "xlsx"):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                feature_df.reset_index().to_excel(writer, index=False, sheet_name="FeatureMatrix")
                future_df.reset_index().to_excel(writer, index=False, sheet_name="FutureFrame")
                pd.DataFrame([model_params_safe]).to_excel(writer, index=False, sheet_name="ModelParams")
                pd.DataFrame([series_profile_safe]).to_excel(writer, index=False, sheet_name="SeriesProfile")
            _downloads[export_token] = buf.getvalue()
            export_ext = "xlsx"
            export_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("feature_matrix.csv", feature_df.reset_index().to_csv(index=False))
                zf.writestr("future_frame.csv", future_df.reset_index().to_csv(index=False))
                zf.writestr("model_params.json", json.dumps(model_params_safe, indent=2))
                zf.writestr("series_profile.json", json.dumps(series_profile_safe, indent=2))
                zf.writestr("report.txt", prep_res.metadata.get("report", ""))
            _downloads[export_token] = zbuf.getvalue()
            export_ext = "zip"
            export_mime = "application/zip"

        y_full = series.dropna().astype(float)
        holdout_n = int(split_counts.get("holdout", 0))
        if holdout_n <= 0:
            holdout_n = max(3, min(max(1, int(body.horizon)), max(3, len(y_full) // 5)))

        if len(y_full) > holdout_n + 6:
            y_train = y_full.iloc[:-holdout_n]
            y_test = y_full.iloc[-holdout_n:]
            future_n = int(len(future_df))

            exog_all = None
            if indep_df is not None or event_df is not None:
                ex_parts = []
                if indep_df is not None and len(indep_df.columns):
                    ex_parts.append(indep_df.copy())
                if event_df is not None and len(event_df.columns):
                    ex_parts.append(event_df.copy())
                if ex_parts:
                    exog_all = pd.concat(ex_parts, axis=1).reindex(y_full.index).ffill().bfill().fillna(0.0)

            def _register_model(name: str, yhat_test: Optional[np.ndarray], yhat_future: Optional[np.ndarray], note: str = ""):
                nonlocal best_model_name, best_test_pred, best_future_pred
                if yhat_test is None:
                    model_comparison.append({"model": name, "mae": None, "rmse": None, "mape": None, "status": f"failed {note}".strip()})
                    return
                yhat_test_arr = np.asarray(yhat_test, dtype=float)
                yhat_future_arr = (np.asarray(yhat_future, dtype=float) if yhat_future is not None else None)
                # Enforce non-negative forecasts unless explicitly allowed.
                if not allow_negative_forecast:
                    yhat_test_arr = _clip_non_negative(yhat_test_arr)
                    yhat_future_arr = _clip_non_negative(yhat_future_arr)

                m = _forecast_metrics(y_test.values, yhat_test_arr)
                row = {"model": name, **m, "status": "ok" if not note else note}
                model_comparison.append(row)
                model_predictions[name] = {
                    "holdout_pred": yhat_test_arr.tolist(),
                    "future_pred": (yhat_future_arr.tolist() if yhat_future_arr is not None else []),
                    "metrics": m,
                    "status": row["status"],
                }
                if m.get("mape") is not None:
                    current_best = min(
                        [r["mape"] for r in model_comparison if r.get("mape") is not None],
                        default=None
                    )
                    if current_best is not None and m["mape"] == current_best:
                        best_model_name = name
                        best_test_pred = yhat_test_arr
                        best_future_pred = yhat_future_arr

            if interm_cls in ("Intermittent", "Lumpy"):
                intermittent_specs = [
                    ("Croston", "classic"),
                    ("SBA", "sba"),
                    ("TSB", "tsb"),
                ]
                for model_name, variant in intermittent_specs:
                    try:
                        yhat_test = _croston_family_forecast(y_train, holdout_n, variant=variant)
                        yhat_future = _croston_family_forecast(y_full, future_n, variant=variant)
                        note = ""
                        if exog_all is not None and exog_all.shape[1] > 0:
                            note = "intermittent baseline (exog not used)"
                        _register_model(model_name, yhat_test, yhat_future, note)
                    except Exception as e:
                        _register_model(model_name, None, None, str(e))

            # ETS
            try:
                if _HAS_STATSMODELS and ExponentialSmoothing is not None:
                    p = int(period_used) if period_used else 0
                    seasonal = "add" if (p > 1 and len(y_train) >= (2 * p)) else None
                    sp = p if seasonal else None

                    def _fit_ets(y, n_steps):
                        """Fit ETS with MLE; fall back to fixed params if MLE diverges."""
                        import warnings as _w
                        model = ExponentialSmoothing(
                            y, trend="add", seasonal=seasonal, seasonal_periods=sp
                        )
                        # Guard: degenerate series (constant or all-zero) → fixed params
                        if float(np.std(y)) < 1e-8:
                            with _w.catch_warnings():
                                _w.simplefilter("ignore")
                                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                                    fit = model.fit(optimized=False, smoothing_level=0.1,
                                                    smoothing_trend=0.01,
                                                    smoothing_seasonal=0.01 if seasonal else None)
                        else:
                            try:
                                with _w.catch_warnings():
                                    _w.simplefilter("ignore")
                                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                                        fit = model.fit(optimized=True)
                                # Reject fit if it produced non-finite parameters
                                if not np.all(np.isfinite(fit.params.values()
                                              if hasattr(fit.params, "values") else list(fit.params))):
                                    raise ValueError("non-finite ETS params")
                            except Exception:
                                with _w.catch_warnings():
                                    _w.simplefilter("ignore")
                                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                                        fit = model.fit(optimized=False, smoothing_level=0.3,
                                                        smoothing_trend=0.05,
                                                        smoothing_seasonal=0.05 if seasonal else None)
                        return np.asarray(fit.forecast(n_steps), dtype=float)

                    yhat_test   = _fit_ets(y_train, holdout_n)
                    yhat_future = _fit_ets(y_full,  future_n)
                    _register_model("ETS", yhat_test, yhat_future)
                else:
                    _register_model("ETS", None, None, "statsmodels unavailable")
            except Exception as e:
                _register_model("ETS", None, None, str(e))

            # ARIMA
            arima_test_pred: Optional[np.ndarray] = None
            arima_future_pred: Optional[np.ndarray] = None
            try:
                if _HAS_STATSMODELS and SARIMAX is not None:
                    d_fit = max(0, min(2, int(d_order)))
                    arima_fit = SARIMAX(y_train, order=(1, d_fit, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    yhat_test = np.asarray(arima_fit.forecast(holdout_n), dtype=float)
                    arima_full = SARIMAX(y_full, order=(1, d_fit, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    yhat_future = np.asarray(arima_full.forecast(future_n), dtype=float)
                    arima_test_pred = yhat_test
                    arima_future_pred = yhat_future
                    _register_model("ARIMA", yhat_test, yhat_future)
                else:
                    _register_model("ARIMA", None, None, "statsmodels unavailable")
            except Exception as e:
                _register_model("ARIMA", None, None, str(e))

            # ARIMAX
            try:
                if _HAS_STATSMODELS and SARIMAX is not None and exog_all is not None and exog_all.shape[1] > 0:
                    d_fit = max(0, min(2, int(d_order)))
                    ex_train = exog_all.iloc[:-holdout_n]
                    ex_test = exog_all.iloc[-holdout_n:]
                    arimax_fit = SARIMAX(y_train, exog=ex_train, order=(1, d_fit, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    yhat_test = np.asarray(arimax_fit.forecast(holdout_n, exog=ex_test), dtype=float)
                    ex_future = pd.DataFrame(0.0, index=future_df.index, columns=exog_all.columns)
                    arimax_full = SARIMAX(y_full, exog=exog_all, order=(1, d_fit, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                    yhat_future = np.asarray(arimax_full.forecast(future_n, exog=ex_future), dtype=float)
                    arimax_note = ""
                    if (not allow_negative_forecast) and (_contains_negative(yhat_test) or _contains_negative(yhat_future)):
                        # Exogenous impact can destabilize sign; ignore exog if it drives negatives.
                        if arima_test_pred is not None and arima_future_pred is not None:
                            yhat_test = arima_test_pred
                            yhat_future = arima_future_pred
                            arimax_note = "exog ignored: negative forecast prevented"
                        else:
                            yhat_test = _clip_non_negative(yhat_test)
                            yhat_future = _clip_non_negative(yhat_future)
                            arimax_note = "exog ignored fallback: clipped non-negative"
                    _register_model("ARIMAX", yhat_test, yhat_future, arimax_note)
                else:
                    _register_model("ARIMAX", None, None, "no exogenous vars")
            except Exception as e:
                _register_model("ARIMAX", None, None, str(e))

            # Linear Regression with lag features.
            try:
                ex_cols = exog_all.columns.tolist() if exog_all is not None else []
                lr_df = pd.DataFrame(index=y_full.index)
                lr_df["y"] = y_full.values
                lr_df["lag1"] = lr_df["y"].shift(1)
                lr_df["lag2"] = lr_df["y"].shift(2)
                p = int(period_used) if period_used and int(period_used) > 1 else 0
                if p > 1:
                    lr_df[f"lag{p}"] = lr_df["y"].shift(p)
                for c in ex_cols:
                    lr_df[f"exog__{c}"] = exog_all[c].values
                lr_df = lr_df.dropna()

                train_cut = y_train.index[-1]
                lr_train = lr_df[lr_df.index <= train_cut]
                lr_test = lr_df[lr_df.index.isin(y_test.index)]
                if len(lr_train) < 5 or len(lr_test) == 0:
                    _register_model("LinearRegression", None, None, "insufficient rows")
                else:
                    x_cols_full = [c for c in lr_df.columns if c != "y"]
                    x_cols_lag_only = [c for c in x_cols_full if not c.startswith("exog__")]

                    def _run_lr(x_cols_used: List[str]):
                        lr_model = LinearRegression()
                        lr_model.fit(lr_train[x_cols_used].values, lr_train["y"].values)
                        yhat_test_local = lr_model.predict(lr_test[x_cols_used].values)

                        # Iterative future forecast.
                        hist = y_full.copy()
                        fut_preds_local = []
                        ex_future = pd.DataFrame(0.0, index=future_df.index, columns=ex_cols) if ex_cols else None
                        for dt in future_df.index:
                            row = {
                                "lag1": float(hist.iloc[-1]) if len(hist) >= 1 else 0.0,
                                "lag2": float(hist.iloc[-2]) if len(hist) >= 2 else float(hist.iloc[-1]) if len(hist) >= 1 else 0.0,
                            }
                            if p > 1:
                                row[f"lag{p}"] = float(hist.iloc[-p]) if len(hist) >= p else float(hist.iloc[-1]) if len(hist) >= 1 else 0.0
                            for c in ex_cols:
                                row[f"exog__{c}"] = float(ex_future.loc[dt, c]) if ex_future is not None else 0.0
                            x_row = np.array([row[col] for col in x_cols_used], dtype=float).reshape(1, -1)
                            pred = float(lr_model.predict(x_row)[0])
                            fut_preds_local.append(pred)
                            hist = pd.concat([hist, pd.Series([pred], index=[dt])])
                        return np.asarray(yhat_test_local, dtype=float), np.asarray(fut_preds_local, dtype=float)

                    yhat_test, yhat_future = _run_lr(x_cols_full)
                    lr_note = ""
                    if (not allow_negative_forecast) and ex_cols and (_contains_negative(yhat_test) or _contains_negative(yhat_future)):
                        yhat_test, yhat_future = _run_lr(x_cols_lag_only)
                        lr_note = "exog ignored: negative forecast prevented"
                    if (not allow_negative_forecast) and (_contains_negative(yhat_test) or _contains_negative(yhat_future)):
                        yhat_test = _clip_non_negative(yhat_test)
                        yhat_future = _clip_non_negative(yhat_future)
                        lr_note = (lr_note + "; clipped non-negative").strip("; ").strip()

                    _register_model("LinearRegression", yhat_test, yhat_future, lr_note)
            except Exception as e:
                _register_model("LinearRegression", None, None, str(e))

            # # FLANN with basis-expanded lag/exogenous features.
            # try:
            #     flann_res = _run_flann_forecast(
            #         y_full=y_full,
            #         y_train=y_train,
            #         y_test=y_test,
            #         future_idx=future_df.index,
            #         exog_all=exog_all,
            #         period_used=period_used,
            #     )
            #     if flann_res is None:
            #         _register_model("FLANN", None, None, "insufficient rows")
            #     else:
            #         flann_note = "basis-expanded lag model"
            #         if exog_all is not None and exog_all.shape[1] > 0:
            #             flann_note += " + exog"
            #         _register_model(
            #             "FLANN",
            #             flann_res.get("holdout_pred"),
            #             flann_res.get("future_pred"),
            #             flann_note,
            #         )
            # except Exception as e:
            #     _register_model("FLANN", None, None, str(e))
            # ── FLANN / RecurrentFLANN / RVFL ────────────────────────────────
            # Shared kwargs drawn from the request body.
            _flann_kwargs = dict(
                y_full=y_full,
                y_train=y_train,
                y_test=y_test,
                future_idx=future_df.index,
                exog_all=exog_all,
                period_used=period_used,
                basis_family=body.flann_basis,
                order=body.flann_order,
                ridge_lambda=body.flann_ridge_lambda,
                recurrent_depth=body.flann_recurrent_depth,
                rvfl_n_hidden=body.rvfl_n_hidden,
                rvfl_activation=body.rvfl_activation,
            )

            # Decide which variants to run.
            variants_to_run = (
                list(FLANN_VARIANTS)            # all three
                if body.run_all_flann_variants
                else [body.flann_variant]       # just the requested one
            )

            for _variant in variants_to_run:
                # Canonical model name shown in the comparison table.
                _model_label = {
                    "flann":           "FLANN",
                    "recurrent_flann": "RecurrentFLANN",
                    "rvfl":            "RVFL",
                }.get(_variant, _variant.upper())

                try:
                    _res = _run_flann_forecast(variant=_variant, **_flann_kwargs)
                    if _res is None:
                        _register_model(_model_label, None, None, "insufficient rows")
                    else:
                        _exog_note = " + exog" if (exog_all is not None
                                                   and exog_all.shape[1] > 0) else ""
                        _basis_note = f"{body.flann_basis}[{body.flann_order}]"
                        _full_note = f"{_basis_note}{_exog_note}"
                        if _variant == "recurrent_flann":
                            _full_note += f" depth={body.flann_recurrent_depth}"
                        elif _variant == "rvfl":
                            _full_note += (
                                f" H={body.rvfl_n_hidden} act={body.rvfl_activation}"
                            )
                        _register_model(
                            _model_label,
                            _res.get("holdout_pred"),
                            _res.get("future_pred"),
                            _full_note,
                        )
                except Exception as _e:
                    _register_model(_model_label, None, None, str(_e))

            if body.enable_combination_models:
                # Collect individual models that have both valid holdout and future preds.
                _exp_holdout_len = len(y_test)
                _exp_future_len = future_n
                print(f"Expected holdout length: {_exp_holdout_len}, future length: {_exp_future_len}")
                _combo_pool = {
                    name: pred
                    for name, pred in model_predictions.items()
                    if (
                        pred.get("metrics", {}).get("mape") is not None
                        and len(pred.get("holdout_pred") or []) == _exp_holdout_len
                        and len(pred.get("future_pred") or []) == _exp_future_len
                    )
                }
                print(f"Models eligible for combination: {list(_combo_pool.keys())}")
                if len(_combo_pool) >= 2:
                    _cnames = list(_combo_pool.keys())
                    _h_mat = np.array([_combo_pool[n]["holdout_pred"] for n in _cnames], dtype=float)
                    _f_mat = np.array([_combo_pool[n]["future_pred"] for n in _cnames], dtype=float)
                    _mapes_arr = np.array([_combo_pool[n]["metrics"]["mape"] for n in _cnames], dtype=float)

                    # 1. Simple Average — equal-weight blend of all eligible models.
                    _register_model(
                        "Combo:SimpleAvg",
                        np.mean(_h_mat, axis=0),
                        np.mean(_f_mat, axis=0),
                    )
                    print("Registered Combo:SimpleAvg")
                    # 2. Inverse-MAPE Weighted Average — better models get higher weight.
                    _safe_mapes = np.where(_mapes_arr > 0, _mapes_arr, 1e-9)
                    _inv_w = 1.0 / _safe_mapes
                    _inv_w = _inv_w / _inv_w.sum()
                    _register_model(
                        "Combo:WeightedAvg",
                        (_inv_w[:, None] * _h_mat).sum(axis=0),
                        (_inv_w[:, None] * _f_mat).sum(axis=0),
                    )
                    print("Registered Combo:WeightedAvg")
                    # 3. Median Ensemble — robust to a single model being an outlier.
                    _register_model(
                        "Combo:Median",
                        np.median(_h_mat, axis=0),
                        np.median(_f_mat, axis=0),
                    )
                    print("Registered Combo:Median")
                    # 4. BestTrio — simple average of the 3 lowest-MAPE individual models.
                    _n_trio = min(3, len(_cnames))
                    if _n_trio >= 2:
                        _top_idx = np.argsort(_mapes_arr)[:_n_trio]
                        _register_model(
                            f"Combo:Top{_n_trio}Avg",
                            np.mean(_h_mat[_top_idx], axis=0),
                            np.mean(_f_mat[_top_idx], axis=0),
                        )
                        print(f"Registered Combo:Top{_n_trio}Avg")
            # ─────────────────────────────────────────────────────────────────────

        # Timeline with split-aware actuals and a baseline model forecast overlay.
        obs_idx = list(feature_df.index)
        fut_idx = list(future_df.index)
        all_idx = obs_idx + fut_idx
        all_labels = [str(x)[:10] for x in all_idx]

        obs_vals = [float(v) if not pd.isna(v) else None for v in series.reindex(obs_idx).values]
        split_obs = feature_df["split"].astype(str).tolist() if "split" in feature_df.columns else (["train"] * len(obs_idx))
        split_all = split_obs + (["future"] * len(fut_idx))

        train_vals = []
        val_vals = []
        test_vals = []
        holdout_vals = []
        for i, sp in enumerate(split_obs):
            v = obs_vals[i]
            train_vals.append(v if sp == "train" else None)
            val_vals.append(v if sp == "validation" else None)
            test_vals.append(v if sp == "test" else None)
            holdout_vals.append(v if sp == "holdout" else None)
        train_vals.extend([None] * len(fut_idx))
        val_vals.extend([None] * len(fut_idx))
        test_vals.extend([None] * len(fut_idx))
        holdout_vals.extend([None] * len(fut_idx))

        # Forecast target dates: holdout + future.
        forecast_targets = [ix for ix, sp in zip(obs_idx, split_obs) if sp == "holdout"] + fut_idx
        non_holdout_vals = [v for v, sp in zip(obs_vals, split_obs) if sp != "holdout" and v is not None]
        holdout_targets = [ix for ix, sp in zip(obs_idx, split_obs) if sp == "holdout"]
        holdout_actual = [v for v, sp in zip(obs_vals, split_obs) if sp == "holdout" and v is not None]
        recommended_candidate = _resolve_recommended_candidate(
            model_rec,
            [row["model"] for row in model_comparison if row.get("mape") is not None]
        )
        selected_model_name = best_model_name if best_model_name in model_predictions else (
            recommended_candidate if recommended_candidate in model_predictions else None
        )

        if (len(y_test) + future_n) and selected_model_name is None and non_holdout_vals:
            last = float(non_holdout_vals[-1])
            drift = 0.0
            if len(non_holdout_vals) > 1 and Ft > 0.5:
                drift = (float(non_holdout_vals[-1]) - float(non_holdout_vals[0])) / (len(non_holdout_vals) - 1)

            season_template: Optional[np.ndarray] = None
            p = int(period_used) if period_used else 0
            if Fs > 0.5 and p > 1 and len(non_holdout_vals) >= p:
                season_template = np.array(non_holdout_vals[-p:], dtype=float)
                season_template = season_template - season_template.mean()

            baseline_holdout: List[float] = []
            baseline_future: List[float] = []
            baseline_holdout_n = int(len(y_test))
            baseline_total_n = baseline_holdout_n + int(future_n)
            for i in range(baseline_total_n):
                pred = last + drift * (i + 1)
                if season_template is not None and len(season_template):
                    pred += float(season_template[i % len(season_template)])
                if min(non_holdout_vals) >= 0:
                    pred = max(0.0, pred)
                pred = float(round(pred, 6))
                if i < baseline_holdout_n:
                    baseline_holdout.append(pred)
                else:
                    baseline_future.append(pred)
            m = _forecast_metrics(y_test.values, np.asarray(baseline_holdout, dtype=float))
            model_comparison.append({"model": "Baseline", **m, "status": "fallback"})
            model_predictions["Baseline"] = {
                "holdout_pred": baseline_holdout,
                "future_pred": baseline_future,
                "metrics": m,
                "status": "fallback",
            }
            if selected_model_name is None:
                selected_model_name = "Baseline"

        def _build_model_output(
            model_name: str,
            holdout_pred_vals: List[float],
            future_pred_vals: List[float],
            status: str,
            metrics: Dict[str, Optional[float]],
        ) -> Dict[str, Any]:
            forecast_map: Dict[str, float] = {}
            for dt, pred in zip(holdout_targets, holdout_pred_vals):
                forecast_map[str(dt)[:10]] = float(round(float(pred), 6))
            for dt, pred in zip(fut_idx, future_pred_vals):
                forecast_map[str(dt)[:10]] = float(round(float(pred), 6))

            forecast_path = [forecast_map.get(lbl) for lbl in all_labels]
            future_only = [None] * len(obs_idx) + [
                float(round(float(pred), 4)) if pred is not None else None for pred in future_pred_vals
            ]

            err_sigma_local = 0.0
            if holdout_actual and holdout_pred_vals and len(holdout_actual) == len(holdout_pred_vals):
                errs = [float(a) - float(p) for a, p in zip(holdout_actual, holdout_pred_vals)]
                if len(errs) >= 2:
                    err_sigma_local = float(np.std(errs, ddof=1))
            if err_sigma_local <= 0:
                base_vals = np.array([v for v in non_holdout_vals if v is not None], dtype=float)
                if len(base_vals) >= 3:
                    diff = np.diff(base_vals)
                    err_sigma_local = float(np.std(diff, ddof=1)) if len(diff) > 1 else float(np.std(base_vals, ddof=1) * 0.1)
                elif len(base_vals) >= 2:
                    err_sigma_local = float(np.std(base_vals, ddof=1) * 0.1)
                else:
                    err_sigma_local = 1.0

            z_95_local = 1.96
            lower = [None] * len(obs_idx)
            upper = [None] * len(obs_idx)
            for h, pred in enumerate(future_pred_vals, start=1):
                if pred is None:
                    lower.append(None)
                    upper.append(None)
                    continue
                band = z_95_local * err_sigma_local * (h ** 0.5)
                lo = float(pred) - band
                hi = float(pred) + band
                if non_holdout_vals and min(non_holdout_vals) >= 0:
                    lo = max(0.0, lo)
                lower.append(round(float(lo), 4))
                upper.append(round(float(hi), 4))

            return {
                "model": model_name,
                "status": status,
                "metrics": _safe(metrics),
                "forecast_path": [round(float(v), 4) if v is not None else None for v in forecast_path],
                "future_forecast": future_only,
                "future_lower_95": lower,
                "future_upper_95": upper,
            }

        model_outputs: Dict[str, Dict[str, Any]] = {}
        for model_name, pred_info in model_predictions.items():
            model_outputs[model_name] = _build_model_output(
                model_name=model_name,
                holdout_pred_vals=pred_info.get("holdout_pred", []),
                future_pred_vals=pred_info.get("future_pred", []),
                status=str(pred_info.get("status", "ok")),
                metrics=pred_info.get("metrics", {}),
            )

        selected_output = model_outputs.get(selected_model_name or "")
        model_forecast = selected_output.get("forecast_path", []) if selected_output else [None] * len(all_labels)
        observed_all = obs_vals + ([None] * len(fut_idx))

        holdout_start_label = None
        for ix, sp in zip(obs_idx, split_obs):
            if sp == "holdout":
                holdout_start_label = str(ix)[:10]
                break
        horizon_start_label = str(fut_idx[0])[:10] if len(fut_idx) else None

        future_forecast = selected_output.get("future_forecast", []) if selected_output else ([None] * len(all_labels))
        future_lower = selected_output.get("future_lower_95", []) if selected_output else ([None] * len(all_labels))
        future_upper = selected_output.get("future_upper_95", []) if selected_output else ([None] * len(all_labels))

        return {
            "ok": True,
            "mode": mode,
            "dep_col": dep_col,
            "node_path": node_path,
            "recommended_model": model_rec,
            "recommended_candidate_model": recommended_candidate,
            "selected_model": selected_model_name,
            "available_models": [row["model"] for row in model_comparison if row["model"] in model_outputs],
            "model_outputs": _safe(model_outputs),
            "timeline": {
                "labels": all_labels,
                "observed": [round(float(v), 4) if v is not None else None for v in observed_all],
                "train": [round(float(v), 4) if v is not None else None for v in train_vals],
                "validation": [round(float(v), 4) if v is not None else None for v in val_vals],
                "test": [round(float(v), 4) if v is not None else None for v in test_vals],
                "holdout": [round(float(v), 4) if v is not None else None for v in holdout_vals],
                "model_forecast": [round(float(v), 4) if v is not None else None for v in model_forecast],
                "future_forecast": future_forecast,
                "future_lower_95": future_lower,
                "future_upper_95": future_upper,
                "split_labels": split_all,
                "selected_model": selected_model_name,
                "holdout_start": holdout_start_label,
                "horizon_start": horizon_start_label,
                "horizon_n": int(len(fut_idx)),
                "confidence_level": 0.95,
            },
            "series_profile": series_profile_safe,
            "model_params": model_params_safe,
            "best_model": best_model_name,
            "model_comparison": model_comparison,
            "split_counts": _safe(split_counts),
            "feature_columns": list(feature_df.columns),
            "future_columns": list(future_df.columns),
            "feature_preview": _df_to_records(feature_df, 20),
            "future_preview": _df_to_records(future_df, 20),
            "download_token": export_token,
            "download_ext": export_ext,
            "download_mime": export_mime,
            "report": prep_res.metadata.get("report", ""),
            "preprocessing": _safe(preprocessing),
            "warnings": (prep_stage_warnings + (prep_res.warnings or [])),
            "allow_negative_forecast": allow_negative_forecast,
        }


    # ── Level Stability Analysis ──────────────────────────────────────────────

    class LevelStabilityReq(BaseModel):
        token: str
        level_col: str                     # the hierarchy column that defines series at this level
        parent_path: Dict[str, str] = {}   # filter to a sub-tree (empty = all)
        value_col: Optional[str] = None
        agg_method: str = "sum"
        interval: str = "M"               # D=daily, W=weekly, M=monthly, Q=quarterly,
                                          # 6M=semi-annual, Y=annual, native=no resampling
        max_series: int = 30              # cap displayed series

    @app.post("/api/level-stability")
    async def level_stability(body: LevelStabilityReq):
        """
        Returns all child series at `level_col` (optionally filtered by parent_path),
        each resampled to `interval`. Includes:
          - per-series time-indexed values  (for superimposed plot)
          - seasonal profile per series     (avg value per calendar sub-period)
          - stability stats                 (CV, year-over-year change, range)
          - cross-series stats at each timestamp (mean, std, min, max, CV envelope)
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

        # ── 1. Filter to parent path ──────────────────────────────────────────
        filtered = df.copy()
        for col, val in body.parent_path.items():
            if col in filtered.columns:
                filtered = filtered[filtered[col].astype(str) == str(val)]
        if len(filtered) == 0:
            raise HTTPException(404, "No data for parent path")

        # ── 2. Group by level_col value → one series per child ───────────────
        # Aggregate any remaining hierarchy cols below level_col
        level_col_idx = hier_cols.index(body.level_col) if body.level_col in hier_cols else 0
        child_hier    = hier_cols[level_col_idx + 1:] if level_col_idx + 1 < len(hier_cols) else []
        groupby_cols  = [body.level_col]

        child_series: Dict[str, pd.Series] = {}
        for child_val, grp in filtered.groupby(body.level_col):
            child_val_str = str(child_val)
            if child_hier:
                s = grp.groupby(level=0)[val_col].agg(body.agg_method)
            else:
                s = grp.groupby(level=0)[val_col].agg(body.agg_method)
            child_series[child_val_str] = s.sort_index().dropna()

        # cap
        series_names = sorted(child_series.keys())[:body.max_series]

        # ── 3. Resample each series to target interval ─────────────────────────
        INTERVAL_MAP = {
            "D":  ("D",  "Day"),
            "W":  ("W",  "Week"),
            "M":  ("MS", "Month"),
            "Q":  ("QS", "Quarter"),
            "6M": ("6MS","Semi-year"),
            "Y":  ("YS", "Year"),
            "native": (None, "Native"),
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

        # ── 4. Build unified time axis (union of all indices) ─────────────────
        all_dates: set = set()
        for s in resampled.values():
            all_dates.update(s.index.tolist())
        sorted_dates = sorted(all_dates)
        date_labels  = [str(d)[:10] for d in sorted_dates]

        # ── 5. Series data for superimposed plot ──────────────────────────────
        series_data: Dict[str, List] = {}
        for name, s in resampled.items():
            s_reindexed = s.reindex(sorted_dates)
            series_data[name] = [
                round(float(v), 4) if pd.notna(v) else None
                for v in s_reindexed.values
            ]

        # ── 6. Cross-series envelope stats at each timestamp ─────────────────
        matrix = np.full((len(sorted_dates), len(series_names)), np.nan)
        for j, name in enumerate(series_names):
            s = resampled[name].reindex(sorted_dates)
            matrix[:, j] = s.values

        env_mean = [round(float(np.nanmean(matrix[i])), 4) if not np.all(np.isnan(matrix[i])) else None for i in range(len(sorted_dates))]
        env_std  = [round(float(np.nanstd(matrix[i])),  4) if not np.all(np.isnan(matrix[i])) else None for i in range(len(sorted_dates))]
        env_min  = [round(float(np.nanmin(matrix[i])),  4) if not np.all(np.isnan(matrix[i])) else None for i in range(len(sorted_dates))]
        env_max  = [round(float(np.nanmax(matrix[i])),  4) if not np.all(np.isnan(matrix[i])) else None for i in range(len(sorted_dates))]
        env_cv   = [
            round(float(np.nanstd(matrix[i]) / (abs(np.nanmean(matrix[i])) + 1e-12)), 4)
            if env_mean[i] else None
            for i in range(len(sorted_dates))
        ]

        # ── 7. Seasonal profiles ──────────────────────────────────────────────
        # Season = sub-period label (month name, quarter, day-of-week, etc.)
        seasonal_profiles: Dict[str, Dict] = {}

        def get_season_label(dt: pd.Timestamp, interval: str) -> str:
            if interval == "D":
                return dt.strftime("%a")           # Mon, Tue …
            elif interval == "W":
                return f"W{dt.isocalendar().week:02d}"
            elif interval == "M":
                return dt.strftime("%b")            # Jan, Feb …
            elif interval == "Q":
                return f"Q{dt.quarter}"
            elif interval == "6M":
                return "H1" if dt.month <= 6 else "H2"
            elif interval == "Y":
                return str(dt.year)
            else:
                return str(dt)[:7]

        # Collect season → year → value for each series
        season_order_map = {
            "D": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            "W": [f"W{i:02d}" for i in range(1, 54)],
            "M": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            "Q": ["Q1","Q2","Q3","Q4"],
            "6M": ["H1","H2"],
            "Y": [],   # dynamically ordered
            "native": [],
        }
        ordered_seasons = season_order_map.get(body.interval, [])

        for name, s in resampled.items():
            season_vals: Dict[str, List[float]] = {}
            for ts, val in s.items():
                if pd.isna(val):
                    continue
                sl = get_season_label(ts, body.interval)
                season_vals.setdefault(sl, []).append(float(val))
            # compute mean per season
            season_avg = {sl: round(float(np.mean(vals)), 4) for sl, vals in season_vals.items()}
            # stability: coefficient of variation per season
            season_cv  = {
                sl: round(float(np.std(vals) / (abs(np.mean(vals)) + 1e-12)), 4)
                for sl, vals in season_vals.items() if len(vals) > 1
            }
            seasonal_profiles[name] = {
                "season_avg": season_avg,
                "season_cv":  season_cv,
                "seasons_present": list(season_avg.keys()),
            }

        # Build combined seasonal avg across all series for each season
        all_seasons: set = set()
        for p in seasonal_profiles.values():
            all_seasons.update(p["season_avg"].keys())
        if ordered_seasons:
            combined_season_labels = [s for s in ordered_seasons if s in all_seasons]
            extra = sorted(all_seasons - set(ordered_seasons))
            combined_season_labels += extra
        else:
            combined_season_labels = sorted(all_seasons)

        combined_season_avg: Dict[str, float] = {}
        combined_season_cv:  Dict[str, float] = {}
        for sl in combined_season_labels:
            vals = []
            for p in seasonal_profiles.values():
                if sl in p["season_avg"]:
                    vals.append(p["season_avg"][sl])
            if vals:
                combined_season_avg[sl] = round(float(np.mean(vals)), 4)
                combined_season_cv[sl]  = round(float(np.std(vals) / (abs(np.mean(vals)) + 1e-12)), 4)

        # ── 8. Per-series stability stats ─────────────────────────────────────
        stability_stats: Dict[str, Dict] = {}
        for name, s in resampled.items():
            vals = s.dropna().values.astype(float)
            if len(vals) < 2:
                continue
            mean_v = float(np.mean(vals))
            std_v  = float(np.std(vals))
            cv     = std_v / (abs(mean_v) + 1e-12)
            # Year-over-year change if enough data
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
                "std":   round(std_v, 4),
                "cv":    round(cv, 4),
                "min":   round(float(np.min(vals)), 4),
                "max":   round(float(np.max(vals)), 4),
                "range": round(float(np.max(vals) - np.min(vals)), 4),
                "n_obs": int(len(vals)),
                "avg_yoy_pct_change": yoy,
                "stability": "stable" if cv < 0.2 else "moderate" if cv < 0.5 else "volatile",
            }

        # ── 9. YoY matrix (series × year) ────────────────────────────────────
        yoy_matrix: Dict[str, Dict[str, float]] = {}
        all_years: set = set()
        for name, s in resampled.items():
            yearly = s.resample("YS").sum() if body.agg_method == "sum" else s.resample("YS").mean()
            yearly = yearly.dropna()
            row = {str(ts.year): round(float(v), 2) for ts, v in yearly.items()}
            yoy_matrix[name] = row
            all_years.update(row.keys())
        yoy_years = sorted(all_years)

        return {
            "level_col":    body.level_col,
            "series_names": series_names,
            "n_series":     len(series_names),
            "interval":     body.interval,
            "interval_label": freq_label,
            "val_col":      val_col,
            "date_labels":  date_labels,
            "series_data":  series_data,
            "envelope": {
                "mean": env_mean, "std": env_std,
                "min":  env_min,  "max": env_max, "cv": env_cv,
            },
            "seasonal_profiles":       _safe(seasonal_profiles),
            "combined_season_labels":  combined_season_labels,
            "combined_season_avg":     _safe(combined_season_avg),
            "combined_season_cv":      _safe(combined_season_cv),
            "stability_stats":         _safe(stability_stats),
            "yoy_matrix":              _safe(yoy_matrix),
            "yoy_years":               yoy_years,
            "parent_path":             body.parent_path,
        }

    # ── Download ──────────────────────────────────────────────────────────────

    @app.get("/api/download/{token}")
    async def download(token: str, ext: str = "csv"):
        data = _downloads.get(token)
        if not data:
            raise HTTPException(404, "Download token expired or not found")
        if ext == "xlsx":
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif ext == "zip":
            mime = "application/zip"
        else:
            mime = "text/csv"
        return Response(
            content=data,
            media_type=mime,
            headers={"Content-Disposition": f"attachment; filename=prepared_data.{ext}"},
        )

    # ── Serve SPA ─────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def root():
        base_dir = Path(__file__).parent
        candidate_paths = [
            base_dir / "ui" / "index.html",
            base_dir / "index.html",
        ]
        for html_path in candidate_paths:
            if html_path.exists():
                return HTMLResponse(html_path.read_text(encoding="utf-8"))
        return HTMLResponse(
            "<h1>TemporalMind UI not found. Place index.html in ./ui/ or project root.</h1>",
            status_code=404,
        )

  
    async def health():
        return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import uvicorn
        log.info("Starting TemporalMind server on http://localhost:8080")
        uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
    except ImportError:
        
        log.error("uvicorn not installed. Run: pip install uvicorn")
