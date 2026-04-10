from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agents.flann_family import run_flann_family_forecast


def _series_stats(s: pd.Series) -> dict:
    s_clean = s.dropna()
    return {
        "n": int(len(s)),
        "n_missing": int(s.isna().sum()),
        "mean": round(float(s_clean.mean()), 4) if len(s_clean) else None,
        "std": round(float(s_clean.std()), 4) if len(s_clean) > 1 else None,
        "min": round(float(s_clean.min()), 4) if len(s_clean) else None,
        "max": round(float(s_clean.max()), 4) if len(s_clean) else None,
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


def _upgrade_model_for_exog(base_rec: str, has_exog: bool) -> str:
    if not has_exog:
        return base_rec
    mapping = {
        "ETS / ARIMA": "ARIMAX / LightGBM (with exog)",
        "ARIMA(p,d,q)": "ARIMAX (with exog)",
        "ARIMA(p,1,q)": "ARIMAX(p,1,q) + exog",
        "Holt-Winters / SARIMA": "SARIMAX (with exog)",
        "ETS(A,N,A) / SARIMA": "SARIMAX (with exog)",
        "Holt / ARIMA(p,1,0)": "ARIMAX + exog",
        "FLANN": "FLANN + exog",
        "RecurrentFLANN": "RecurrentFLANN + exog",
        "RVFL": "RVFL + exog",
        "Croston / SBA": "Croston + event regressors",
        "TSB / Zero-Inflated": "TSB + event regressors",
    }
    for k, v in mapping.items():
        if k in base_rec:
            return v
    return base_rec + " + exog"


def _resolve_recommended_candidate(model_rec: str, available_models: List[str]) -> Optional[str]:
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
        (("rvfl",), ["RVFL", "FLANN"]),
        (("flann",), ["FLANN", "RecurrentFLANN", "RVFL"]),
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
