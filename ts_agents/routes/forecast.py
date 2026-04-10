"""
routes/forecast.py  –  Per-series forecast preparation endpoint
================================================================
Routes
------
POST /api/forecast-prepare  – per-series forecast preparation (model params + future frame)
"""

from __future__ import annotations

import io
import uuid
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAS_STATSMODELS = True
except Exception:
    ExponentialSmoothing = None
    SARIMAX = None
    _HAS_STATSMODELS = False

from agents.decomposition_agent import DecompositionAgent
from agents.forecast_preparation_agent import ForecastPreparationAgent
from agents.intermittency_agent import IntermittencyAgent
from agents.interval_advisor_agent import IntervalAdvisorAgent
from agents.missing_values_agent import MissingValuesAgent
from agents.outlier_detection_agent import OutlierDetectionAgent
from agents.flann_family import SUPPORTED_VARIANTS as FLANN_VARIANTS
from core.runtime import DOWNLOADS as _downloads, SESSIONS as _sessions
from utils.api_helper import (
    _croston_family_forecast,
    _df_to_records,
    _forecast_metrics,
    _hierarchy_session_frame,
    _first_session_frame,
    _recommend_model,
    _resolve_recommended_candidate,
    _run_flann_forecast,
    _safe,
    _upgrade_model_for_exog,
)

router = APIRouter()


# ── Request model ──────────────────────────────────────────────────────────────

class ForecastPrepareReq(BaseModel):
    token:                    str
    mode:                     str   = "single"       # single | hierarchy
    dep_col:                  Optional[str] = None
    node_path:                Dict[str, str] = Field(default_factory=dict)
    interval_mode:            str   = "session"      # session | advisor | manual
    target_freq:              Optional[str] = None
    quantity_type:            str   = "flow"          # flow | stock | rate
    accumulation_method:      str   = "auto"
    transform:                str   = "auto"
    scale_method:             str   = "minmax"
    apply_missing_treatment:  bool  = True
    missing_method:           str   = "auto"
    missing_period:           int   = 7
    missing_zero_as_missing:  bool  = False
    apply_outlier_treatment:  bool  = True
    outlier_methods:          List[str] = Field(default_factory=lambda: ["iqr", "zscore", "isof"])
    outlier_treatment:        str   = "cap"           # cap | remove | keep
    rolling_windows:          List[int] = Field(default_factory=lambda: [7, 14, 28])
    add_calendar:             bool  = True
    train_ratio:              float = 0.70
    val_ratio:                float = 0.15
    horizon:                  int   = 13
    n_holdout:                int   = 0
    output_format:            str   = "excel"         # excel | csv (zip bundle)
    flann_variant:            str   = "flann"
    flann_basis:              str   = "mixed"
    flann_order:              int   = 3
    flann_ridge_lambda:       float = 1e-2
    flann_recurrent_depth:    int   = 1
    rvfl_n_hidden:            int   = 64
    rvfl_activation:          str   = "sigmoid"
    run_all_flann_variants:   bool  = False
    enable_combination_models:bool  = False
    allow_negative_forecast:  bool  = False


# ── Route ──────────────────────────────────────────────────────────────────────

@router.post("/api/forecast-prepare")
async def forecast_prepare(body: ForecastPrepareReq):
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404, "Session not found")

    hier_cols  = sess.get("hierarchy_cols", [])
    mode       = body.mode if body.mode in ("single", "hierarchy") else "single"
    node_path  = {k: str(v) for k, v in (body.node_path or {}).items() if v is not None and str(v) != ""}

    work_df = _hierarchy_session_frame(sess) if mode == "hierarchy" else _first_session_frame(sess)
    if work_df is None or not isinstance(work_df, pd.DataFrame):
        raise HTTPException(400, "No prepared frame available in session")

    dep_cols = sess.get("dependent_cols") or sess.get("value_cols", [])
    dep_col  = body.dep_col or (dep_cols[0] if dep_cols else None)
    if not dep_col or dep_col not in work_df.columns:
        raise HTTPException(400, f"Dependent column not found: {dep_col}")

    indep_cols = [c for c in sess.get("independent_cols", []) if c in work_df.columns]
    event_cols = [c for c in sess.get("event_cols", [])       if c in work_df.columns]

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

    remaining_hier = [c for c in hier_cols if c not in node_path] if mode == "hierarchy" else []
    if remaining_hier:
        agg_spec: Dict[str, str] = {dep_col: "sum"}
        for c in indep_cols: agg_spec[c] = "mean"
        for c in event_cols: agg_spec[c] = "max"
        grouped   = source_df.groupby(level=0).agg(agg_spec).sort_index()
        series    = grouped[dep_col].dropna()
        indep_df  = grouped[indep_cols] if indep_cols else None
        event_df  = grouped[event_cols] if event_cols else None
    else:
        series   = source_df[dep_col].dropna().sort_index()
        indep_df = source_df[indep_cols].reindex(series.index)      if indep_cols else None
        event_df = source_df[event_cols].reindex(series.index).fillna(0) if event_cols else None

    prep_stage_warnings: List[str] = []
    preprocessing: Dict[str, Any] = {}

    # ── Optional interval selection ───────────────────────────────────────────
    chosen_freq   = sess.get("target_freq") or sess.get("detected_freq")
    interval_mode = (body.interval_mode or "session").lower().strip()
    if interval_mode == "manual" and body.target_freq:
        chosen_freq = str(body.target_freq).strip()
    elif interval_mode == "advisor":
        advisor_res = IntervalAdvisorAgent().execute(
            series=series, native_freq=sess.get("detected_freq"), top_n=3
        )
        if advisor_res.ok:
            chosen_freq = advisor_res.metadata.get("best_interval") or chosen_freq
            preprocessing["interval_advisor"] = {
                "best_interval": advisor_res.metadata.get("best_interval"),
                "best_alias":    advisor_res.metadata.get("best_alias"),
            }
            prep_stage_warnings.extend(advisor_res.warnings or [])
        else:
            prep_stage_warnings.append("Interval advisor failed; falling back to session frequency.")

    # ── Optional accumulation ─────────────────────────────────────────────────
    if chosen_freq and isinstance(series.index, pd.DatetimeIndex):
        inferred = None
        try:
            inferred = pd.infer_freq(series.index)
        except Exception:
            pass
        if inferred != chosen_freq:
            method = (body.accumulation_method or "auto").lower()
            qty    = (body.quantity_type or "flow").lower()
            if method == "auto":
                method = {"flow": "sum", "stock": "last", "rate": "mean"}.get(qty, "sum")
            try:
                dep_resampler = series.sort_index().resample(chosen_freq)
                method_map = {
                    "sum": dep_resampler.sum, "mean": dep_resampler.mean,
                    "median": dep_resampler.median, "last": dep_resampler.last,
                    "first": dep_resampler.first, "max": dep_resampler.max,
                    "min": dep_resampler.min,
                }
                if method in method_map:
                    series = method_map[method]()
                else:
                    prep_stage_warnings.append(f"Unknown accumulation method '{method}', using sum.")
                    series = dep_resampler.sum()
                series = series.dropna()

                if indep_df is not None and len(indep_df.columns):
                    indep_df = indep_df.reindex(series.index.union(indep_df.index)).sort_index().resample(chosen_freq).mean().reindex(series.index)
                if event_df is not None and len(event_df.columns):
                    event_df = event_df.reindex(series.index.union(event_df.index)).sort_index().resample(chosen_freq).max().reindex(series.index).fillna(0.0)

                preprocessing["accumulation"] = {"applied": True, "target_freq": chosen_freq, "method": method, "quantity_type": qty}
            except Exception as e:
                prep_stage_warnings.append(f"Accumulation failed ({e}); continuing with original frequency.")
                preprocessing["accumulation"] = {"applied": False, "target_freq": chosen_freq}
        else:
            preprocessing["accumulation"] = {"applied": False, "target_freq": chosen_freq, "reason": "already_at_frequency"}

    # ── Optional missing-value treatment ─────────────────────────────────────
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

    # ── Optional outlier treatment ────────────────────────────────────────────
    if bool(body.apply_outlier_treatment):
        out_res = OutlierDetectionAgent().execute(
            series=series, methods=(body.outlier_methods or ["iqr", "zscore", "isof"]), contamination=0.05
        )
        if out_res.ok:
            treated       = series.copy()
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
                    refill = MissingValuesAgent().execute(
                        series=treated,
                        method=(body.missing_method or "linear"),
                        period=max(2, int(body.missing_period)),
                        zero_as_missing=False,
                    )
                    if refill.ok:
                        treated = refill.data["imputed"]
                        prep_stage_warnings.extend(refill.warnings or [])
            series = treated.dropna().sort_index()
            preprocessing["outliers"] = {"treatment": out_treatment, "summary": _safe(out_res.metadata.get("summary", {}))}
            prep_stage_warnings.extend(out_res.warnings or [])
        else:
            prep_stage_warnings.append("Outlier treatment failed; continuing with original series.")

    if len(series) < 10:
        raise HTTPException(400, f"Series too short for forecast preparation ({len(series)} obs)")

    # ── Decomposition + intermittency ─────────────────────────────────────────
    decomp_res  = DecompositionAgent().execute(series=series)
    interm_res  = IntermittencyAgent().execute(series=series)

    Ft          = decomp_res.metadata.get("trend_strength_Ft",    0.0) if decomp_res.ok else 0.0
    Fs          = decomp_res.metadata.get("seasonal_strength_Fs", 0.0) if decomp_res.ok else 0.0
    d_order     = decomp_res.metadata.get("differencing_order",   0)   if decomp_res.ok else 0
    period_used = decomp_res.metadata.get("period_used",          12)  if decomp_res.ok else 12
    acf_lags    = decomp_res.metadata.get("acf_significant_lags")      if decomp_res.ok else None
    interm_cls  = interm_res.metadata.get("summary", {}).get("classification", "Smooth") if interm_res.ok else "Smooth"
    base_rec    = _recommend_model(interm_cls, Ft > 0.5, Fs > 0.5, d_order)
    model_rec   = _upgrade_model_for_exog(base_rec, bool(indep_cols or event_cols))

    if indep_df is not None and len(indep_df.columns):
        indep_df = indep_df.reindex(series.index).ffill().bfill().fillna(0.0)
    if event_df is not None and len(event_df.columns):
        event_df = event_df.reindex(series.index).fillna(0.0)

    # ── ForecastPreparationAgent ──────────────────────────────────────────────
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
        Ft=Ft, Fs=Fs, d=d_order, period=period_used,
    )
    if not prep_res.ok:
        raise HTTPException(500, prep_res.errors[0] if prep_res.errors else "Forecast preparation failed")

    feature_df  = prep_res.data["feature_matrix"]
    future_df   = prep_res.data["future_frame"]
    split_counts = feature_df["split"].value_counts().to_dict() if "split" in feature_df.columns else {}
    split_counts["future"] = int(len(future_df))

    sess["forecast_prep_result"] = prep_res
    sess["forecast_feature_df"]  = feature_df
    sess["forecast_future_df"]   = future_df

    series_profile_safe = _safe(prep_res.metadata.get("series_profile", {}))
    model_params_safe   = _safe(prep_res.data.get("model_params", {}))

    # ── Export file ───────────────────────────────────────────────────────────
    export_token  = str(uuid.uuid4())[:8]
    output_format = (body.output_format or "excel").lower()
    if output_format in ("excel", "xlsx"):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            feature_df.reset_index().to_excel(writer, index=False, sheet_name="FeatureMatrix")
            future_df.reset_index().to_excel(writer, index=False, sheet_name="FutureFrame")
            pd.DataFrame([model_params_safe]).to_excel(writer, index=False, sheet_name="ModelParams")
            pd.DataFrame([series_profile_safe]).to_excel(writer, index=False, sheet_name="SeriesProfile")
        _downloads[export_token] = buf.getvalue()
        export_ext  = "xlsx"
        export_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("feature_matrix.csv", feature_df.reset_index().to_csv(index=False))
            zf.writestr("future_frame.csv",   future_df.reset_index().to_csv(index=False))
            zf.writestr("model_params.json",  str(model_params_safe))
            zf.writestr("series_profile.json",str(series_profile_safe))
            zf.writestr("report.txt",          prep_res.metadata.get("report", ""))
        _downloads[export_token] = zbuf.getvalue()
        export_ext  = "zip"
        export_mime = "application/zip"

    # ── Model comparison on holdout ───────────────────────────────────────────
    model_comparison:  List[Dict[str, Any]] = []
    model_predictions: Dict[str, Dict[str, Any]] = {}
    best_model_name  = "Baseline"
    best_test_pred:   Optional[np.ndarray] = None
    best_future_pred: Optional[np.ndarray] = None
    allow_negative   = bool(body.allow_negative_forecast)

    def _contains_negative(vals):
        if vals is None: return False
        arr = np.asarray(vals, dtype=float)
        return bool(np.any(np.isfinite(arr) & (arr < 0)))

    def _clip_non_negative(vals):
        if vals is None: return None
        arr = np.asarray(vals, dtype=float)
        return np.where(np.isfinite(arr), np.maximum(arr, 0.0), arr) if arr.size else arr

    y_full    = series.dropna().astype(float)
    holdout_n = int(split_counts.get("holdout", 0))
    if holdout_n <= 0:
        holdout_n = max(3, min(max(1, int(body.horizon)), max(3, len(y_full) // 5)))

    def _register_model(name: str, yhat_test, yhat_future, note: str = ""):
        nonlocal best_model_name, best_test_pred, best_future_pred
        if yhat_test is None:
            model_comparison.append({"model": name, "mae": None, "rmse": None, "mape": None, "status": f"failed {note}".strip()})
            return
        yhat_test_arr   = np.asarray(yhat_test,   dtype=float)
        yhat_future_arr = np.asarray(yhat_future,  dtype=float) if yhat_future is not None else None
        if not allow_negative:
            yhat_test_arr   = _clip_non_negative(yhat_test_arr)
            yhat_future_arr = _clip_non_negative(yhat_future_arr)
        m   = _forecast_metrics(y_test.values, yhat_test_arr)
        row = {"model": name, **m, "status": "ok" if not note else note}
        model_comparison.append(row)
        model_predictions[name] = {
            "holdout_pred": yhat_test_arr.tolist(),
            "future_pred":  yhat_future_arr.tolist() if yhat_future_arr is not None else [],
            "metrics":      m,
            "status":       row["status"],
        }
        if m.get("mape") is not None:
            current_best = min([r["mape"] for r in model_comparison if r.get("mape") is not None], default=None)
            if current_best is not None and m["mape"] == current_best:
                best_model_name  = name
                best_test_pred   = yhat_test_arr
                best_future_pred = yhat_future_arr

    if len(y_full) > holdout_n + 6:
        y_train  = y_full.iloc[:-holdout_n]
        y_test   = y_full.iloc[-holdout_n:]
        future_n = int(len(future_df))

        exog_all = None
        if indep_df is not None or event_df is not None:
            ex_parts = []
            if indep_df is not None and len(indep_df.columns): ex_parts.append(indep_df.copy())
            if event_df is not None and len(event_df.columns): ex_parts.append(event_df.copy())
            if ex_parts:
                exog_all = pd.concat(ex_parts, axis=1).reindex(y_full.index).ffill().bfill().fillna(0.0)

        # Croston family for intermittent series
        if interm_cls in ("Intermittent", "Lumpy"):
            for model_name, variant in [("Croston", "classic"), ("SBA", "sba"), ("TSB", "tsb")]:
                try:
                    yhat_test   = _croston_family_forecast(y_train, holdout_n, variant=variant)
                    yhat_future = _croston_family_forecast(y_full,  future_n,  variant=variant)
                    note = "intermittent baseline (exog not used)" if exog_all is not None and exog_all.shape[1] > 0 else ""
                    _register_model(model_name, yhat_test, yhat_future, note)
                except Exception as e:
                    _register_model(model_name, None, None, str(e))

        # ETS
        try:
            if _HAS_STATSMODELS and ExponentialSmoothing is not None:
                import warnings as _w
                p        = int(period_used) if period_used else 0
                seasonal = "add" if (p > 1 and len(y_train) >= (2 * p)) else None
                sp       = p if seasonal else None

                def _fit_ets(y, n_steps):
                    model = ExponentialSmoothing(y, trend="add", seasonal=seasonal, seasonal_periods=sp)
                    if float(np.std(y)) < 1e-8:
                        with _w.catch_warnings():
                            _w.simplefilter("ignore")
                            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                                fit = model.fit(optimized=False, smoothing_level=0.1, smoothing_trend=0.01,
                                                smoothing_seasonal=0.01 if seasonal else None)
                    else:
                        try:
                            with _w.catch_warnings():
                                _w.simplefilter("ignore")
                                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                                    fit = model.fit(optimized=True)
                            if not np.all(np.isfinite(fit.params.values() if hasattr(fit.params, "values") else list(fit.params))):
                                raise ValueError("non-finite ETS params")
                        except Exception:
                            with _w.catch_warnings():
                                _w.simplefilter("ignore")
                                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                                    fit = model.fit(optimized=False, smoothing_level=0.3, smoothing_trend=0.05,
                                                    smoothing_seasonal=0.05 if seasonal else None)
                    return np.asarray(fit.forecast(n_steps), dtype=float)

                _register_model("ETS", _fit_ets(y_train, holdout_n), _fit_ets(y_full, future_n))
            else:
                _register_model("ETS", None, None, "statsmodels unavailable")
        except Exception as e:
            _register_model("ETS", None, None, str(e))

        # ARIMA
        arima_test_pred = arima_future_pred = None
        try:
            if _HAS_STATSMODELS and SARIMAX is not None:
                d_fit = max(0, min(2, int(d_order)))
                arima_fit    = SARIMAX(y_train, order=(1, d_fit, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                yhat_test    = np.asarray(arima_fit.forecast(holdout_n), dtype=float)
                arima_full   = SARIMAX(y_full,  order=(1, d_fit, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                yhat_future  = np.asarray(arima_full.forecast(future_n),  dtype=float)
                arima_test_pred   = yhat_test
                arima_future_pred = yhat_future
                _register_model("ARIMA", yhat_test, yhat_future)
            else:
                _register_model("ARIMA", None, None, "statsmodels unavailable")
        except Exception as e:
            _register_model("ARIMA", None, None, str(e))

        # ARIMAX
        try:
            if _HAS_STATSMODELS and SARIMAX is not None and exog_all is not None and exog_all.shape[1] > 0:
                d_fit      = max(0, min(2, int(d_order)))
                ex_train   = exog_all.iloc[:-holdout_n]
                ex_test    = exog_all.iloc[-holdout_n:]
                ex_future  = pd.DataFrame(0.0, index=future_df.index, columns=exog_all.columns)
                arimax_fit = SARIMAX(y_train, exog=ex_train, order=(1, d_fit, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                yhat_test  = np.asarray(arimax_fit.forecast(holdout_n, exog=ex_test), dtype=float)
                arimax_full= SARIMAX(y_full, exog=exog_all, order=(1, d_fit, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                yhat_future= np.asarray(arimax_full.forecast(future_n, exog=ex_future), dtype=float)
                arimax_note = ""
                if (not allow_negative) and (_contains_negative(yhat_test) or _contains_negative(yhat_future)):
                    if arima_test_pred is not None and arima_future_pred is not None:
                        yhat_test   = arima_test_pred
                        yhat_future = arima_future_pred
                        arimax_note = "exog ignored: negative forecast prevented"
                    else:
                        yhat_test   = _clip_non_negative(yhat_test)
                        yhat_future = _clip_non_negative(yhat_future)
                        arimax_note = "exog ignored fallback: clipped non-negative"
                _register_model("ARIMAX", yhat_test, yhat_future, arimax_note)
            else:
                _register_model("ARIMAX", None, None, "no exogenous vars")
        except Exception as e:
            _register_model("ARIMAX", None, None, str(e))

        # Linear Regression
        try:
            ex_cols = exog_all.columns.tolist() if exog_all is not None else []
            lr_df   = pd.DataFrame(index=y_full.index)
            lr_df["y"]    = y_full.values
            lr_df["lag1"] = lr_df["y"].shift(1)
            lr_df["lag2"] = lr_df["y"].shift(2)
            p = int(period_used) if period_used and int(period_used) > 1 else 0
            if p > 1:
                lr_df[f"lag{p}"] = lr_df["y"].shift(p)
            for c in ex_cols:
                lr_df[f"exog__{c}"] = exog_all[c].values
            lr_df = lr_df.dropna()

            train_cut  = y_train.index[-1]
            lr_train   = lr_df[lr_df.index <= train_cut]
            lr_test    = lr_df[lr_df.index.isin(y_test.index)]
            if len(lr_train) < 5 or len(lr_test) == 0:
                _register_model("LinearRegression", None, None, "insufficient rows")
            else:
                x_cols_full     = [c for c in lr_df.columns if c != "y"]
                x_cols_lag_only = [c for c in x_cols_full if not c.startswith("exog__")]

                def _run_lr(x_cols_used):
                    lr_model = LinearRegression()
                    lr_model.fit(lr_train[x_cols_used].values, lr_train["y"].values)
                    yhat_test_local = lr_model.predict(lr_test[x_cols_used].values)
                    hist = y_full.copy()
                    fut_preds_local = []
                    ex_future_lr = pd.DataFrame(0.0, index=future_df.index, columns=ex_cols) if ex_cols else None
                    for dt in future_df.index:
                        row = {
                            "lag1": float(hist.iloc[-1]) if len(hist) >= 1 else 0.0,
                            "lag2": float(hist.iloc[-2]) if len(hist) >= 2 else float(hist.iloc[-1]) if len(hist) >= 1 else 0.0,
                        }
                        if p > 1:
                            row[f"lag{p}"] = float(hist.iloc[-p]) if len(hist) >= p else float(hist.iloc[-1]) if len(hist) >= 1 else 0.0
                        for c in ex_cols:
                            row[f"exog__{c}"] = float(ex_future_lr.loc[dt, c]) if ex_future_lr is not None else 0.0
                        x_row = np.array([row[col] for col in x_cols_used], dtype=float).reshape(1, -1)
                        pred  = float(lr_model.predict(x_row)[0])
                        fut_preds_local.append(pred)
                        hist = pd.concat([hist, pd.Series([pred], index=[dt])])
                    return np.asarray(yhat_test_local, dtype=float), np.asarray(fut_preds_local, dtype=float)

                yhat_test, yhat_future = _run_lr(x_cols_full)
                lr_note = ""
                if (not allow_negative) and ex_cols and (_contains_negative(yhat_test) or _contains_negative(yhat_future)):
                    yhat_test, yhat_future = _run_lr(x_cols_lag_only)
                    lr_note = "exog ignored: negative forecast prevented"
                if (not allow_negative) and (_contains_negative(yhat_test) or _contains_negative(yhat_future)):
                    yhat_test   = _clip_non_negative(yhat_test)
                    yhat_future = _clip_non_negative(yhat_future)
                    lr_note = (lr_note + "; clipped non-negative").strip("; ").strip()
                _register_model("LinearRegression", yhat_test, yhat_future, lr_note)
        except Exception as e:
            _register_model("LinearRegression", None, None, str(e))

        # FLANN / RecurrentFLANN / RVFL
        _flann_kwargs = dict(
            y_full=y_full, y_train=y_train, y_test=y_test,
            future_idx=future_df.index, exog_all=exog_all, period_used=period_used,
            basis_family=body.flann_basis, order=body.flann_order,
            ridge_lambda=body.flann_ridge_lambda, recurrent_depth=body.flann_recurrent_depth,
            rvfl_n_hidden=body.rvfl_n_hidden, rvfl_activation=body.rvfl_activation,
        )
        variants_to_run = list(FLANN_VARIANTS) if body.run_all_flann_variants else [body.flann_variant]
        for _variant in variants_to_run:
            _label = {"flann": "FLANN", "recurrent_flann": "RecurrentFLANN", "rvfl": "RVFL"}.get(_variant, _variant.upper())
            try:
                _res = _run_flann_forecast(variant=_variant, **_flann_kwargs)
                if _res is None:
                    _register_model(_label, None, None, "insufficient rows")
                else:
                    _exog_note  = " + exog" if (exog_all is not None and exog_all.shape[1] > 0) else ""
                    _basis_note = f"{body.flann_basis}[{body.flann_order}]"
                    _full_note  = _basis_note + _exog_note
                    if _variant == "recurrent_flann": _full_note += f" depth={body.flann_recurrent_depth}"
                    elif _variant == "rvfl":          _full_note += f" H={body.rvfl_n_hidden} act={body.rvfl_activation}"
                    _register_model(_label, _res.get("holdout_pred"), _res.get("future_pred"), _full_note)
            except Exception as _e:
                _register_model(_label, None, None, str(_e))

        # Combination models
        if body.enable_combination_models:
            _combo_pool = {
                name: pred for name, pred in model_predictions.items()
                if (pred.get("metrics", {}).get("mape") is not None
                    and len(pred.get("holdout_pred") or []) == len(y_test)
                    and len(pred.get("future_pred")  or []) == future_n)
            }
            if len(_combo_pool) >= 2:
                _cnames    = list(_combo_pool.keys())
                _h_mat     = np.array([_combo_pool[n]["holdout_pred"] for n in _cnames], dtype=float)
                _f_mat     = np.array([_combo_pool[n]["future_pred"]  for n in _cnames], dtype=float)
                _mapes_arr = np.array([_combo_pool[n]["metrics"]["mape"] for n in _cnames], dtype=float)
                _register_model("Combo:SimpleAvg",   np.mean(_h_mat, axis=0), np.mean(_f_mat, axis=0))
                _safe_mapes = np.where(_mapes_arr > 0, _mapes_arr, 1e-9)
                _inv_w = (1.0 / _safe_mapes); _inv_w /= _inv_w.sum()
                _register_model("Combo:WeightedAvg", (_inv_w[:, None] * _h_mat).sum(axis=0), (_inv_w[:, None] * _f_mat).sum(axis=0))
                _register_model("Combo:Median",      np.median(_h_mat, axis=0), np.median(_f_mat, axis=0))
                _n_trio = min(3, len(_cnames))
                if _n_trio >= 2:
                    _top_idx = np.argsort(_mapes_arr)[:_n_trio]
                    _register_model(f"Combo:Top{_n_trio}Avg", np.mean(_h_mat[_top_idx], axis=0), np.mean(_f_mat[_top_idx], axis=0))

    # ── Build timeline output ─────────────────────────────────────────────────
    obs_idx  = list(feature_df.index)
    fut_idx  = list(future_df.index)
    all_idx  = obs_idx + fut_idx
    all_labels = [str(x)[:10] for x in all_idx]

    obs_vals  = [float(v) if not pd.isna(v) else None for v in series.reindex(obs_idx).values]
    split_obs = feature_df["split"].astype(str).tolist() if "split" in feature_df.columns else (["train"] * len(obs_idx))
    split_all = split_obs + (["future"] * len(fut_idx))

    train_vals = []; val_vals = []; test_vals = []; holdout_vals = []
    for i, sp in enumerate(split_obs):
        v = obs_vals[i]
        train_vals.append(v if sp == "train" else None)
        val_vals.append(v if sp == "validation" else None)
        test_vals.append(v if sp == "test" else None)
        holdout_vals.append(v if sp == "holdout" else None)
    for _ in fut_idx:
        train_vals.append(None); val_vals.append(None); test_vals.append(None); holdout_vals.append(None)

    holdout_targets  = [ix for ix, sp in zip(obs_idx, split_obs) if sp == "holdout"]
    holdout_actual   = [v  for v,  sp in zip(obs_vals, split_obs) if sp == "holdout" and v is not None]
    non_holdout_vals = [v  for v,  sp in zip(obs_vals, split_obs) if sp != "holdout" and v is not None]
    forecast_targets = holdout_targets + fut_idx

    recommended_candidate = _resolve_recommended_candidate(
        model_rec, [row["model"] for row in model_comparison if row.get("mape") is not None]
    )
    selected_model_name = (
        best_model_name if best_model_name in model_predictions else
        (recommended_candidate if recommended_candidate in model_predictions else None)
    )

    # Baseline fallback
    if (len(y_test) + future_n if len(y_full) > holdout_n + 6 else 0) and selected_model_name is None and non_holdout_vals:
        last  = float(non_holdout_vals[-1])
        drift = 0.0
        if len(non_holdout_vals) > 1 and Ft > 0.5:
            drift = (float(non_holdout_vals[-1]) - float(non_holdout_vals[0])) / (len(non_holdout_vals) - 1)
        p = int(period_used) if period_used else 0
        season_template = None
        if Fs > 0.5 and p > 1 and len(non_holdout_vals) >= p:
            season_template = np.array(non_holdout_vals[-p:], dtype=float)
            season_template -= season_template.mean()
        y_test_local   = y_full.iloc[-holdout_n:] if len(y_full) > holdout_n else y_full
        future_n_local = len(future_df)
        baseline_holdout: List[float] = []
        baseline_future:  List[float] = []
        for i in range(int(len(y_test_local)) + int(future_n_local)):
            pred = last + drift * (i + 1)
            if season_template is not None and len(season_template):
                pred += float(season_template[i % len(season_template)])
            if non_holdout_vals and min(non_holdout_vals) >= 0:
                pred = max(0.0, pred)
            (baseline_holdout if i < len(y_test_local) else baseline_future).append(round(float(pred), 6))
        m = _forecast_metrics(y_test_local.values, np.asarray(baseline_holdout, dtype=float))
        model_comparison.append({"model": "Baseline", **m, "status": "fallback"})
        model_predictions["Baseline"] = {"holdout_pred": baseline_holdout, "future_pred": baseline_future, "metrics": m, "status": "fallback"}
        selected_model_name = "Baseline"

    def _build_model_output(model_name, holdout_pred_vals, future_pred_vals, status, metrics):
        forecast_map = {}
        for dt, pred in zip(holdout_targets, holdout_pred_vals):
            forecast_map[str(dt)[:10]] = float(round(float(pred), 6))
        for dt, pred in zip(fut_idx, future_pred_vals):
            forecast_map[str(dt)[:10]] = float(round(float(pred), 6))
        forecast_path = [forecast_map.get(lbl) for lbl in all_labels]
        future_only   = [None] * len(obs_idx) + [
            float(round(float(pred), 4)) if pred is not None else None for pred in future_pred_vals
        ]
        err_sigma = 0.0
        if holdout_actual and holdout_pred_vals and len(holdout_actual) == len(holdout_pred_vals):
            errs = [float(a) - float(p) for a, p in zip(holdout_actual, holdout_pred_vals)]
            if len(errs) >= 2: err_sigma = float(np.std(errs, ddof=1))
        if err_sigma <= 0:
            base_arr = np.array([v for v in non_holdout_vals if v is not None], dtype=float)
            if len(base_arr) >= 3:
                diff = np.diff(base_arr)
                err_sigma = float(np.std(diff, ddof=1)) if len(diff) > 1 else float(np.std(base_arr, ddof=1) * 0.1)
            elif len(base_arr) >= 2: err_sigma = float(np.std(base_arr, ddof=1) * 0.1)
            else: err_sigma = 1.0
        lower = [None] * len(obs_idx); upper = [None] * len(obs_idx)
        for h, pred in enumerate(future_pred_vals, start=1):
            if pred is None: lower.append(None); upper.append(None); continue
            band = 1.96 * err_sigma * (h ** 0.5)
            lo = float(pred) - band; hi = float(pred) + band
            if non_holdout_vals and min(non_holdout_vals) >= 0: lo = max(0.0, lo)
            lower.append(round(float(lo), 4)); upper.append(round(float(hi), 4))
        return {"model": model_name, "status": status, "metrics": _safe(metrics),
                "forecast_path": [round(float(v), 4) if v is not None else None for v in forecast_path],
                "future_forecast": future_only, "future_lower_95": lower, "future_upper_95": upper}

    model_outputs: Dict[str, Dict[str, Any]] = {
        name: _build_model_output(name, pred["holdout_pred"], pred["future_pred"], str(pred.get("status", "ok")), pred.get("metrics", {}))
        for name, pred in model_predictions.items()
    }

    selected_output      = model_outputs.get(selected_model_name or "")
    model_forecast       = selected_output.get("forecast_path", [])   if selected_output else [None] * len(all_labels)
    future_forecast      = selected_output.get("future_forecast", []) if selected_output else [None] * len(all_labels)
    future_lower         = selected_output.get("future_lower_95", []) if selected_output else [None] * len(all_labels)
    future_upper         = selected_output.get("future_upper_95", []) if selected_output else [None] * len(all_labels)
    observed_all         = obs_vals + ([None] * len(fut_idx))
    holdout_start_label  = next((str(ix)[:10] for ix, sp in zip(obs_idx, split_obs) if sp == "holdout"), None)
    horizon_start_label  = str(fut_idx[0])[:10] if fut_idx else None

    return {
        "ok":                        True,
        "mode":                      mode,
        "dep_col":                   dep_col,
        "node_path":                 node_path,
        "recommended_model":         model_rec,
        "recommended_candidate_model":recommended_candidate,
        "selected_model":            selected_model_name,
        "available_models":          [row["model"] for row in model_comparison if row["model"] in model_outputs],
        "model_outputs":             _safe(model_outputs),
        "timeline": {
            "labels":          all_labels,
            "observed":        [round(float(v), 4) if v is not None else None for v in observed_all],
            "train":           [round(float(v), 4) if v is not None else None for v in train_vals],
            "validation":      [round(float(v), 4) if v is not None else None for v in val_vals],
            "test":            [round(float(v), 4) if v is not None else None for v in test_vals],
            "holdout":         [round(float(v), 4) if v is not None else None for v in holdout_vals],
            "model_forecast":  [round(float(v), 4) if v is not None else None for v in model_forecast],
            "future_forecast": future_forecast,
            "future_lower_95": future_lower,
            "future_upper_95": future_upper,
            "split_labels":    split_all,
            "selected_model":  selected_model_name,
            "holdout_start":   holdout_start_label,
            "horizon_start":   horizon_start_label,
            "horizon_n":       int(len(fut_idx)),
            "confidence_level":0.95,
        },
        "series_profile":    series_profile_safe,
        "model_params":      model_params_safe,
        "best_model":        best_model_name,
        "model_comparison":  model_comparison,
        "split_counts":      _safe(split_counts),
        "feature_columns":   list(feature_df.columns),
        "future_columns":    list(future_df.columns),
        "feature_preview":   _df_to_records(feature_df, 20),
        "future_preview":    _df_to_records(future_df,  20),
        "download_token":    export_token,
        "download_ext":      export_ext,
        "download_mime":     export_mime,
        "report":            prep_res.metadata.get("report", ""),
        "preprocessing":     _safe(preprocessing),
        "warnings":          prep_stage_warnings + (prep_res.warnings or []),
        "allow_negative_forecast": allow_negative,
    }
