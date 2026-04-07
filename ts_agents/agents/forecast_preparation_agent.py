"""
agents/forecast_preparation_agent.py
──────────────────────────────────────
Produces a complete, forecast-ready dataset for ONE series (identified by
a hierarchy node path or a simple column name) with:

  ┌─────────────────────────────────────────────────────────────────┐
  │  TIMELINE  (ordered by time, no shuffle)                        │
  │                                                                 │
  │  │←── train ──────────────────│← val →│← holdout/test →│future│
  │  0                          n_train  n_val          n_obs  +H  │
  └─────────────────────────────────────────────────────────────────┘

  train      – used to fit the model
  validation – used to tune hyperparameters (early stopping, etc.)
  holdout    – withheld completely; used to evaluate true out-of-sample
               accuracy at the chosen horizon H
  future     – H empty rows with populated feature columns (calendar,
               exog forecasts) ready for the model to predict into

Key outputs
───────────
  feature_matrix     : full DataFrame with split column
  future_frame       : H-row DataFrame of forecast-period features
  model_params       : dict of suggested model hyperparameters
  series_profile     : rich metadata (n_obs, freq, Ft, Fs, ADI, CV², etc.)
  inverse_transform  : dict with enough info to undo scaling/differencing

Parameters
──────────
  series          pd.Series             target (DatetimeIndex, clean)
  node_path       dict                  hierarchy path (for labelling)
  dep_col         str                   column name
  indep_df        pd.DataFrame | None   exogenous continuous vars (aligned)
  event_df        pd.DataFrame | None   binary event cols (aligned)
  transform       str                   auto|none|log|diff|log_diff|boxcox
  scale_method    str                   minmax|standard|robust|none
  freq            str                   pandas offset alias e.g. 'W','MS'
  n_holdout       int                   periods held out at the end
  horizon         int                   future forecast periods
  train_ratio     float                 fraction of non-holdout for training
  val_ratio       float                 fraction of non-holdout for validation
  rolling_windows list[int]
  add_calendar    bool
  acf_lags        list[int] | None      from DecompositionAgent
"""

from __future__ import annotations

import warnings as _warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.base_agent import AgentResult, AgentStatus, BaseAgent
from core.context_store import ContextStore

try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    from scipy.stats import boxcox
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

_SCALERS = {
    "minmax":   lambda: MinMaxScaler(),
    "standard": lambda: StandardScaler(),
    "robust":   lambda: RobustScaler(),
}


# ── Model parameter scaffolding ───────────────────────────────────────────────

def _suggest_model_params(
    model_rec: str,
    interm_cls: str,
    Ft: float,
    Fs: float,
    d: int,
    period: int,
    n_obs: int,
    horizon: int,
    has_exog: bool,
) -> Dict[str, Any]:
    """
    Returns a structured dict of suggested parameters for the recommended model.
    These are STARTING POINTS — not tuned values.
    """
    base: Dict[str, Any] = {
        "model": model_rec,
        "horizon": horizon,
        "frequency": None,    # filled by caller
        "period": period,
        "has_exogenous": has_exog,
    }

    name = model_rec.lower()

    # ── ARIMA / ARIMAX ────────────────────────────────────────────────────────
    if "arima" in name or "sarima" in name:
        p = min(3, max(1, period // 4))
        q = 1
        P = 1 if Fs > 0.3 else 0
        Q = 1 if Fs > 0.3 else 0
        D = 1 if Fs > 0.3 else 0
        base["arima"] = {
            "p": p, "d": d, "q": q,
            "P": P, "D": D, "Q": Q,
            "m": period,
            "trend": "c" if Ft > 0.3 else "n",
            "information_criterion": "aic",
            "auto_arima": True,
            "stepwise": True,
            "max_p": 5, "max_q": 5, "max_P": 2, "max_Q": 2,
        }

    # ── ETS ───────────────────────────────────────────────────────────────────
    if "ets" in name or "holt" in name:
        error = "mul" if interm_cls == "Erratic" else "add"
        trend = "add" if Ft > 0.3 else None
        seasonal = "add" if Fs > 0.3 else None
        base["ets"] = {
            "error": error,
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": period if seasonal else None,
            "damped_trend": Ft > 0.5,
            "optimise": True,
        }

    # ── Croston / SBA ─────────────────────────────────────────────────────────
    if "croston" in name or "sba" in name or "tsb" in name:
        base["croston"] = {
            "alpha": 0.1,
            "beta":  0.1 if "tsb" in name else None,
            "variant": "sba" if "sba" in name else ("tsb" if "tsb" in name else "classic"),
        }

    # ── LightGBM / ML ─────────────────────────────────────────────────────────
    if "lightgbm" in name or "ml" in name or "xgb" in name:
        base["lightgbm"] = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": max(5, n_obs // 50),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "regression",
            "metric": "rmse",
            "early_stopping_rounds": 20,
        }

    # ── TBATS ─────────────────────────────────────────────────────────────────
    if "tbats" in name:
        base["tbats"] = {
            "use_box_cox": None,   # auto
            "use_trend": Ft > 0.2,
            "use_damped_trend": Ft > 0.5,
            "seasonal_periods": [period] if Fs > 0.3 else None,
            "use_arma_errors": True,
        }

    if "flann" in name or "rvfl" in name:
        base["flann_family"] = {
            # Which variant to run by default (can be overridden by the caller)
            "variant":  (
                "recurrent_flann" if "recurrent" in name
                else "rvfl"       if "rvfl"      in name
                else "flann"
            ),
            # Available choices shown in the UI
            "available_variants": ["flann", "recurrent_flann", "rvfl"],
            # Basis expansion
            "basis_family":       "mixed",        # polynomial|trig|chebyshev|legendre|mixed
            "available_basis":    [
                "polynomial", "trigonometric", "chebyshev", "legendre", "mixed"
            ],
            "expansion_order":    3,              # polynomial degree / trig harmonics
            "ridge_lambda":       0.01,           # L2 regularisation for output layer
            "use_exogenous":      has_exog,
        }
        base["flann"] = {
                    "description": (
                        "Functional Link ANN — non-linear basis expansion of lag/exog "
                        "inputs, single linear output layer solved by ridge regression. "
                        "No hidden layer, no back-prop. O(n·D²) training."
                    ),
                    "basis_family":    "mixed",
                    "expansion_order": 3,
                    "ridge_lambda":    0.01,
                    "lag_features":    [1, 2, 3],
                    "rolling_features": [3, 7],
                    "use_exogenous":   has_exog,
                    "multi_step":      "iterative",   # recursive prediction for h>1
                    "recommended_when": (
                        "small-to-medium n, smooth or erratic series, "
                        "interpretability matters"
                    ),
        }

        base["recurrent_flann"] = {
            "description": (
                "FLANN with a recurrent feedback state. The previous R "
                "predictions are appended to the input vector before basis "
                "expansion, capturing short-memory dynamics without gradient "
                "descent. Training uses teacher-forcing."
            ),
            "basis_family":     "mixed",
            "expansion_order":  3,
            "ridge_lambda":     0.01,
            "recurrent_depth":  min(3, max(1, period // 4)),  # auto from period
            "use_exogenous":    has_exog,
            "multi_step":       "recurrent",
            "recommended_when": (
                "series with strong temporal autocorrelation, "
                "periodic or trending behaviour, medium n"
            ),
        }
        # Hidden-layer size heuristic: roughly 2×input_dim, capped for small n
        _rvfl_hidden = max(16, min(128, n_obs // 4))

        base["rvfl"] = {
            "description": (
                "Random Vector Functional Link. Hidden-layer weights are drawn "
                "ONCE from a uniform random distribution and FIXED. Only the "
                "output weights are learned (ridge). Extremely fast; often "
                "matches deep networks on small-to-medium tabular time-series."
            ),
            "n_hidden":        _rvfl_hidden,
            "activation":      "sigmoid",        # sigmoid | relu | tanh | sin
            "available_activations": ["sigmoid", "relu", "tanh", "sin"],
            "ridge_lambda":    0.01,
            "random_seed":     42,
            "direct_links":    True,             # original inputs bypass hidden layer
            "use_exogenous":   has_exog,
            "multi_step":      "iterative",
            "recommended_when": (
                "very small n (<100), need fast training, "
                "ensemble diversity when combined with FLANN"
            ),
        }
    # ── Universal evaluation metrics ─────────────────────────────────────────
    base["evaluation"] = {
        "metrics": ["MAE", "RMSE", "MAPE", "sMAPE", "MASE"],
        "cv_folds": max(2, min(5, n_obs // (horizon * 3))),
        "cv_step_size": horizon,
        "holdout_periods": None,   # filled by caller
    }

    return base


# ── AGENT ─────────────────────────────────────────────────────────────────────

class ForecastPreparationAgent(BaseAgent):

    def __init__(self, context_store: Optional[ContextStore] = None):
        super().__init__("ForecastPreparationAgent", context_store)

    def validate_inputs(self, series: Any = None, **kwargs):
        if series is None:
            raise ValueError("ForecastPreparationAgent requires 'series'.")
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn + scipy required.")

    def _run(
        self,
        series: pd.Series,
        node_path: Optional[Dict[str, str]] = None,
        dep_col: str = "y",
        indep_df: Optional[pd.DataFrame] = None,
        event_df: Optional[pd.DataFrame] = None,
        transform: str = "auto",
        scale_method: str = "minmax",
        freq: Optional[str] = None,
        n_holdout: int = 0,
        horizon: int = 13,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        rolling_windows: Optional[List[int]] = None,
        add_calendar: bool = True,
        acf_lags: Optional[List[int]] = None,
        model_rec: str = "ARIMA",
        interm_cls: str = "Smooth",
        Ft: float = 0.0,
        Fs: float = 0.0,
        d: int = 0,
        period: int = 12,
        **kwargs,
    ) -> AgentResult:

        warn_list: List[str] = []
        series = series.dropna().sort_index().astype(float)
        n_total = len(series)
        rolling_windows = rolling_windows or [7, 14, 28]

        if n_total < 10:
            return AgentResult(
                agent_name=self.name, status=AgentStatus.FAILED,
                errors=[f"Series too short ({n_total} obs). Need ≥ 10."]
            )

        # ── 1. Carve out holdout before any fitting ───────────────────────────
        n_holdout = min(n_holdout, n_total // 3)   # cap at 1/3 of series
        if n_holdout > 0:
            series_fit   = series.iloc[:-n_holdout]
            series_hold  = series.iloc[-n_holdout:]
        else:
            series_fit   = series
            series_hold  = pd.Series([], dtype=float, name=series.name)

        # ── 2. Transform ──────────────────────────────────────────────────────
        if transform == "auto":
            transform, d = self._auto_transform(series_fit, warn_list)

        y_fit, transform_meta = self._apply_transform(series_fit, transform, d, warn_list)
        transform_meta["original_last_value"] = float(series_fit.iloc[-1])
        transform_meta["original_last_diff_value"] = (
            float(series_fit.diff().iloc[-1]) if d > 0 else None
        )

        # ── 3. Feature matrix on fit portion ─────────────────────────────────
        feat = pd.DataFrame({dep_col: y_fit})

        # lags
        lag_list = self._select_lags(y_fit, acf_lags, warn_list)
        for lag in lag_list:
            feat[f"lag_{lag}"] = feat[dep_col].shift(lag)

        # rolling
        for w in rolling_windows:
            if w >= len(feat) // 2:
                continue
            feat[f"roll_mean_{w}"] = feat[dep_col].shift(1).rolling(w).mean()
            feat[f"roll_std_{w}"]  = feat[dep_col].shift(1).rolling(w).std()
            feat[f"roll_min_{w}"]  = feat[dep_col].shift(1).rolling(w).min()
            feat[f"roll_max_{w}"]  = feat[dep_col].shift(1).rolling(w).max()

        # calendar
        if add_calendar and isinstance(feat.index, pd.DatetimeIndex):
            feat = self._add_calendar_features(feat)

        # exogenous — attach aligned indep / event cols
        exog_cols: List[str] = []
        if indep_df is not None:
            for ic in indep_df.columns:
                s = indep_df[ic].reindex(feat.index)
                feat[f"indep__{ic}"] = s.values
                feat[f"indep_lag1__{ic}"] = s.shift(1).values
                exog_cols += [f"indep__{ic}", f"indep_lag1__{ic}"]

        if event_df is not None:
            for ec in event_df.columns:
                s = event_df[ec].reindex(feat.index).fillna(0)
                feat[f"event__{ec}"] = s.values
                feat[f"event_roll7__{ec}"] = s.rolling(7, min_periods=1).sum().values
                exog_cols += [f"event__{ec}", f"event_roll7__{ec}"]

        # drop NaN rows
        n_before = len(feat)
        feat = feat.dropna()
        if n_before - len(feat):
            warn_list.append(f"Dropped {n_before-len(feat)} NaN rows after feature creation.")

        # ── 4. Train / Val split (on fit portion only) ────────────────────────
        n_fit = len(feat)
        n_train = max(1, int(n_fit * train_ratio))
        n_val   = max(0, int(n_fit * val_ratio))
        n_test  = n_fit - n_train - n_val   # remaining in fit (pre-holdout test window)

        splits_idx = {
            "train":      list(range(n_train)),
            "validation": list(range(n_train, n_train + n_val)),
            "test":       list(range(n_train + n_val, n_fit)),
        }
        split_col = (
            ["train"]      * n_train +
            ["validation"] * n_val   +
            ["test"]       * n_test
        )
        feat["split"] = split_col

        # ── 5. Scale (fit on train only) ──────────────────────────────────────
        X_cols = [c for c in feat.columns if c not in (dep_col, "split")]
        y_col  = dep_col
        scaler_X = scaler_y = None
        feat_scaled = feat.copy()

        if scale_method != "none" and _HAS_SKLEARN and n_train > 0:
            sc = _SCALERS.get(scale_method)
            if sc:
                scaler_X = sc()
                scaler_y = sc()
                train_X = feat.iloc[splits_idx["train"]][X_cols].values
                train_y = feat.iloc[splits_idx["train"]][[y_col]].values
                if len(train_X) and train_X.shape[1]:
                    scaler_X.fit(train_X)
                    feat_scaled[X_cols] = scaler_X.transform(feat[X_cols].values)
                if len(train_y):
                    scaler_y.fit(train_y)
                    feat_scaled[[y_col]] = scaler_y.transform(feat[[y_col]].values)

        # ── 6. Holdout frame ──────────────────────────────────────────────────
        hold_frame = pd.DataFrame()
        if len(series_hold):
            hold_y, _ = self._apply_transform(series_hold, transform, d, [])
            hold_feat = pd.DataFrame({dep_col: hold_y})
            for lag in lag_list:
                # concat for proper lag computation
                combined = pd.concat([y_fit, hold_y])
                hold_feat[f"lag_{lag}"] = combined.shift(lag).iloc[-len(series_hold):]
            for w in rolling_windows:
                if w >= len(feat) // 2:
                    continue
                combined = pd.concat([y_fit, hold_y])
                hold_feat[f"roll_mean_{w}"] = combined.shift(1).rolling(w).mean().iloc[-len(series_hold):]
                hold_feat[f"roll_std_{w}"]  = combined.shift(1).rolling(w).std().iloc[-len(series_hold):]
                hold_feat[f"roll_min_{w}"]  = combined.shift(1).rolling(w).min().iloc[-len(series_hold):]
                hold_feat[f"roll_max_{w}"]  = combined.shift(1).rolling(w).max().iloc[-len(series_hold):]
            if add_calendar and isinstance(hold_feat.index, pd.DatetimeIndex):
                hold_feat = self._add_calendar_features(hold_feat)
            hold_feat["split"] = "holdout"
            hold_frame = hold_feat

        # ── 7. Future frame (H blank rows after last date) ────────────────────
        last_date  = series.index[-1]
        inferred_freq = freq or pd.infer_freq(series.index) or "D"
        future_idx = pd.date_range(
            start=last_date, periods=horizon + 1, freq=inferred_freq
        )[1:]

        future_feat = pd.DataFrame(index=future_idx)
        future_feat[dep_col] = np.nan   # target unknown — to be predicted

        # Future lags: use last known values from series
        # Build a continuation array for proper lag computation
        last_known = series.iloc[-(max(lag_list or [1]) + max(rolling_windows or [1]) + 1):]
        if d > 0:
            last_known_tr = np.log1p(last_known) if "log" in transform else last_known
            for diff_i in range(d):
                last_known_tr = last_known_tr.diff()
        else:
            last_known_tr = last_known.copy()

        future_series_placeholder = pd.concat([
            last_known_tr,
            pd.Series(np.nan, index=future_idx),
        ])
        for lag in lag_list:
            future_feat[f"lag_{lag}"] = future_series_placeholder.shift(lag).iloc[-horizon:].values
        for w in rolling_windows:
            future_feat[f"roll_mean_{w}"] = future_series_placeholder.shift(1).rolling(w).mean().iloc[-horizon:].values
            future_feat[f"roll_std_{w}"]  = future_series_placeholder.shift(1).rolling(w).std().iloc[-horizon:].values
            future_feat[f"roll_min_{w}"]  = future_series_placeholder.shift(1).rolling(w).min().iloc[-horizon:].values
            future_feat[f"roll_max_{w}"]  = future_series_placeholder.shift(1).rolling(w).max().iloc[-horizon:].values

        if add_calendar and isinstance(future_feat.index, pd.DatetimeIndex):
            future_feat = self._add_calendar_features(future_feat)

        # exog for future — fill NaN (user can supply actuals later)
        for col in exog_cols:
            future_feat[col] = np.nan

        future_feat["split"] = "future"

        # ── 8. Model parameter scaffold ───────────────────────────────────────
        model_params = _suggest_model_params(
            model_rec=model_rec, interm_cls=interm_cls,
            Ft=Ft, Fs=Fs, d=d, period=period,
            n_obs=n_total, horizon=horizon, has_exog=bool(exog_cols),
        )
        model_params["frequency"] = inferred_freq
        model_params["evaluation"]["holdout_periods"] = n_holdout

        # ── 9. Series profile ─────────────────────────────────────────────────
        series_profile = {
            "node_path": node_path or {},
            "dep_col": dep_col,
            "n_total": n_total,
            "n_fit": n_fit,
            "n_train": n_train,
            "n_val": n_val,
            "n_test_in_fit": n_test,
            "n_holdout": n_holdout,
            "horizon": horizon,
            "freq": inferred_freq,
            "date_range_fit": (str(series_fit.index[0])[:10], str(series_fit.index[-1])[:10]),
            "date_range_holdout": (
                str(series_hold.index[0])[:10] if len(series_hold) else None,
                str(series_hold.index[-1])[:10] if len(series_hold) else None,
            ),
            "future_range": (str(future_idx[0])[:10], str(future_idx[-1])[:10]),
            "transform": transform,
            "diff_order": d,
            "scale_method": scale_method,
            "model_rec": model_rec,
            "interm_cls": interm_cls,
            "Ft": round(Ft, 4),
            "Fs": round(Fs, 4),
            "period": period,
            "n_features": len(X_cols),
            "lag_features": [f"lag_{l}" for l in lag_list],
            "exog_cols": exog_cols,
            "transform_meta": transform_meta,
        }

        # ── 10. Combine all splits ────────────────────────────────────────────
        full_frame = feat_scaled.copy()
        if len(hold_frame):
            # scale holdout with same scalers
            hf = hold_frame.copy()
            common_X = [c for c in X_cols if c in hf.columns]
            if scaler_X and common_X:
                try:
                    hf[common_X] = scaler_X.transform(hf[common_X].values)
                except Exception:
                    pass
            if scaler_y and dep_col in hf.columns:
                try:
                    hf[[dep_col]] = scaler_y.transform(hf[[dep_col]].fillna(0).values)
                except Exception:
                    pass
            full_frame = pd.concat([full_frame, hf])

        # Add node path columns for identification
        if node_path:
            for k, v in node_path.items():
                full_frame[f"hier_{k}"] = v
                future_feat[f"hier_{k}"] = v
        full_frame["series_id"] = dep_col
        future_feat["series_id"] = dep_col

        report = self._build_report(series_profile)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={
                "feature_matrix": full_frame,
                "future_frame": future_feat,
                "X_train": feat_scaled.iloc[splits_idx["train"]][X_cols],
                "y_train": feat_scaled.iloc[splits_idx["train"]][y_col],
                "X_val":   feat_scaled.iloc[splits_idx["validation"]][X_cols] if splits_idx["validation"] else pd.DataFrame(),
                "y_val":   feat_scaled.iloc[splits_idx["validation"]][y_col]  if splits_idx["validation"] else pd.Series(dtype=float),
                "X_test":  feat_scaled.iloc[splits_idx["test"]][X_cols]       if splits_idx["test"] else pd.DataFrame(),
                "y_test":  feat_scaled.iloc[splits_idx["test"]][y_col]        if splits_idx["test"] else pd.Series(dtype=float),
                "X_holdout": hold_frame[[c for c in X_cols if c in hold_frame.columns]] if len(hold_frame) else pd.DataFrame(),
                "y_holdout": hold_frame[dep_col] if len(hold_frame) else pd.Series(dtype=float),
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "model_params": model_params,
            },
            metadata={
                "series_profile": series_profile,
                "report": report,
                "transform_meta": transform_meta,
            },
            warnings=warn_list,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _auto_transform(series: pd.Series, warn_list: List[str]) -> Tuple[str, int]:
        vals = series.dropna().values
        if (vals <= 0).any():
            return "diff", 1
        skew = float(pd.Series(vals).skew())
        if abs(skew) > 1.0:
            return "log", 0
        return "none", 0

    @staticmethod
    def _apply_transform(
        series: pd.Series, transform: str, diff_order: int, warn_list: List[str]
    ) -> Tuple[pd.Series, Dict]:
        meta: Dict = {"transform": transform, "diff_order": diff_order}
        y = series.copy().astype(float)

        if transform in ("log", "log_diff"):
            if (y <= 0).any():
                shift = abs(y.min()) + 1
                y += shift
                meta["log_shift"] = float(shift)
            y = np.log1p(y)
            meta["log_applied"] = True

        if transform == "boxcox":
            try:
                y_arr, lam = boxcox(y.values + 1e-6)
                y = pd.Series(y_arr, index=series.index, name=series.name)
                meta["boxcox_lambda"] = round(float(lam), 6)
            except Exception as e:
                y = np.log1p(y)
                meta["log_applied"] = True
                if warn_list is not None:
                    warn_list.append(f"Box-Cox fallback: {e}")

        if transform in ("diff", "log_diff") or diff_order > 0:
            for _ in range(diff_order):
                y = y.diff()
            meta["diff_applied"] = True

        return y, meta

    @staticmethod
    def _select_lags(series: pd.Series, acf_lags: Optional[List[int]],
                     warn_list: List[str], max_lags: int = 8) -> List[int]:
        if acf_lags:
            return sorted(acf_lags[:max_lags])
        n = len(series)
        default = [1, 2, 3, 7, 14, 28, 30]
        chosen = [l for l in default if l < n // 3]
        if warn_list is not None:
            warn_list.append(f"Using default lags: {chosen}")
        return chosen or [1]

    @staticmethod
    def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        try:
            df["day_of_week"]    = idx.dayofweek
            df["day_of_month"]   = idx.day
            df["week_of_year"]   = idx.isocalendar().week.astype(int)
            df["month"]          = idx.month
            df["quarter"]        = idx.quarter
            df["is_weekend"]     = (idx.dayofweek >= 5).astype(int)
            df["is_month_start"] = idx.is_month_start.astype(int)
            df["is_month_end"]   = idx.is_month_end.astype(int)
            df["month_sin"]      = np.sin(2 * np.pi * idx.month / 12)
            df["month_cos"]      = np.cos(2 * np.pi * idx.month / 12)
            df["dow_sin"]        = np.sin(2 * np.pi * idx.dayofweek / 7)
            df["dow_cos"]        = np.cos(2 * np.pi * idx.dayofweek / 7)
        except Exception:
            pass
        return df

    @staticmethod
    def _build_report(p: Dict) -> str:
        node_str = " › ".join(f"{k}={v}" for k, v in p.get("node_path", {}).items()) or "global"
        lines = [
            "═══ Forecast Preparation Report ═══",
            f"  Series          : {p['dep_col']}  [{node_str}]",
            f"  Total obs       : {p['n_total']}",
            f"  Transform       : {p['transform']}  (d={p['diff_order']})",
            f"  Scaling         : {p['scale_method']}",
            f"  Horizon         : {p['horizon']} periods",
            f"  Frequency       : {p['freq']}",

            f"  Model rec       : {p['model_rec']}  ({p['interm_cls']})",
            (f"FLANN variant: {p['flann_variant']} (basis={p.get('flann_basis', 'mixed')}[{p.get('flann_order', 3)}] λ={p.get('flann_ridge_lambda', 0.01)})" 
          if any(x in p.get('model_rec', '').lower() for x in ['flann', 'rvfl']) 
          else f"Ft={p['Ft']:.3f} Fs={p['Fs']:.3f} period={p['period']}"),
            "",
            "  Split breakdown:",
            f"    Train      : {p['n_train']}  {p['date_range_fit'][0]}",
            f"    Validation : {p['n_val']}",
            f"    Test       : {p['n_test_in_fit']}",
            f"    Holdout    : {p['n_holdout']}  " + (p['date_range_holdout'][0] or '—'),
            f"    Future     : {p['horizon']}  {p['future_range'][0]} → {p['future_range'][1]}",
            "",
            f"  Features        : {p['n_features']}",
            f"  Lags            : {p['lag_features']}",
            f"  Exog cols       : {p['exog_cols'] or 'none'}",
        ]
        return "\n".join(lines)
