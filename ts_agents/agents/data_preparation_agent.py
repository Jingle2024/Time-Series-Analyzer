"""
agents/data_preparation_agent.py
─────────────────────────────────
Transforms a clean, analyzed time series into a forecast-ready feature matrix.

Steps
-----
1. Stationarity transforms  : differencing, log(1+y), Box-Cox
2. Scaling / normalisation  : MinMax, Standard, Robust
3. Lag features             : significant ACF lags
4. Rolling window features  : mean, std, min, max over configurable windows
5. Calendar features        : day-of-week, week-of-year, month, quarter,
                              is_weekend, cyclical Fourier encoding
6. Train / Val / Test split : strict temporal split (no shuffle)
7. Output                   : feature matrix X, target y, split indices, scaler objects
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.base_agent import AgentResult, AgentStatus, BaseAgent
from core.context_store import ContextStore

try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    from scipy.stats import boxcox
    from scipy.special import inv_boxcox
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


_SCALERS = {
    "minmax":   lambda: MinMaxScaler(),
    "standard": lambda: StandardScaler(),
    "robust":   lambda: RobustScaler(),
}


class DataPreparationAgent(BaseAgent):
    """
    Parameters
    ----------
    series           : pd.Series  – cleaned, imputed target series
    target_col       : str        – name for target column in output
    transform        : str        – 'none'|'log'|'diff'|'log_diff'|'boxcox'
    diff_order       : int        – differencing order (1 or 2)
    scale_method     : str        – 'minmax'|'standard'|'robust'|'none'
    lag_list         : list[int]  – explicit lags; None = use ACF significant lags
    rolling_windows  : list[int]  – rolling window sizes (default [7, 14, 28])
    add_calendar     : bool       – add calendar features (default True)
    train_ratio      : float      – train proportion (default 0.7)
    val_ratio        : float      – validation proportion (default 0.15)
    horizon          : int        – forecast horizon (periods); used for documentation
    """

    def __init__(self, context_store: Optional[ContextStore] = None):
        super().__init__("DataPreparationAgent", context_store)

    def validate_inputs(self, series: Any = None, **kwargs):
        if series is None:
            raise ValueError("DataPreparationAgent requires 'series'.")
        if not _HAS_SKLEARN:
            raise ImportError("sklearn and scipy required. pip install scikit-learn scipy")

    def _run(
        self,
        series: pd.Series,
        target_col: str = "y",
        transform: str = "auto",
        diff_order: int = 0,
        scale_method: str = "minmax",
        lag_list: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        add_calendar: bool = True,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        horizon: int = 1,
        acf_lags: Optional[List[int]] = None,   # from DecompositionAgent output
        **kwargs,
    ) -> AgentResult:

        warn_list: List[str] = []
        series = series.dropna().sort_index().astype(float)
        rolling_windows = rolling_windows or [7, 14, 28]

        # ── 1. transform ──────────────────────────────────────────────────────
        if transform == "auto":
            transform, diff_order = self._auto_transform(series, warn_list)
            warn_list.append(f"Auto-selected transform: '{transform}', diff_order={diff_order}")

        y_transformed, transform_meta = self._apply_transform(series, transform, diff_order, warn_list)

        # ── 2. build feature DataFrame ────────────────────────────────────────
        df = pd.DataFrame({target_col: y_transformed})

        # ── 3. lag features ───────────────────────────────────────────────────
        if lag_list is None:
            lag_list = self._select_lags(y_transformed, acf_lags, warn_list)
        for lag in lag_list:
            df[f"lag_{lag}"] = df[target_col].shift(lag)

        # ── 4. rolling features ───────────────────────────────────────────────
        for w in rolling_windows:
            if w >= len(df) // 2:
                warn_list.append(f"Rolling window {w} is ≥ half the series length; skipping.")
                continue
            df[f"roll_mean_{w}"] = df[target_col].shift(1).rolling(w).mean()
            df[f"roll_std_{w}"]  = df[target_col].shift(1).rolling(w).std()
            df[f"roll_min_{w}"]  = df[target_col].shift(1).rolling(w).min()
            df[f"roll_max_{w}"]  = df[target_col].shift(1).rolling(w).max()

        # ── 5. calendar features ──────────────────────────────────────────────
        if add_calendar and isinstance(df.index, pd.DatetimeIndex):
            df = self._add_calendar_features(df, warn_list)

        # ── 6. drop NaN rows from lag/rolling creation ─────────────────────────
        n_before_drop = len(df)
        df = df.dropna()
        n_dropped = n_before_drop - len(df)
        if n_dropped > 0:
            warn_list.append(f"Dropped {n_dropped} rows due to NaN from lag/rolling features.")

        # ── 7. split ──────────────────────────────────────────────────────────
        splits = self._temporal_split(df, train_ratio, val_ratio)

        # ── 8. scale ──────────────────────────────────────────────────────────
        X_cols = [c for c in df.columns if c != target_col]
        y_col = target_col
        scaler_X = scaler_y = None

        df_out = df.copy()
        if scale_method != "none" and _HAS_SKLEARN:
            scaler_cls = _SCALERS.get(scale_method)
            if scaler_cls:
                scaler_X = scaler_cls()
                scaler_y = scaler_cls()
                train_idx = splits["train_idx"]
                # fit on train only
                scaler_X.fit(df.iloc[train_idx][X_cols].values)
                scaler_y.fit(df.iloc[train_idx][[y_col]].values)
                df_out[X_cols] = scaler_X.transform(df[X_cols].values)
                df_out[[y_col]] = scaler_y.transform(df[[y_col]].values)

        # ── 9. assemble outputs ───────────────────────────────────────────────
        feature_names = X_cols
        summary = {
            "target_col": target_col,
            "transform": transform,
            "diff_order": diff_order,
            "scale_method": scale_method,
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "n_obs_final": len(df_out),
            "lag_features": [f"lag_{l}" for l in lag_list],
            "rolling_features": [c for c in feature_names if c.startswith("roll_")],
            "calendar_features": [c for c in feature_names if not c.startswith(("lag_", "roll_"))],
            "splits": {k: {"start": str(df_out.index[v[0]]) if v else None,
                           "end": str(df_out.index[v[-1]]) if v else None,
                           "n": len(v)}
                       for k, v in splits.items() if k.endswith("_idx")},
            "horizon": horizon,
            "transform_meta": transform_meta,
        }

        report = self._build_report(summary)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={
                "feature_matrix": df_out,
                "X_train": df_out.iloc[splits["train_idx"]][X_cols],
                "y_train": df_out.iloc[splits["train_idx"]][y_col],
                "X_val":   df_out.iloc[splits["val_idx"]][X_cols],
                "y_val":   df_out.iloc[splits["val_idx"]][y_col],
                "X_test":  df_out.iloc[splits["test_idx"]][X_cols],
                "y_test":  df_out.iloc[splits["test_idx"]][y_col],
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "transform_meta": transform_meta,
            },
            metadata={"summary": summary, "report": report},
            warnings=warn_list,
        )

    # ── transform ─────────────────────────────────────────────────────────────

    @staticmethod
    def _auto_transform(series: pd.Series, warn_list: List[str]) -> Tuple[str, int]:
        """Heuristic: if series has positive values and high skewness, use log."""
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
                y = y + shift
                meta["log_shift"] = float(shift)
                warn_list.append(f"Log transform: shifted series by {shift} to ensure positivity.")
            y = np.log1p(y)
            meta["log_applied"] = True

        if transform == "boxcox":
            try:
                y_arr, lam = boxcox(y.values + 1e-6)
                y = pd.Series(y_arr, index=series.index, name=series.name)
                meta["boxcox_lambda"] = round(float(lam), 6)
            except Exception as e:
                warn_list.append(f"Box-Cox failed ({e}); using log instead.")
                y = np.log1p(y)
                meta["log_applied"] = True

        if transform in ("diff", "log_diff") or diff_order > 0:
            for _ in range(diff_order):
                y = y.diff()
            meta["diff_applied"] = True

        return y, meta

    # ── lag selection ─────────────────────────────────────────────────────────

    @staticmethod
    def _select_lags(
        series: pd.Series,
        acf_lags: Optional[List[int]],
        warn_list: List[str],
        max_lags: int = 8,
    ) -> List[int]:
        if acf_lags:
            return sorted(acf_lags[:max_lags])
        # fallback: use a set of common lags
        n = len(series)
        default = [1, 2, 3, 7, 14, 28, 30]
        chosen = [l for l in default if l < n // 3]
        warn_list.append(f"Using default lags: {chosen}")
        return chosen

    # ── rolling features ──────────────────────────────────────────────────────

    # (handled inline in _run)

    # ── calendar features ─────────────────────────────────────────────────────

    @staticmethod
    def _add_calendar_features(df: pd.DataFrame, warn_list: List[str]) -> pd.DataFrame:
        idx = df.index
        try:
            df["day_of_week"]   = idx.dayofweek                          # 0=Mon
            df["day_of_month"]  = idx.day
            df["week_of_year"]  = idx.isocalendar().week.astype(int)
            df["month"]         = idx.month
            df["quarter"]       = idx.quarter
            df["is_weekend"]    = (idx.dayofweek >= 5).astype(int)
            df["is_month_start"]= idx.is_month_start.astype(int)
            df["is_month_end"]  = idx.is_month_end.astype(int)

            # Fourier cyclical encoding for month (annual seasonality)
            df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
            df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
            # day-of-week cyclical
            df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
            df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
        except Exception as e:
            warn_list.append(f"Calendar features failed partially: {e}")
        return df

    # ── temporal split ────────────────────────────────────────────────────────

    @staticmethod
    def _temporal_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Dict:
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        return {
            "train_idx": list(range(n_train)),
            "val_idx":   list(range(n_train, n_train + n_val)),
            "test_idx":  list(range(n_train + n_val, n)),
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
        }

    @staticmethod
    def _build_report(summary: Dict) -> str:
        lines = [
            "═══ Data Preparation Report ═══",
            f"  Transform          : {summary['transform']}  (diff_order={summary['diff_order']})",
            f"  Scaling            : {summary['scale_method']}",
            f"  Total features     : {summary['n_features']}",
            f"  Final observations : {summary['n_obs_final']}",
            f"  Forecast horizon   : {summary['horizon']}",
            "",
            "  Feature groups:",
            f"    Lag features     : {len(summary['lag_features'])} → {summary['lag_features']}",
            f"    Rolling features : {len(summary['rolling_features'])}",
            f"    Calendar features: {len(summary['calendar_features'])}",
            "",
            "  Train/Val/Test split (temporal):",
        ]
        for split, info in summary["splits"].items():
            lines.append(
                f"    {split:10s}: n={info['n']:5d}  [{info['start']}  →  {info['end']}]"
            )
        return "\n".join(lines)
