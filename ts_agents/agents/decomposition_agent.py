"""
agents/decomposition_agent.py
──────────────────────────────
Full time series decomposition using pure numpy/scipy (no statsmodels required).
Falls back to statsmodels automatically if it is installed.

Components extracted
---------------------
  trend     – LOESS-like trend via Savitzky-Golay smoothing
  seasonal  – period-averaged seasonal component (STL-style)
  cycle     – Hodrick-Prescott filtered cycle
  residual  – y - trend - seasonal

Tests run
----------
  Stationarity  : ADF (augmented Dickey-Fuller), KPSS
  Trend         : OLS linear regression, Theil-Sen robust slope
  Residual      : Ljung-Box autocorrelation, ARCH-LM heteroskedasticity,
                  Shapiro-Wilk normality (for n ≤ 5000)

Component strength metrics (Hyndman & Athanasopoulos)
-------------------------------------------------------
  Ft = max(0, 1 - Var(R) / Var(T+R))   trend strength  in [0,1]
  Fs = max(0, 1 - Var(R) / Var(S+R))   seasonal strength in [0,1]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import shapiro, theilslopes

from core.base_agent import AgentResult, AgentStatus, BaseAgent
from core.context_store import ContextStore
from core.stats_compat import (
    adf_test, kpss_test, stl_decompose, hp_filter,
    acf, pacf, ljungbox, arch_lm, ols_fit,
)


class DecompositionAgent(BaseAgent):

    def __init__(self, context_store=None):
        super().__init__("DecompositionAgent", context_store)

    def validate_inputs(self, series: Any = None, **kwargs):
        if series is None:
            raise ValueError("DecompositionAgent requires 'series' (pd.Series).")

    def _run(
        self,
        series: pd.Series,
        period: Optional[int] = None,
        model: str = "additive",
        hp_lambda: float = 1600.0,
        run_stationarity: bool = True,
        **kwargs,
    ) -> AgentResult:

        warn_list: List[str] = []
        series = series.dropna().sort_index()
        name = series.name or "series"

        if len(series) < 14:
            return AgentResult(
                agent_name=self.name, status=AgentStatus.FAILED,
                errors=[f"Series '{name}' too short (< 14 obs)."]
            )

        y = series.values.astype(float)

        # 1. stationarity
        stationarity: Dict = {}
        d_order = 0
        if run_stationarity:
            stationarity, d_order = self._stationarity_tests(y, warn_list)

        # 2. period detection
        if period is None:
            period = self._detect_period(y, series.index)
            warn_list.append(f"Auto-detected period: {period}")
        period = max(2, min(int(period), len(y) // 2))

        # 3. STL-style decomposition
        decomp = stl_decompose(y, period=period, robust=True)
        trend_s    = pd.Series(decomp["trend"],    index=series.index, name="trend")
        seasonal_s = pd.Series(decomp["seasonal"], index=series.index, name="seasonal")
        residual_s = pd.Series(decomp["residual"], index=series.index, name="residual")

        # 4. HP filter (cycle)
        cycle_arr, hp_arr = hp_filter(y, lamb=hp_lambda)
        cycle_s   = pd.Series(cycle_arr, index=series.index, name="cycle_hp")
        hp_trend_s= pd.Series(hp_arr,    index=series.index, name="hp_trend")

        # 5. trend analysis
        trend_stats = self._trend_analysis(y)

        # 6. residual diagnostics
        resid_diag = self._residual_diagnostics(decomp["residual"], warn_list)

        # 7. component strengths
        Ft = float(max(0.0, 1 - np.var(decomp["residual"]) /
                       (np.var(decomp["trend"] + decomp["residual"]) + 1e-12)))
        Fs = float(max(0.0, 1 - np.var(decomp["residual"]) /
                       (np.var(decomp["seasonal"] + decomp["residual"]) + 1e-12)))

        # 8. significant ACF/PACF lags
        sig_acf, sig_pacf = self._sig_lags(y, len(y))

        interpretation = self._interpret(Ft, Fs, stationarity, trend_stats, resid_diag)
        report = self._build_report(
            name, len(y), stationarity, d_order, period, Ft, Fs,
            trend_stats, resid_diag,
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={
                "original": series,
                "trend": trend_s,
                "seasonal": seasonal_s,
                "cycle": cycle_s,
                "residual": residual_s,
                "hp_trend": hp_trend_s,
            },
            metadata={
                "series_name": name,
                "n_obs": len(series),
                "period_used": period,
                "model": model,
                "stationarity": stationarity,
                "differencing_order": d_order,
                "trend_strength_Ft": round(Ft, 4),
                "seasonal_strength_Fs": round(Fs, 4),
                "trend_stats": trend_stats,
                "residual_diagnostics": resid_diag,
                "acf_significant_lags": sig_acf,
                "pacf_significant_lags": sig_pacf,
                "interpretation": interpretation,
                "report": report,
            },
            warnings=warn_list,
        )

    @staticmethod
    def _stationarity_tests(y: np.ndarray, warn_list: List[str]) -> Tuple[Dict, int]:
        results = {}
        try:
            results["ADF"] = adf_test(y)
        except Exception as e:
            warn_list.append(f"ADF test failed: {e}")
        try:
            results["KPSS"] = kpss_test(y)
        except Exception as e:
            warn_list.append(f"KPSS test failed: {e}")

        adf_ok  = results.get("ADF",  {}).get("stationary", True)
        kpss_ok = results.get("KPSS", {}).get("stationary", True)
        d_order = 0 if (adf_ok and kpss_ok) else 1
        if not adf_ok and kpss_ok:
            warn_list.append("Conflicting stationarity; d=1 assumed.")
        return results, d_order

    @staticmethod
    def _trend_analysis(y: np.ndarray) -> Dict:
        stats: Dict = {"n": len(y)}
        x = np.arange(len(y), dtype=float)
        X = np.column_stack([np.ones(len(y)), x])
        try:
            res = ols_fit(X, y)
            stats.update({
                "ols_slope":       round(float(res.params[1]), 6),
                "ols_pvalue":      round(float(res.pvalues[1]), 4),
                "ols_r2":          round(float(res.rsquared), 4),
                "trend_direction": ("upward" if res.params[1] > 0 else
                                    "downward" if res.params[1] < 0 else "flat"),
                "trend_significant": bool(res.pvalues[1] < 0.05),
            })
        except Exception:
            pass
        try:
            ts = theilslopes(y, x)
            stats["theilsen_slope"] = round(float(ts.slope), 6)
        except Exception:
            pass
        return stats

    @staticmethod
    def _residual_diagnostics(resid: np.ndarray, warn_list: List[str]) -> Dict:
        diag: Dict = {}
        r = resid[~np.isnan(resid)]
        if len(r) < 8:
            return diag
        try:
            lb = ljungbox(r, lags=10)
            diag["ljungbox_lb_stat"]        = lb["lb_stat"]
            diag["ljungbox_p_value"]         = lb["lb_pvalue"]
            diag["residuals_autocorrelated"] = lb["lb_pvalue"] < 0.05
        except Exception as e:
            warn_list.append(f"Ljung-Box failed: {e}")
        try:
            if len(r) <= 5000:
                sw_stat, sw_p = shapiro(r)
                diag["shapiro_stat"]     = round(float(sw_stat), 4)
                diag["shapiro_p"]        = round(float(sw_p), 4)
                diag["residuals_normal"] = sw_p > 0.05
        except Exception:
            pass
        try:
            a = arch_lm(r)
            diag["arch_lm_stat"]    = a["lm_stat"]
            diag["arch_lm_p"]       = a["lm_p"]
            diag["heteroskedastic"] = a["lm_p"] < 0.05
        except Exception:
            pass
        return diag

    @staticmethod
    def _sig_lags(y: np.ndarray, n: int, n_lags: int = 40) -> Tuple[List[int], List[int]]:
        n_lags = min(n_lags, n // 2 - 1)
        thresh = 1.96 / np.sqrt(n)
        try:
            acf_v  = acf(y,  nlags=n_lags)
            pacf_v = pacf(y, nlags=n_lags)
            return (
                [i for i, v in enumerate(acf_v[1:],  1) if abs(v) > thresh][:10],
                [i for i, v in enumerate(pacf_v[1:], 1) if abs(v) > thresh][:10],
            )
        except Exception:
            return [], []

    @staticmethod
    def _detect_period(y: np.ndarray, index) -> int:
        y_c = y - np.nanmean(y)
        fft = np.abs(np.fft.rfft(y_c))
        fft[0] = 0
        freqs = np.fft.rfftfreq(len(y))
        top_idx = int(np.argmax(fft[1:])) + 1
        if freqs[top_idx] > 0:
            period = int(round(1.0 / freqs[top_idx]))
            return max(2, min(period, len(y) // 3))
        if hasattr(index, "freq") and index.freq is not None:
            defaults = {"H": 24, "D": 7, "W": 52, "M": 12, "MS": 12, "Q": 4, "QS": 4}
            for k, v in defaults.items():
                if str(index.freq).startswith(k):
                    return v
        return 7

    @staticmethod
    def _interpret(Ft, Fs, stationarity, trend_stats, resid_diag) -> str:
        parts = [
            f"Ft={Ft:.2f} → " + ("Strong" if Ft > 0.6 else "Moderate" if Ft > 0.3 else "Weak") + " trend",
            f"Fs={Fs:.2f} → " + ("Strong" if Fs > 0.6 else "Moderate" if Fs > 0.3 else "Weak") + " seasonality",
        ]
        if "ADF" in stationarity:
            parts.append("ADF: " + ("Stationary" if stationarity["ADF"].get("stationary") else "Non-stationary"))
        if trend_stats.get("trend_significant"):
            parts.append(f"Trend: {trend_stats.get('trend_direction')} (p={trend_stats.get('ols_pvalue')})")
        if resid_diag.get("residuals_autocorrelated"):
            parts.append("⚠ Residuals autocorrelated")
        if resid_diag.get("heteroskedastic"):
            parts.append("⚠ ARCH effects")
        return "; ".join(parts)

    @staticmethod
    def _build_report(name, n_obs, stationarity, d_order, period, Ft, Fs,
                      trend_stats, resid_diag) -> str:
        lines = [
            f"═══ Decomposition Report: {name} ═══",
            f"  Observations      : {n_obs}",
            f"  Seasonal period   : {period}",
            f"  Trend strength Ft : {Ft:.4f}  ({'strong' if Ft > 0.6 else 'moderate' if Ft > 0.3 else 'weak'})",
            f"  Seasonal str.  Fs : {Fs:.4f}  ({'strong' if Fs > 0.6 else 'moderate' if Fs > 0.3 else 'weak'})",
            "", "  Stationarity:",
        ]
        for test, res in stationarity.items():
            lines.append(
                f"    {test:6s}: stat={res.get('statistic', '?'):8}  "
                f"p={res.get('p_value', '?')}  stationary={res.get('stationary', '?')}"
            )
        lines += [
            f"  Differencing order : d={d_order}",
            "", "  Trend (OLS):",
            f"    slope={trend_stats.get('ols_slope', '?')}  p={trend_stats.get('ols_pvalue', '?')}  "
            f"R²={trend_stats.get('ols_r2', '?')}  dir={trend_stats.get('trend_direction', '?')}",
            "", "  Residual diagnostics:",
            f"    Ljung-Box  p={resid_diag.get('ljungbox_p_value', '?')}  "
            f"autocorr={resid_diag.get('residuals_autocorrelated', '?')}",
            f"    Shapiro-W  p={resid_diag.get('shapiro_p', '?')}  "
            f"normal={resid_diag.get('residuals_normal', '?')}",
            f"    ARCH-LM    p={resid_diag.get('arch_lm_p', '?')}  "
            f"heteroskedastic={resid_diag.get('heteroskedastic', '?')}",
        ]
        return "\n".join(lines)
