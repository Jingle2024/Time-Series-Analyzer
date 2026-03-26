"""
agents/intermittency_agent.py
──────────────────────────────
Classifies demand series by intermittency using the
Syntetos-Boylan (2005) framework:

  ADI = Average Demand Interval  (mean periods between non-zero obs)
  CV² = Coefficient of Variation squared of non-zero demand sizes

  Classification matrix:
  ┌──────────────────────┬──────────────────────┐
  │  ADI < 1.32          │  ADI ≥ 1.32          │
  ├──────────────────────┼──────────────────────┤
  │  Smooth  (CV² < 0.49)│ Intermittent (CV²<0.49)│
  │  Erratic (CV² ≥ 0.49)│ Lumpy    (CV² ≥ 0.49) │
  └──────────────────────┴──────────────────────┘

Model recommendations per class
─────────────────────────────────
  Smooth      → ARIMA, ETS, Linear regression
  Erratic     → ETS (multiplicative error), TBATS
  Intermittent→ Croston, SBA (Syntetos-Boylan approx.)
  Lumpy       → TSB (Teunter-Syntetos-Babai), iETS, bootstrapping
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.base_agent import AgentResult, AgentStatus, BaseAgent
from core.context_store import ContextStore

# ── Syntetos-Boylan thresholds ────────────────────────────────────────────────
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49

MODEL_RECS = {
    "Smooth":       ["ARIMA", "ETS (additive)", "Linear Regression", "Holt-Winters"],
    "Erratic":      ["ETS (multiplicative error)", "TBATS", "BATS", "LightGBM"],
    "Intermittent": ["Croston", "SBA (Syntetos-Boylan Approx.)", "TSB"],
    "Lumpy":        ["TSB (Teunter-Syntetos-Babai)", "iETS", "Bootstrap aggregation", "Zero-inflated models"],
}


class IntermittencyAgent(BaseAgent):
    """
    Accepts a demand series and returns:
      - ADI, CV², classification label
      - Demand occurrence model (Bernoulli / Negative Binomial)
      - Model recommendations
      - Croston decomposition (demand size + demand interval series)
    """

    def __init__(self, context_store: Optional[ContextStore] = None):
        super().__init__("IntermittencyAgent", context_store)

    def validate_inputs(self, series: Any = None, **kwargs):
        if series is None:
            raise ValueError("IntermittencyAgent requires 'series'.")

    def _run(
        self,
        series: pd.Series,
        zero_threshold: float = 0.0,   # values ≤ this are treated as zero-demand
        **kwargs,
    ) -> AgentResult:

        series = series.dropna().sort_index().astype(float)
        name = series.name or "series"

        # ── basic stats ───────────────────────────────────────────────────────
        n = len(series)
        nonzero_mask = series.values > zero_threshold
        n_nonzero = int(nonzero_mask.sum())
        n_zero = n - n_nonzero
        pct_zero = round(n_zero / n * 100, 2)

        if n_nonzero == 0:
            return AgentResult(
                agent_name=self.name, status=AgentStatus.FAILED,
                errors=[f"Series '{name}' is all zeros — cannot classify intermittency."]
            )

        # ── ADI ───────────────────────────────────────────────────────────────
        # average number of periods between consecutive non-zero observations
        adi = n / n_nonzero

        # ── CV² ───────────────────────────────────────────────────────────────
        nonzero_demands = series.values[nonzero_mask]
        mu_nz = float(np.mean(nonzero_demands))
        sigma_nz = float(np.std(nonzero_demands, ddof=1)) if n_nonzero > 1 else 0.0
        cv2 = (sigma_nz / (mu_nz + 1e-12)) ** 2

        # ── classification ────────────────────────────────────────────────────
        label = self._classify(adi, cv2)
        model_recs = MODEL_RECS[label]

        # ── Croston decomposition ─────────────────────────────────────────────
        demand_sizes, demand_intervals = self._croston_decompose(series, zero_threshold)

        # ── Croston forecast (basic) ──────────────────────────────────────────
        croston_forecast = self._croston_forecast(series, alpha=0.1, zero_threshold=zero_threshold)

        # ── demand distribution fit ───────────────────────────────────────────
        dist_info = self._fit_demand_distribution(nonzero_demands)

        summary = {
            "series_name": name,
            "n_obs": n,
            "n_nonzero": n_nonzero,
            "n_zero": n_zero,
            "pct_zero": pct_zero,
            "ADI": round(adi, 4),
            "CV2": round(cv2, 4),
            "classification": label,
            "model_recommendations": model_recs,
            "mean_nonzero_demand": round(mu_nz, 4),
            "std_nonzero_demand": round(sigma_nz, 4),
            "demand_distribution": dist_info,
            "croston_forecast_1step": round(float(croston_forecast), 4) if croston_forecast else None,
        }

        report = self._build_report(summary)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={
                "demand_sizes": demand_sizes,
                "demand_intervals": demand_intervals,
                "croston_forecast": croston_forecast,
            },
            metadata={"summary": summary, "report": report},
        )

    # ── classification ────────────────────────────────────────────────────────

    @staticmethod
    def _classify(adi: float, cv2: float) -> str:
        if adi < ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
            return "Smooth"
        elif adi < ADI_THRESHOLD and cv2 >= CV2_THRESHOLD:
            return "Erratic"
        elif adi >= ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
            return "Intermittent"
        else:
            return "Lumpy"

    # ── Croston decomposition ─────────────────────────────────────────────────

    @staticmethod
    def _croston_decompose(
        series: pd.Series, zero_threshold: float
    ) -> tuple[pd.Series, pd.Series]:
        """
        Extract two sub-series:
          demand_sizes     – non-zero demand values (when demand occurs)
          demand_intervals – inter-demand intervals (periods between non-zero obs)
        """
        nonzero_idx = np.where(series.values > zero_threshold)[0]
        if len(nonzero_idx) == 0:
            empty = pd.Series([], dtype=float)
            return empty, empty

        sizes = series.iloc[nonzero_idx].rename("demand_size")
        intervals = pd.Series(
            np.diff(nonzero_idx, prepend=nonzero_idx[0]),
            index=series.index[nonzero_idx],
            name="demand_interval",
        )
        return sizes, intervals

    # ── Croston forecast ──────────────────────────────────────────────────────

    @staticmethod
    def _croston_forecast(
        series: pd.Series, alpha: float = 0.1, zero_threshold: float = 0.0
    ) -> Optional[float]:
        """
        Basic Croston (1972):
          z_t = smoothed demand size
          p_t = smoothed inter-demand interval
          forecast = z_t / p_t
        """
        y = series.values.astype(float)
        nonzero_idx = np.where(y > zero_threshold)[0]
        if len(nonzero_idx) < 2:
            return None
        z = float(y[nonzero_idx[0]])
        p = 1.0
        last_nonzero = nonzero_idx[0]
        for i in nonzero_idx[1:]:
            q = float(i - last_nonzero)
            z = alpha * y[i] + (1 - alpha) * z
            p = alpha * q + (1 - alpha) * p
            last_nonzero = i
        return z / (p + 1e-12)

    # ── distribution fit ──────────────────────────────────────────────────────

    @staticmethod
    def _fit_demand_distribution(nonzero_demands: np.ndarray) -> Dict:
        if len(nonzero_demands) < 4:
            return {"note": "Too few observations for distribution fitting"}
        mean = float(np.mean(nonzero_demands))
        var = float(np.var(nonzero_demands, ddof=1))
        dispersion = var / (mean + 1e-12)
        if dispersion < 0.8:
            dist = "Binomial / underdispersed"
        elif dispersion < 1.2:
            dist = "Poisson"
        else:
            dist = "Negative Binomial (overdispersed)"
        return {
            "suggested_distribution": dist,
            "mean": round(mean, 4),
            "variance": round(var, 4),
            "dispersion_ratio": round(dispersion, 4),
        }

    @staticmethod
    def _build_report(summary: Dict) -> str:
        lines = [
            f"═══ Intermittency Report: {summary['series_name']} ═══",
            f"  Observations   : {summary['n_obs']}",
            f"  Non-zero       : {summary['n_nonzero']} ({100-summary['pct_zero']:.1f}%)",
            f"  Zero demand    : {summary['n_zero']} ({summary['pct_zero']}%)",
            f"  ADI            : {summary['ADI']}  (threshold={ADI_THRESHOLD})",
            f"  CV²            : {summary['CV2']}  (threshold={CV2_THRESHOLD})",
            f"  Classification : ★ {summary['classification']}",
            "",
            f"  Mean non-zero demand : {summary['mean_nonzero_demand']}",
            f"  Std  non-zero demand : {summary['std_nonzero_demand']}",
            f"  Demand distribution  : {summary['demand_distribution'].get('suggested_distribution','?')}",
            f"  Croston 1-step fcst  : {summary['croston_forecast_1step']}",
            "",
            "  Recommended models:",
        ]
        for m in summary["model_recommendations"]:
            lines.append(f"    → {m}")
        return "\n".join(lines)
