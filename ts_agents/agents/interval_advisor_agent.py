"""
agents/interval_advisor_agent.py
─────────────────────────────────
Recommends the best analysis interval(s) for a time series using:
  1. FFT / spectral density → dominant frequencies
  2. Autocorrelation function (ACF) → lag peaks
  3. Signal-to-noise ratio comparison across candidate intervals
  4. Coverage check → minimum 2 full seasonal cycles required

Returns a ranked list of intervals with scores, rationale, and an
information-loss estimate for each coarser resolution.
"""

from __future__ import annotations

import warnings as _warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from core.base_agent import AgentResult, AgentStatus, BaseAgent
from core.context_store import ContextStore

# ── candidate interval catalogue ─────────────────────────────────────────────

CANDIDATE_INTERVALS: List[Dict[str, Any]] = [
    {"alias": "Hourly",    "freq": "H",  "periods_per_year": 8760},
    {"alias": "Daily",     "freq": "D",  "periods_per_year": 365},
    {"alias": "Weekly",    "freq": "W",  "periods_per_year": 52},
    {"alias": "Bi-weekly", "freq": "2W", "periods_per_year": 26},
    {"alias": "Monthly",   "freq": "MS", "periods_per_year": 12},
    {"alias": "Quarterly", "freq": "QS", "periods_per_year": 4},
    {"alias": "Yearly",    "freq": "AS", "periods_per_year": 1},
]

FREQ_TO_DAYS = {"H": 1/24, "D": 1, "W": 7, "2W": 14, "MS": 30.44, "QS": 91.3, "AS": 365.25}


class IntervalAdvisorAgent(BaseAgent):

    def __init__(self, context_store: Optional[ContextStore] = None):
        super().__init__("IntervalAdvisorAgent", context_store)

    def validate_inputs(self, series: Any = None, **kwargs):
        if series is None:
            raise ValueError("IntervalAdvisorAgent requires 'series' (pd.Series).")

    def _run(
        self,
        series: pd.Series,
        native_freq: Optional[str] = None,
        top_n: int = 3,
        **kwargs,
    ) -> AgentResult:

        series = series.dropna().sort_index()
        if len(series) < 10:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=["Series too short (< 10 observations) for interval analysis."],
            )

        warn_list: List[str] = []

        # ── 1. normalise to float array ───────────────────────────────────────
        y = series.values.astype(float)
        y_norm = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)

        # ── 2. FFT → dominant period in "steps" ───────────────────────────────
        fft_periods = self._fft_dominant_periods(y_norm)

        # ── 3. native step size in days ───────────────────────────────────────
        native_step_days = self._native_step_days(series)

        # ── 4. score each candidate interval ─────────────────────────────────
        scored = []
        for cand in CANDIDATE_INTERVALS:
            cand_days = FREQ_TO_DAYS.get(cand["freq"], 30)
            if cand_days < native_step_days * 0.9:
                # can't resample to finer than native
                continue
            score, rationale, info_loss = self._score_interval(
                series, y, y_norm, cand, cand_days,
                native_step_days, fft_periods, warn_list,
            )
            scored.append({**cand, "score": score, "rationale": rationale,
                           "info_loss_pct": info_loss})

        if not scored:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=["No valid candidate intervals found."],
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        recommendations = scored[:top_n]

        summary = self._build_summary(recommendations, fft_periods, native_step_days)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data=recommendations,
            metadata={
                "best_interval": recommendations[0]["freq"],
                "best_alias": recommendations[0]["alias"],
                "fft_dominant_periods_steps": fft_periods,
                "native_step_days": native_step_days,
                "native_freq": native_freq,
                "summary": summary,
                "all_scored": scored,
            },
            warnings=warn_list,
        )

    # ── scoring logic ─────────────────────────────────────────────────────────

    def _score_interval(
        self,
        series: pd.Series,
        y: np.ndarray,
        y_norm: np.ndarray,
        cand: Dict,
        cand_days: float,
        native_step_days: float,
        fft_periods: List[float],
        warn_list: List[str],
    ) -> Tuple[float, str, float]:

        rationale_parts = []
        score = 0.0

        # ── a. resample to candidate interval ─────────────────────────────────
        try:
            resampled = series.resample(cand["freq"]).sum()
            resampled = resampled.dropna()
        except Exception:
            return 0.0, "Resampling failed.", 100.0

        n = len(resampled)
        if n < 8:
            return 0.0, "Too few periods after resampling.", 100.0

        # ── b. coverage: need ≥ 2 full cycles of dominant seasonal period ─────
        yr = cand["periods_per_year"]
        coverage_cycles = n / yr if yr > 0 else 0
        if coverage_cycles >= 2:
            score += 25
            rationale_parts.append(f"Good coverage ({coverage_cycles:.1f} annual cycles)")
        elif coverage_cycles >= 1:
            score += 10
            rationale_parts.append(f"Marginal coverage ({coverage_cycles:.1f} annual cycles)")
        else:
            score -= 10
            rationale_parts.append(f"Insufficient coverage ({coverage_cycles:.2f} annual cycles)")

        # ── c. SNR ────────────────────────────────────────────────────────────
        snr = self._snr(resampled.values.astype(float))
        snr_score = min(30, snr * 3)
        score += snr_score
        rationale_parts.append(f"SNR={snr:.2f}")

        # ── d. ACF at lag-1 (want meaningful autocorrelation) ─────────────────
        acf1 = self._acf_lag1(resampled.values.astype(float))
        acf_score = abs(acf1) * 20
        score += acf_score
        rationale_parts.append(f"ACF(1)={acf1:.2f}")

        # ── e. alignment with FFT dominant periods ────────────────────────────
        ratio = cand_days / max(native_step_days, 1e-9)
        for dom_period in fft_periods:
            alignment = 1 / (1 + abs(ratio - dom_period) / (dom_period + 1e-6))
            score += alignment * 15
            if alignment > 0.7:
                rationale_parts.append(f"Aligns with dominant period ~{dom_period:.0f} steps")

        # ── f. information loss ───────────────────────────────────────────────
        info_loss = max(0.0, min(100.0, (1 - native_step_days / max(cand_days, 1e-9)) * 100)) \
            if cand_days > native_step_days else 0.0

        return round(score, 2), "; ".join(rationale_parts), round(info_loss, 1)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fft_dominant_periods(y_norm: np.ndarray, top_k: int = 3) -> List[float]:
        n = len(y_norm)
        fft_vals = np.abs(np.fft.rfft(y_norm))
        freqs = np.fft.rfftfreq(n)
        fft_vals[0] = 0  # remove DC
        idx_sorted = np.argsort(fft_vals)[::-1]
        periods = []
        for idx in idx_sorted:
            if freqs[idx] > 0:
                period = 1.0 / freqs[idx]
                if 2 < period < n / 2:
                    periods.append(round(period, 1))
                    if len(periods) == top_k:
                        break
        return periods

    @staticmethod
    def _native_step_days(series: pd.Series) -> float:
        if len(series) < 2:
            return 1.0
        gaps = series.index.to_series().diff().dropna()
        median_gap = gaps.median()
        return pd.Timedelta(median_gap).total_seconds() / 86400

    @staticmethod
    def _snr(y: np.ndarray) -> float:
        if len(y) < 4:
            return 0.0
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            trend = pd.Series(y).rolling(max(2, len(y) // 5), center=True, min_periods=1).mean().values
        signal_var = np.var(trend)
        noise_var = np.var(y - trend) + 1e-12
        return float(signal_var / noise_var)

    @staticmethod
    def _acf_lag1(y: np.ndarray) -> float:
        if len(y) < 3:
            return 0.0
        y_c = y - np.mean(y)
        denom = np.dot(y_c, y_c)
        if denom == 0:
            return 0.0
        return float(np.dot(y_c[:-1], y_c[1:]) / denom)

    @staticmethod
    def _build_summary(recs: List[Dict], fft_periods: List[float], step_days: float) -> str:
        best = recs[0]
        lines = [
            f"Recommended interval: {best['alias']} ({best['freq']})  score={best['score']}",
            f"Rationale: {best['rationale']}",
            f"Info loss vs native: {best['info_loss_pct']}%",
            f"FFT dominant periods (in native steps): {fft_periods}",
            f"Native step: {step_days:.2f} days",
        ]
        if len(recs) > 1:
            lines.append("Alternatives: " + ", ".join(
                f"{r['alias']}(score={r['score']})" for r in recs[1:]
            ))
        return "\n".join(lines)
