"""
agents/missing_values_agent.py
────────────────────────────────
Classifies missing data patterns (MCAR / MAR / MNAR),
distinguishes true zeros from structural missings, and
applies the most appropriate imputation strategy.

Imputation methods
------------------
  forward_fill  – propagate last observed value
  backward_fill – propagate next observed value
  linear        – linear interpolation
  spline        – cubic spline interpolation
  seasonal      – fill using same-period values (e.g. same weekday)
  knn           – K-nearest-neighbours (sklearn)
  mean / median – simple global statistics
  zero          – fill with 0 (for structural zeros)
  auto          – choose best method automatically
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.base_agent import AgentResult, AgentStatus, BaseAgent
from core.context_store import ContextStore

try:
    from sklearn.impute import KNNImputer
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class MissingValuesAgent(BaseAgent):
    """
    Parameters
    ----------
    series         : pd.Series
    method         : imputation method (see module docstring)
    period         : seasonal period for seasonal imputation
    knn_neighbors  : k for KNN (default 5)
    zero_as_missing: treat 0 as missing? (default False)
    """

    def __init__(self, context_store: Optional[ContextStore] = None):
        super().__init__("MissingValuesAgent", context_store)

    def validate_inputs(self, series: Any = None, **kwargs):
        if series is None:
            raise ValueError("MissingValuesAgent requires 'series'.")

    def _run(
        self,
        series: pd.Series,
        method: str = "auto",
        period: int = 7,
        knn_neighbors: int = 5,
        zero_as_missing: bool = False,
        **kwargs,
    ) -> AgentResult:

        warn_list: List[str] = []
        series = series.sort_index()
        name = series.name or "series"

        # ── optionally treat zeros as missing ─────────────────────────────────
        s = series.copy().astype(float)
        if zero_as_missing:
            n_zeros = (s == 0).sum()
            s[s == 0] = np.nan
            warn_list.append(f"Treated {n_zeros} zeros as missing.")

        # ── gap analysis ──────────────────────────────────────────────────────
        gap_stats = self._gap_analysis(s)

        # ── pattern classification ────────────────────────────────────────────
        pattern = self._classify_pattern(s, gap_stats, warn_list)

        # ── choose method ─────────────────────────────────────────────────────
        chosen_method = method
        if method == "auto":
            chosen_method = self._auto_method(pattern, gap_stats, warn_list)
            warn_list.append(f"Auto-selected method: {chosen_method}")

        # ── impute ────────────────────────────────────────────────────────────
        imputed = self._impute(s, chosen_method, period, knn_neighbors, warn_list)

        # ── completeness report ───────────────────────────────────────────────
        n_missing_before = int(s.isna().sum())
        n_missing_after = int(imputed.isna().sum())

        completeness = {
            "n_obs_total": len(s),
            "n_missing_before": n_missing_before,
            "pct_missing_before": round(n_missing_before / len(s) * 100, 2),
            "n_missing_after": n_missing_after,
            "pct_missing_after": round(n_missing_after / len(s) * 100, 2),
            "method_used": chosen_method,
            "pattern": pattern,
            "gap_stats": gap_stats,
        }

        report = self._build_report(name, completeness)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={"imputed": imputed, "original": series, "was_missing": s.isna()},
            metadata={"completeness": completeness, "report": report},
            warnings=warn_list,
        )

    # ── gap analysis ──────────────────────────────────────────────────────────

    @staticmethod
    def _gap_analysis(s: pd.Series) -> Dict:
        missing_mask = s.isna()
        n_missing = int(missing_mask.sum())
        if n_missing == 0:
            return {"n_missing": 0, "n_gaps": 0, "max_gap": 0, "avg_gap": 0, "gap_lengths": []}

        # find consecutive runs of NaN
        gaps = []
        in_gap = False
        gap_len = 0
        for v in missing_mask:
            if v:
                in_gap = True
                gap_len += 1
            else:
                if in_gap:
                    gaps.append(gap_len)
                    gap_len = 0
                    in_gap = False
        if in_gap:
            gaps.append(gap_len)

        return {
            "n_missing": n_missing,
            "n_gaps": len(gaps),
            "max_gap": int(max(gaps)) if gaps else 0,
            "avg_gap": round(float(np.mean(gaps)), 2) if gaps else 0,
            "gap_lengths": gaps,
        }

    # ── pattern classification ────────────────────────────────────────────────

    @staticmethod
    def _classify_pattern(s: pd.Series, gap_stats: Dict, warn_list: List[str]) -> str:
        """
        Simplified classification:
          MCAR – missing randomly scattered, no pattern
          MAR  – missingness correlates with time position (e.g. end-of-series gaps)
          MNAR – missingness correlates with value (e.g. large values are missing)
        """
        n_missing = gap_stats["n_missing"]
        if n_missing == 0:
            return "COMPLETE"

        # Check if missings are concentrated at the end (MAR heuristic)
        n = len(s)
        missing_positions = np.where(s.isna())[0]
        later_half_count = int((missing_positions >= n // 2).sum())
        if later_half_count / n_missing > 0.7:
            return "MAR (end-of-series)"

        # Check MNAR: are missing values preceded by extreme values?
        obs = s.dropna()
        hi_thresh = obs.quantile(0.9) if len(obs) > 5 else np.inf
        before_missing = []
        for pos in missing_positions:
            if pos > 0 and not pd.isna(s.iloc[pos - 1]):
                before_missing.append(s.iloc[pos - 1])
        if before_missing:
            frac_high = sum(1 for v in before_missing if v > hi_thresh) / len(before_missing)
            if frac_high > 0.4:
                warn_list.append("MNAR suspected: high values tend to precede missing entries.")
                return "MNAR"

        return "MCAR"

    # ── auto method selection ─────────────────────────────────────────────────

    @staticmethod
    def _auto_method(pattern: str, gap_stats: Dict, warn_list: List[str]) -> str:
        max_gap = gap_stats.get("max_gap", 0)
        pct = gap_stats["n_missing"] / max(1, gap_stats.get("n_obs_total", gap_stats["n_missing"] + 1))
        if gap_stats["n_missing"] == 0:
            return "none"
        if max_gap <= 2:
            return "linear"
        if max_gap <= 7:
            return "spline"
        if pattern in ("MAR (end-of-series)",):
            return "forward_fill"
        if _HAS_SKLEARN and pct < 0.30:
            return "knn"
        return "seasonal"

    # ── imputation ────────────────────────────────────────────────────────────

    def _impute(
        self, s: pd.Series, method: str, period: int, knn_neighbors: int, warn_list: List[str]
    ) -> pd.Series:
        if method == "none" or s.isna().sum() == 0:
            return s.copy()
        if method == "forward_fill":
            return s.ffill()
        if method == "backward_fill":
            return s.bfill()
        if method == "mean":
            return s.fillna(s.mean())
        if method == "median":
            return s.fillna(s.median())
        if method == "zero":
            return s.fillna(0)
        if method == "linear":
            return s.interpolate(method="linear", limit_direction="both")
        if method == "spline":
            try:
                return s.interpolate(method="spline", order=3, limit_direction="both")
            except Exception as e:
                warn_list.append(f"Spline failed ({e}); falling back to linear.")
                return s.interpolate(method="linear", limit_direction="both")
        if method == "seasonal":
            return self._seasonal_impute(s, period, warn_list)
        if method == "knn":
            return self._knn_impute(s, knn_neighbors, warn_list)
        warn_list.append(f"Unknown method '{method}'; using linear.")
        return s.interpolate(method="linear", limit_direction="both")

    @staticmethod
    def _seasonal_impute(s: pd.Series, period: int, warn_list: List[str]) -> pd.Series:
        """Fill each missing value with the mean of same-position values in other cycles."""
        result = s.copy()
        n = len(s)
        for i in range(n):
            if pd.isna(result.iloc[i]):
                same_positions = [j for j in range(i % period, n, period) if j != i and not pd.isna(s.iloc[j])]
                if same_positions:
                    result.iloc[i] = float(np.mean(s.iloc[same_positions].values))
                else:
                    warn_list.append(f"No seasonal peers for index {i}; using global mean.")
                    result.iloc[i] = float(s.mean())
        return result

    @staticmethod
    def _knn_impute(s: pd.Series, k: int, warn_list: List[str]) -> pd.Series:
        if not _HAS_SKLEARN:
            warn_list.append("sklearn not available; falling back to linear.")
            return s.interpolate(method="linear", limit_direction="both")
        X = s.values.reshape(-1, 1)
        imp = KNNImputer(n_neighbors=min(k, int(s.notna().sum() - 1)))
        imputed = imp.fit_transform(X).flatten()
        return pd.Series(imputed, index=s.index, name=s.name)

    @staticmethod
    def _build_report(name: str, completeness: Dict) -> str:
        lines = [
            f"═══ Missing Values Report: {name} ═══",
            f"  Total observations  : {completeness['n_obs_total']}",
            f"  Missing before impute: {completeness['n_missing_before']} ({completeness['pct_missing_before']}%)",
            f"  Missing after impute : {completeness['n_missing_after']} ({completeness['pct_missing_after']}%)",
            f"  Pattern classified  : {completeness['pattern']}",
            f"  Method used         : {completeness['method_used']}",
            "",
            "  Gap analysis:",
            f"    Number of gaps: {completeness['gap_stats']['n_gaps']}",
            f"    Max gap length: {completeness['gap_stats']['max_gap']} periods",
            f"    Avg gap length: {completeness['gap_stats']['avg_gap']} periods",
        ]
        return "\n".join(lines)
