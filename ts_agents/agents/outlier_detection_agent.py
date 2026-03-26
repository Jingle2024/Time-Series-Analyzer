"""
agents/outlier_detection_agent.py
───────────────────────────────────
Detects point, contextual, and collective outliers using multiple methods.
Each method assigns a binary flag; a consensus score is produced.

Methods
-------
  iqr         – 1.5×IQR fence (fast, robust)
  zscore      – |z| > threshold  (assumes normality)
  isof        – Isolation Forest (sklearn) — unsupervised, multivariate-capable
  lof         – Local Outlier Factor (sklearn)
  residual    – applied on STL residuals (best for contextual outliers)

Severity scoring
----------------
  severity = fraction of methods that flag the point  (0.0 – 1.0)
  HIGH   if severity >= 0.6
  MEDIUM if severity >= 0.4
  LOW    otherwise

Treatment recommendations per outlier:
  - SHORT isolated spike         → cap to fence value
  - LONG run / collective        → investigate & flag
  - Post-decomposition residual  → remove or impute
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.base_agent import AgentResult, AgentStatus, BaseAgent
from core.context_store import ContextStore

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ── individual detectors ──────────────────────────────────────────────────────

def _detect_iqr(y: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    q1, q3 = np.nanpercentile(y, 25), np.nanpercentile(y, 75)
    iqr = q3 - q1
    return (y < q1 - multiplier * iqr) | (y > q3 + multiplier * iqr)


def _detect_zscore(y: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    mu, sigma = np.nanmean(y), np.nanstd(y)
    if sigma == 0:
        return np.zeros(len(y), dtype=bool)
    return np.abs((y - mu) / sigma) > threshold


def _detect_isolation_forest(y: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    if not _HAS_SKLEARN:
        return np.zeros(len(y), dtype=bool)
    X = y.reshape(-1, 1)
    mask = np.isnan(X).flatten()
    flags = np.zeros(len(y), dtype=bool)
    X_clean = X[~mask]
    if len(X_clean) < 10:
        return flags
    clf = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    preds = clf.fit_predict(X_clean)
    clean_flags = preds == -1
    flags[~mask] = clean_flags
    return flags


def _detect_lof(y: np.ndarray, n_neighbors: int = 20, contamination: float = 0.05) -> np.ndarray:
    if not _HAS_SKLEARN:
        return np.zeros(len(y), dtype=bool)
    X = y.reshape(-1, 1)
    mask = np.isnan(X).flatten()
    flags = np.zeros(len(y), dtype=bool)
    X_clean = X[~mask]
    n_neighbors = min(n_neighbors, max(2, len(X_clean) - 1))
    if len(X_clean) < 5:
        return flags
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    preds = clf.fit_predict(X_clean)
    flags[~mask] = preds == -1
    return flags


def _detect_residual(residual: Optional[np.ndarray], multiplier: float = 3.0) -> np.ndarray:
    if residual is None:
        return None
    mu = np.nanmean(residual)
    sigma = np.nanstd(residual)
    if sigma == 0:
        return np.zeros(len(residual), dtype=bool)
    return np.abs((residual - mu) / sigma) > multiplier


# ── agent ─────────────────────────────────────────────────────────────────────

class OutlierDetectionAgent(BaseAgent):
    """
    Parameters
    ----------
    series       : pd.Series  — the target series
    residual     : pd.Series  — optional STL residual (for contextual detection)
    methods      : list of method names to use (default: all)
    iqr_mult     : IQR multiplier (default 1.5)
    z_thresh     : Z-score threshold (default 3.0)
    contamination: expected fraction of outliers (default 0.05)
    """

    def __init__(self, context_store: Optional[ContextStore] = None):
        super().__init__("OutlierDetectionAgent", context_store)

    def validate_inputs(self, series: Any = None, **kwargs):
        if series is None:
            raise ValueError("OutlierDetectionAgent requires 'series'.")

    def _run(
        self,
        series: pd.Series,
        residual: Optional[pd.Series] = None,
        methods: Optional[List[str]] = None,
        iqr_mult: float = 1.5,
        z_thresh: float = 3.0,
        contamination: float = 0.05,
        **kwargs,
    ) -> AgentResult:

        warn_list: List[str] = []
        series = series.dropna().sort_index()
        name = series.name or "series"
        y = series.values.astype(float)
        n = len(y)

        if methods is None:
            methods = ["iqr", "zscore", "isof", "lof", "residual"]

        if not _HAS_SKLEARN and any(m in methods for m in ["isof", "lof"]):
            warn_list.append("sklearn not installed; skipping Isolation Forest and LOF.")
            methods = [m for m in methods if m not in ("isof", "lof")]

        # ── run detectors ─────────────────────────────────────────────────────
        flag_matrix: Dict[str, np.ndarray] = {}

        if "iqr" in methods:
            flag_matrix["iqr"] = _detect_iqr(y, iqr_mult)

        if "zscore" in methods:
            flag_matrix["zscore"] = _detect_zscore(y, z_thresh)

        if "isof" in methods:
            flag_matrix["isof"] = _detect_isolation_forest(y, contamination)

        if "lof" in methods:
            flag_matrix["lof"] = _detect_lof(y, contamination=contamination)

        if "residual" in methods:
            if residual is not None:
                res_arr = residual.reindex(series.index).values.astype(float)
                rf = _detect_residual(res_arr, multiplier=z_thresh)
                if rf is not None:
                    flag_matrix["residual"] = rf
            else:
                warn_list.append("No residual series provided; skipping residual-based detection.")

        if not flag_matrix:
            return AgentResult(
                agent_name=self.name, status=AgentStatus.FAILED,
                errors=["No detection methods could run."],
            )

        # ── consensus score ───────────────────────────────────────────────────
        stack = np.column_stack([v.astype(float) for v in flag_matrix.values()])
        severity_score = stack.mean(axis=1)   # fraction of methods flagging each point
        is_outlier = severity_score >= (1 / (len(flag_matrix) + 1))  # flagged by ≥ 1 method

        # ── build outlier table ───────────────────────────────────────────────
        outlier_records = []
        for i, ts in enumerate(series.index):
            if is_outlier[i]:
                sev = severity_score[i]
                level = "HIGH" if sev >= 0.6 else "MEDIUM" if sev >= 0.4 else "LOW"
                # fences for capping
                q1 = np.nanpercentile(y, 25)
                q3 = np.nanpercentile(y, 75)
                iqr = q3 - q1
                upper = q3 + iqr_mult * iqr
                lower = q1 - iqr_mult * iqr
                treatment = (
                    f"cap to [{lower:.2f}, {upper:.2f}]"
                    if level in ("LOW", "MEDIUM")
                    else "investigate / impute"
                )
                outlier_records.append({
                    "timestamp": ts,
                    "value": round(float(y[i]), 4),
                    "severity": level,
                    "severity_score": round(float(sev), 3),
                    "flagged_by": [m for m, flags in flag_matrix.items() if flags[i]],
                    "treatment": treatment,
                })

        outlier_df = pd.DataFrame(outlier_records)

        # ── fence values for capping ──────────────────────────────────────────
        q1 = float(np.nanpercentile(y, 25))
        q3 = float(np.nanpercentile(y, 75))
        iqr_val = q3 - q1
        fences = {
            "lower_1.5iqr": round(q1 - 1.5 * iqr_val, 4),
            "upper_1.5iqr": round(q3 + 1.5 * iqr_val, 4),
            "lower_3iqr":   round(q1 - 3.0 * iqr_val, 4),
            "upper_3iqr":   round(q3 + 3.0 * iqr_val, 4),
        }

        summary = {
            "series_name": name,
            "n_obs": n,
            "n_outliers": int(is_outlier.sum()),
            "pct_outliers": round(float(is_outlier.mean() * 100), 2),
            "high_severity": int((severity_score >= 0.6).sum()),
            "medium_severity": int(((severity_score >= 0.4) & (severity_score < 0.6)).sum()),
            "low_severity": int(((severity_score > 0) & (severity_score < 0.4)).sum()),
            "methods_used": list(flag_matrix.keys()),
            "fences": fences,
        }

        report = self._build_report(summary, outlier_df)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={
                "outlier_table": outlier_df,
                "severity_scores": pd.Series(severity_score, index=series.index, name="severity"),
                "is_outlier": pd.Series(is_outlier, index=series.index, name="is_outlier"),
                "flag_matrix": {m: pd.Series(f, index=series.index) for m, f in flag_matrix.items()},
            },
            metadata={"summary": summary, "report": report},
            warnings=warn_list,
        )

    @staticmethod
    def _build_report(summary: Dict, outlier_df: pd.DataFrame) -> str:
        lines = [
            "═══ Outlier Detection Report ═══",
            f"  Series       : {summary['series_name']}",
            f"  Observations : {summary['n_obs']}",
            f"  Methods      : {', '.join(summary['methods_used'])}",
            f"  Outliers     : {summary['n_outliers']} ({summary['pct_outliers']}%)",
            f"    HIGH       : {summary['high_severity']}",
            f"    MEDIUM     : {summary['medium_severity']}",
            f"    LOW        : {summary['low_severity']}",
            "",
            "  Fences (IQR-based):",
        ]
        for k, v in summary["fences"].items():
            lines.append(f"    {k:20s}: {v}")
        if len(outlier_df) > 0:
            lines.append("\n  Top outliers:")
            top = outlier_df.sort_values("severity_score", ascending=False).head(10)
            for _, row in top.iterrows():
                lines.append(
                    f"    {str(row['timestamp'])[:19]}  val={row['value']:10.4f}"
                    f"  {row['severity']:6s} (score={row['severity_score']})  → {row['treatment']}"
                )
        return "\n".join(lines)
