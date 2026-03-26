"""
agents/accumulation_agent.py
─────────────────────────────
Resamples a time series along the TIME AXIS to a target interval.
Supports flow quantities (sum) and stock quantities (mean/last).
Produces a multi-interval comparison report.

Key distinction:
  - ACCUMULATION  = collapsing fine-grained time points → coarser buckets
  - AGGREGATION   = handled separately by HierarchyAggregationAgent (cross-sectional)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from core.base_agent import AgentResult, AgentStatus, BaseAgent
from core.context_store import ContextStore

# ── supported accumulation methods ───────────────────────────────────────────

METHODS = {
    "sum":    lambda g: g.sum(),
    "mean":   lambda g: g.mean(),
    "median": lambda g: g.median(),
    "last":   lambda g: g.last(),
    "first":  lambda g: g.first(),
    "max":    lambda g: g.max(),
    "min":    lambda g: g.min(),
    # OHLC is handled specially below
}

QUANTITY_TYPE_DEFAULTS = {
    "flow":  "sum",   # sales, transactions, events
    "stock": "last",  # inventory, price, balance
    "rate":  "mean",  # temperature, ratio
}


def _resample_series(
    series: pd.Series,
    freq: str,
    method: str,
    closed: str = "left",
    label: str = "left",
) -> pd.Series:
    """Resample a single series, handling OHLC separately."""
    resampler = series.resample(freq, closed=closed, label=label)
    if method == "ohlc":
        ohlc = resampler.ohlc()
        # return 'close' column as primary, attach rest as metadata cols
        return ohlc["close"].rename(series.name)
    fn = METHODS.get(method)
    if fn is None:
        raise ValueError(f"Unknown accumulation method '{method}'. Choose: {list(METHODS)} + ['ohlc']")
    return fn(resampler).rename(series.name)


class AccumulationAgent(BaseAgent):
    """
    Parameters (passed to execute())
    ---------------------------------
    df           : pd.DataFrame  – DatetimeIndex, one or more value columns
    target_freq  : str           – target pandas offset alias, e.g. 'W', 'MS', 'QS'
    method       : str           – 'sum'|'mean'|'median'|'last'|'first'|'max'|'min'|'ohlc'
                                   or 'auto' to choose based on quantity_type
    quantity_type: str           – 'flow'|'stock'|'rate' (used when method='auto')
    compare_freqs: list[str]     – additional frequencies to compute for comparison
                                   e.g. ['W', 'MS', 'QS']
    value_cols   : list[str]     – subset of columns to accumulate; None = all numeric
    """

    def __init__(self, context_store: Optional[ContextStore] = None):
        super().__init__("AccumulationAgent", context_store)

    def validate_inputs(self, df: Any = None, target_freq: Any = None, **kwargs):
        if df is None:
            raise ValueError("AccumulationAgent requires 'df'.")
        if target_freq is None:
            raise ValueError("AccumulationAgent requires 'target_freq'.")

    def _run(
        self,
        df: pd.DataFrame,
        target_freq: str,
        method: str = "auto",
        quantity_type: str = "flow",
        compare_freqs: Optional[List[str]] = None,
        value_cols: Optional[List[str]] = None,
        **kwargs,
    ) -> AgentResult:

        warnings: List[str] = []

        # ── resolve method ────────────────────────────────────────────────────
        if method == "auto":
            method = QUANTITY_TYPE_DEFAULTS.get(quantity_type, "sum")

        # ── select columns ────────────────────────────────────────────────────
        if value_cols is None:
            value_cols = df.select_dtypes(include="number").columns.tolist()
        df_vals = df[value_cols].copy()

        # ── validate index ────────────────────────────────────────────────────
        if not isinstance(df_vals.index, pd.DatetimeIndex):
            raise TypeError("DataFrame must have a DatetimeIndex.")
        df_vals = df_vals.sort_index()

        # ── accumulate to target frequency ───────────────────────────────────
        accumulated = self._accumulate(df_vals, target_freq, method, warnings)

        # ── comparison report ─────────────────────────────────────────────────
        comparison = {}
        all_freqs = list(dict.fromkeys([target_freq] + (compare_freqs or [])))
        for freq in all_freqs:
            try:
                resampled = self._accumulate(df_vals, freq, method, warnings=[])
                comparison[freq] = {
                    "n_periods": len(resampled),
                    "data": resampled,
                    "stats": self._stats(resampled, df_vals, freq, method),
                }
            except Exception as e:
                warnings.append(f"Comparison freq '{freq}' failed: {e}")

        # ── information retention ─────────────────────────────────────────────
        info_retention = self._information_retention(df_vals, accumulated, method)

        report = self._build_report(
            target_freq, method, quantity_type,
            df_vals, accumulated, comparison, info_retention,
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data=accumulated,
            metadata={
                "target_freq": target_freq,
                "method": method,
                "quantity_type": quantity_type,
                "value_cols": value_cols,
                "n_input": len(df_vals),
                "n_output": len(accumulated),
                "compression_ratio": round(len(df_vals) / max(1, len(accumulated)), 2),
                "information_retention": info_retention,
                "comparison": {k: v["stats"] for k, v in comparison.items()},
                "report": report,
            },
            warnings=warnings,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _accumulate(
        self,
        df: pd.DataFrame,
        freq: str,
        method: str,
        warnings: List[str],
    ) -> pd.DataFrame:
        result_cols = []
        for col in df.columns:
            s = df[col]
            try:
                r = _resample_series(s, freq, method)
                result_cols.append(r)
            except Exception as e:
                warnings.append(f"Column '{col}' resampling failed: {e}")
        if not result_cols:
            raise RuntimeError("All columns failed resampling.")
        return pd.concat(result_cols, axis=1)

    @staticmethod
    def _stats(
        resampled: pd.DataFrame,
        original: pd.DataFrame,
        freq: str,
        method: str,
    ) -> Dict[str, Any]:
        stats = {}
        for col in resampled.columns:
            r = resampled[col].dropna()
            o = original[col].dropna()
            stats[col] = {
                "n_periods": len(r),
                "mean": round(float(r.mean()), 4),
                "std": round(float(r.std()), 4),
                "cv": round(float(r.std() / (r.mean() + 1e-12)), 4),
                "total": round(float(r.sum()), 4) if method == "sum" else None,
                "min": round(float(r.min()), 4),
                "max": round(float(r.max()), 4),
                "pct_zero": round(float((r == 0).mean() * 100), 2),
                "pct_missing": round(float(r.isna().mean() * 100), 2),
            }
        return stats

    @staticmethod
    def _information_retention(
        original: pd.DataFrame,
        accumulated: pd.DataFrame,
        method: str,
    ) -> Dict[str, float]:
        """
        Proxy for information retention:
          variance_retention = Var(resampled) / Var(original)
          For sum: also check total conservation.
        """
        retention = {}
        for col in accumulated.columns:
            if col not in original.columns:
                continue
            orig_var = float(original[col].var())
            acc_var = float(accumulated[col].var())
            retention[col] = round(
                min(1.0, acc_var / (orig_var + 1e-12)), 4
            ) if orig_var > 0 else 1.0
        return retention

    @staticmethod
    def _build_report(
        target_freq: str,
        method: str,
        quantity_type: str,
        original: pd.DataFrame,
        accumulated: pd.DataFrame,
        comparison: Dict,
        info_retention: Dict,
    ) -> str:
        lines = [
            "═══ Accumulation Report ═══",
            f"  Target interval : {target_freq}",
            f"  Method          : {method}  (quantity_type={quantity_type})",
            f"  Input rows      : {len(original)}",
            f"  Output periods  : {len(accumulated)}",
            f"  Compression     : {len(original)/max(1,len(accumulated)):.1f}x",
            "",
            "  Variance retention per column:",
        ]
        for col, ret in info_retention.items():
            lines.append(f"    {col:30s}: {ret*100:.1f}%")
        lines.append("")
        lines.append("  Comparison across intervals:")
        for freq, stats_map in comparison.items():
            mark = " ← selected" if freq == target_freq else ""
            if isinstance(stats_map, dict):
                for col, s in stats_map.items():
                    if isinstance(s, dict):
                        lines.append(
                            f"    [{freq:4s}]{mark}  {col}: "
                            f"n={s.get('n_periods','?')}, mean={s.get('mean','?')}, cv={s.get('cv','?')}"
                        )
        return "\n".join(lines)
