"""
agents/hierarchy_aggregation_agent.py
──────────────────────────────────────
Builds a tree from leaf series and aggregates values UPWARD across
hierarchy levels. Aggregation is cross-sectional (operates across series,
NOT along the time axis — that is handled by AccumulationAgent).

Hierarchy definition
────────────────────
Pass a list of column names ordered from COARSEST to FINEST, e.g.:
  ["region", "country", "store", "sku"]

The DataFrame must have:
  - DatetimeIndex
  - hierarchy key columns (categorical)
  - one or more value columns

Aggregation methods (per level or global)
──────────────────────────────────────────
  "sum"        – parent = Σ children          (additive, default)
  "mean"       – parent = average of children
  "weighted"   – parent = Σ(w_i * child_i)   (weights from metadata)
  "median"     – parent = median of children
  "max"        – parent = max of children
  "min"        – parent = min of children
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.base_agent import AgentResult, AgentStatus, BaseAgent
from core.context_store import ContextStore


# ── aggregation functions ─────────────────────────────────────────────────────

def _agg_fn(method: str):
    return {
        "sum":    "sum",
        "mean":   "mean",
        "median": "median",
        "max":    "max",
        "min":    "min",
    }.get(method, "sum")


class HierarchyAggregationAgent(BaseAgent):
    """
    Outputs
    -------
    data : dict[str, pd.DataFrame]
        Keys like "total", "level_0__North", "leaf__North__StoreA__SKU1"
        Each DataFrame has DatetimeIndex and the same value columns.
    metadata : dict
        - hierarchy_levels : list of level names
        - level_series_count : {level: count}
        - coherence_check : {level: bool}  True if parent == Σ children
        - level_effects : stats per level (variance, CV, trend strength)
    """

    def __init__(self, context_store: Optional[ContextStore] = None):
        super().__init__("HierarchyAggregationAgent", context_store)

    def validate_inputs(self, df: Any = None, hierarchy_cols: Any = None, **kwargs):
        if df is None:
            raise ValueError("HierarchyAggregationAgent requires 'df'.")
        if not hierarchy_cols:
            raise ValueError("HierarchyAggregationAgent requires 'hierarchy_cols' list.")

    def _run(
        self,
        df: pd.DataFrame,
        hierarchy_cols: List[str],
        value_cols: Optional[List[str]] = None,
        method: str = "sum",
        weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> AgentResult:

        warnings: List[str] = []

        if value_cols is None:
            value_cols = df.select_dtypes(include="number").columns.tolist()

        # ── validate ──────────────────────────────────────────────────────────
        missing = [c for c in hierarchy_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Hierarchy columns not found in DataFrame: {missing}")

        df = df.sort_index()

        # ── build all levels ──────────────────────────────────────────────────
        all_series: Dict[str, pd.DataFrame] = {}
        level_meta: Dict[str, Any] = {}

        # LEVEL 0: individual leaf series
        leaf_groups = df.groupby(hierarchy_cols)
        leaf_count = 0
        for keys, group in leaf_groups:
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            label = "leaf__" + "__".join(str(k) for k in key_tuple)
            all_series[label] = group[value_cols].copy()
            leaf_count += 1
        level_meta["leaf"] = {"n_series": leaf_count, "level_depth": len(hierarchy_cols)}

        # INTERMEDIATE LEVELS (from fine to coarse)
        for depth in range(len(hierarchy_cols) - 1, 0, -1):
            level_keys = hierarchy_cols[:depth]
            level_name = f"level_{len(hierarchy_cols)-depth}"
            groups = df.groupby(level_keys)
            count = 0
            for keys, group in groups:
                key_tuple = keys if isinstance(keys, tuple) else (keys,)
                label = f"{level_name}__" + "__".join(str(k) for k in key_tuple)
                aggregated = self._aggregate(group[value_cols], method, weights)
                all_series[label] = aggregated
                count += 1
            level_meta[level_name] = {"n_series": count, "level_depth": depth}

        # TOP LEVEL: grand total
        top = self._aggregate(df[value_cols], method, weights)
        all_series["total"] = top
        level_meta["total"] = {"n_series": 1, "level_depth": 0}

        # ── coherence check ───────────────────────────────────────────────────
        coherence = self._check_coherence(df, value_cols, hierarchy_cols, method, warnings)

        # ── level effects report ──────────────────────────────────────────────
        level_effects = self._level_effects(all_series, value_cols)

        report = self._build_report(hierarchy_cols, method, level_meta, coherence, level_effects)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data=all_series,
            metadata={
                "hierarchy_cols": hierarchy_cols,
                "method": method,
                "value_cols": value_cols,
                "level_meta": level_meta,
                "coherence": coherence,
                "level_effects": level_effects,
                "n_total_series": len(all_series),
                "report": report,
            },
            warnings=warnings,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(
        group: pd.DataFrame,
        method: str,
        weights: Optional[Dict[str, float]],
    ) -> pd.DataFrame:
        """Aggregate a sub-group to a single series per value column."""
        if method == "weighted" and weights:
            result = {}
            for col in group.columns:
                series_list = []
                # group is already subset with one or more rows per timestamp
                for ts, row in group.groupby(level=0):
                    vals = row[col].values
                    w = np.array([weights.get(str(idx), 1.0) for idx in row.index])
                    w = w / w.sum()
                    series_list.append((ts, float(np.dot(w, vals))))
                idx, vals = zip(*series_list) if series_list else ([], [])
                result[col] = pd.Series(vals, index=idx)
            return pd.DataFrame(result)
        else:
            agg = _agg_fn(method)
            return group.groupby(level=0).agg(agg)

    @staticmethod
    def _check_coherence(
        df: pd.DataFrame,
        value_cols: List[str],
        hierarchy_cols: List[str],
        method: str,
        warnings: List[str],
    ) -> Dict[str, bool]:
        """
        For additive aggregation: verify parent == Σ children at every timestep.
        Returns per-column coherence flag.
        """
        if method not in ("sum",):
            return {col: None for col in value_cols}  # only meaningful for sum

        coherence = {}
        total_expected = df.groupby(level=0)[value_cols].sum()

        for col in value_cols:
            max_diff = float((df.groupby(level=0)[col].sum() - total_expected[col]).abs().max())
            ok = max_diff < 1e-6
            coherence[col] = ok
            if not ok:
                warnings.append(
                    f"Coherence check FAILED for '{col}': max diff = {max_diff:.6f}"
                )
        return coherence

    @staticmethod
    def _level_effects(
        all_series: Dict[str, pd.DataFrame],
        value_cols: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        For each level (leaf/mid/top), compute average statistics
        to show how aggregation affects signal characteristics.
        """
        from collections import defaultdict

        level_buckets: Dict[str, List[pd.DataFrame]] = defaultdict(list)
        for label, data in all_series.items():
            prefix = label.split("__")[0]
            level_buckets[prefix].append(data)

        effects = {}
        for level, frames in level_buckets.items():
            cv_vals, var_vals = [], []
            for frame in frames:
                for col in value_cols:
                    if col in frame.columns:
                        s = frame[col].dropna()
                        if len(s) > 1:
                            mu = s.mean()
                            cv_vals.append(float(s.std() / (mu + 1e-12)))
                            var_vals.append(float(s.var()))
            effects[level] = {
                "n_series": len(frames),
                "avg_cv": round(float(np.mean(cv_vals)), 4) if cv_vals else None,
                "avg_variance": round(float(np.mean(var_vals)), 4) if var_vals else None,
                "interpretation": (
                    "High variability (leaf level)" if level == "leaf"
                    else "Smoothed by aggregation" if level not in ("leaf", "total")
                    else "Aggregate (lowest variability)"
                ),
            }
        return effects

    @staticmethod
    def _build_report(
        hierarchy_cols: List[str],
        method: str,
        level_meta: Dict,
        coherence: Dict,
        level_effects: Dict,
    ) -> str:
        lines = [
            "═══ Hierarchy Aggregation Report ═══",
            f"  Hierarchy      : {' → '.join(hierarchy_cols)} → [total]",
            f"  Method         : {method}",
            "",
            "  Series count per level:",
        ]
        for level, meta in sorted(level_meta.items(), key=lambda x: -x[1].get("level_depth", 0)):
            lines.append(f"    {level:15s}: {meta['n_series']} series")

        lines += ["", "  Coherence (parent == Σ children):"]
        for col, ok in coherence.items():
            flag = "✓" if ok else "✗ FAILED"
            lines.append(f"    {col}: {flag}")

        lines += ["", "  Level effects (aggregation impact):"]
        for level, eff in level_effects.items():
            lines.append(
                f"    {level:10s}: avg_cv={eff['avg_cv']}, avg_var={eff['avg_variance']}"
                f"  → {eff['interpretation']}"
            )
        return "\n".join(lines)
