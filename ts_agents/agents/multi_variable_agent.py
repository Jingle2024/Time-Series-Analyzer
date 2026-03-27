"""
agents/multi_variable_agent.py
───────────────────────────────
Manages multi-variable time series datasets where columns can serve
different roles in a forecasting context:

  Dependent   – the target variable to be forecast (one per model)
  Independent – explanatory / exogenous variables (continuous, numeric)
  Event       – binary {0,1} or Boolean flags representing events,
                holidays, promotions, interventions, etc.
  Hierarchy   – categorical group-by keys (not value series)
  Timestamp   – the time index column

Key capabilities
─────────────────
1. Column role auto-suggestion
   - Timestamp : datetime-parseable or named with date/time hints
   - Event     : numeric column with only {0,1} values (or {True,False})
   - Hierarchy : low-cardinality categorical (string/object dtype, nunique < 30% of rows)
   - Dependent : first non-event numeric column (or user-designated)
   - Independent: remaining numeric columns

2. Cross-correlation analysis
   - Pearson correlation matrix across all numeric value cols
   - Lag-0, lag-k CCF (cross-correlation function) between each
     independent/event col and the dependent col
   - Granger-causality proxy: lagged R² contribution
   - Lead/lag detection: which lag gives the max |CCF|

3. Event impact analysis
   - Average value of dependent var on event vs non-event periods
   - Effect size (Cohen's d) per event column
   - Time-to-peak after event (post-event window analysis)

4. Feature recommendation for forecasting
   - Which independent cols are most correlated at which lag
   - Which event cols show statistically significant lift
   - Suggested lag window per independent variable
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.base_agent import AgentResult, AgentStatus, BaseAgent
from core.context_store import ContextStore


# ── column role constants ─────────────────────────────────────────────────────
ROLE_DEPENDENT   = "dependent"
ROLE_INDEPENDENT = "independent"
ROLE_EVENT       = "event"
ROLE_HIERARCHY   = "hierarchy"
ROLE_TIMESTAMP   = "timestamp"
ROLE_IGNORE      = "ignore"

ALL_ROLES = [ROLE_DEPENDENT, ROLE_INDEPENDENT, ROLE_EVENT,
             ROLE_HIERARCHY, ROLE_TIMESTAMP, ROLE_IGNORE]


# ── auto-detection helpers ────────────────────────────────────────────────────

def detect_event_columns(df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
    """
    A column is an Event candidate if:
      - It is numeric AND
      - Its unique non-null values are a subset of {0, 1} (or {0.0, 1.0})
        OR it is boolean dtype
      - AND it has at least some 1s (not all zeros)
    """
    events = []
    for col in numeric_cols:
        s = df[col].dropna()
        if s.dtype == bool:
            events.append(col)
            continue
        uniq = set(s.unique())
        # allow {0}, {1}, {0,1}, {0.0, 1.0}, {0.0}, {1.0}
        normalised = {round(float(v), 6) for v in uniq}
        if normalised <= {0.0, 1.0} and 1.0 in normalised:
            events.append(col)
    return events


def detect_hierarchy_columns(df: pd.DataFrame, excluded: set) -> List[str]:
    """Low-cardinality categorical columns."""
    n = len(df)
    result = []
    for col in df.columns:
        if col in excluded:
            continue
        if df[col].dtype == object or str(df[col].dtype).startswith("category"):
            nu = df[col].nunique()
            if nu < max(2, n * 0.30):
                result.append(col)
    return result


def suggest_roles(
    df: pd.DataFrame,
    ts_col: Optional[str] = None,
) -> Dict[str, str]:
    """
    Returns a dict mapping column_name → suggested_role.
    """
    suggestions: Dict[str, str] = {}
    excluded = set()

    # timestamp
    if ts_col:
        suggestions[ts_col] = ROLE_TIMESTAMP
        excluded.add(ts_col)
    # also check index name
    if hasattr(df.index, "name") and df.index.name:
        suggestions[df.index.name] = ROLE_TIMESTAMP

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in excluded]
    hier_cols = detect_hierarchy_columns(df, excluded | set(numeric_cols))

    for col in hier_cols:
        suggestions[col] = ROLE_HIERARCHY
        excluded.add(col)

    event_cols = detect_event_columns(df, numeric_cols)
    for col in event_cols:
        suggestions[col] = ROLE_EVENT
        excluded.add(col)

    non_event_numeric = [c for c in numeric_cols if c not in excluded]
    for i, col in enumerate(non_event_numeric):
        suggestions[col] = ROLE_DEPENDENT if i == 0 else ROLE_INDEPENDENT

    return suggestions


# ── cross-correlation ─────────────────────────────────────────────────────────

def pearson_corr_matrix(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    sub = df[cols].dropna(how="all")
    corr = sub.corr(method="pearson")
    result = {}
    for c in cols:
        result[c] = {}
        for c2 in cols:
            v = corr.loc[c, c2] if c in corr.index and c2 in corr.columns else None
            result[c][c2] = round(float(v), 4) if v is not None and not np.isnan(v) else None
    return result


def cross_correlation_function(
    x: np.ndarray,
    y: np.ndarray,
    max_lags: int = 20,
) -> Dict[str, Any]:
    """
    Compute CCF(y, x at lag k):  correlation between y_t and x_{t-k}.
    Positive lag = x leads y (x causes y).
    Returns lags array, ccf values, best lag, and max |ccf|.
    """
    n = min(len(x), len(y))
    x = x[-n:].astype(float)
    y = y[-n:].astype(float)

    # Normalise
    x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
    y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)

    max_lags = min(max_lags, n // 3)
    lags = list(range(-max_lags, max_lags + 1))
    ccf_vals = []
    for lag in lags:
        if lag < 0:
            # x lags y: y_t correlated with x_{t+|lag|}
            corr = float(np.nanmean(y[:n+lag] * x[-lag:]))
        elif lag > 0:
            # x leads y: y_t correlated with x_{t-lag}
            corr = float(np.nanmean(y[lag:] * x[:n-lag]))
        else:
            corr = float(np.nanmean(y * x))
        ccf_vals.append(round(corr, 4))

    abs_vals = [abs(v) for v in ccf_vals]
    best_idx = int(np.argmax(abs_vals))
    best_lag = lags[best_idx]
    max_ccf  = ccf_vals[best_idx]

    # Significance threshold: 1.96 / sqrt(n)
    sig_thresh = round(1.96 / np.sqrt(n), 4)

    return {
        "lags": lags,
        "ccf":  ccf_vals,
        "best_lag": best_lag,
        "max_ccf": round(max_ccf, 4),
        "sig_threshold": sig_thresh,
        "significant": abs(max_ccf) > sig_thresh,
        "interpretation": _interp_ccf(best_lag, max_ccf, sig_thresh),
    }


def _interp_ccf(best_lag: int, max_ccf: float, thresh: float) -> str:
    if abs(max_ccf) <= thresh:
        return "No significant cross-correlation detected"
    direction = "positive" if max_ccf > 0 else "negative"
    if best_lag > 0:
        return f"Independent leads dependent by {best_lag} period(s) — {direction} association (likely causal)"
    elif best_lag < 0:
        return f"Independent lags dependent by {abs(best_lag)} period(s) — {direction} association (lagging indicator)"
    else:
        return f"Contemporaneous {direction} association (same-period effect)"


def event_impact_analysis(
    dep: pd.Series,
    event: pd.Series,
    window_after: int = 5,
) -> Dict[str, Any]:
    """
    Compare dependent variable on event vs non-event periods.
    Also computes a post-event window average.
    """
    aligned = pd.DataFrame({"dep": dep, "event": event}).dropna()
    on_event  = aligned.loc[aligned["event"] == 1, "dep"]
    off_event = aligned.loc[aligned["event"] == 0, "dep"]

    if len(on_event) == 0 or len(off_event) == 0:
        return {"error": "Insufficient data for event impact analysis"}

    mu_on  = float(on_event.mean())
    mu_off = float(off_event.mean())
    # Cohen's d
    pooled_std = float(np.sqrt((on_event.std()**2 + off_event.std()**2) / 2)) + 1e-12
    cohens_d   = (mu_on - mu_off) / pooled_std
    lift_pct   = (mu_on - mu_off) / (abs(mu_off) + 1e-12) * 100

    # post-event window: average of dep for `window_after` periods after each event
    event_idx = aligned.index[aligned["event"] == 1].tolist()
    post_avgs = []
    for ts in event_idx:
        pos = aligned.index.get_loc(ts)
        end = min(pos + window_after + 1, len(aligned))
        window_vals = aligned["dep"].iloc[pos+1:end].values
        if len(window_vals):
            post_avgs.append(float(window_vals.mean()))
    post_event_avg = round(float(np.mean(post_avgs)), 4) if post_avgs else None

    return {
        "n_event_periods": int(len(on_event)),
        "n_non_event_periods": int(len(off_event)),
        "mean_on_event": round(mu_on, 4),
        "mean_off_event": round(mu_off, 4),
        "lift_pct": round(lift_pct, 2),
        "cohens_d": round(cohens_d, 4),
        "effect_size": "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small",
        "post_event_avg": post_event_avg,
        "window_after": window_after,
    }


def granger_proxy(
    dep: np.ndarray,
    indep: np.ndarray,
    max_lag: int = 8,
) -> Dict[str, Any]:
    """
    Simplified Granger-causality proxy:
    For each lag k, fit y_t ~ y_{t-1} + x_{t-k} and measure R² increment
    over the AR(1) baseline. Returns the lag with the best R² gain.
    """
    n = min(len(dep), len(indep))
    dep   = dep[-n:].astype(float)
    indep = indep[-n:].astype(float)

    def ols_r2(X, y):
        try:
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            ss_res = np.sum((y - X @ b)**2)
            ss_tot = np.sum((y - np.mean(y))**2) + 1e-12
            return max(0.0, float(1 - ss_res / ss_tot))
        except Exception:
            return 0.0

    max_lag = min(max_lag, n // 4)
    if max_lag < 1:
        return {"best_lag": 0, "r2_gain": 0.0, "useful": False}

    # Baseline AR(1): y ~ [1, y_{t-1}]
    y_base = dep[1:]
    X_base = np.column_stack([np.ones(len(y_base)), dep[:-1]])
    r2_base = ols_r2(X_base, y_base)

    best_lag, best_gain = 0, 0.0
    r2_per_lag = {}
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        y_r = dep[lag:]
        x_r = indep[:n - lag]
        ar1 = dep[lag-1:n-1]
        if len(y_r) < 10:
            break
        X_aug = np.column_stack([np.ones(len(y_r)), ar1, x_r])
        r2_aug = ols_r2(X_aug, y_r)
        gain = r2_aug - r2_base
        r2_per_lag[lag] = round(gain, 4)
        if gain > best_gain:
            best_gain = gain
            best_lag = lag

    return {
        "best_lag": best_lag,
        "r2_gain": round(best_gain, 4),
        "r2_base": round(r2_base, 4),
        "r2_per_lag": r2_per_lag,
        "useful": best_gain > 0.01,
    }


# ── AGENT ─────────────────────────────────────────────────────────────────────

class MultiVariableAgent(BaseAgent):
    """
    Parameters
    ----------
    df             : pd.DataFrame with DatetimeIndex
    roles          : dict  col → role  (ROLE_* constants)
                     If None, roles are auto-suggested.
    dependent_col  : str   override the dependent column
    max_ccf_lags   : int   max lags for cross-correlation (default 20)
    event_window   : int   post-event window periods (default 5)
    granger_lags   : int   max lags for Granger proxy (default 8)
    """

    def __init__(self, context_store: Optional[ContextStore] = None):
        super().__init__("MultiVariableAgent", context_store)

    def validate_inputs(self, df: Any = None, **kwargs):
        if df is None:
            raise ValueError("MultiVariableAgent requires 'df'.")

    def _run(
        self,
        df: pd.DataFrame,
        roles: Optional[Dict[str, str]] = None,
        dependent_col: Optional[str] = None,
        max_ccf_lags: int = 20,
        event_window: int = 5,
        granger_lags: int = 8,
        **kwargs,
    ) -> AgentResult:

        warnings: List[str] = []

        # ── 1. resolve roles ─────────────────────────────────────────────────
        if roles is None:
            roles = suggest_roles(df)
            warnings.append("Column roles auto-detected")

        # allow override of dependent col
        if dependent_col and dependent_col in df.columns:
            roles[dependent_col] = ROLE_DEPENDENT
            # demote any other dependent to independent
            for c, r in roles.items():
                if c != dependent_col and r == ROLE_DEPENDENT:
                    roles[c] = ROLE_INDEPENDENT

        dep_cols   = [c for c, r in roles.items() if r == ROLE_DEPENDENT   and c in df.columns]
        indep_cols = [c for c, r in roles.items() if r == ROLE_INDEPENDENT and c in df.columns]
        event_cols = [c for c, r in roles.items() if r == ROLE_EVENT       and c in df.columns]

        if not dep_cols:
            return AgentResult(
                agent_name=self.name, status=AgentStatus.FAILED,
                errors=["No dependent column found in roles."]
            )

        dep_col = dep_cols[0]
        dep_series = df[dep_col].dropna()

        # ── 2. per-variable stats ─────────────────────────────────────────────
        all_numeric = dep_cols + indep_cols + event_cols
        var_stats: Dict[str, Any] = {}
        for col in all_numeric:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            var_stats[col] = {
                "role": roles.get(col, ROLE_IGNORE),
                "n": int(len(s)),
                "n_missing": int(df[col].isna().sum()),
                "mean": round(float(s.mean()), 4) if len(s) else None,
                "std":  round(float(s.std()),  4) if len(s) > 1 else None,
                "min":  round(float(s.min()),  4) if len(s) else None,
                "max":  round(float(s.max()),  4) if len(s) else None,
                "pct_zero": round(float((s == 0).mean() * 100), 2) if len(s) else 0,
                "pct_one":  round(float((s == 1).mean() * 100), 2) if len(s) else 0,
                "is_binary": roles.get(col) == ROLE_EVENT,
            }

        # ── 3. Pearson correlation matrix ────────────────────────────────────
        corr_cols = [c for c in all_numeric if c in df.columns]
        corr_matrix = pearson_corr_matrix(df, corr_cols) if len(corr_cols) >= 2 else {}

        # ── 4. Cross-correlation: each indep/event vs dep ─────────────────────
        ccf_results: Dict[str, Any] = {}
        for col in indep_cols + event_cols:
            if col not in df.columns:
                continue
            try:
                x = df[col].reindex(dep_series.index).fillna(0).values
                y = dep_series.values
                ccf_results[col] = cross_correlation_function(x, y, max_lags=max_ccf_lags)
            except Exception as e:
                warnings.append(f"CCF failed for '{col}': {e}")

        # ── 5. Event impact analysis ─────────────────────────────────────────
        event_impacts: Dict[str, Any] = {}
        for col in event_cols:
            if col not in df.columns:
                continue
            try:
                event_series = df[col].reindex(dep_series.index).fillna(0)
                event_impacts[col] = event_impact_analysis(
                    dep_series, event_series, window_after=event_window
                )
            except Exception as e:
                warnings.append(f"Event impact failed for '{col}': {e}")

        # ── 6. Granger-causality proxy ────────────────────────────────────────
        granger_results: Dict[str, Any] = {}
        for col in indep_cols:
            if col not in df.columns:
                continue
            try:
                x = df[col].reindex(dep_series.index).fillna(method="ffill").fillna(0).values
                granger_results[col] = granger_proxy(dep_series.values, x, max_lag=granger_lags)
            except Exception as e:
                warnings.append(f"Granger proxy failed for '{col}': {e}")

        # ── 7. Feature recommendations ────────────────────────────────────────
        feature_recs = self._build_feature_recs(
            dep_col, indep_cols, event_cols,
            ccf_results, granger_results, event_impacts,
        )

        # ── 8. Report ─────────────────────────────────────────────────────────
        report = self._build_report(
            dep_col, indep_cols, event_cols, corr_matrix,
            ccf_results, event_impacts, feature_recs,
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data={
                "roles": roles,
                "var_stats": var_stats,
                "corr_matrix": corr_matrix,
                "ccf_results": ccf_results,
                "event_impacts": event_impacts,
                "granger_results": granger_results,
                "feature_recommendations": feature_recs,
            },
            metadata={
                "dependent_col": dep_col,
                "independent_cols": indep_cols,
                "event_cols": event_cols,
                "n_variables": len(all_numeric),
                "report": report,
            },
            warnings=warnings,
        )

    # ── feature recommendation builder ───────────────────────────────────────

    @staticmethod
    def _build_feature_recs(
        dep_col: str,
        indep_cols: List[str],
        event_cols: List[str],
        ccf_results: Dict,
        granger_results: Dict,
        event_impacts: Dict,
    ) -> List[Dict[str, Any]]:
        recs = []
        for col in indep_cols:
            ccf = ccf_results.get(col, {})
            gr  = granger_results.get(col, {})
            if ccf.get("significant") or gr.get("useful"):
                recs.append({
                    "variable": col,
                    "role": ROLE_INDEPENDENT,
                    "recommended_lag": ccf.get("best_lag", 0),
                    "max_ccf": ccf.get("max_ccf", 0),
                    "r2_gain": gr.get("r2_gain", 0),
                    "granger_lag": gr.get("best_lag", 0),
                    "significant": True,
                    "reason": ccf.get("interpretation", ""),
                    "include_in_model": True,
                })
            else:
                recs.append({
                    "variable": col,
                    "role": ROLE_INDEPENDENT,
                    "recommended_lag": ccf.get("best_lag", 0),
                    "max_ccf": ccf.get("max_ccf", 0),
                    "r2_gain": 0,
                    "significant": False,
                    "reason": "No significant relationship detected",
                    "include_in_model": False,
                })

        for col in event_cols:
            impact = event_impacts.get(col, {})
            eff = impact.get("effect_size", "small")
            sig = eff in ("large", "medium") or abs(impact.get("cohens_d", 0)) > 0.3
            recs.append({
                "variable": col,
                "role": ROLE_EVENT,
                "recommended_lag": 0,
                "lift_pct": impact.get("lift_pct", 0),
                "cohens_d": impact.get("cohens_d", 0),
                "effect_size": eff,
                "significant": sig,
                "reason": f"Lift = {impact.get('lift_pct', 0):.1f}%, effect={eff}",
                "include_in_model": sig,
            })

        # sort by significance then max_ccf / lift
        recs.sort(key=lambda r: (not r["significant"], -abs(r.get("max_ccf") or r.get("lift_pct", 0) or 0)))
        return recs

    @staticmethod
    def _build_report(
        dep_col, indep_cols, event_cols,
        corr_matrix, ccf_results, event_impacts, feature_recs,
    ) -> str:
        lines = [
            "═══ Multi-Variable Analysis Report ═══",
            f"  Dependent    : {dep_col}",
            f"  Independent  : {', '.join(indep_cols) or 'none'}",
            f"  Event vars   : {', '.join(event_cols) or 'none'}",
            "",
            "  Cross-correlations with dependent:",
        ]
        for col, res in ccf_results.items():
            sig = "✓" if res.get("significant") else "—"
            lines.append(
                f"    {sig} {col:30s}: best_lag={res['best_lag']:+3d}  max_ccf={res['max_ccf']:+.3f}  {res['interpretation'][:60]}"
            )
        if event_impacts:
            lines += ["", "  Event impacts:"]
            for col, imp in event_impacts.items():
                lines.append(
                    f"    {col:30s}: lift={imp.get('lift_pct',0):+.1f}%  d={imp.get('cohens_d',0):.3f}  [{imp.get('effect_size','?')}]"
                )
        lines += ["", "  Feature recommendations:"]
        for rec in feature_recs:
            inc = "✓ INCLUDE" if rec["include_in_model"] else "✗ skip"
            lines.append(f"    {inc:12s} {rec['variable']:30s}  {rec['reason'][:60]}")
        return "\n".join(lines)
