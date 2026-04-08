"""
agents/ingestion_agent.py
─────────────────────────
Loads time series data from CSV, Excel, or JSON (dict or list).
Infers or accepts explicit column mapping:
  timestamp_col  – name of the datetime column
  value_cols     – list of value column names
  hierarchy_cols – ordered list of hierarchy key columns (coarsest → finest)

Outputs a canonical pandas DataFrame with a DatetimeIndex.
"""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from core.base_agent import AgentResult, AgentStatus, BaseAgent
from core.context_store import ContextStore

logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

_TIMESTAMP_HINTS = [
    "date", "datetime", "time", "timestamp", "ts",
    "period", "week", "month", "year",
]
_VALUE_HINTS = [
    "value", "qty", "quantity", "amount", "sales",
    "demand", "vol", "volume", "count", "revenue",
]


def _guess_timestamp_col(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for hint in _TIMESTAMP_HINTS:
        for col_lower, col_orig in cols_lower.items():
            if hint in col_lower:
                return col_orig
    # fall back: first column that parses as datetime
    for col in df.columns:
        try:
            parsed = _parse_datetime_series(df[col].iloc[:10])
            if float(parsed.notna().mean()) >= 0.8:
                return col
        except Exception:
            pass
    raise ValueError(
        "Cannot infer timestamp column. Pass timestamp_col= explicitly."
    )


def _guess_value_cols(df: pd.DataFrame, ts_col: str) -> List[str]:
    candidates = []
    for col in df.columns:
        if col == ts_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)
    if not candidates:
        raise ValueError("No numeric columns found to use as value columns.")
    return candidates


def _guess_hierarchy_cols(df: pd.DataFrame, ts_col: str, value_cols: List[str]) -> List[str]:
    excluded = {ts_col} | set(value_cols)
    return [c for c in df.columns if c not in excluded and df[c].dtype == object]


_AMBIGUOUS_DATE_RE = re.compile(r"^\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s*$")


def _looks_ambiguous_date_series(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    return bool(sample.map(lambda v: bool(_AMBIGUOUS_DATE_RE.match(v))).mean() >= 0.8)


def _score_datetime_parse(parsed: pd.Series) -> tuple:
    valid = parsed.dropna()
    valid_n = int(valid.shape[0])
    if valid_n == 0:
        return (0, 0.0, 0, 0, 0)

    uniq = pd.Series(valid).sort_values().drop_duplicates()
    inferred = None
    try:
        inferred = pd.infer_freq(pd.DatetimeIndex(uniq))
    except Exception:
        inferred = None

    dominant_gap_ratio = 0.0
    if len(uniq) >= 3:
        diffs = uniq.diff().dropna()
        if len(diffs):
            dominant_gap_ratio = float(diffs.value_counts(normalize=True).iloc[0])

    month_start_count = 0
    day_one_count = 0
    if valid_n:
        month_start_count = int((valid.dt.is_month_start).sum())
        day_one_count = int((valid.dt.day == 1).sum())

    # Higher is better. Prefer more successful parses, then a clean inferred
    # frequency, then month-start alignment, then regular spacing.
    return (
        valid_n,
        1 if inferred else 0,
        month_start_count,
        day_one_count,
        dominant_gap_ratio,
    )


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    ambiguous = _looks_ambiguous_date_series(series)
    candidates = []

    parse_options = [None]
    if ambiguous:
        parse_options.extend([True, False])

    for dayfirst in parse_options:
        kwargs = {"errors": "coerce"}
        if dayfirst is not None:
            kwargs["dayfirst"] = dayfirst
        parsed = pd.to_datetime(series, **kwargs)
        candidates.append((_score_datetime_parse(parsed), dayfirst, parsed))

    best_score, best_dayfirst, best_parsed = max(candidates, key=lambda item: item[0])
    logger.info("Datetime parse selected dayfirst=%s score=%s", best_dayfirst, best_score)
    return best_parsed


# ── agent ─────────────────────────────────────────────────────────────────────

class IngestionAgent(BaseAgent):
    """
    Supported sources
    -----------------
    - file path   : str / Path ending in .csv, .xlsx, .xls
    - raw string  : CSV text
    - dict / list : JSON-style records (from an API response)
    - DataFrame   : pass-through
    """

    def __init__(self, context_store: Optional[ContextStore] = None):
        super().__init__("IngestionAgent", context_store)

    def validate_inputs(self, source: Any = None, **kwargs) -> None:
        if source is None:
            raise ValueError("IngestionAgent requires 'source' argument.")

    def _run(
        self,
        source: Union[str, Path, dict, list, pd.DataFrame],
        timestamp_col: Optional[str] = None,
        value_cols: Optional[List[str]] = None,
        hierarchy_cols: Optional[List[str]] = None,
        freq_hint: Optional[str] = None,       # e.g. 'D', 'W', 'M'
        **kwargs,
    ) -> AgentResult:

        warnings: List[str] = []

        # ── 1. load raw data ──────────────────────────────────────────────────
        df = self._load(source, warnings)

        # ── 2. column mapping ─────────────────────────────────────────────────
        ts_col = timestamp_col or _guess_timestamp_col(df)
        val_cols = value_cols or _guess_value_cols(df, ts_col)
        hier_cols = hierarchy_cols if hierarchy_cols is not None else _guess_hierarchy_cols(
            df, ts_col, val_cols
        )

        # ── 3. parse & set datetime index ────────────────────────────────────
        df[ts_col] = _parse_datetime_series(df[ts_col])
        df = df.sort_values(ts_col).reset_index(drop=True)
        df = df.set_index(ts_col)
        df.index.name = "timestamp"

        # ── 4. coerce value columns to float ─────────────────────────────────
        for col in val_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n_bad = df[col].isna().sum()
            df[col] = df[col].fillna(0)
            if n_bad:
                warnings.append(f"Column '{col}': {n_bad} values could not be parsed → NaN")

        # ── 5. detect native frequency ────────────────────────────────────────
        detected_freq = self._detect_freq(df, freq_hint, warnings)

        # ── 6. schema report ──────────────────────────────────────────────────
        schema = {
            "n_rows": len(df),
            "n_series": len(val_cols) * max(1, df[hier_cols].nunique().prod() if hier_cols else 1),
            "timestamp_col": "timestamp (index)",
            "value_cols": val_cols,
            "hierarchy_cols": hier_cols,
            "date_range": (str(df.index.min()), str(df.index.max())),
            "detected_freq": detected_freq,
            "missing_pct": {
                col: round(df[col].isna().mean() * 100, 2) for col in val_cols
            },
        }

        logger.info("Ingestion schema: %s", schema)

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data=df,
            metadata={
                "schema": schema,
                "value_cols": val_cols,
                "hierarchy_cols": hier_cols,
                "detected_freq": detected_freq,
            },
            warnings=warnings,
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _load(self, source: Any, warnings: List[str]) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()

        if isinstance(source, (dict, list)):
            records = source if isinstance(source, list) else [source]
            return pd.DataFrame(records)

        path = Path(source) if isinstance(source, str) and not source.startswith("{") else None

        if path and path.exists():
            suffix = path.suffix.lower()
            if suffix == ".csv":
                return pd.read_csv(path)
            elif suffix in (".xlsx", ".xls"):
                return pd.read_excel(path)
            else:
                raise ValueError(f"Unsupported file extension: {suffix}")

        # treat as raw CSV string
        if isinstance(source, str):
            return pd.read_csv(io.StringIO(source))

        raise ValueError(f"Cannot load source of type {type(source)}")

    @staticmethod
    def _detect_freq(df: pd.DataFrame, hint: Optional[str], warnings: List[str]) -> str:
        if hint:
            return hint
        idx = pd.DatetimeIndex(pd.Series(df.index).dropna().sort_values().drop_duplicates())
        try:
            inferred = pd.infer_freq(idx)
            if inferred:
                return inferred
        except Exception:
            pass
        # fallback: compute median gap
        if len(idx) >= 2:
            gaps = pd.Series(idx).diff().dropna()
            median_gap = gaps.median()
            td = pd.Timedelta(median_gap)
            days = td.days
            if days <= 1:
                return "D"
            elif days <= 7:
                return "W"
            elif days <= 31:
                return "MS"
            elif days <= 92:
                return "QS"
            else:
                return "AS"
        warnings.append("Could not detect frequency; defaulting to 'D'")
        return "D"
