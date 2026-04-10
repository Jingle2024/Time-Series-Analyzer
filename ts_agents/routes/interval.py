"""
routes/interval.py  –  Interval advice & accumulation endpoints
===============================================================
Routes
------
POST /api/interval-advice  – get interval recommendations
POST /api/accumulate       – resample to chosen interval
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.interval_advisor_agent import IntervalAdvisorAgent
from agents.accumulation_agent import AccumulationAgent
from core.runtime import SESSIONS as _sessions
from utils.api_helper import _df_to_records, _safe

router = APIRouter()


# ── Interval advice ────────────────────────────────────────────────────────────

class IntervalReq(BaseModel):
    token: str
    top_n: int = 3


@router.post("/api/interval-advice")
async def interval_advice(body: IntervalReq):
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404, "No ingested data for this token")
    df: pd.DataFrame = sess["df"]
    val_cols = sess.get("value_cols", [])
    col      = val_cols[0] if val_cols else df.select_dtypes("number").columns[0]

    series = df[col].dropna()
    agent  = IntervalAdvisorAgent()
    result = agent.execute(series=series, native_freq=sess.get("detected_freq"), top_n=body.top_n)
    if not result.ok:
        raise HTTPException(500, result.errors[0])

    recs = result.data
    return {
        "recommendations": [
            {
                "alias":          r["alias"],
                "freq":           r["freq"],
                "score":          r["score"],
                "info_loss_pct":  r["info_loss_pct"],
                "rationale":      r["rationale"],
            }
            for r in recs
        ],
        "best_freq":        result.metadata["best_interval"],
        "best_alias":       result.metadata["best_alias"],
        "summary":          result.metadata["summary"],
        "native_step_days": result.metadata["native_step_days"],
    }


# ── Accumulate ─────────────────────────────────────────────────────────────────

class AccumReq(BaseModel):
    token:         str
    target_freq:   str
    method:        str = "auto"
    quantity_type: str = "flow"


@router.post("/api/accumulate")
async def accumulate(body: AccumReq):
    sess = _sessions.get(body.token)
    if not sess or "df" not in sess:
        raise HTTPException(404)
    df       = sess["df"]
    val_cols = sess.get("value_cols")

    agent  = AccumulationAgent()
    result = agent.execute(
        df=df,
        target_freq=body.target_freq,
        method=body.method,
        quantity_type=body.quantity_type,
        compare_freqs=None,
        value_cols=val_cols,
    )
    if not result.ok:
        raise HTTPException(500, result.errors[0])

    sess["accumulated_df"] = result.data
    sess["target_freq"]    = body.target_freq

    comparison: dict = {}
    for freq, stats in result.metadata.get("comparison", {}).items():
        for col, s in stats.items():
            comparison.setdefault(col, {})[freq] = _safe(s)

    return {
        "n_input":              result.metadata["n_input"],
        "n_output":             result.metadata["n_output"],
        "compression_ratio":    result.metadata["compression_ratio"],
        "information_retention":_safe(result.metadata["information_retention"]),
        "comparison":           comparison,
        "preview":              _df_to_records(result.data.reset_index(), 12),
        "report":               result.metadata.get("report", ""),
    }
