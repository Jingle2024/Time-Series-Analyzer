"""
routes/ingestion.py  –  Upload & schema-confirmation endpoints
==============================================================
Routes
------
POST /api/upload          – ingest file, detect schema, return column info
POST /api/confirm-schema  – accept user column mapping
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from agents.ingestion_agent import IngestionAgent
from agents.multi_variable_agent import (
    ROLE_DEPENDENT, ROLE_EVENT, ROLE_HIERARCHY, ROLE_IGNORE,
    ROLE_INDEPENDENT, ROLE_TIMESTAMP,
    detect_event_columns, detect_hierarchy_columns, suggest_roles,
)
from core.runtime import SESSIONS as _sessions
from utils.api_helper import _df_to_records, _looks_like_timestamp_name, _safe

router = APIRouter()


# ── Upload ─────────────────────────────────────────────────────────────────────

@router.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        raw   = await file.read()
        fname = file.filename or "upload"
        suffix = Path(fname).suffix.lower()

        if suffix in (".xlsx", ".xls"):
            df_raw = pd.read_excel(io.BytesIO(raw))
        elif suffix == ".csv":
            df_raw = pd.read_csv(io.BytesIO(raw))
        else:
            raise HTTPException(400, "Only CSV and Excel files are supported.")

        import uuid
        token = str(uuid.uuid4())[:8]
        _sessions[token] = {"raw_df": df_raw, "filename": fname}

        cols    = list(df_raw.columns)
        dtypes  = {c: str(df_raw[c].dtype) for c in cols}
        nunique = {c: int(df_raw[c].nunique()) for c in cols}
        n_rows  = len(df_raw)

        ts_candidates, val_candidates = [], []
        for c in cols:
            col_lower = c.lower()
            if pd.api.types.is_datetime64_any_dtype(df_raw[c]):
                ts_candidates.append(c)
                continue
            if pd.api.types.is_numeric_dtype(df_raw[c]):
                val_candidates.append(c)
                continue
            try:
                sample = df_raw[c].dropna().head(10)
                if len(sample):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        parsed = pd.to_datetime(sample.astype(str), errors="coerce")
                else:
                    parsed = pd.Series(dtype="datetime64[ns]")
                if len(parsed) and float(parsed.notna().mean()) >= 0.8:
                    ts_candidates.append(c)
                    continue
            except Exception:
                pass
            if _looks_like_timestamp_name(col_lower):
                ts_candidates.append(c)

        hier_candidates = detect_hierarchy_columns(
            df_raw,
            excluded=set(ts_candidates) | set(val_candidates),
        )
        is_hierarchy  = len(hier_candidates) > 0
        needs_confirm = (
            len(ts_candidates) != 1
            or len(val_candidates) == 0
            or (is_hierarchy and len(hier_candidates) > 4)
        )

        event_candidates  = detect_event_columns(df_raw, val_candidates)
        suggested_roles: Dict[str, str] = {}
        for c in cols:
            if c in ts_candidates:
                suggested_roles[c] = ROLE_TIMESTAMP
            elif c in hier_candidates:
                suggested_roles[c] = ROLE_HIERARCHY
            elif c in event_candidates:
                suggested_roles[c] = ROLE_EVENT
            elif c in val_candidates:
                non_event_vals = [v for v in val_candidates if v not in event_candidates]
                idx = non_event_vals.index(c) if c in non_event_vals else -1
                suggested_roles[c] = ROLE_DEPENDENT if idx == 0 else ROLE_INDEPENDENT
            else:
                suggested_roles[c] = ROLE_IGNORE

        return {
            "token":                 token,
            "filename":              fname,
            "n_rows":                n_rows,
            "columns":               cols,
            "dtypes":                dtypes,
            "nunique":               nunique,
            "ts_candidates":         ts_candidates,
            "value_candidates":      val_candidates,
            "hierarchy_candidates":  hier_candidates,
            "event_candidates":      event_candidates,
            "suggested_roles":       suggested_roles,
            "is_hierarchy":          is_hierarchy,
            "needs_confirm":         needs_confirm,
            "preview":               _df_to_records(df_raw, 8),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Confirm schema ─────────────────────────────────────────────────────────────

class SchemaConfirm(BaseModel):
    token:           str
    timestamp_col:   str
    value_cols:      List[str]
    hierarchy_cols:  List[str] = []
    variable_roles:  Dict[str, str] = {}


@router.post("/api/confirm-schema")
async def confirm_schema(body: SchemaConfirm):
    sess = _sessions.get(body.token)
    if not sess:
        raise HTTPException(404, "Session not found")
    df_raw = sess["raw_df"]

    if body.variable_roles:
        all_val = [
            c for c, r in body.variable_roles.items()
            if r in (ROLE_DEPENDENT, ROLE_INDEPENDENT, ROLE_EVENT)
        ]
        if all_val:
            body = body.model_copy(update={"value_cols": all_val})

    agent  = IngestionAgent()
    result = agent.execute(
        source=df_raw,
        timestamp_col=body.timestamp_col,
        value_cols=body.value_cols,
        hierarchy_cols=body.hierarchy_cols,
    )
    if not result.ok:
        raise HTTPException(400, result.errors[0] if result.errors else "Ingestion failed")

    df: pd.DataFrame = result.data
    roles = body.variable_roles or {}

    dep_cols   = [c for c, r in roles.items() if r == ROLE_DEPENDENT]
    indep_cols = [c for c, r in roles.items() if r == ROLE_INDEPENDENT]
    event_cols = [c for c, r in roles.items() if r == ROLE_EVENT]

    if not roles:
        roles      = suggest_roles(df, ts_col=body.timestamp_col)
        dep_cols   = [c for c, r in roles.items() if r == ROLE_DEPENDENT]
        indep_cols = [c for c, r in roles.items() if r == ROLE_INDEPENDENT]
        event_cols = [c for c, r in roles.items() if r == ROLE_EVENT]

    sess["df"]              = df
    sess["schema"]          = result.metadata
    sess["value_cols"]      = body.value_cols
    sess["hierarchy_cols"]  = body.hierarchy_cols
    sess["detected_freq"]   = result.metadata["detected_freq"]
    sess["variable_roles"]  = roles
    sess["dependent_cols"]  = dep_cols or (body.value_cols[:1] if body.value_cols else [])
    sess["independent_cols"]= indep_cols
    sess["event_cols"]      = event_cols

    return {
        "ok":               True,
        "schema":           _safe(result.metadata["schema"]),
        "warnings":         result.warnings,
        "preview":          _df_to_records(df.reset_index(), 10),
        "variable_roles":   roles,
        "dependent_cols":   dep_cols,
        "independent_cols": indep_cols,
        "event_cols":       event_cols,
    }
