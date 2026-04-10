"""
api.py  –  TemporalMind FastAPI entry-point
===========================================
This file is intentionally thin: it creates the FastAPI app,
registers CORS middleware, and mounts every route module.

All business logic lives in:
  routes/ingestion.py      – POST /api/upload, /api/confirm-schema
  routes/interval.py       – POST /api/interval-advice, /api/accumulate
  routes/analysis.py       – POST /api/analyze, /api/variable-roles, /api/cross-correlation
  routes/hierarchy.py      – POST /api/hierarchy, /api/hierarchy-tree,
                                   /api/hierarchy-children, /api/analyze-node,
                                   /api/level-stability
  routes/treatment.py      – POST /api/missing-values, /api/outliers
  routes/prepare.py        – POST /api/prepare
  routes/forecast.py       – POST /api/forecast-prepare
  routes/system.py         – GET  /api/download/{token}, GET /, GET /health

Shared helpers live in:
  util/api-helper.py       – pure utility functions (no FastAPI dependency)

Run:  python api.py          (serves on http://localhost:8080)
Docs: http://localhost:8080/docs
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import (
    analysis,
    forecast,
    hierarchy,
    ingestion,
    interval,
    prepare,
    system,
    treatment,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("server")

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="TemporalMind API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount route modules ────────────────────────────────────────────────────────

app.include_router(ingestion.router)   # /api/upload, /api/confirm-schema
app.include_router(interval.router)    # /api/interval-advice, /api/accumulate
app.include_router(analysis.router)    # /api/analyze, /api/variable-roles, /api/cross-correlation
app.include_router(hierarchy.router)   # /api/hierarchy, /api/hierarchy-tree,
                                       # /api/hierarchy-children, /api/analyze-node,
                                       # /api/level-stability
app.include_router(treatment.router)   # /api/missing-values, /api/outliers
app.include_router(prepare.router)     # /api/prepare
app.include_router(forecast.router)    # /api/forecast-prepare
app.include_router(system.router)      # /api/download/{token}, /, /health

# ── Dev server ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import uvicorn
        log.info("Starting TemporalMind server on http://localhost:8080")
        uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)
    except ImportError:
        log.error("uvicorn not installed. Run: pip install uvicorn")
