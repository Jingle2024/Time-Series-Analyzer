"""
routes/system.py  –  System / utility endpoints
================================================
Routes
------
GET /api/download/{token}  – stream the prepared file
GET /                      – serve the SPA (index.html)
GET /health                – liveness check
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response

from core.runtime import DOWNLOADS as _downloads

router = APIRouter()


@router.get("/api/download/{token}")
async def download(token: str, ext: str = "csv"):
    data = _downloads.get(token)
    if not data:
        raise HTTPException(404, "Download token expired or not found")
    if ext == "xlsx":
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif ext == "zip":
        mime = "application/zip"
    else:
        mime = "text/csv"
    return Response(
        content=data,
        media_type=mime,
        headers={"Content-Disposition": f"attachment; filename=prepared_data.{ext}"},
    )


@router.get("/", response_class=HTMLResponse)
async def root():
    base_dir = Path(__file__).resolve().parents[1]   # project root (two levels up from routes/)
    candidate_paths = [
        base_dir / "ui" / "index.html",
        base_dir / "index.html",
    ]
    for html_path in candidate_paths:
        if html_path.exists():
            return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>TemporalMind UI not found. Place index.html in ./ui/ or project root.</h1>",
        status_code=404,
    )


@router.get("/health")
async def health():
    return {"status": "ok"}
