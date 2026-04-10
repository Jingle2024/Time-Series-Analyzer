"""Minimal TemporalMind server bootstrap."""

from __future__ import annotations

import logging

from routes.api import app

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("server")


if __name__ == "__main__":
    try:
        import uvicorn

        log.info("Starting TemporalMind server on http://localhost:8080")
        uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
    except ImportError:
        log.error("uvicorn not installed. Run: pip install uvicorn")
