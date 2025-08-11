from __future__ import annotations

import json
import os
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import ContentAnalysisPipeline

# logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("server")

# Ensure subprocess support on Windows for Playwright
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

BASE_DIR = Path(__file__).parent.resolve()
WEB_DIR = BASE_DIR / "web"
OUTPUT_DIR = BASE_DIR / "output"
REPORTS_DIR = BASE_DIR / "reports"

app = FastAPI(title="Sentiment Observatory API", version="0.1.0")

# If you plan to host frontend separately, enable CORS here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mounts
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")
if OUTPUT_DIR.exists() or True:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
if REPORTS_DIR.exists() or True:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


class AnalyzeRequest(BaseModel):
    url: str
    # advanced omitted; server always runs advanced analysis


def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        raise ValueError("Empty URL")
    if not (u.startswith("http://") or u.startswith("https://")):
        u = "https://" + u
    return u


def _run_pipeline_and_save(url: str, advanced: bool) -> dict:
    pipeline = ContentAnalysisPipeline(enable_advanced=advanced)
    report = pipeline.run_pipeline(url)
    pipeline.save_report(report)
    return report


@app.get("/")
async def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found. Ensure web/index.html exists.")
    return FileResponse(str(index_path))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest, request: Request) -> JSONResponse:
    req_id = str(uuid.uuid4())[:8]
    try:
        os.environ["HEADLESS"] = "true"
        os.environ["SCRAPER_TIMEOUT_MS"] = "20000"
        os.environ["SERVER_CONTEXT"] = "true"

        url = _normalize_url(req.url)
        logger.info("[%s] analyze url=%s advanced=true", req_id, url)
        report = await asyncio.to_thread(_run_pipeline_and_save, url, True)
        return JSONResponse(content={"report": report, "request_id": req_id})
    except Exception as e:
        logger.error("[%s] analyze error: %s", req_id, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/appalyze")
async def appalyze(req: AnalyzeRequest, request: Request) -> JSONResponse:
    # alias for clients using the older/typo route
    return await analyze(req, request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
