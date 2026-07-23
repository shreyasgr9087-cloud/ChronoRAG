"""
ChronoRAG API server — replaces app.py (Streamlit) entirely.

Run:
    uvicorn server:app --reload --port 8000

Endpoints:
    POST /api/query      { "query": "..." }  -> full pipeline result (dict)
    GET  /api/health      liveness check

In production, this also serves the built React app (chronorag-ui/dist)
as static files, so you can ship ONE process instead of two.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import execute_chronorag_pipeline

app = FastAPI(title="ChronoRAG API", version="1.0.0")

# Dev CORS: Vite runs on :5173 by default. Tighten this for production
# (or drop it entirely once the frontend is served from this same app).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/query")
def query(payload: QueryRequest):
    q = payload.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    try:
        return execute_chronorag_pipeline(q)
    except Exception as e:
        # Surface pipeline errors as a clean 500 instead of a stack trace leak.
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")


# --- Optional: serve the built frontend from this same process ---
# After running `npm run build` inside chronorag-ui/, its output lands in
# chronorag-ui/dist/. Point FRONTEND_DIST at that folder (env var or edit
# below) to serve the whole app from http://localhost:8000 with no
# separate Vite server or CORS needed.
FRONTEND_DIST = os.getenv("FRONTEND_DIST", "../chronorag-ui/dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
