"""BirdWatcher FastAPI entrypoint."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from db.session import init_db
from pipeline.worker import start_worker
from routers import (
    art,
    biome,
    corrections,
    detections,
    ingest,
    push,
    species,
    stats,
    tavern,
    territory,
)

# uvicorn sets up its own loggers (uvicorn, uvicorn.access) but does NOT
# configure the root logger. Our app modules use logging.getLogger(__name__)
# — without this basicConfig their log.info(...) calls drop on the floor.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    scheduler = start_worker()
    # TODO Phase 4: also start Haikubox poller on the same scheduler.
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


app = FastAPI(title="BirdWatcher", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://birdwatcher.ryanhoulette.com", "http://localhost:5270"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])
app.include_router(detections.router, prefix="/api/detections", tags=["detections"])
app.include_router(push.router, prefix="/api/push", tags=["push"])
app.include_router(corrections.router, prefix="/api/corrections", tags=["corrections"])
app.include_router(species.router, prefix="/api/species", tags=["species"])
app.include_router(stats.router, prefix="/api/stats", tags=["stats"])
app.include_router(art.router, prefix="/api/art", tags=["art"])
app.include_router(tavern.router, prefix="/api/tavern", tags=["tavern"])
app.include_router(biome.router, prefix="/api/biome", tags=["biome"])
app.include_router(territory.router, prefix="/api/territory", tags=["territory"])

class SafeStaticFiles(StaticFiles):
    """Serve media files while strictly blocking database, secrets, and non-media files."""

    BLOCKED_EXTENSIONS = frozenset({
        ".db", ".db-wal", ".db-shm", ".sqlite", ".sqlite3",
        ".py", ".pyc", ".env", ".key", ".pem", ".json", ".txt", ".yaml", ".yml",
    })
    ALLOWED_DIRECTORIES = frozenset({"crops", "clips", "heatmaps", "calibration", "frames"})

    async def get_response(self, path: str, scope):
        lower = path.lower()
        # Direct check for database or sensitive file extensions
        if any(lower.endswith(ext) for ext in self.BLOCKED_EXTENSIONS) or "birdwatcher.db" in lower:
            from starlette.responses import PlainTextResponse
            return PlainTextResponse("Not Found", status_code=404)
        # Ensure only allowed media directories (crops, clips, heatmaps, calibration, frames) are served
        parts = Path(path).parts
        if parts and parts[0] not in self.ALLOWED_DIRECTORIES:
            from starlette.responses import PlainTextResponse
            return PlainTextResponse("Not Found", status_code=404)
        return await super().get_response(path, scope)


# Serve uploaded clips/crops at /media (the PWA reads from here) with sensitive file protections
app.mount("/media", SafeStaticFiles(directory=DATA_DIR), name="media")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
