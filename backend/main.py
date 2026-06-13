"""BirdWatcher FastAPI entrypoint."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from db.session import init_db
from pipeline.worker import start_worker
from routers import corrections, detections, ingest, push, species, stats

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

app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])
app.include_router(detections.router, prefix="/api/detections", tags=["detections"])
app.include_router(push.router, prefix="/api/push", tags=["push"])
app.include_router(corrections.router, prefix="/api/corrections", tags=["corrections"])
app.include_router(species.router, prefix="/api/species", tags=["species"])
app.include_router(stats.router, prefix="/api/stats", tags=["stats"])

# Serve uploaded clips/crops at /media (the PWA reads from here)
app.mount("/media", StaticFiles(directory=DATA_DIR), name="media")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
