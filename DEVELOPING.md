# Developing BirdWatcher

A guide for the next human (or Claude) who returns to this codebase six months
from now and needs to remember how the pieces fit together. Pair this with
`README.md` (the product blurb) and `DEPLOY.md` (the cloud-VM provisioning
runbook); this file is everything in between.

## The one-paragraph mental model

A Reolink RLC-811WA WiFi camera pointed at the bird feeders. On motion it
POSTs a ~5–10 s video clip to our backend. The backend extracts frames,
runs YOLO11 to find each bird in each frame, tracks them across frames with
a simple IoU tracker, classifies each track's best crop with a fine-grained
bird species model, and re-ranks the predictions using two priors: an
allow-list + monthly distribution derived from the user's Haikubox
(BirdNET audio) detection history, and recent audio "did you hear this
species in the last 90 s" boosts. A React PWA displays the resulting feed
and pings the user's phone via Web Push when a species hasn't been seen in
30 days. Users correct misidentifications via a searchable picker; those
corrections feed a (still-stubbed) periodic fine-tune of the classifier.

## Architecture

```
   Reolink RLC-811WA       Haikubox (BirdNET audio)
   on motion → POST clip   poll detections every 30 s
              │                       │
              ▼                       ▼
   ┌─────────────────────────────────────────────────┐
   │  FastAPI backend                                │
   │  - /api/ingest/motion   (clip upload)           │
   │  - /api/detections      (PWA read)              │
   │  - /api/species         (picker)                │
   │  - /api/corrections     (active learning)       │
   │  - /api/push/{subscribe, vapid_public_key}      │
   │                                                 │
   │  APScheduler workers:                           │
   │   • pipeline.worker — pulls unprocessed Visit   │
   │     rows; runs YOLO → tracker → classifier →    │
   │     fusion → DB. Pushes if rare.                │
   │   • ingest.haikubox — polls Haikubox API every  │
   │     30 s; caches into haikubox_detections.      │
   │                                                 │
   │  SQLite (data/birdwatcher.db) + image files     │
   │  (data/clips/, data/crops/).                    │
   │                                                 │
   │  Caddy reverse-proxy in front, TLS via LE.      │
   └────────────┬────────────────────────────────────┘
                │ HTTPS
                ▼
            Android PWA (Vite + React + TS, vite-plugin-pwa)
```

## Project layout

```
BirdWatcher/
├── README.md                ← product intro
├── DEPLOY.md                ← cloud-VM bringup
├── DEVELOPING.md            ← (this file)
├── Makefile                 ← common dev tasks
├── docker-compose.yml       ← prod stack (api + caddy)
├── Caddyfile                ← TLS + reverse proxy
├── .github/workflows/ci.yml ← pytest + frontend build on PR/push
│
├── backend/                 ← Python 3.12 / FastAPI
│   ├── main.py              ← app entrypoint, lifespan, routers
│   ├── settings.py          ← env-driven config (pydantic-settings)
│   ├── requirements.txt     ← runtime + ML deps
│   ├── Dockerfile           ← python:3.12-slim + OpenCV runtime
│   ├── .env.example         ← required env vars
│   │
│   ├── db/
│   │   ├── session.py       ← SQLAlchemy engine, sessionmaker, init_db
│   │   ├── models.py        ← Species, Visit, Detection, HaikuboxDetection,
│   │   │                       PushSubscription, Correction
│   │   └── utils.py         ← utcnow() helper (naive UTC)
│   │
│   ├── routers/             ← FastAPI routes, one module per resource
│   │   ├── ingest.py        ← POST /api/ingest/motion (camera upload)
│   │   ├── detections.py    ← GET /api/detections (PWA feed read)
│   │   ├── species.py       ← GET /api/species (picker data)
│   │   ├── corrections.py   ← POST /api/corrections
│   │   └── push.py          ← Web Push subscription mgmt + VAPID key
│   │
│   ├── ingest/              ← External data sources
│   │   └── haikubox.py      ← poll the Haikubox REST API
│   │
│   ├── pipeline/            ← The classification pipeline
│   │   ├── frames.py        ← OpenCV frame extraction at ~3 fps
│   │   ├── detect.py        ← YOLO11-nano singleton, bird-only
│   │   ├── track.py         ← IoU tracker (no ML deps, pure)
│   │   ├── classify.py      ← HF transformers classifier + NA allow-list
│   │   ├── calibration.py   ← Load yard_priors.json with fallbacks
│   │   ├── fuse.py          ← Bayesian fusion (visual × audio × seasonal)
│   │   ├── notify.py        ← rarity decision + Web Push dispatch
│   │   ├── process.py       ← Orchestrator: extract → detect → ... → DB
│   │   └── worker.py        ← APScheduler: polls Visit rows; runs process
│   │
│   ├── scripts/             ← One-shot CLI tools
│   │   ├── smoke_test.py    ← Run pipeline against a clip locally
│   │   ├── generate_vapid_keys.py
│   │   ├── calibrate_from_haikubox.py
│   │   ├── fetch_models.py  ← Pre-warm YOLO + classifier weights
│   │   └── retrain_classifier.py  ← STUB; prints correction summary
│   │
│   ├── tests/               ← pytest, no ML deps required
│   │   ├── conftest.py      ← autouse: blank calibration path per test
│   │   ├── test_track.py    ← IoU tracker
│   │   ├── test_fuse.py     ← Bayesian fusion
│   │   ├── test_calibration.py
│   │   ├── test_notify.py
│   │   └── test_species_and_corrections.py
│   │
│   ├── data/                ← gitignored: SQLite, clips, crops, yard_priors
│   ├── models/              ← gitignored: HF / YOLO weight cache
│   └── secrets/             ← gitignored: vapid_private.pem
│
├── frontend/                ← Vite + React 18 + TS PWA
│   ├── package.json + lock
│   ├── vite.config.ts       ← injectManifest mode for custom service worker
│   ├── tsconfig*.json
│   ├── tailwind.config.js
│   ├── index.html
│   └── src/
│       ├── main.tsx, App.tsx
│       ├── sw.ts            ← Custom service worker: push + click handlers
│       ├── index.css
│       ├── lib/
│       │   ├── api.ts       ← fetchDetections, fetchSpecies, submitCorrection
│       │   └── push.ts      ← Web Push subscribe/unsubscribe
│       ├── pages/
│       │   ├── Feed.tsx, Species.tsx, Settings.tsx
│       └── components/
│           ├── DetectionCard.tsx  ← per-detection card + "Wrong species?"
│           ├── SpeciesPicker.tsx  ← searchable combobox
│           └── AudioBadge.tsx
│
└── scripts/                 ← Deploy automation (runs on Mac, not VM)
    ├── deploy_to_server.sh  ← rsync + ssh into bootstrap
    └── bootstrap_server.sh  ← runs ON the VM; installs, builds, calibrates
```

## The pipeline end-to-end

Trace one motion event from camera to phone notification:

1. **Camera fires motion event.** Reolink uploads a multipart clip to
   `POST /api/ingest/motion` (`routers/ingest.py`). The route writes the
   clip to `data/clips/<timestamp>_<orig>.mp4` and creates a `Visit` row
   with `processed_at = NULL` and the clip path.

2. **Worker tick.** `pipeline/worker.py` is an APScheduler job firing
   every 5 s inside the same process. It queries `Visit` rows where
   `processed_at IS NULL`, takes one, and hands it to
   `pipeline/process.py::process_visit`.

3. **Frame extraction.** `pipeline/frames.py` uses OpenCV to decode the
   clip at ~3 fps. Each `Frame` knows its index, timestamp, and BGR pixels.

4. **Per-frame detection.** For every frame, `pipeline/detect.py` runs
   YOLO11-nano restricted to COCO class 14 ("bird"), confidence ≥ 0.30.
   Returns `BirdDetection(bbox, confidence, frame_index)`.

5. **Tracking.** `pipeline/track.py::Tracker` is a greedy IoU matcher
   (`MATCH_IOU_THRESHOLD = 0.30`, `MAX_MISSED_FRAMES = 3`). It assigns
   detections to ongoing tracks across frames; the output is a list of
   `Track`s each containing the per-frame detections of one bird.

6. **Per-track classification.** For each track, `process.py` selects up
   to 3 crops (best by `area × confidence`, plus 2 evenly-spaced others)
   and hands each to `pipeline/classify.py::classify_bird`. The classifier:
   - Loads `dennisjooo/Birds-Classifier-EfficientNetB2` (525 classes).
   - Filters via `NA_BACKYARD_ALLOWLIST` (or the calibration-derived list
     from `pipeline/calibration.py::get_allowlist()` if a `yard_priors.json`
     exists).
   - If the post-softmax mass on allow-listed species is below 10 %, returns
     `[]` (the crop is treated as not-a-bird and the track is skipped).
   - Otherwise returns top-5 normalized predictions.

7. **Multi-crop voting.** `process.py::_average_predictions` averages
   the per-crop top-5 distributions into a single per-track top-5.

8. **Bayesian fusion.** `pipeline/fuse.py::fuse` re-ranks the top-5 by
   multiplying with two priors:
   - Audio: 3× boost if `HaikuboxDetection` exists for this species in
     the last 90 s (`audio_correlation_window_seconds`).
   - Seasonal: monthly multiplier from `calibration.get_monthly_multiplier`
     (yard-specific) or `_SEASONAL_PRIORS` (hand-coded fallback).
   Renormalizes within the top-5.

9. **Persistence.** `process.py` writes a `Detection` row: top-1 species
   id, fused confidence, full top-5 in `raw_predictions` JSON, `audio_confirmed`
   flag, the crop file path, and the track id.

10. **Push notification.** Immediately after `db.flush()`, `process.py`
    calls `pipeline/notify.py::dispatch_for_detection`. That checks
    `is_rare(species, when, window_days)` per `PushSubscription` row and
    fires a Web Push via `pywebpush` if no recent prior detection exists.
    The service worker (`frontend/src/sw.ts`) shows the system notification.

11. **PWA refresh.** `frontend/src/pages/Feed.tsx` polls
    `GET /api/detections` every 15 s via TanStack Query. The new detection
    appears at the top.

The Haikubox poller is a separate APScheduler job (also in
`pipeline/worker.py`) that runs `ingest/haikubox.py::poll_once` every 30 s
and inserts new audio detections so step 8 has fresh data to fuse with.

## Running locally

```bash
# One-time
cp backend/.env.example backend/.env
# Fill in HAIKUBOX_API_KEY and HAIKUBOX_SERIAL

# Tests (no ML deps needed; 37 tests in ~1s)
make test

# Full local stack (Docker)
make build         # ~5-10 min first time; layers cached after
make smoke         # exercise the pipeline against a sample bird video
make calibrate     # build yard_priors.json from your Haikubox

# Frontend dev with hot reload (assumes Docker backend on :8000)
make fe-dev
```

`make help` lists every target.

## Common "how do I…"

### Add a new API endpoint

1. Create or extend a module in `backend/routers/`.
2. Mount it in `backend/main.py`:
   ```python
   from routers import species
   app.include_router(species.router, prefix="/api/species", tags=["species"])
   ```
3. Write a test in `backend/tests/test_<thing>.py`. Mirror the pattern in
   `test_species_and_corrections.py` (in-memory SQLite + `StaticPool` +
   `app.dependency_overrides[get_db]`).

### Add a new pipeline stage

The pipeline is linear inside `pipeline/process.py::process_visit`. Slot
your stage between the right two existing ones and pass data via the local
variables (`per_crop_predictions`, `fused`, etc.). If the stage needs to
share state with another process, model it as an `APScheduler` job in
`pipeline/worker.py`.

For ML model loading, follow the singleton pattern in
`pipeline/detect.py::_get_model` or `pipeline/classify.py::_load`: a
module-level lock-guarded `_model` variable, lazy-loaded on first call,
with a deferred heavy import behind `TYPE_CHECKING` so tests can import
the module without torch.

### Add a species to the hand-coded allow-list fallback

Edit `NA_BACKYARD_ALLOWLIST` in `backend/pipeline/classify.py`. Match the
gpiosenka dataset's uppercase spelling (consult the classifier's
`id2label` if unsure). Hyphenation differences are forgiven by
`_hyphen_insensitive`.

This only matters when no calibration file exists — in production the
yard-derived list from `yard_priors.json` takes precedence.

### Tighten or relax push notification frequency

The rarity window is per-subscription (`PushSubscription.notify_window_days`,
default 30). The Settings page in the PWA exposes a slider; the backend
honors whatever each subscriber requested. To change the default for new
subscribers, edit `db/models.py` and `routers/push.py::SubscribeRequest`.

To change the rarity logic itself (e.g. "first sighting ever" instead of
"first in N days"), modify `pipeline/notify.py::is_rare`.

### Swap the classifier model

1. Pick a new HF model. If it's a `transformers` image-classification
   model and its labels are recognizable English species names, you can
   override at runtime by setting `BIRD_CLASSIFIER_MODEL` in
   `backend/.env` — no code change needed.
2. If the model uses a different framework (ONNX, fastai checkpoint,
   etc.), rewrite `pipeline/classify.py::_load` to load it. Keep the
   public surface (`classify_bird(crop_bgr) → list[SpeciesPrediction]`)
   identical so nothing else in the pipeline needs to change.
3. Re-run the smoke test (`make smoke`) and inspect the top-1 labels.
4. See `README.md` "Future improvements" for two specific candidates
   that were evaluated and deferred.

### Refresh yard calibration

`make calibrate` re-fetches and overwrites `yard_priors.json`. The loader
in `pipeline/calibration.py` watches the file's mtime, so changes are
picked up live without a backend restart. Plan to do this every few
months as the local bird population shifts seasonally.

## Tests

37 unit tests, organized by module:

| File | Coverage |
|---|---|
| `test_track.py` | IoU geometry, single/multi-bird tracking, gap bridging, best-crop scoring |
| `test_fuse.py` | Bayesian renormalization, audio close-call flips, seasonal boost/suppression |
| `test_calibration.py` | Load + cache + fallback when file missing / malformed |
| `test_notify.py` | Rarity decision over various detection histories |
| `test_species_and_corrections.py` | GET /api/species sources, POST correction happy/edge paths |

None of them load torch/transformers/ultralytics — the heavy modules use
lazy imports + `TYPE_CHECKING` so they only resolve when actually called.
That keeps the pytest run under 2 s and CI's pip install small.

The `conftest.py` autouse fixture points `calibration.CALIBRATION_PATH` at
a non-existent file in `tmp_path` for every test, so the host machine's
real `yard_priors.json` never pollutes test runs. Tests that want a
calibration in scope override that path themselves.

## Debugging recipes

### "Camera uploads succeed but no detections appear in the feed"

Most likely the worker is rejecting every crop as `not_a_bird`. Tail logs:

```bash
make logs | grep -E "tracks|rejected|not_a_bird"
```

If you see `all crops rejected as not_a_bird`, either YOLO is finding
non-bird objects (false positives — increase `BIRD_CONFIDENCE_THRESHOLD`
in `pipeline/detect.py`) or the classifier's `IN_RANGE_THRESHOLD` in
`pipeline/classify.py` is too strict — lower it from 0.10 to 0.05.

If you see no `pipeline.process` logs at all, the worker isn't running.
`docker compose restart api`.

### "Push notifications never arrive"

Run `make logs | grep -i push` while triggering a new species detection.
Common causes:

- `VAPID_PUBLIC_KEY` in `.env` doesn't match the key the browser
  subscribed with. Each VAPID regen invalidates every subscription.
  Confirm parity:
  ```bash
  ssh ryan@birdwatcher.ryanhoulette.com 'grep VAPID_PUBLIC_KEY ~/BirdWatcher/backend/.env'
  curl https://birdwatcher.ryanhoulette.com/api/push/vapid_public_key
  ```
- The species hasn't been seen in `notify_window_days`. The subscription
  records its own window — check the row in `push_subscriptions`.
- Android Chrome silently revoked permission. Reinstall the PWA.

### "Calibration script fails / returns 404 a lot"

The Haikubox `/daily-count` endpoint has an undocumented gap window —
returns 404 for ~7–14 days ago, works both more recent and further back.
The script logs each 404 but keeps going; 200 days of attempted fetches
typically yield ~180+ usable days. If you see 404 on every date,
re-check `HAIKUBOX_SERIAL` and `HAIKUBOX_API_KEY` in `.env`.

### "Wrong species" picker is empty in the PWA

`GET /api/species` returns an empty list. Either:

- No yard calibration exists yet and the hand-coded fallback isn't being
  reached — check `pipeline/classify.py::NA_BACKYARD_ALLOWLIST` is
  non-empty (it is in the committed code).
- Backend can't reach the calibration file. Verify the volume mount in
  `docker-compose.yml` includes `backend/data:/app/data`.

### A test fails after I bumped a dependency

Likely a SQLAlchemy or pydantic breaking change. Common spots:
`Query.get(...)` was removed in 2.x — use `session.get(Model, pk)`.
`utcnow()` is in `db/utils.py`; never call `datetime.utcnow()` directly.

## Deploy

See `DEPLOY.md` for the full 12-step runbook. Shortened workflow once
the VM is set up:

```bash
make deploy   # rsync + bootstrap (idempotent)
make logs     # confirm it came up
```

`scripts/bootstrap_server.sh` is idempotent — re-running skips installed
deps and cached Docker layers. New code reaches the VM via rsync; the
script always rebuilds the image and frontend bundle and re-up's the
stack.

## Deferred work

Stashed in `README.md` "Future improvements":

- **Classifier swap** to `prithivMLmods/Bird-Species-Classifier-526`
  (SigLIP-2 backbone) or an iNat-2021 derivative. Defer until ~50
  real-yard labeled crops are accumulated via active learning.
- **Scheduled yard calibration refresh** (currently manual via
  `make calibrate`; should become an APScheduler job).
- **Active-learning fine-tune** (`scripts/retrain_classifier.py` is
  currently a stub printing the correction summary; trigger the real
  fine-tune around ~500 corrections).

## Conventions

- **Naive UTC everywhere on disk.** Always call `utcnow()` from
  `db.utils`, never `datetime.utcnow()` (deprecated) or
  `datetime.now(timezone.utc)` (timezone-aware, breaks DB comparisons).
- **Defensive parsing of external APIs.** The Haikubox API in particular
  has weak schema docs; see `ingest/haikubox.py::_SPECIES_KEYS` for the
  fallback-key pattern we use. Tolerate shape drift.
- **Lazy-load heavy ML imports.** Use `TYPE_CHECKING` blocks for
  torch/transformers/ultralytics imports at module level; do the real
  import inside the function that needs them. Keeps `pytest` fast.
- **Don't `--amend` commits.** Each phase gets its own commit; the
  history reads as a project timeline.
- **`make` is the API.** Anything you find yourself typing twice should
  become a Makefile target.
