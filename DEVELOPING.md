# Developing BirdWatcher

A guide for the next human (or Claude) who returns to this codebase six months
from now and needs to remember how the pieces fit together. Pair this with
`README.md` (the product blurb) and `DEPLOY.md` (the cloud-VM provisioning
runbook); this file is everything in between.

## The one-paragraph mental model

A Reolink RLC-811WA WiFi camera pointed at the bird feeders. On motion it
uploads a JPG snapshot and a short MP4 clip over FTPS to a pure-ftpd
container on our VM. The backend's filesystem-scan worker picks new files
out of the FTPS drop directory, extracts frames at ~4 fps, runs **tiled**
YOLO11-small over each frame (4K downsampling drops small birds otherwise),
tracks detections across frames with a simple IoU tracker, ranks each
track's crops by area × confidence × Laplacian-variance sharpness, hands
the top three crops to a fine-grained bird species model, averages those
top-5 distributions, and re-ranks with two priors: an allow-list + monthly
distribution derived from the user's Haikubox (BirdNET audio) detection
history, and recent audio "did you hear this species in the last 90 s"
boosts. Tracks where the classifier rejects every crop are still persisted
as `species_id=NULL` ("Unidentified") so the user can hand-label them. A
React PWA displays the resulting feed (infinite-scroll) and pings the
user's phone via Web Push when a species hasn't been seen in 30 days.
Users correct misidentifications via a searchable picker — which spans
both yard-heard species and a curated North American list, plus a "Not a
bird" sentinel for YOLO false positives. Those corrections feed a (still-
stubbed) periodic fine-tune of the classifier.

## Architecture

```
   Reolink RLC-811WA              Haikubox (BirdNET audio)
   on motion → FTPS upload        poll detections every 30 s
   (JPG snapshot + MP4 clip)              │
              │                            │
              ▼                            ▼
   ┌────────────────────────────────────────────────────┐
   │  pure-ftpd (TLS, port 22222 + passive 30000-30009) │
   │  writes to data/clips/upload/YYYY/MM/DD/*.{jpg,mp4}│
   └────────────┬───────────────────────────────────────┘
                │ filesystem scan
                ▼
   ┌────────────────────────────────────────────────────┐
   │  FastAPI backend                                   │
   │  - /api/detections      (PWA read, paginated)      │
   │  - /api/species         (picker: yard + NA list)   │
   │  - /api/corrections     (active learning)          │
   │  - /api/push/{subscribe, vapid_public_key}         │
   │  - /api/ingest/motion   (webhook ping, no body)    │
   │                                                    │
   │  APScheduler workers:                              │
   │   • pipeline.worker — scans data/clips/ for new    │
   │     files (≥90 s old), creates Visit rows, runs    │
   │     tiled-YOLO → tracker → sharpness rank →        │
   │     classify → fuse → DB. Pushes if rare.          │
   │     SkipFile exception path retires files too      │
   │     big / missing / corrupt so the queue drains.   │
   │   • ingest.haikubox — polls Haikubox API every     │
   │     30 s; caches into haikubox_detections.         │
   │                                                    │
   │  SQLite (data/birdwatcher.db) + image files        │
   │  (data/clips/, data/crops/). mem_limit: 4g.        │
   │                                                    │
   │  Caddy reverse-proxy in front, TLS via LE.         │
   └────────────┬───────────────────────────────────────┘
                │ HTTPS
                ▼
            Android PWA (Vite + React + TS, vite-plugin-pwa)
            - Feed: useInfiniteQuery + IntersectionObserver
            - Picker: SpeciesPicker w/ "Not a bird" sentinel
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
│   ├── na_birds.py          ← Curated ~200-species NA bird list (picker)
│   │
│   ├── pipeline/            ← The classification pipeline
│   │   ├── frames.py        ← OpenCV frame extraction at ~4 fps;
│   │   │                       raises SkipFile if MP4 > 15 MB
│   │   ├── detect.py        ← Tiled YOLO11-small (1024-px tiles, 20%
│   │   │                       overlap, NMS merge), bird-only
│   │   ├── track.py         ← IoU tracker (no ML deps, pure)
│   │   ├── classify.py      ← HF transformers classifier + NA allow-list
│   │   ├── calibration.py   ← Load yard_priors.json with fallbacks
│   │   ├── fuse.py          ← Bayesian fusion (visual × audio × seasonal)
│   │   ├── notify.py        ← rarity decision + Web Push dispatch
│   │   ├── process.py       ← Orchestrator: extract → detect → track →
│   │   │                       sharpness-rank → classify × 3 → vote →
│   │   │                       fuse → DB. Classifier-rejected tracks
│   │   │                       still persist as species_id=NULL.
│   │   ├── exceptions.py    ← SkipFile (permanent-skip sentinel)
│   │   └── worker.py        ← APScheduler: filesystem scan of
│   │                          data/clips/ → creates Visits → runs process
│   │
│   ├── scripts/             ← One-shot CLI tools
│   │   ├── smoke_test.py    ← Run pipeline against a clip locally
│   │   ├── generate_vapid_keys.py
│   │   ├── calibrate_from_haikubox.py
│   │   ├── benchmark_classifiers.py  ← A/B current vs candidate
│   │   │                                classifier on labeled corrections
│   │   ├── fetch_models.py  ← Pre-warm YOLO + classifier weights
│   │   └── retrain_classifier.py  ← STUB; prints correction summary
│   │
│   ├── tests/               ← pytest, no ML deps required
│   │   ├── conftest.py      ← autouse: blank calibration path per test
│   │   ├── test_track.py    ← IoU tracker
│   │   ├── test_fuse.py     ← Bayesian fusion
│   │   ├── test_calibration.py
│   │   ├── test_notify.py
│   │   ├── test_process.py  ← classifier-rejection persistence path,
│   │   │                       Laplacian-variance ranking
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

1. **Camera fires motion event.** Reolink uploads a JPG snapshot and an
   MP4 clip over FTPS into `data/clips/upload/YYYY/MM/DD/Birdfeeder_NN_YYYYMMDDHHMMSS.{jpg,mp4}`
   (capture time embedded in the filename, in whatever TZ the camera is
   configured for — we run the camera in GMT+0 so the parser can treat it
   as naive UTC).

2. **Worker tick.** `pipeline/worker.py` is an APScheduler job firing
   every 5 s inside the same process. It recursively scans `data/clips/`
   for files that are ≥ `MIN_FILE_AGE_SECONDS` (90 s) old — long enough
   that residential-bandwidth FTPS uploads have finished — and creates a
   `Visit` row for each new path (capture time parsed from filename,
   `processed_at = NULL`). It then picks the oldest pending Visit and
   hands it to `pipeline/process.py::process_visit`. Two exception paths:
   `SkipFile` → mark `processed_at` permanently so the row drops out of
   the queue (file gone / too big / corrupt); anything else → leave
   pending and retry next tick.

3. **Frame extraction.** `pipeline/frames.py` uses OpenCV to decode the
   clip at ~4 fps (Reolink records at 20 fps so this is every ~3rd frame).
   Each `Frame` knows its index, timestamp, and BGR pixels. The frame
   image is consumed and released within the per-frame loop — we don't
   cache full frames anymore (see step 4). Raises `SkipFile` if an MP4
   exceeds `MAX_VIDEO_BYTES` (15 MB).

4. **Per-frame detection.** For every frame, `pipeline/detect.py` runs
   **tiled** YOLO11-small (`yolo11s.pt`, ~10M params) restricted to COCO
   class 14 ("bird"), confidence ≥ `BIRD_CONFIDENCE_THRESHOLD` (0.20).
   The frame is sliced into 1024-px tiles with 20 % overlap, YOLO runs
   at native scale on each tile, and detections are NMS-merged at IoU
   0.50. (Untiled inference on downsampled 4K loses small birds entirely;
   see LESSONS.md.) Tiles overlap by 20 %. Cross-tile duplicates AND
   tile-seam fragments are handled by `_nmm` (Non-Maximum Merging):
   standard duplicates merge into the union bbox, and two half-bird
   detections from adjacent tiles (near-zero gap on one axis with strong
   perpendicular alignment) merge into one whole-bird bbox — so a bird
   that straddles a seam ends up as one whole-bird crop, not two halves.
   Returns `BirdDetection(bbox, confidence, frame_index, crop)`. The crop
   is extracted from the full-res frame at this point with 30 % padding
   (so even tightly-fit YOLO boxes show a visually complete bird) and
   stored on the detection itself, so the frame can be released before
   the next one is decoded — caching ~120 KB crops instead of ~25 MB
   frames is what lets us afford 6 fps sampling without OOM.

5. **Tracking.** `pipeline/track.py::Tracker` is a greedy IoU matcher
   (`MATCH_IOU_THRESHOLD = 0.30`, `MAX_MISSED_FRAMES = 3`). It assigns
   detections to ongoing tracks across frames; the output is a list of
   `Track`s each containing the per-frame detections of one bird.

6. **Sharpness-aware crop ranking.** For each track,
   `process.py::_rank_detections` scores every detection by
   `area × confidence × (Laplacian-variance + 1)`. Laplacian variance is
   a classic focus measure — motion-blurred or out-of-focus crops score
   low, in-focus perched frames win. The best crop is saved to disk under
   `data/crops/v{visit:08d}_t{track:04d}.jpg`; the top 3 are handed to
   the classifier.

7. **Per-track classification.** Each ranked crop goes through
   `pipeline/classify.py::classify_bird`. The classifier:
   - Loads `dennisjooo/Birds-Classifier-EfficientNetB2` (525 classes).
   - Filters via `NA_BACKYARD_ALLOWLIST` (or the calibration-derived list
     from `pipeline/calibration.py::get_allowlist()` if a `yard_priors.json`
     exists).
   - If the post-softmax mass on allow-listed species is below
     `IN_RANGE_THRESHOLD` (0.10), returns `[]` (the crop is rejected).
   - Otherwise returns top-5 normalized predictions (per-base-species
     aggregated across plumage variants).

   **Classifier-rejection path:** if **every** crop in the track is
   rejected, `process.py` still writes a `Detection` row with
   `species_id=NULL` and `confidence=0.0`. These show up in the feed as
   "Unidentified" so the user can hand-label via the picker — the highest-
   value active-learning examples.

8. **Multi-crop voting.** `process.py::_average_predictions` averages
   the per-crop top-5 distributions into a single per-track top-5.

9. **Bayesian fusion.** `pipeline/fuse.py::fuse` re-ranks the top-5 by
   multiplying with two priors:
   - Audio: 3× boost if `HaikuboxDetection` exists for this species in
     the last 90 s (`audio_correlation_window_seconds`).
   - Seasonal: monthly multiplier from `calibration.get_monthly_multiplier`
     (yard-specific) or `_SEASONAL_PRIORS` (hand-coded fallback).
   Renormalizes within the top-5.

10. **Persistence.** `process.py` writes a `Detection` row: top-1 species
    id, fused confidence, full top-5 in `raw_predictions` JSON,
    `audio_confirmed` flag, the crop file path, the bbox, and the track id.

11. **Push notification.** Immediately after `db.flush()`, `process.py`
    calls `pipeline/notify.py::dispatch_for_detection`. That checks
    `is_rare(species, when, window_days)` per `PushSubscription` row and
    fires a Web Push via `pywebpush` if no recent prior detection exists.
    The service worker (`frontend/src/sw.ts`) shows the system notification.

12. **PWA refresh.** `frontend/src/pages/Feed.tsx` uses
    `useInfiniteQuery` against `GET /api/detections?before_id=...` with
    cursor-based pagination (`PAGE_SIZE = 50`) and an IntersectionObserver
    sentinel that fires `fetchNextPage()` on scroll. A 30 s `refetch()`
    interval pulls in fresh captures at the top.

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

56 unit tests, organized by module:

| File | Coverage |
|---|---|
| `test_track.py` | IoU geometry, single/multi-bird tracking, gap bridging, best-crop scoring |
| `test_fuse.py` | Bayesian renormalization, audio close-call flips, seasonal boost/suppression |
| `test_calibration.py` | Load + cache + fallback when file missing / malformed |
| `test_notify.py` | Rarity decision over various detection histories |
| `test_process.py` | Classifier-rejection → species_id=NULL persistence, Laplacian-variance ranking |
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

- **Active-learning fine-tune** (`scripts/retrain_classifier.py` is a
  stub). Real species head needs ~50 corrections per species (currently
  ~26 across all real species). The "Not a bird" pile (~210 corrections)
  is closer to feasible for a YOLO false-positive head fine-tune.
- **Classifier upgrade.** A/B benchmark via `scripts/benchmark_classifiers.py`
  showed prithivMLmods/Bird-Species-Classifier-526 (SigLIP-2) at 14 % top-1
  vs dennisjooo at 3 %. Held off swapping because the absolute numbers say
  the image-quality ceiling matters more than the model; revisit after
  multi-frame voting + sharpness ranking show their gains.
- **Scheduled yard calibration refresh** (currently manual via
  `make calibrate`; should become an APScheduler job).
- **Audio-correlation backfill** (Haikubox poller only fetches the last
  hour; backlogged visits miss audio confirmation).

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
