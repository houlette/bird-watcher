# Developing BirdWatcher

A guide for the next human (or Claude) who returns to this codebase six months
from now and needs to remember how the pieces fit together. Pair this with
`README.md` (the product blurb) and `DEPLOY.md` (the cloud-VM provisioning
runbook); this file is everything in between.

## The one-paragraph mental model

A Reolink RLC-810WA WiFi camera pointed at the bird feeders. On motion it
uploads a JPG snapshot and a short MP4 clip over FTPS to a pure-ftpd
container on our VM. The backend's filesystem-scan worker picks new files
out of the FTPS drop directory, **gates on daylight** (no point running
inference on nighttime IR/grayscale), extracts frames at ~3 fps for the
first 10 s, runs **tiled YOLO11-small** over each frame (4K downsampling
drops small birds otherwise), runs the YOLO output through a **scene
mask** that drops bbox centers in NAB-clustered cells unless YOLO
confidence is strong, tracks the survivors across frames with a simple
**IoU tracker**, ranks each track's crops by area × confidence ×
Laplacian-variance sharpness, optionally **phase-correlation-aligns the
top crops and averages** them into a denoised composite, **CLAHE**-
normalizes for lighting, hands the result to a fine-grained
**EfficientNet-B2** classifier (allow-list filtered, post-softmax mass
threshold), averages those top-5 distributions, and re-ranks with **four
priors**: an allow-list + monthly distribution derived from Haikubox
audio, a recent-audio "did you hear this species in the last 90 s"
boost, a hand-coded seasonal table fallback, and a **per-species
log-normal size prior** over bbox `max(w,h)` (aspect-ratio gated,
perch-perspective scaled). Tracks where the classifier rejects every
crop are still persisted as `species_id=NULL` ("Unidentified") so the
user can hand-label them. A React PWA displays the resulting feed and
pings the user's phone via Web Push when a species hasn't been seen in
30 days. Users correct misidentifications via a searchable picker that
spans yard-heard species, a curated North American list, **four
family-level catch-all labels** (Sparrow / Warbler / Woodpecker /
Finch), and `Not a bird` / `Unknown bird` sentinels. For the
"Unidentified" backlog, a one-shot script asks **Claude (Opus 4.8 with
vision)** to label each crop, auto-commits HIGH-confidence answers,
and queues MEDIUM-confidence ones for a one-tap review filter in the
PWA. The Stats page surfaces the full pipeline funnel as nightly
snapshots plus on-demand today's row, including server-side-rendered
**heatmaps** showing where in the camera frame the birds actually are.

## Architecture

```
   Reolink RLC-810WA              Haikubox (BirdNET audio)
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
   │  - /api/detections      paginated feed read        │
   │  - /api/species         picker: yard + NA +        │
   │                         family-level catch-alls    │
   │  - /api/corrections     user labels                │
   │  - /api/corrections/bulk multi-detection labeling  │
   │  - /api/corrections/confirm  user-confirmed label  │
   │  - /api/corrections/llm-confirm  promote MEDIUM    │
   │  - /api/push/{subscribe, vapid_public_key}         │
   │  - /api/ingest/motion   (webhook ping, no body)    │
   │  - /api/stats           pipeline funnel snapshots  │
   │                                                    │
   │  APScheduler workers:                              │
   │   • pipeline.worker — every 5 s: scans clips/,     │
   │     daylight-gates, runs the pipeline on the       │
   │     newest pending Visit, marks processed/skipped  │
   │   • haikubox poller — every 30 s, last hour        │
   │   • clip cleanup — hourly, 24h retention           │
   │   • frame cleanup — daily, 14 day retention        │
   │     (labeled-detection frames preserved forever)   │
   │   • nightly stats — cron 02:15 UTC                 │
   │   • nightly heatmap render — cron 02:20 UTC        │
   │                                                    │
   │  SQLite (data/birdwatcher.db) + image files        │
   │  (data/clips/, data/crops/, data/frames/,          │
   │   data/calibration/, data/heatmaps/,               │
   │   data/llm_classify_results/). mem_limit: 4g.      │
   │                                                    │
   │  Caddy reverse-proxy in front, TLS via LE.         │
   └────────────┬───────────────────────────────────────┘
                │ HTTPS
                ▼
            React 18 PWA (Vite + TS, vite-plugin-pwa)
            - Feed: useInfiniteQuery + IntersectionObserver
                    + filter chips (All / Unidentified /
                    Awaiting review / LLM-labeled HIGH /
                    LLM-labeled MEDIUM (review) / species)
                    + batch-edit mode + bulk-action bar
            - Labels: NAB-review mode (filter only)
            - Stats: headline cards, funnel/rates line charts,
                     classifier-vs-corrections, per-species
                     accuracy, top species, hour-of-day heatmap,
                     YOLO-conf histogram split by NAB vs species,
                     bird-location heatmaps with three tabs
            - Settings: push notifications
            - SpeciesPicker: pinned sentinels + classifier
                     suggestions + families + yard + extra NA
            - DetectionCard: blurred letterbox backdrop,
                     image tap → ImageZoom, top-K alternates,
                     LLM rationale line when source=llm-claude,
                     ✓/🚫/✏️ row in MEDIUM-review and
                     Awaiting-review filters
```

## Project layout

```
BirdWatcher/
├── README.md
├── DEPLOY.md
├── DEVELOPING.md            ← (this file)
├── LESSONS.md
├── Makefile
├── docker-compose.yml
├── Caddyfile
├── .github/workflows/ci.yml
│
├── backend/
│   ├── main.py
│   ├── settings.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .env.example
│   │
│   ├── db/
│   │   ├── session.py       SQLAlchemy engine, init_db,
│   │   │                    additive-column migrations
│   │   ├── models.py        Species, Visit, Detection,
│   │   │                    HaikuboxDetection, PushSubscription,
│   │   │                    Correction, PipelineStatsDaily
│   │   ├── families.py      family-level catch-all definitions
│   │   │                    + DB seeder
│   │   └── utils.py         utcnow() helper
│   │
│   ├── routers/
│   │   ├── ingest.py        POST /api/ingest/motion
│   │   ├── detections.py    GET  /api/detections + filters
│   │   ├── species.py       GET  /api/species (yard/extra/families)
│   │   ├── corrections.py   POST + /bulk + /confirm + /llm-confirm
│   │   ├── push.py          Web Push subscription mgmt + VAPID
│   │   └── stats.py         GET  /api/stats?days=30
│   │
│   ├── ingest/
│   │   └── haikubox.py
│   │
│   ├── na_birds.py
│   │
│   ├── pipeline/            The classification pipeline
│   │   ├── frames.py        OpenCV frame extraction, 3 fps, 10 s cap
│   │   ├── detect.py        Tiled YOLO11-small + NMS + NMM
│   │   ├── scene_mask.py    100×100 grid NAB-cluster suppression
│   │   ├── track.py         IoU tracker
│   │   ├── classify.py      EfficientNet-B2 + allow-list +
│   │   │                    in-range threshold + CLAHE pre-proc
│   │   ├── daylight.py      sunrise/sunset gate (astral)
│   │   ├── calibration.py   yard_priors.json loader (mtime-cached)
│   │   ├── fuse.py          Bayesian fusion: visual × audio ×
│   │   │                    seasonal × size_mult
│   │   ├── size_prior.py    per-species log-normal over max(w,h),
│   │   │                    aspect-gated + perch-scaled
│   │   ├── notify.py        rarity decision + Web Push dispatch
│   │   ├── process.py       Orchestrator: extract → detect →
│   │   │                    scene_mask → track → sharpness rank →
│   │   │                    [optional multi-frame phase-correlation
│   │   │                    fusion] → classify → vote → fuse →
│   │   │                    DB → push + capture track_bboxes,
│   │   │                    scene_mask_suppressed count
│   │   ├── stats.py         daily funnel aggregation,
│   │   │                    global totals, classifier accuracy
│   │   │                    (top1 vs corrected_name)
│   │   ├── exceptions.py    SkipFile sentinel
│   │   └── worker.py        APScheduler: visit processing,
│   │                        Haikubox poll, clip + frame retention,
│   │                        nightly stats compute,
│   │                        nightly heatmap render
│   │
│   ├── scripts/
│   │   ├── smoke_test.py            run pipeline on one clip
│   │   ├── generate_vapid_keys.py
│   │   ├── fetch_models.py          pre-warm YOLO + classifier
│   │   ├── calibrate_from_haikubox.py   build yard_priors.json
│   │   ├── benchmark_classifiers.py     A/B on labeled corrections
│   │   ├── retrain_classifier.py        STUB
│   │   ├── llm_classify_unidentified.py Claude vision → Corrections
│   │   ├── analyze_bird_locations.py    heatmap renderer (PNG)
│   │   ├── backfill_stats.py            one-shot historical snapshot
│   │   ├── backfill_llm_rationale.py    backfill old llm-claude rows
│   │   ├── backfill_llm_medium.py       JSONL → llm-claude-medium
│   │   ├── depth/                       size-prior experiments
│   │   │   ├── build_depth_map.py
│   │   │   ├── species_sizes.py         Cornell length table
│   │   │   ├── size_from_bbox.py        scale-free size proxy
│   │   │   ├── validate_size_signal.py
│   │   │   ├── compare_proxies.py
│   │   │   ├── compare_track_smoothed.py
│   │   │   └── calibrate_size_priors.py production calibration
│   │   │                                 used by pipeline.size_prior
│   │   └── sweep/                       regression triage + OFAT
│   │       └── …                        replay harness, scoring,
│   │                                    multi-config experiments
│   │
│   ├── tests/                17 test modules, 141 passing
│   ├── data/                 gitignored
│   ├── models/               gitignored
│   └── secrets/              gitignored
│
├── frontend/
│   └── src/
│       ├── main.tsx, App.tsx
│       ├── sw.ts
│       ├── lib/
│       │   ├── api.ts        fetchDetections (many filter params),
│       │   │                 submitCorrection, bulkCorrection,
│       │   │                 confirmClassifierLabel,
│       │   │                 confirmLlmCorrection,
│       │   │                 fetchStats, fetchSpecies
│       │   └── push.ts
│       ├── pages/
│       │   ├── Feed.tsx, Species.tsx, Settings.tsx, Stats.tsx
│       └── components/
│           ├── DetectionCard.tsx
│           ├── SpeciesPicker.tsx
│           ├── FilterPicker.tsx
│           ├── BulkActionBar.tsx
│           ├── ImageZoom.tsx
│           └── AudioBadge.tsx
│
└── scripts/                  Deploy automation (runs on Mac)
    ├── deploy_to_server.sh
    └── bootstrap_server.sh
```

## The pipeline end-to-end

Trace one motion event from camera to phone notification:

1. **Camera fires motion event.** Reolink uploads a JPG snapshot and an
   MP4 clip over FTPS into
   `data/clips/upload/YYYY/MM/DD/Birdfeeder_NN_YYYYMMDDHHMMSS.{jpg,mp4}`.
   The camera runs in GMT+0 so the parser treats the filename timestamp as
   naive UTC. Years before 2000 (the camera sometimes loses NTP and writes
   `1970…`) fall back to file mtime.

2. **Worker tick.** `pipeline/worker.py` is an APScheduler job firing
   every `POLL_INTERVAL_SECONDS` (5 s). It scans `data/clips/` for files
   ≥ `MIN_FILE_AGE_SECONDS` (90 s) old (FTPS upload typically completes
   within 16–80 s on residential bandwidth), creates a `Visit` row for
   each new path (capture time parsed from filename), and picks the
   **newest** pending Visit to process (counter-intuitive for a queue —
   during a backlog drain, fresh visits should reach the feed faster
   than 5-day-old ones).

3. **Daylight gate.** `pipeline/daylight.py::is_daylight` uses the
   `astral` library against the configured camera lat/long (Boston, MA in
   settings) to determine if the capture time falls between sunrise and
   sunset + a buffer. Out-of-range visits are marked
   `processing_error="skipped: captured outside daylight hours"` and
   short-circuited. At night the Reolink switches to grayscale IR which
   the classifier was never trained on, and birds aren't active anyway —
   no point burning CPU.

4. **Frame extraction.** `pipeline/frames.py::extract_frames` uses OpenCV
   to decode the clip at `target_fps = 3` for the first
   `MAX_PROCESS_DURATION_SECONDS` (10 s). At a 20-fps source that's every
   ~7th frame — enough sharpness candidates per visit while keeping
   per-clip YOLO cost bounded. Raises `SkipFile` if the MP4 exceeds
   `MAX_VIDEO_BYTES`. The frame image is consumed and released within the
   per-frame loop; we never cache full frames (each is ~25 MB at 4K).

5. **Per-frame detection (tiled YOLO).** `pipeline/detect.py::detect_birds`
   runs **YOLO11-small** (`yolo11s.pt`, ~10M params) restricted to COCO
   class 14 ("bird"), confidence ≥ `BIRD_CONFIDENCE_THRESHOLD = 0.35`. The
   3840 × 2160 frame is sliced into `TILE_PX = 1024` tiles with
   `TILE_OVERLAP_PX = 205` (20 %) of overlap; YOLO runs at native scale on
   each tile. Untiled inference on downsampled 4K loses small birds — a
   100-px-wide bird on a 1024-resized frame collapses to ~17 px, well
   below YOLO's detection floor.

   - **NMS** (Non-Maximum Suppression) at IoU 0.50 within each tile.
   - **NMM** (Non-Maximum Merging) across tiles: ordinary cross-tile
     duplicates merge into the union bbox, and **tile-seam fragments**
     (two half-bird detections with near-zero gap on one axis and strong
     perpendicular alignment) merge into one whole-bird bbox.

   Each surviving detection has its **crop extracted from the full-res
   frame with 30 % padding** and stored on the `BirdDetection` so the
   frame can be released before the next iteration — caching ~120 KB
   crops scales with bird-count, not frame-count.

6. **Scene-mask suppression.** `pipeline/scene_mask.py::filter_detections`
   drops detections whose bbox center falls inside a "hot" 100 × 100 px
   grid cell, where "hot" means ≥ `MIN_NABS_PER_CELL` (10) user-labeled
   "Not a bird" detections in the last `LOOKBACK_DAYS` (14). A detection
   with YOLO confidence ≥ `OVERRIDE_YOLO_CONFIDENCE` (0.65) is kept
   regardless — a confident bird at the hummingbird feeder shouldn't get
   dropped just because the location usually contains glints. The mask
   refreshes every hour from the DB. Both the kept list and the
   suppressed count are returned; the count is persisted to
   `Visit.scene_mask_suppressed` so the Stats page can surface the
   otherwise-invisible "real bird at a hot cell got dropped" failure
   mode.

7. **Tracking.** `pipeline/track.py::Tracker` is a greedy IoU matcher
   (`MATCH_IOU_THRESHOLD = 0.30`, `MAX_MISSED_FRAMES = 3`). It assigns
   detections to ongoing tracks across frames; the output is a list of
   `Track`s, each containing the per-frame detections of one bird.

8. **Sharpness-aware crop ranking.** For each track,
   `process.py::_rank_detections` scores every detection by
   `area × confidence × (Laplacian-variance + 1)`. Laplacian variance is
   a classic focus measure — motion-blurred or out-of-focus crops score
   low; in-focus perched frames win. The **top 3** become classifier
   candidates.

9. **Multi-frame fusion (optional).** When `_USE_MULTI_FRAME_FUSION = True`
   (current default), `process.py::_fuse_crops` phase-correlates each of
   the top 2 crops against the top-1 anchor (`cv2.phaseCorrelate`),
   re-aligns them, and averages all three into a single denoised
   composite. Pixel-wise averaging of well-aligned crops cancels random
   noise / motion blur while preserving plumage detail. Falls back to the
   anchor crop alone when alignment confidence is low. Reduces classifier
   inference cost by 3× per track AND consistently produces a cleaner
   crop than any individual frame.

10. **CLAHE pre-processing for the classifier.** `pipeline/classify.py`
    runs **Contrast-Limited Adaptive Histogram Equalization** on the L
    channel of LAB (`clipLimit=2.0`, `tileGridSize=(8, 8)`) before
    classification. This lifts shadowed feather detail without blowing
    out highlights or shifting colors. CLAHE in RGB shifts hues;
    L-channel-only is colour-safe. The display polish in
    `process.py::_polish_for_display` does the same thing on the
    user-visible crop saved to disk (sharpen pass was tried at 0.5 → 0.3
    → 0.15 amount and consistently read as crunchy; disabled).

11. **Per-track classification.**
    `pipeline/classify.py::classify_bird` loads
    `dennisjooo/Birds-Classifier-EfficientNetB2` (525 classes) and:

    - Filters via `NA_BACKYARD_ALLOWLIST` ∪ `NA_BIRD_SPECIES`, OR a
      yard-calibrated allow-list from `pipeline/calibration.py::get_allowlist()`
      when `data/calibration/yard_priors.json` exists.
    - If the post-softmax mass on allow-listed species is below
      `IN_RANGE_THRESHOLD = 0.10`, returns `[]` (crop rejected).
    - Otherwise returns top-5 normalized predictions, aggregated across
      per-base-species plumage variants (`_normalize_for_display` handles
      apostrophe-tolerant matching + a curated typo-fix table for known
      classifier-label oddities like `BLACKBURNIAM WARBLER`).

    **Classifier-rejection path:** if every crop in the track (or the
    fused composite) is rejected, `process.py` still writes a `Detection`
    row with `species_id=NULL` and `confidence=0.0`. These show up in
    the feed under the "Unidentified" filter so the user can hand-label
    them — the highest-value active-learning examples.

12. **Multi-crop voting.** `process.py::_average_predictions` averages
    the per-crop top-5 distributions into a single per-track top-5. In
    fused-crop mode there's only one crop, so this is a passthrough.

13. **Bayesian fusion.** `pipeline/fuse.py::fuse` re-ranks the top-5 by
    multiplying with four priors:

    - **Audio**: `AUDIO_BOOST = 3.0` if a `HaikuboxDetection` exists for
      this species within
      `settings.audio_correlation_window_seconds` (90 s) of capture time.
    - **Seasonal**: monthly multiplier from
      `calibration.get_monthly_multiplier(species, month)` (yard-derived
      from Haikubox monthly distribution) or `_SEASONAL_PRIORS`
      (hand-coded eastern-NA fallback for Junco, Hummingbird, etc.).
    - **Size**: `size_prior.size_multiplier(species, bbox)` —
      log-normal density of observed `max(w, h)` under the species's
      fit, gated by aspect ratio and scaled by foot-y perspective bin.
      See "Models & algorithms" below for the full formula. Floors at
      `SIZE_FLOOR = 0.33`.

    The product is renormalized within the top-5 to a posterior
    distribution.

14. **Persistence.** `process.py` writes a `Detection` row: top-1
    species id, fused confidence, full top-5 in `raw_predictions` JSON,
    `audio_confirmed` flag, crop file path, bbox, track id, **the full
    list of per-frame bboxes in the track** (`track_bboxes` JSON, used
    by the size-prior smoothing experiment), and YOLO confidence of the
    saved crop. The visit's `scene_mask_suppressed` count is also
    persisted.

15. **Push notification.** Immediately after `db.flush()`,
    `process.py` calls `pipeline/notify.py::dispatch_for_detection`.
    That checks `is_rare(species, when, window_days)` per
    `PushSubscription` row and fires a Web Push via `pywebpush` if no
    recent prior detection exists. The service worker
    (`frontend/src/sw.ts`) shows the system notification.

16. **PWA refresh.** `frontend/src/pages/Feed.tsx` uses
    `useInfiniteQuery` against `GET /api/detections?before=…` with
    compound cursor pagination (`captured_at|detection_id`) and an
    IntersectionObserver sentinel that fires `fetchNextPage()` on
    scroll. A 30 s `refetchInterval` pulls in fresh captures at the top.

The Haikubox poller is a separate APScheduler job
(`ingest/haikubox.py::poll_once`) that runs every 30 s and inserts new
audio detections so step 13 has fresh data to fuse with. Currently
fetches only the last hour, so backlogged visits don't get audio
confirmation — see `LESSONS.md` "Audio-correlation backfill" for the
known gap.

## Models & algorithms reference

A one-line summary of every algorithm in the pipeline + its constants.

| Stage | Algorithm / model | Where | Key constants |
|---|---|---|---|
| Detection | YOLO11-small | `pipeline/detect.py` | `BIRD_CONFIDENCE_THRESHOLD=0.35`, COCO class 14 |
| Tiled inference | 1024×1024 tiles, 20 % overlap, NMS@0.50 within tile | `pipeline/detect.py::_tile` | `TILE_PX=1024`, `TILE_OVERLAP_PX=205` |
| Cross-tile dedup | NMM (Non-Maximum Merging) — union bbox for ordinary overlaps; seam-stitching for half-bird fragments | `pipeline/detect.py::_nmm` | `TILE_SEAM_GAP_PX=20` |
| Scene mask | 100×100 px grid, hot cells = ≥10 NAB labels in 14 days; high-confidence override | `pipeline/scene_mask.py` | `GRID_PX=100`, `MIN_NABS_PER_CELL=10`, `LOOKBACK_DAYS=14`, `OVERRIDE_YOLO_CONFIDENCE=0.65` |
| Tracker | Greedy IoU matcher | `pipeline/track.py` | `MATCH_IOU_THRESHOLD=0.30`, `MAX_MISSED_FRAMES=3` |
| Sharpness rank | `area × confidence × (Laplacian-variance + 1)` | `pipeline/process.py::_rank_detections` | `cv2.Laplacian(gray, cv2.CV_64F).var()` |
| Multi-frame fusion | Phase-correlation alignment + pixel averaging, top-3 crops → one composite | `pipeline/process.py::_fuse_crops` | `_USE_MULTI_FRAME_FUSION=True` |
| Lighting normalization | CLAHE on L channel of LAB | `pipeline/classify.py::_preprocess_for_classifier` | `clipLimit=2.0`, `tileGridSize=(8,8)` |
| Classifier | EfficientNet-B2 — `dennisjooo/Birds-Classifier-EfficientNetB2` | `pipeline/classify.py` | 525 classes, top-5 returned |
| Allow-list filter | Yard-calibrated set or `NA_BACKYARD_ALLOWLIST ∪ NA_BIRD_SPECIES`; reject if in-range mass below threshold | `pipeline/classify.py`, `pipeline/calibration.py` | `IN_RANGE_THRESHOLD=0.10`, `MIN_DETECTIONS_FOR_ALLOWLIST=5` |
| Per-track vote | Softmax-averaged top-5 across surviving crops | `pipeline/process.py::_average_predictions` | – |
| Audio prior | 3× boost if Haikubox heard species within 90 s | `pipeline/fuse.py` | `AUDIO_BOOST=3.0`, `AUDIO_FLOOR=1.0`, window 90 s |
| Seasonal prior | Yard-calibrated monthly multiplier or hand-coded fallback | `pipeline/calibration.py::get_monthly_multiplier`, `pipeline/fuse.py::_SEASONAL_PRIORS` | `MAX_SEASONAL_MULTIPLIER=4.0`, `MIN=0.1` |
| Size prior | Log-normal density of `max(w,h)` under per-species μ_log/σ_log fit, ASPECT-gated, perch-scaled | `pipeline/size_prior.py` | `SIZE_FLOOR=0.33`, aspect_bounds `[0.40, 2.50]`, 5 perch bins anchored on Mourning Dove |
| Daylight gate | Astral library sunrise/sunset + buffer | `pipeline/daylight.py` | `PRE_SUNSET_BUFFER_MINUTES=15`, IANA timezone |
| Rarity (push) | "First in N days" per-subscription | `pipeline/notify.py::is_rare` | default `notify_window_days=30` |
| LLM backlog classify | Claude Opus 4.8 vision + structured-output JSON schema + prompt caching | `scripts/llm_classify_unidentified.py` | `MODEL="claude-opus-4-8"`, `EFFORT="medium"`, HIGH auto-commits, MEDIUM queues for review, LOW skipped |
| Heatmap rendering | numpy `histogram2d` + scipy `gaussian_filter` (σ=2), matplotlib overlay on a clean background frame | `scripts/analyze_bird_locations.py` | bins=96; size-grid cell=240 px, minimum 5 samples |

## The `Correction.source` taxonomy

The DB schema allows multiple kinds of labels. The `source` column on
`Correction` carries provenance:

| `source` | Meaning |
|---|---|
| `NULL` | User-via-UI correction (the picker). Default for new POSTs. |
| `user-confirmed` | User reviewed an `Awaiting review` card and tapped ✓ — records true-positive agreement with the production classifier. Without this we couldn't measure classifier-TP rate (every other Correction is implicitly a disagreement). |
| `llm-claude` | HIGH-confidence Claude call (≥99 % spot-check accuracy on our data). Auto-committed by `scripts/llm_classify_unidentified.py`. |
| `llm-claude-medium` | MEDIUM-confidence Claude call. Auto-committed but flagged for one-tap review via the `LLM-labeled MEDIUM (review)` filter. |
| `llm-claude-confirmed` | User tapped ✓ on a `llm-claude-medium` card — promotes the source tag and drops the row out of the review queue. |

These tags must propagate to any future fine-tune script: LLM-generated
labels carry Claude's biases and shouldn't be confused with human
ground-truth.

## Family-level labels

`db/families.py` defines four catch-all labels for when the user
recognizes the broad type but can't ID the species:

- **Sparrow** (House, Song, White-throated, Chipping, Dark-eyed Junco, …)
- **Warbler** (Yellow-rumped, Pine, Common Yellowthroat, Ovenbird, …)
- **Woodpecker** (Downy, Hairy, Red-bellied, Flicker, Pileated)
- **Finch** (House, Purple, American Goldfinch)

These are seeded as `Species` rows with `is_family=True` by
`db.session.init_db()` on every boot (idempotent). They appear in the
picker's "Families" section and travel with member-species lists in the
`/api/species` response so the UI can show "e.g., House, Song,
Junco…" hints. Classifier accuracy on the Stats page grants **partial
credit** when the user says "Sparrow" and the model's top-1 was any
member species.

## The Stats page

`backend/pipeline/stats.py` computes the funnel daily; the nightly
worker cron at 02:15 UTC writes a `PipelineStatsDaily` row per UTC
date. The endpoint `GET /api/stats?days=N` returns the last N daily
snapshots **plus today's row recomputed live** (so the page is always
current) and a `totals` block with all-time metrics.

Daily metrics:

- Funnel: clips_received, clips_daylight, clips_with_detections,
  detections_total, detections_labeled_by_classifier (read from
  `raw_predictions[0].species` since user corrections overwrite
  `species_id`), detections_user_corrected,
  corrections_{nab,unknown,real_species}
- Classifier accuracy: `classifier_correct` (top-1 matches the user's
  correction, with family-level partial credit) / `classifier_eligible`
  (corrections to real species or families)
- Operational: visits_with_processing_error,
  detections_audio_confirmed, detections_scene_mask_suppressed
- Payload extras: hour-of-day histogram, YOLO-confidence histogram
  split by NAB vs species

`scripts/backfill_stats.py` populates historical days from existing
visit data.

The Stats page also embeds three bird-location heatmaps rendered
nightly at 02:20 UTC by `scripts/analyze_bird_locations.py` and served
as static PNGs under `/media/heatmaps/` (the standard `/media/` mount
serves anything under `data/`).

## Database migrations

`backend/db/session.py::_apply_additive_migrations` runs on every
`init_db()` and ALTERs the table to add any column listed in
`_ADDITIVE_COLUMNS` that isn't already present. Tuples are
`(table, column, sql_type)`. Existing rows get NULL — model definitions
mark these columns nullable. Current entries:

- `detections.yolo_confidence` — raw YOLO confidence on the saved crop
- `detections.track_bboxes` — full per-frame bbox history for the track
- `visits.scene_mask_suppressed` — per-visit count
- `pipeline_stats_daily.detections_scene_mask_suppressed` — daily aggregate
- `species.is_family` — family marker
- `corrections.source` — provenance
- `corrections.rationale` — LLM rationale, NULL for user/UI rows

There's no Alembic. The schema is small and the DB is one SQLite file;
additive-only migrations let deploys stay rsync-and-restart.

## Running locally

```bash
# One-time
cp backend/.env.example backend/.env
# Fill in HAIKUBOX_API_KEY and HAIKUBOX_SERIAL (and ANTHROPIC_API_KEY if
# you'll run the LLM backlog script)

# Tests (no ML deps needed; 141 tests, ~2 s)
make test

# Full local stack (Docker)
make build         # ~5-10 min first time
make smoke         # exercise pipeline against a sample bird video
make calibrate     # build yard_priors.json from Haikubox

# Frontend dev with hot reload (assumes Docker backend on :8000)
make fe-dev
```

## Common "how do I…"

### Add a new API endpoint

1. Create or extend a module in `backend/routers/`.
2. Mount it in `backend/main.py`:
   ```python
   from routers import stats
   app.include_router(stats.router, prefix="/api/stats", tags=["stats"])
   ```
3. Write a test mirroring `tests/test_species_and_corrections.py`
   (in-memory SQLite + `StaticPool` + `app.dependency_overrides[get_db]`).

### Add a new pipeline stage

The pipeline is linear inside `pipeline/process.py::process_visit`. Slot
your stage between the right two existing ones and pass data via the
local variables. For ML model loading, follow the singleton pattern in
`pipeline/detect.py::_get_model` (module-level lock-guarded `_model`,
lazy-loaded, deferred heavy import) so tests can import the module
without torch.

### Add a new persistent column

1. Add the mapped column to the model in `db/models.py` (nullable).
2. Append a tuple to `_ADDITIVE_COLUMNS` in `db/session.py` for the
   matching SQL type.
3. Restart — the migration runs on `init_db()`.

### Add a new family-level label

1. Add an entry to `FAMILY_MEMBERS` in `db/families.py` with the
   member-species list.
2. Restart — `ensure_family_species_rows` seeds the new row on boot.

### Refresh yard calibration

`make calibrate` re-fetches and overwrites `data/calibration/yard_priors.json`.
`pipeline/calibration.py` mtime-watches the file so changes are picked
up live without a backend restart. Plan to do this every few months.

### Recalibrate size priors

After accumulating new real-species corrections:
```bash
docker compose exec api python scripts/depth/calibrate_size_priors.py
```
Writes `data/calibration/size_priors.json`. The runtime loader
mtime-watches like the yard priors. Must be re-run after any camera
move (bbox dimensions shift).

### Backlog-classify with Claude

```bash
docker compose exec api python scripts/llm_classify_unidentified.py \
  --limit 20 --auto-commit
```
HIGH-confidence labels commit immediately as `source="llm-claude"`
Corrections (the user reviews them via the `LLM-labeled HIGH` filter).
MEDIUM commit as `source="llm-claude-medium"` and surface in the
`LLM-labeled MEDIUM (review)` filter with ✓/🚫/✏️ buttons. LOW is
skipped. Output JSONL persists under
`data/llm_classify_results/` for audit.

### Swap the classifier model

1. Set `BIRD_CLASSIFIER_MODEL` in `backend/.env` if it's a
   `transformers` image classifier with recognizable label strings.
2. For non-HF models, rewrite `pipeline/classify.py::_load`. Keep
   `classify_bird(crop_bgr) → list[SpeciesPrediction]` identical so
   nothing else needs to change.
3. Re-run `scripts/benchmark_classifiers.py` to A/B against the
   current model on the labeled-correction set.

## Tests

141 tests across 17 modules, organized by component:

| File | Coverage |
|---|---|
| `test_track.py` | IoU geometry, tracking, gap bridging, best-crop scoring |
| `test_fuse.py` | Bayesian renormalization, audio close-call flips, seasonal |
| `test_fuse_crops.py` | Phase-correlation alignment + averaging |
| `test_calibration.py` | Yard-priors loader + cache + fallback |
| `test_notify.py` | Rarity decision |
| `test_process.py` | Classifier-rejection persistence, Laplacian ranking |
| `test_classify_normalize.py` | Apostrophe-tolerant name matching + typo fixes |
| `test_polish_for_display.py` | CLAHE pre-write display polish |
| `test_scene_mask.py` | Hot-cell computation, suppression with confidence override |
| `test_detect.py` | Tile geometry, NMS, NMM (incl. seam-stitching) |
| `test_daylight.py` | Sunrise/sunset gate with timezone |
| `test_size_prior.py` | Log-normal multiplier, aspect gating, perch scaling |
| `test_stats.py` | Daily funnel, accuracy partial credit, endpoint shape |
| `test_families.py` | Family seeder, picker exposure, partial credit |
| `test_species_and_corrections.py` | Picker endpoint, single + bulk correction paths |
| `test_worker_scan.py` | Clip-retention, frame-retention, filename parser |

None load torch/transformers/ultralytics — heavy modules use lazy
imports + `TYPE_CHECKING` so they only resolve when actually called.
That keeps the pytest run at ~2 s and CI's pip install lean.

The `conftest.py` autouse fixture points
`calibration.CALIBRATION_PATH` at a non-existent file per test so the
host's real `yard_priors.json` never pollutes test runs.

## Debugging recipes

### "Camera uploads succeed but no detections appear in the feed"

1. Worker not running: `make logs` → look for any
   `pipeline.worker` line. `docker compose restart api` if absent.
2. Daylight gate: visits at night get
   `processing_error="skipped: …daylight…"`. Check `visits` rows.
3. Scene-mask suppression: spike in `Visit.scene_mask_suppressed` for
   the relevant time window. Check the Stats page's funnel chart for
   the dashed red "Mask-suppressed" line.
4. Classifier rejection: `make logs | grep -E "tracks|rejected"`. If
   you see `all crops rejected`, lower `IN_RANGE_THRESHOLD` in
   `pipeline/classify.py` from 0.10 to 0.05, OR check whether the
   actual species is in the yard allow-list.

### "Push notifications never arrive"

- `VAPID_PUBLIC_KEY` in `.env` doesn't match the key the browser
  subscribed with. Each VAPID regen invalidates every subscription.
- Species hasn't been seen in `notify_window_days`. Check the row in
  `push_subscriptions`.
- Android Chrome silently revoked permission. Reinstall the PWA.

### "Stats page chart looks wrong"

Today's row recomputes live; yesterday's-and-older come from
`PipelineStatsDaily`. If a metric definition changed, you may need to
re-run `scripts/backfill_stats.py` against existing visits to update
old snapshots.

### "Heatmaps look y-flipped"

`scripts/analyze_bird_locations.py` uses `matplotlib.imshow` with
`extent=(0, W, H, 0)` to flip y for image-style display. The contour
function doesn't honor extent the same way; we pass an explicit (X, Y)
meshgrid. If you add a new overlay, do the same — see
`_render_heatmap`.

### A test fails after I bumped a dependency

Likely a SQLAlchemy or pydantic breaking change. `Query.get(...)` was
removed in 2.x — use `session.get(Model, pk)`. `utcnow()` is in
`db/utils.py`; never call `datetime.utcnow()` directly.

## Deploy

See `DEPLOY.md` for the full runbook. Shortened workflow once the VM
is set up:

```bash
make deploy   # rsync + bootstrap (idempotent)
make logs
```

`scripts/deploy_to_server.sh` rsyncs the repo and runs
`bootstrap_server.sh` on the VM, which rebuilds the API image (if
requirements or Dockerfile changed) and the frontend bundle, then
re-up's the stack. New code reaches the VM via rsync; calibration
files and the SQLite DB are bind-mounted under `data/` so they survive
rebuilds.

## Conventions

- **Naive UTC everywhere on disk.** Always call `utcnow()` from
  `db.utils`, never `datetime.utcnow()` (deprecated) or
  `datetime.now(timezone.utc)` (timezone-aware, breaks DB comparisons).
- **Defensive parsing of external APIs.** The Haikubox API in particular
  has weak schema docs; see `ingest/haikubox.py::_SPECIES_KEYS` for the
  fallback-key pattern. Tolerate shape drift.
- **Lazy-load heavy ML imports.** Use `TYPE_CHECKING` blocks for
  torch/transformers/ultralytics at module level; do the real import
  inside the function that needs them. Keeps `pytest` fast.
- **Don't `--amend` commits.** Each phase gets its own commit; the
  history reads as a project timeline.
- **`make` is the API.** Anything you find yourself typing twice should
  become a Makefile target.
- **Additive migrations only.** Add a column; never rename or drop. The
  DB is one SQLite file; adding to `_ADDITIVE_COLUMNS` runs the ALTER
  on every boot.
- **LLM-generated labels carry the `source` tag.** Any future training
  script must filter on `source` to avoid the
  "model labels its own training data" anti-pattern.
