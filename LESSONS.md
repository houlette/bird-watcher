# Lessons from building BirdWatcher

Things we learned the hard way over the course of this project, organized
by area. Companion to `DEVELOPING.md` (which says *how* the code works);
this document says *what surprised us along the way and why* so the next
person (or the next Claude session) doesn't pay the same tuition.

## Hardware (cameras, etc.)

- **Reolink's "AI animal" class is useless for small birds.** It was
  designed for security use cases (large mammals near doors). Their own
  docs suggest setting a minimum-object-size filter that excludes small
  birds. Use plain motion detection and filter in software.
- **Auto-tracking PTZ cameras chase one subject.** Pointed at a feeder
  with 5 chickadees, the camera hunts whichever bird moved last. Fixed
  varifocal lens with manual zoom is correct for multi-bird scenes.
- **"32MP" on trail cams is interpolated.** Real sensor is usually 4–8MP
  upscaled. Doesn't add detail.
- **Focal length matters more than megapixels** for pixels-on-bird. A
  4K sensor with a varifocal at the right zoom beats an 8K-claimed sensor
  with a wide lens.
- **Don't run Ethernet between floors when you can avoid it.** A WiFi
  camera + plug-in adapter is dramatically easier than running cable + a
  PoE switch, and it works fine for 4K motion clips.

## Cloud / VM

- **Don't reach for AWS for personal projects.** Spot instances lose data
  on termination (wrong for stateful), Lambda's cold starts wreck always-on
  polling workloads, and on-demand t4g costs ~3× Hetzner CAX.
- **Hetzner CAX21 (ARM shared) is a serious bargain.** Our Docker images
  are multi-arch; all our deps (torch, ultralytics, opencv, transformers)
  ship arm64 wheels. ~$6/mo gets 4 vCPU / 8 GB / 80 GB SSD, plenty for
  the workload.
- **Hetzner has TWO firewalls.** The Cloud Firewall in the dashboard sits
  in front of the VM's own UFW. Adding a port to UFW alone is not enough
  — Hetzner's upstream drops the traffic. Every port we open needs to be
  in both places, and removing a port from the dashboard while leaving
  UFW open silently breaks reachability.
- **`adduser --disabled-password` + `usermod -aG sudo` is half a config.**
  The user has sudo group membership but no password to authenticate
  with, so every sudo prompt blocks. Add `NOPASSWD: ALL` for the user
  in `/etc/sudoers.d/` or the user can't sudo. Best done from the
  provider's web-console-as-root before locking down root SSH.
- **Don't power-cycle a wedged VM through the OS** if the OS itself is
  unresponsive. Use the cloud dashboard's hard reset. The in-browser
  console doesn't help if the kernel can't schedule shell commands.

## Docker / Compose

- **`${VAR}` in docker-compose.yml resolves from the project-root `.env`,
  NOT from the `env_file:` field on a service.** This is the most
  surprising thing about compose's env handling. Symptom: our SFTP
  container ran with the literal default password `changeme` because
  `${SFTP_PASSWORD}` couldn't find the variable. We mirror
  `backend/.env` → `./.env` in bootstrap to bridge the gap.
- **`rsync --delete-after` will nuke things you generate on the remote.**
  We lost the VAPID private key on every redeploy because the local
  `backend/secrets/` was empty and rsync diligently mirrored that
  emptiness. Always exclude generated paths.
- **`docker compose up -d` doesn't remove orphans by default.** If you
  rename a service (we renamed `sftp` → `ftp`), the old container keeps
  running. Use `--remove-orphans` or `docker compose down` first.
- **Set `mem_limit:` on containers that run ML inference.** A torch
  process holding a 4K video's worth of decoded frames + model weights
  can OOM an 8 GB VM in seconds, taking everything else down with it.
  4 GB cap (half the host) is a reasonable default. Pair with a
  `SkipFile` exception path so the worker doesn't loop on the same
  problematic file forever.
- **`restart: unless-stopped` + crashing container = infinite OOM
  cycle.** When the API container kept getting OOM-killed and
  auto-restarted on a backlog of large MP4s, the VM became unresponsive
  for half an hour because the docker daemon was constantly fighting
  for memory. Two safety nets needed: bounded memory per container, and
  a way to permanently retire problematic work items.

## Reolink-specific gotchas

- **The webhook is metadata-only.** Reolink's HTTP push fires a 262-byte
  JSON `alarm` payload on motion. It NEVER includes the clip. If you
  want the clip, FTP/FTPS/SFTP is the only mechanism.
- **Reolink firmware doesn't support SFTP.** The UI calls it "FTP
  Settings," and the protocol options are FTP and FTPS (FTP-over-TLS).
  Setting up an SFTP server (atmoz/sftp) wastes a day debugging
  "Connection closed by remote host" before realizing the camera is
  speaking FTP at an SSH server.
- **Reolink's FTP CAN do motion-triggered video uploads** — the Reolink
  docs we found online were misleading on this. The camera UI's
  "Upload" dropdown is "Video & Image", and the schedule tab has an
  Alarm trigger that fires both. We initially thought we'd be limited
  to snapshot bursts only.
- **Reolink uploads into a date-stamped subdirectory tree.** Pattern:
  `<remote_dir>/YYYY/MM/DD/<channel>_<NN>_<YYYYMMDDHHMMSS>.{jpg,mp4}`.
  Our filesystem scan had to be recursive. We also parse the timestamp
  out of the filename for accurate `Visit.started_at`.
- **Capture timestamps in the filename are UTC.** Cross-referenced
  against the webhook's `alarmTime: ...+0000` field. Don't try to do
  timezone math.
- **Max file size in FTP video settings controls chunk length.** Long
  motion events split into multiple files. Set this to match your
  pipeline's per-clip memory budget (we use 15 MB).

## Computer vision / ML

- **YOLO11-nano misses small birds.** A male cardinal at ~100 px on a
  4K frame doesn't classify as "bird" at any confidence threshold,
  even with imgsz=1280. YOLO11-medium also misses. The bird is
  visually present but below the model's effective detection scale
  after downsampling.
- **Use tiled inference for small objects on high-res frames.** Slice
  the 4K frame into overlapping ~1024-px tiles, run YOLO at native
  scale, NMS-merge. SAHI does this; we ended up writing it inline
  because SAHI's category_mapping handling for yolov8 kept KeyError'ing.
  ~40 lines of hand-rolled tiling is less debt than the dependency.
- **YOLO will classify a red bird as "orange"** (the fruit class) at
  low confidence. We saw this on cardinals. Doesn't help us; just a
  flag that the model isn't really seeing the bird.
- **General-purpose classifiers emit exotic non-NA species.** The
  `dennisjooo` 525-species classifier confidently identified a New
  England cardinal as "Fiordland Penguin" and "Baikal Teal." Post-
  filter to an allow-list of locally-plausible species, or use a
  region-specific classifier.
- **The Haikubox API only returns the last hour of audio detections.**
  This means audio correlation only works for live-incoming data, not
  backfills. We'd need to extend the poller's lookback to retroactively
  match audio to old visual detections.
- **Decoded video frames in memory are enormous.** 4K BGR is ~25 MB
  per frame; a 30-second clip at 3 fps is 90 frames = 2.2 GB. If you
  cache all of them so the classifier can read crops out of any one,
  you OOM. Either stream-process (process and release each frame) or
  cap the file size up front.

## Backend / Python

- **`uvicorn` doesn't configure the root logger.** Your `log.info()`
  calls in app code go nowhere until `logging.basicConfig()` runs in
  `main.py`. Symptom: the diagnostic log lines you added to debug an
  issue are invisible.
- **`datetime.utcnow()` is deprecated as of 3.12.** Use
  `datetime.now(timezone.utc).replace(tzinfo=None)` (we wrap as
  `db.utils.utcnow`). Don't switch DB columns to timezone-aware unless
  you're ready to migrate every existing row.
- **`Session.query(Model).get(pk)` is legacy in SQLAlchemy 2.x.** Use
  `session.get(Model, pk)` to silence the warning.
- **In-memory SQLite tests with FastAPI's TestClient need
  `connect_args={"check_same_thread": False}` AND
  `poolclass=StaticPool`.** Without StaticPool, each thread gets its
  own `:memory:` database, so the test thread's seeded data is
  invisible to the request-handling thread.
- **Lazy-import heavy ML deps via `TYPE_CHECKING`.** Importing torch
  at module-load time turns a 1-second pytest run into a 10-second one
  and breaks CI's lightweight install. Defer the import to inside the
  function that needs it.
- **A `FileNotFoundError` in the worker should be a permanent skip,
  not a retryable error.** Our worker treated it as transient, so a
  single dead `Visit` row at the head of the queue blocked the next
  ~50 valid rows from being processed. Pattern: `SkipFile` exception
  → mark `processed_at` so the row drops out of the queue.

## Frontend / PWA / TypeScript

- **`vite-plugin-pwa` needs `injectManifest` (not `generateSW`) if
  you want to own the service worker.** generateSW gives Workbox the
  pen; injectManifest lets you write `src/sw.ts` and have Workbox
  inject the precache manifest into it. Custom push handlers need
  injectManifest.
- **`PushManager.subscribe`'s `applicationServerKey` requires
  `BufferSource` in modern lib.dom.d.ts**, which excludes the
  `Uint8Array<ArrayBufferLike>` that our `urlBase64ToUint8Array`
  returns. Cast at the call site.
- **`tsconfig.node.json` references `"node"` in `types:`.** If you
  forget `@types/node` in devDependencies, `npm run build` fails on a
  clean install. Easy to miss because local development has it
  globally.
- **`npm ci` requires `package-lock.json`.** We committed `package.json`
  without the lock file and the remote build failed. Generate the lock
  locally (`npm install`) and commit it.

## Git / secrets

- **`git filter-repo` is the way to remove secrets from history.**
  `git filter-branch` is deprecated; BFG is fine but Scala-based and
  less flexible. `filter-repo` is Python, fast, drop-in.
- **`filter-repo` rewrites ALL refs by default.** The "backup" branch
  you create just before running it gets rewritten too. The original
  commits are reachable via reflog for ~30 days then GC'd; that's
  your actual safety net.
- **Rotate the secret anyway, even after scrubbing history.** Local
  history is clean, but the original value was visible in your shell
  history, the cloud VM's `.env`, the model context of the AI you were
  pairing with, etc. Scrubbing git ≠ unleaking.

## Process / debugging

- **Smoke-test early with synthetic data.** Before our camera arrived,
  we ran the pipeline against a Wikimedia bird video. That caught the
  classifier hallucinating non-NA species — much cheaper to find then
  than during a live install.
- **Run `npm run build` locally before deploying.** Caught two
  TypeScript errors that would have failed CI and wasted a deploy
  cycle.
- **When debugging "0 detections," look at the actual image first.**
  We spent an hour tuning YOLO confidence thresholds for a frame whose
  "cardinal" turned out to be a stationary red hummingbird feeder. No
  amount of model tuning would help.
- **`Connection closed by remote host` during SSH banner exchange
  means protocol/algorithm mismatch.** Not a credentials issue —
  authentication hasn't even been attempted yet. Either:
  - Client and server can't agree on KEX/cipher/hostkey algorithms
    (old IoT vs modern OpenSSH), or
  - Client is speaking a completely different protocol at an
    SSH-speaking port (Reolink doing FTP at our SFTP server).
- **Stale wakeup prompts are noise; ignore them.** When a scheduled
  wakeup fires after the work it was supposed to check is already done,
  acknowledge the staleness and move on. Don't redo the check.
- **Don't trust web docs for IoT device behavior.** The Reolink "FTP
  doesn't support motion-triggered MP4" claim came from their own
  support article and was contradicted by the actual camera UI. The
  source of truth is the device in your hand.

## Architectural calls that paid off

- **Storage volumes mounted from host, not docker-managed.** Made it
  trivial to inspect captured clips with `ls`, scp them off for
  debugging, etc.
- **APScheduler running inside the FastAPI process** rather than as a
  separate worker container. Simpler ops, simpler debugging, and our
  workload is light enough that one process is fine.
- **Visit/Detection schema separating ingestion from processing.**
  Lets the worker rebuild the pending queue from disk (filesystem
  scan) without losing data on restarts.
- **Fallback chains everywhere** — calibration falls back to a hand-
  coded allow-list, capture-time parser falls back to file mtime,
  classifier falls back to default model, push falls back to no-op
  when VAPID isn't configured. Means the system is always shippable
  even when half the upstream pieces aren't set up yet.

## Things to revisit if revived later

- **Audio correlation backfill.** Currently the Haikubox poller only
  fetches the last hour. Once or twice a day fetching a 24-hour window
  would let backlog detections benefit from audio confirmation.
- **Stream-process frames instead of caching.** Lets us lift the 15 MB
  video-size cap and process longer motion events without OOM risk.
- **Real-time push of detections via Server-Sent Events.** The PWA
  currently polls `/api/detections` every 15s. SSE would feel snappier.
- **Active-learning retraining loop.** `scripts/retrain_classifier.py`
  is a stub. Once ~500 user corrections accumulate, wiring it up will
  shift the system from "generic classifier filtered by allow-list"
  to "fine-tuned classifier of this yard's birds." Likely worth ~5
  accuracy points on lookalikes.
- **Per-species expected count alerts.** "I haven't seen a chickadee
  in 5 days" is more useful than "ooh, a new cardinal." The
  infrastructure (push subscriptions, threshold per subscription)
  could carry this with a new endpoint and a Settings UI.
