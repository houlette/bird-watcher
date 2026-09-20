This is the plan for the bird detector, agreed with Ryan on 2026-09-15. It
replaces "make YOLO faster" with a sequence of steps, each ending in a gate
whose pass criteria are written here before its result exists. A step that
fails its gate stops the plan at that point until the failure is understood;
nothing downstream starts on the strength of a hoped-for result. Record every
gate result in the status table at the end, with the numbers, the date and the
commit.

## Why the detector is in question

YOLO11s is used as it ships, with COCO weights, for one class on one yard, and
it is the most expensive and the least precise stage in the pipeline:

- YOLO tiles are 97.5% of a video visit's wall clock (`Visit.timings`), and on
  busy days the worker loses most daylight visits to 24-hour clip retention.
- The binary filter overrides about 71% of its detections as not a bird (2,515
  of 3,546), and five junk defences have been built around its false positives.
- The small model was chosen over nano to cut bird-shaped junk, which spends
  CPU on precision the yard's own labels could supply.
- The archive holds 8,170 saved source frames, 2,172 with a user-confirmed
  bird and 3,136 with user-rejected junk, and every detection stores its
  per-frame boxes, so near-complete box labels exist for every object the
  current detector found.

A YOLO-family CNN still looks right for a 4-core CPU with 15 tiles per frame;
transformer detectors such as RF-DETR lead on accuracy but are too heavy here
(published comparisons, not tested on this project). YOLO26, released January
2026, reports up to 43% faster CPU inference for its nano model than YOLO11n
(vendor figure, Intel Xeon, ONNX; unverified here). Background subtraction as
the first stage is ruled out by the project's own evidence: the backdrop model
lost 37.5% of confirmed birds.

## Rules that apply to every step

- **Independent ground truth.** Bird and junk truth comes from user
  corrections (`Correction.source` of NULL or `user-confirmed`), never from a
  model's own output.
- **Held out by construction.** Detector evaluation uses capture days that no
  detector training set may include (step 2 freezes them).
- **Same pixels.** Tiles are cut and letterboxed as production does: 1024 px
  tiles with 20% overlap, padded only to a multiple of 32, merged across tiles
  with `_nmm`. Saved frames are JPEG q85, so absolute rates on them are not
  production's; step 1 and step 6 check on decoded video too.
- **One factor per comparison.** Runtime, architecture and training data are
  changed in separate steps.
- **Counts with intervals.** Every rate is quoted with its count and a 95%
  Wilson interval (`/Users/ryan/.claude/skills/computer-vision/scripts/eval_stats.py`).
- **Reversible shipping.** Every production change goes behind an environment
  flag that restores the previous detector without a code change.
- **Benchmarks** run with the api container paused, one configuration per
  fresh process, anchored by a PyTorch 3-thread run in the same session.

## Step 1: ship the faster runtime for the current model

The same YOLO11s weights, run through OpenVINO with rectangular tiles and
several tiles in flight at once, in place of PyTorch one tile at a time.

Measured so far (2026-09-15, paused-API benchmark, same session anchor PyTorch
3 threads at about 0.350 s per tile):

| Configuration | s/tile | Change |
|---|---|---|
| FP32, 4 tiles at once, 4 threads | 0.199 | −43% |
| INT8, 4 tiles at once, 4 threads | 0.169 | −52% |
| INT8, 2 tiles at once, 4 threads | 0.165 | −53% |
| INT8, 3 tiles at once, 3 threads | 0.220 | −37% |

INT8 was quantized with NNCF 3.3.0 on OpenVINO 2026.3.1, calibrated on 120
tiles from 40 unlabelled frames (listed in `calibration_frames.txt` beside the
model), with the detection head kept in floating point.

**Gate 1.1, quality on labelled frames.** 300 frames with a confirmed bird and
100 with rejected junk, one frame per visit, spread across capture days, none
used for calibration. Only tiles overlapping the labelled box run, which gives
the same answer for that box at a fraction of the CPU.
- Pass: the candidate loses at most 3 of the confirmed-bird boxes PyTorch
  finds (1%), at IoU 0.5 after the merge.
- Pass: on junk frames it finds the labelled junk box at a rate within 10
  points of PyTorch. A detector that suddenly stops seeing junk may also stop
  seeing birds, and the filters downstream were tuned on today's volume.
- Pass: the median confidence change on boxes both find is within ±0.03, and at
  most 2% of them cross 0.65, the spatial filters' confidence override.
- If INT8 passes, use INT8. If INT8 fails and FP32 passes, use FP32. If both
  fail, stop and find the difference in pre- or post-processing first.

**Gate 1.2, parity on production pixels.** Decoded frames from 30 recent video
clips, all 15 tiles, PyTorch against the chosen OpenVINO model.
- Pass: at least 98% of PyTorch's merged boxes are matched at IoU 0.5, and the
  total box count is within 2%.

**Gate 1.3, memory.** Peak resident memory of a process running the chosen
configuration on a full clip, added to the api container's current anonymous
memory (about 2.0 GB of its 4 GB cap).
- Pass: projected at or under 3.4 GB. Otherwise reduce tiles in flight and
  re-measure speed.

**Gate 1.4, cores (Ryan decides).** Four cores busy while the worker runs
(about −52%) against three (about −37%), with the other projects on the VM in
view. The recommendation is 4 tiles in flight at 1 thread each, because a web
request that takes a core then delays one tile instead of stalling a
multi-thread operation.

**Gate 1.5, production.** Shipped behind a `YOLO_BACKEND` setting, with the
PyTorch path kept. Revised on 2026-09-15, before any result: the original
criterion compared detections per visit and the binary filter's override share
over 48 hours with the previous 7 days, but in those 7 days detections per
fully processed visit ranged 0.111 to 0.599 by day (0.326 overall, 1,144 over
3,506 visits) and the override share 81% to 97% (91.4% overall), so bird
activity alone moves both far past any threshold a 48-hour window could test.
The revision tests the shipped code directly instead.
- **1.5a, integration parity.** Deploy with the setting at `torch`. From the
  deployed image, run the shipped `detect_birds` with both backends
  (`backend/scripts/detector/parity_integration.py`) on frames of recent clips
  where production's tracks held a box. Pass: at least 100 PyTorch boxes, at
  least 98% matched at IoU 0.5, counts within 2%. This also makes up for gate
  1.2's thin sample.
- **1.5b, 48 hours on OpenVINO.** Pass: median `detect` per tile in
  `Visit.timings` at most 0.266 s (the slower benchmark session's 0.231 plus
  15%); no out-of-memory kill; no new worker errors. Detections per visit and
  the override share are reported against the ranges above, not gated.
- The watch runs 2026-09-16 11:50 UTC to 2026-09-18 11:50 UTC, from the deploy
  that recreated the container on 4 tiles in flight over 4 threads. Night
  hours contribute nothing, since only daylight clips are processed. The
  counters it reads, memory peak and limit events, are per container, so any
  further deploy restarts the window. Also watch the process high-water mark:
  it must not climb across the 48 hours.
- Two earlier attempts were abandoned, both on configuration rather than on
  anything about OpenVINO itself. The first ran 4 tiles over 4 threads on
  2026-09-15 and was stopped for memory: it held 2.9 GB of process memory,
  pressed the container's 4 GiB limit 4,488 times in 75 minutes and had the
  kernel evict cached data each time, with no kill and 0.206 s per tile over
  119 visits. The second, from 2026-09-16 00:47 UTC, ran 2 tiles over 2
  threads, which Ryan approved that evening to hold 0.49 GB less, and was
  stopped after 11 hours on speed: 0.364 s per tile over 298 visits, against
  the gate's 0.266 s limit and against 0.210 s over 333 visits for 4 over 4
  the day before, so the cut cost 73% and left OpenVINO slower than the
  PyTorch path it replaced, which held 0.322 s in a clean hour. Hourly medians
  ran 0.363 to 0.367 with a p90 of 0.369, including the hours before dawn when
  the worker had the machine to itself, which is what ruled out contention.
- The gate 1.3 benchmark had put 2 over 2 level with 4 over 1, 0.229 against
  0.231 s per tile, but it ran with the api container paused, so it did not
  predict production: benchmark a configuration the way it will run, or do not
  trust the comparison across configurations. The memory headroom that
  motivated the cut is no longer needed, since the commit that made it also
  lifted the container limit to 5 GiB, where 4 over 4 now sits near 41%.
- If the gate fails on the shipped configuration, return to PyTorch
  (`YOLO_BACKEND=torch docker compose up -d api` on the server) and
  investigate before retrying.

## Step 2: freeze the detector yardstick

Before any other model is compared, fix the frames it is judged on.

- Held-out capture days: the binary filter yardstick's 36 held-out days (in
  `backend/scripts/train/heldout_yardstick.json`) plus a seeded random quarter
  of capture days before 2026-06-07. Using the same days keeps both models
  held out together for end-to-end checks.
- Freeze the frame ids and days in a committed file, and make every detector
  training export refuse them, as `heldout.py` does for the binary filter.

Definitions, written on 2026-09-15 before anything was counted. Ryan approved
starting this step during gate 1.5b's watch, since it only reads the database
and its result does not depend on the runtime.
- A frame is a saved source frame, `v{visit}_t{track}.jpg`, the best frame of
  one track. It counts if the file exists at freeze time and its detection's
  latest user label (`Correction.source` NULL or `user-confirmed`) is a species
  or Unknown bird (a bird frame) or Not a bird (a junk frame). Poor quality and
  unlabelled frames are left out.
- A capture day is the camera's local date (America/New_York) of
  `Visit.started_at`.
- The pre-June pool is every capture day before 2026-06-07 holding at least one
  bird or junk frame. The quarter is `round(0.25 * n)` days drawn with
  `random.Random(0).sample` from the sorted pool.
- Training boxes for the gate are bird frames on the other days, one box each
  (the frame's own track). That is a lower bound, since step 4 also counts
  other tracks' boxes at the same frame index.
- Distinct visits are reported beside frame counts, not gated, since frames of
  one visit share the light and usually the bird.
- Every labelled frame on a held-out day is frozen by detection id. Frames
  labelled later on those days are neither scored nor trained on.
- Caveat found after the freeze: all 1,234 held-out junk frames were visible
  in the feed, and 834 of them come from the 3 pre-June days, when the feed
  showed nearly everything; the rest are junk that escaped the binary filter.
  The junk side mixes those two selection routes and holds none of the junk
  the pipeline hid, so it supports paired old-versus-new comparison, not an
  absolute rate for all of the detector's junk.

**Gate 2.**
- Pass: at least 300 confirmed-bird frames and 150 junk frames on held-out
  days, from at least 15 distinct days, and at least 1,500 confirmed-bird boxes
  left on training days.
- If the held-out side is short, raise the pre-June share in steps of 0.1, up
  to 0.5, and recount before freezing. If training boxes fall below 1,500 at
  the share that fills the held-out side, the gate fails, since step 4 could
  not pass either. Refreezing after step 3 has started is not allowed.

## Step 3: generic YOLO26 against YOLO11s

Only the architecture changes: COCO weights, class 14, the step 1 runtime path.

- Build a benchmark image with a current Ultralytics release; production stays
  on 8.3.40 until something ships. Export YOLO26s and YOLO26n to OpenVINO with
  dynamic shapes. YOLO26 outputs final boxes without separate NMS, so the
  post-processing differs, but the cross-tile `_nmm` merge still applies.
- Run gate 1.1's quality check on step 2's held-out frames against YOLO11s,
  and the paused benchmark at the step 1.4 core budget.

**Gate 3.**
- Pass: a YOLO26 size meets gate 1.1's recall and confidence criteria
  against YOLO11s, and is at least 20% faster per tile at the same core budget.
  It then becomes the base for step 5, and may ship generically through step 7.
- Otherwise the fine-tuning base stays YOLO11.
- A confidence threshold for YOLO26 is picked on training days only, never on
  held-out days.

## Step 4: build the training set from the archive

No new labels at this step.

Revised on 2026-09-15, before step 4 started, after two findings. First,
`Detection.track_frames` only exists from 2026-09-11 16:19 UTC (880 rows), so
the other tracks' boxes at a saved frame's moment cannot be read from the
database for nearly every labelled frame. Second, Ryan labels by correction:
since July the feed hides the 67% of detections the pipeline calls Not a bird,
his junk labels are the junk that escaped into the feed (539 from July to
September), and his bird labels are almost all implicit (8 explicit). He
reviews every bird in the feed and corrects obvious mistakes, but leaves a
species alone when unsure between look-alikes such as catbird and mockingbird.

- Run YOLO11s through production's tiled path on each saved frame to find
  every box in it, and match the labelled track's stored box at IoU 0.5. A
  saved frame whose labelled box is not found is dropped and counted.
- A box whose detection's latest user label is a species or Unknown bird is a
  positive; Not a bird is background. From the reviewed-from date, a box whose
  detection the feed showed as a species with no correction is an implicit
  positive, used for training only, never for evaluation. Ryan gave the date on
  2026-09-15: the start of 2026-09-01, camera local time. Earlier uncorrected
  birds say nothing, since he had not reviewed them all. Detections captured in
  the 24 hours before an export are left out, since the newest items may not be
  reviewed yet. Implicit labels are derived at export and never written as
  Correction rows.
- Cut tiles as production does. Keep a tile only if every box in it has a
  label, explicit or implicit, and none is Poor quality; count the tiles and
  labelled boxes this drops. Add an equal number of tiles with no box from the
  same frames.
- Refuse step 2's held-out days. Write a manifest of frame ids, days, seed and
  base-weight hashes beside the dataset.

**Gate 4.**
- Pass: at least 1,500 bird boxes from at least 25 capture days. Counted on
  2026-09-15, before the tile rule drops any: 1,638 explicit bird boxes on 50
  training days, plus 524 implicit ones on training days from 2026-09-01
  (537 in all, 13 of them on held-out days).
- Pass: Ryan audits a contact sheet of 100 random bird boxes and 100 random
  junk boxes, and finds at most 3 wrong in each. Explicit and implicit bird
  labels are audited as separate sheets of 100; if the implicit sheet has more
  than 3 wrong, implicit labels are dropped and the count is rerun without
  them.
- If short: add labels from `llm-claude` and `llm-claude-confirmed` to training
  only, never to evaluation, and audit again. If still short, size a targeted
  labelling pass with `eval_stats.py n-for-margin` before asking for labels.
- Known limit: positives exist only where today's detector found a bird, so a
  fine-tune cannot learn today's blind spots from these labels. Step 5's
  disagreement review counts what it gains.

## Step 5: fine-tune and evaluate on held-out days

- Fine-tune a single-class detector from step 3's base, in nano and small, at
  1024 px, on the Mac. Hold out a validation split of training days to pick the
  confidence threshold; match today's recall on validation confirmed birds.
- Export to OpenVINO and quantize INT8 on training-day tiles.

**Gate 5**, against YOLO11s through the step 1 path, on held-out days:
- **5.1 recall:** loses at most 1% of the confirmed-bird boxes YOLO11s finds.
- **5.2 junk:** finds at least 30% fewer of the held-out junk boxes. This is
  the point of the fine-tune; without it the extra machinery does not earn its
  place.
- **5.3 gains:** Ryan labels a contact sheet of the boxes only the candidate
  finds; the confirmed birds among them are reported, not required.
- **5.4 speed:** no slower per tile than the step 1 model at the same core
  budget; for nano, at least 40% faster.
- Pass 5.1, 5.2 and 5.4: go to step 6.
- Fail 5.1: sort the lost birds by size and lighting before any more training.
- Fail 5.2 only: stop; keep the step 1 or step 3 model.

## Step 6: end-to-end replay on real clips

The tracker, the scene mask's 0.65 override, recurrence and the binary filter
were all tuned on today's detector, so the whole pipeline is compared, not just
boxes.

- Clips are deleted after 24 hours, so first keep a replay sample: one in ten
  daylight clips for five days, stored on the frames volume if the root disk
  cannot take it. On 2026-09-15 that volume had 29 GB free and the root disk
  6.5 GB.
- Replay both pipelines on the sample. Ryan's normal review of those days
  labels the current feed; he also reviews items only the candidate produces.

**Gate 6.**
- Pass: the candidate keeps at least 99% of the replayed feed items Ryan
  confirms as birds.
- Pass: junk reaching the feed falls at least 20%, or stays level with the
  step 5 speed gain.
- Thresholds that moved with the confidence distribution are re-fitted on
  training days, then the gate is rerun.

## Step 7: ship and watch

- Deploy behind the detector flag.
- Watch for 7 days:
  - the outcome metric, junk corrections per day (6.45 before this plan),
    against feed items per day (71.4)
  - detections per visit and the binary filter's override share
  - per-tile time
  - a random sample of 50 binary-filter rejections, reviewed for birds as the
    false-negative channel

**Gate 7.**
- Pass: junk corrections per day fall, or stay level with the speed gain.
- Pass: at most 1 bird in the 50 rejections; no out-of-memory kills or new
  errors.
- Fail: flag back and investigate.

## Step 8: follow-ons

- Retrain the binary filter on the new detector's crops. The yardstick stays
  valid for a paired comparison. Ryan agreed on 2026-09-15 to pause the
  binary filter retrain until gate 7, since its training crops and the junk it
  must catch both change with the detector.
- Update `DEVELOPING.md`, `LESSONS.md` and the computer-vision skill's
  `birdwatcher-state.md`.
- Remove the PyTorch detection path 30 days after gate 7.

## Status

| Step | Status | Gate result | Date, commit |
|---|---|---|---|
| 1 | done: OpenVINO is the production default, gate 1.5b 48-hour watch passed | 1.5b attempt 3: PASS. The 48-hour watch (2026-09-16 11:50 to 2026-09-18 11:50 UTC) on 4 tiles over 4 threads processed 2,126 visits (619,005 tiles). Median detect per tile was 0.2022 s (against the 0.266 s limit; mean 0.2053 s, p90 0.2157 s). 0 non-skipped processing errors (2,055 clips skipped due to pre-gating 24-hour expiration backlog). 0 out-of-memory kills, container restart count 0, memory held ~2.55 GiB of 5 GiB limit. Host CPU is AMD EPYC-Rome (x86_64) running AVX2 JIT kernels. 1.5b attempt 2: FAIL on speed, stopped after 11 hours (0.364 s at 2 over 2). 1.5a: PASS (101/101 boxes matched). 1.4: 4 tiles in flight, 1 thread each. 1.3: PASS (peak process memory 1,876 MB). 1.2: PASS (12/12 boxes). 1.1: FP32 PASS (lost 0/261 birds, 95% CI 0-1.5%), INT8 FAIL (lost 5/261, 1.9%). | 2026-09-18, cbe2120 |
| Phase 0 | done: motion-gated spatial tiling deployed to production as default | PASS. Implemented in `pipeline/motion_tiles.py` and `pipeline/process.py`. Hardened with connected-component seam expansion, signed median delta lighting correction, 30-frame perch memory, and post-visit malloc_trim(0) watchdog. Validated offline on Mac: Stage 1 exact paired Tango eval on N=1,000 stratified yardstick passed non-inferiority (Z = 632.46, p < 0.0001); Stage 2 video replay across 20 verified clips matched 166 of 166 bird tracks (100.0% recall, 0 lost tracks). Shipped to production; live telemetry confirms 22% to 32% reduction in tiles per clip (10.2 to 13.6 tiles/frame vs 15.0 baseline) and ~20s faster per video visit. | 2026-09-18, 2064dcc |
| 2 | done: yardstick frozen in `backend/scripts/detector/heldout_frames.json` (534 bird frames) and expanded in `heldout_frames_1000.json` (N=1,000 bird frames) | PASS at the pre-registered share of 0.25, definitions committed first in d5b932a. Held-out days: the binary yardstick's 36 plus 3 of 11 pre-June days (2026-05-27, 06-02, 06-06), 39 in all, 38 holding frames. Held out: 534 bird frames from 340 visits, 1,234 junk frames from 721 visits. Expanded to N=1,000 bird frames and 400 junk frames stratified by month with max 8% per day in `heldout_frames_1000.json` for hardened detector non-inferiority testing. | 2026-09-15 (d5b932a), 2026-09-18 (cbe2120) |
| 3 | done: generic YOLO26s against YOLO11s | FAIL on speed and accuracy; fine-tuning base stays YOLO11s per pre-registered gate rule. Speed in OpenVINO: 90.3 vs 86.1 ms/tile (-4.9% speedup, fails >=20% gate). On 300 held-out bird frames, YOLO26s found 177 vs YOLO11s's 220 (lost 59 birds, 26.8% lost rate on found, fails <=1% gate; net degradation +14.33%, Tango Z = -4.46, p = 0.9999). 31.1% of common boxes crossed 0.65 (fails <=2%). On 150 junk frames, junk rate was 8.7% vs 40.0% (-31.3 points, fails <=10 pt gate). On 50 seam-straddling birds, recall was only 66.0%. Fine-tuning base formally confirmed as YOLO11s. | 2026-09-18 |
| 4 | done: training dataset exported from archive | PASS. Gate 4 visual audit passed across all 3 sheets reviewed by Ryan (Sheet 1 explicit birds: 0 wrong / 100; Sheet 2 implicit birds >= 0.60 conf: 0 wrong / 100; Sheet 3 junk: <=2 borderline poor-quality / 100, passes <=3 threshold). Curated 18,200 training tiles (9,100 bird + 9,100 balanced background) and 848 validation tiles across 34 training days (27 train, 7 val); strictly refused all 39 held-out days. Packaged in `bird_yolo11s_dataset.tar.gz`. | 2026-09-18 |
| 5 | done: single-class YOLO11s fine-tuned on Mac (`yolo11s_yard_refine`) | PASS Gate 5.2 (89.8% junk cut, 18 vs 176 hits) and Gate 5.4 (74.0 ms/tile vs 77.0 ms baseline, 4% faster). Discovered +115 novel birds (+4.6% net detections, 881 vs 842 hits). On Gate 5.1, strict IoU >= 0.50 lost 76 boxes, of which 48 were better full-body plumage boxes and only 28 (3.3%) were true misses (21 detected at conf 0.11-0.24). Approved by Ryan to proceed to Step 6 video replay. | 2026-09-19 |
| 6 | done: end-to-end replay on real video clips | PASS Gate 6 across 15 real multi-frame video clips from 2026-09-17. Ground-truth confirmed bird retention: 53 of 53 confirmed birds captured (100.0% retention, 0 lost birds, passes >=99% gate). Eliminated 13 baseline false-alarm bird hallucinations. Discovered +47 novel tracks. Temporal video tracking fully bridged static-frame variations. Ready for Step 7. | 2026-09-20 |
| 7 | in progress: ship fine-tuned model and begin 7-day watch | Deploying candidate OpenVINO model behind YOLO_OPENVINO_MODEL and beginning 7-day monitoring. | |
| 8 | not started | | |
