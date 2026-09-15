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

**Gate 1.5, production, 48 hours after deploy.** Shipped behind a
`YOLO_BACKEND` flag, with the PyTorch path kept.
- Pass: median `detect` per tile in `Visit.timings` within 15% of the
  benchmark for the chosen configuration.
- Pass: detections per fully processed daylight visit within 15% of the
  previous 7 days, and the binary filter's override share within 5 points.
- Pass: no out-of-memory kill and no new worker errors.
- If any fails, set the flag back to PyTorch and investigate before retrying.

## Step 2: freeze the detector yardstick

Before any other model is compared, fix the frames it is judged on.

- Held-out capture days: the binary filter yardstick's 36 held-out days (in
  `backend/scripts/train/heldout_yardstick.json`) plus a seeded random quarter
  of capture days before 2026-06-07. Using the same days keeps both models
  held out together for end-to-end checks.
- Freeze the frame ids and days in a committed file, and make every detector
  training export refuse them, as `heldout.py` does for the binary filter.

**Gate 2.**
- Pass: at least 300 confirmed-bird frames and 150 junk frames on held-out
  days, from at least 15 distinct days, and at least 1,500 confirmed-bird boxes
  left on training days.
- If short, raise the pre-June share and recount before freezing. Refreezing
  after step 3 has started is not allowed.

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

- For each saved frame (visit, sampled frame index), collect every track in
  that visit with a box at that index, from `track_bboxes` and `track_frames`.
- A box whose detection is user-confirmed as a bird is a positive; user-
  rejected junk is background. Exclude a frame if any box in it is unlabelled
  or Poor quality, since it can be neither positive nor background. Count what
  that exclusion costs.
- Cut tiles as production does; keep every tile holding a labelled box, plus
  an equal number of tiles with no box from the same frames.
- Refuse step 2's held-out days. Write a manifest of frame ids, days, seed and
  base-weight hashes beside the dataset.

**Gate 4.**
- Pass: at least 1,500 bird boxes from at least 25 capture days.
- Pass: Ryan audits a contact sheet of 100 random bird boxes and 100 random
  junk boxes, and finds at most 3 wrong in each.
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
  valid for a paired comparison. TODO: Ryan to confirm pausing the binary
  filter retrain until gate 7, since its training crops and the junk it must
  catch both change with the detector.
- Update `DEVELOPING.md`, `LESSONS.md` and the computer-vision skill's
  `birdwatcher-state.md`.
- Remove the PyTorch detection path 30 days after gate 7.

## Status

| Step | Status | Gate result | Date, commit |
|---|---|---|---|
| 1 | in progress: gate 1.2 running on FP32 | 1.1: FP32 PASS (lost 0 of 261 PyTorch-found birds, 95% CI 0-1.5%; junk found 44% vs 44%; confidence change 0.000; identical box counts). INT8 FAIL (lost 5 of 261, 1.9%, CI 0.8-4.4%; 14.5% of boxes cross 0.65; gained 19 birds). Nano FAIL (lost 108 of 261, 41.4%). Continue with FP32. | 2026-09-15 |
| 2 | not started | | |
| 3 | not started | | |
| 4 | not started | | |
| 5 | not started | | |
| 6 | not started | | |
| 7 | not started | | |
| 8 | not started | | |
