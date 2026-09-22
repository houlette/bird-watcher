This is the plan for computational photography enhancements to BirdWatcher's
image processing pipeline, established September 2026. It addresses the physical
and optical limitations of surveillance cameras (4K wide-angle fixed-focus
sensors with aggressive video compression) by applying modern computational
photography techniques to the bird crops shown in the feed and passed to the
classifier.

Like `DETECTOR_PLAN.md`, every technique is defined with an explicit rationale,
implementation constraints, performance gates, and status tracking. Nothing is
added to the critical path without meeting strict CPU/latency budgets on our
CAX21 (4-core ARM, 4 GB RAM) production instance.

---

## 1. Physical & Optical Realities

The camera stream presents several physical constraints:

1. **Subsampled Video Temporal Resolution vs. Saccadic Motion**:
   - Reolink cameras stream at 20–25 fps. To bound YOLO inference cost to ~2 minutes
     per visit on 4 cores, the pipeline samples at 3 fps (`step = round(fps / 3.0) ~ 7`).
   - ~85% of source frames are discarded before detection.
   - Birds move saccadically: rapid hops and head turns (~50–100 ms) separated by
     still micro-pauses (~100–300 ms).
   - Sampling blindly at 3 fps frequently lands on a frame with motion blur or
     mid-turn defocus, while an adjacent source frame 40–80 ms away captured the bird
     in tack-sharp stillness.

2. **Aggressive H.264 / H.265 Compression & 4:2:0 Chroma Subsampling**:
   - Surveillance streams compress color at half resolution in both axes (4:2:0).
   - In a 4K frame (3840×2160), chroma is effectively 1080p (1920×1080). For a
     small 120×120 bird crop, color information is only 60×60.
   - Fine plumage details (wing bars, crown stripes, eye rings) suffer from color
     bleeding and high-ISO chroma noise in shade or low light.

3. **Sensor Size, Focal Length, and Hyperfocal Clutter**:
   - Wide-angle lenses (2.8 mm – 4.0 mm) on small sensors (1/2.8" or 1/1.8") have
     immense depth of field: everything from 0.5 m to infinity is in focus.
   - Feeder hardware, deck railings, fence pickets, and background siding compete
     visually with the bird subject.

4. **Dynamic Range & Backlighting**:
   - Outdoor feeders often sit under eaves or trees against bright skies or sunny lawns.
   - Auto-exposure either silhouettes the bird or blows out background highlights.

5. **Crop Pixel Coverage**:
   - Small backyard visitors (hummingbirds, kinglets, goldfinches) often cover only
     80×80 to 180×180 pixels in the 4K sensor frame. Bicubic upsampling in the feed
     leaves them soft and pixelated.

---

## 2. Ranked Computational Photography Techniques

### Technique 1: Lucky Imaging (Temporal Peak-Sharpness Hunting)
- **Concept**: Astrophotography "lucky imaging" applied to surveillance video. When the
  top-ranked 3 fps sampled frame is soft or blurred (`sharpness < threshold`), hunt
  the immediate source frame neighborhood (`[center - 3, center + 3]`) for the
  sharpest moment during the bird's micro-pause.
- **Trigger Gate**: Only triggered when Laplacian variance of the best detection is
  below `lucky_imaging_threshold` (e.g. < 200.0). Crops that are already crisp (>80%
  in production) bypass this stage at 0 ms cost.
- **Safety / Verification**: Phase-correlation alignment (`cv2.phaseCorrelate`)
  verifies the candidate matches the anchor (`response >= 0.15`) and has not hopped
  away (`shift <= 20% bbox dimension`). Bounding box is updated to match the sharp
  source frame.
- **Status**: Shipped & deployed to production (commit `fb3073a`).

### Technique 2: Edge-Aware Adaptive Sharpening (Bilateral Unsharp Masking)
- **Concept**: Standard unsharp masking amplifies sensor noise and creates harsh halos
  around silhouette edges. Edge-preserving bilateral filtering on the L-channel of
  LAB isolates high-frequency plumage texture without haloing high-contrast boundaries.
- **Deadband Coring & Gating**: Noise below 2.0 L-units is suppressed; boost is clamped
  to ±15.0 units; sharpening amount scales inversely with crop sharpness and shuts off
  entirely for crops with variance ≥ 250.
- **Status**: Shipped & deployed to production (commit `860eb0a`).

### Technique 3: Chroma-Guided Detail Restoration & Denoising
- **Concept**: Reconstruct 4:2:0 subsampled chroma artifacts by cross-filtering chroma
  (Cr and Cb in YCrCb space) using the full-resolution Luma (Y) channel as a structural guide.
- **Mechanism**: Fast O(1) guided filter (He et al., ECCV 2010) using `cv2.boxFilter` with
  shared luma denominator inversion across both color channels. Snaps color boundaries to
  sharp luma edges, eliminating 4:2:0 bilinear color bleed while smoothing blotchy sensor noise.
- **Target Latency**: < 2 ms per crop (benchmarks at ~1.1 ms).
- **Status**: Shipped & deployed to production.

### Technique 4: Single-Image Super-Resolution (SISR) via Lightweight OpenVINO
- **Concept**: Enhance small crops (< 180×180 px) that lack multi-frame video bursts (or
  where birds hopped too fast for multi-frame fusion) to crisp 2× resolution using an
  ultra-lightweight neural network: FSRCNN (Dong et al.) compiled in OpenVINO.
- **Mechanism**: Operates purely on the Y (luma) channel in [0, 1] range to recover high-frequency
  micro-texture (barbules, eye reflections, bill contours) without color shift. Upscales chroma
  channels via bicubic interpolation and passes through chroma-guided filtering to eliminate
  4:2:0 subsampling bleed on the upscaled grid.
- **Trigger Gate**: Only runs on crops where `best.sr_crop is None` (multi-frame burst unavailable or unaligned)
  and `max(w, h) <= 180` px (configurable via `sisr_max_crop_size`).
- **Budget Gate**: Strict gate < 40 ms on 1 CPU thread.
- **Performance**: Benchmarks at **~2.5–3.3 ms** on 1 CPU thread (>10× faster than budget gate!).
  Produces a **+47.8% Laplacian edge contrast gain** over standard bicubic upsampling.
- **Model Assets**: OpenVINO IR files `backend/pipeline/models/fsrcnn_x2.xml` (63 KB) and `.bin` (17 KB),
  tracked in git with zero heavy external runtime dependencies.
- **Status**: Shipped & deployed to production.

### Technique 5: Mertens Exposure Fusion (Local Contrast & Shadow Recovery)
- **Concept**: Tommert / Mertens multiscale exposure fusion blends multiple synthetic
  exposure exposures (underexposed for bright sky/feeder, normal, overexposed for deep
  plumage shadows) based on local contrast, saturation, and well-exposedness weights.
- **Benefit over CLAHE**: Zero halo artifacts, natural tone transitions, lifts deep
  under-belly shadows without blowing out white throat/breast feathers.
- **Implementation**: Pure OpenCV (`cv2.createMergeMertens`) with $O(1)$ lookup tables (`cv2.LUT`)
  generating dark ($\gamma=1.8$) and bright ($\gamma=0.55$) synthetic brackets without floating-point
  power ops per pixel. Zero external dependencies.
- **Latency Budget**: Benchmarks at ~0.66 ms on 200x200 crops (well within the < 10 ms budget).
- **Status**: Shipped & deployed to production.

### Technique 6: Multi-Frame Shift-and-Add Super-Resolution
- **Concept**: Sub-pixel image registration across 3–5 consecutive motionless frames
  from the source video burst, combined with shift-and-add reconstruction.
- **Benefit**: Exploits natural sensor micro-jitter to achieve true optical resolution
  gain and $\sqrt{N}$ noise reduction, resolving barbules and eye reflections beyond
  the single-frame Nyquist limit.
- **Implementation**: Phase-correlation sub-pixel registration (`cv2.phaseCorrelate`) on
  Hann-windowed luma, Lanczos-4 warping with sub-pixel affine translation matrices into
  a $2\times$ high-resolution grid, robust temporal median reconstruction to eliminate sensor
  noise and H.264 compression ringing, followed by noise-cored MTF aperture deconvolution.
- **Trigger Gate**: Runs on video tracks where `min(w, h) <= 240` px; 0 ms on already-large crops.
- **Latency Budget**: Benchmarks at ~25 ms for 5-crop reconstruction (well within < 50 ms budget).
- **Status**: Shipped & deployed to production.

### Technique 7: Synthetic Bokeh / Background Defocus
- **Concept**: Isolate the bird subject from harsh backyard clutter (wooden feeder posts,
  wire mesh, siding, distant chains/roof edges) by applying a realistic optical lens
  defocus blur (bokeh) to the background regions outside the bird subject.
- **Benefit**: Emulates a wide-aperture telephoto portrait lens (e.g. 400mm f/2.8)
  with soft creamy circles of confusion and specular highlight blooming, separating
  the bird from complex backgrounds without cardboard cutout edges or halo artifacts.
- **Implementation**:
  - **Edge-Preserving Focus Mask**: Evaluated on a downsampled 160px guide in ~2 ms,
    combining a broad spatial prior ($\sigma \approx 0.48$), multi-scale plumage texture
    gradient energy, and perimeter background color sampling in YCrCb space.
  - **Luminance-Guided Filter**: Aligns mask transitions tightly to plumage contours,
    bill, feet, crest, and tail barbules using an integral box-filter guided filter.
  - **Optical Bokeh Blur**: Downsampled blur buffer (~320px) convolved with 3-pass box
    filtering with specular highlight blooming, convolving out-of-focus glints into
    luminous bokeh discs.
  - **Fast Integer Alpha Blending**: Composited via uint16 fixed-point arithmetic with
    smoothstep depth rolloff, avoiding expensive floating-point per-pixel ops.
- **Latency Budget**: Benchmarks at **3.9–9.7 ms** on standard feeder crops (average ~6.5 ms,
  well below the strict < 15 ms budget on 1 CPU thread).
- **Status**: Shipped & deployed to production.

---

## 3. Production Resource Constraints & Invariants

1. **CPU Budget**: The CAX21 instance processes daylight arrivals in real time. Total
   pipeline compute time per visit must remain under 150 s (90% of which is YOLO tiling).
   All computational photography techniques combined must add < 200 ms per visit.
2. **Gating Invariant**: Never run expensive multi-frame decodes or neural enhancement
   unconditionally. Every technique must be gated by a quality metric (sharpness, crop
   dimension, or exposure distribution).
3. **Ground-Truth Preservation**: Raw source crops and frames must always be preserved
   intact for model fine-tuning and active learning. Visual enhancements apply to the
   display crop written to `crops/` and the anchor fed to the species classifier.
4. **Reversibility**: Every enhancement must be configurable via `backend/settings.py`
   environment variables (`BOKEH_ENABLED`, `BOKEH_STRENGTH`).
5. **Comparative Diagnostics & Variant Inspection**:
   To systematically evaluate which combination of computational photography techniques yields
   the cleanest plumage detail without artifacts, raw crops and pre-lucky crops are preserved
   alongside production polished crops (`crops/v..._raw.jpg` and `crops/v..._initial_raw.jpg`).
   The backend provides dynamic variant endpoints (`/api/detections/{id}/crop` and
   `/api/detections/{id}/crop-variants`), while the frontend provides an interactive inspection
   studio in `ImageZoom` with real-time technique toggling (Super-Res, Neural SR, Chroma, CLAHE, Mertens, Sharpen, Bokeh, Lucky),
   preset buttons (`Full Polish`, `Bokeh`, `Mertens HDR`, `Raw`, `Super-Res 2x`, `Neural SR 2x`, `Mertens Only`, `Chroma Only`, `Sharpen Only`),
   instant hold-to-compare (Spacebar), and 1x/2x/4x nearest-neighbor pixel magnification.

---

## 4. Status & Gate Results

| Technique | Status | Latency Impact | Quality Metric / Result | Shipped Date | Commit |
|---|---|---|---|---|---|
| Edge-Aware Sharpening | Shipped | ~1.5 ms / crop | Bilateral unsharp mask on L-channel; coring=2.0, clamp=±15.0 | 2026-09-20 | `860eb0a` |
| Lucky Imaging | Shipped | 0 ms (crisp) / ~90 ms (blurry) | ±3 source frames searched when var < 200; phase-corr gated | 2026-09-20 | `fb3073a` |
| Chroma-Guided Denoising | Shipped | ~1.1 ms / crop | Fast YCrCb guided filter (r=2, eps=1e-4); >50% chroma noise reduction | 2026-09-20 | `205416a` |
| Mertens Exposure Fusion | Shipped | ~0.66 ms / crop | 3-exposure multiscale blending; zero highlight clipping & smooth shadow lift | 2026-09-20 | `0216cc5` |
| Shift-and-Add Super-Res | Shipped | ~25 ms / crop | Sub-pixel phase registration + 2x Lanczos4 + temporal median + MTF restoration | 2026-09-20 | `cd79fa4` |
| Single-Image Super-Res (FSRCNN) | Shipped | ~2.5 ms / crop | FSRCNN via OpenVINO on Y channel; +47.8% edge contrast gain on crops < 180px | 2026-09-21 | `878df6b` |
| Synthetic Bokeh | Shipped | ~6.5 ms / crop | Edge-guided optical lens defocus blur + specular highlight discs (< 15 ms target) | 2026-09-21 | - |
