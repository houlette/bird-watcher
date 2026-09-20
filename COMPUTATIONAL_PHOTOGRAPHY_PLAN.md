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

### Technique 4: Single-Image Super-Resolution (SISR) via Lightweight ONNX
- **Concept**: Enhance small crops (< 180×180 px) to crisp 2×/4× resolution for display
  using an ultra-lightweight neural network (e.g. Real-ESRGAN-compact, FastSR, or
  OmniSR) quantized to OpenVINO INT8/FP16.
- **Trigger Gate**: Only runs on crops where `min(w, h) < 180` px.
- **Budget Gate**: Inference must take < 40 ms on 1 CPU thread.
- **Status**: Planned (Step 4).

### Technique 5: Mertens Exposure Fusion (Local Contrast & Shadow Recovery)
- **Concept**: Tommert / Mertens multiscale exposure fusion blends multiple synthetic
  exposure exposures (underexposed for bright sky/feeder, normal, overexposed for deep
  plumage shadows) based on local contrast, saturation, and well-exposedness weights.
- **Benefit over CLAHE**: Zero halo artifacts, natural tone transitions, lifts deep
  under-belly shadows without blowing out white throat/breast feathers.
- **Status**: Planned (Step 5).

### Technique 6: Multi-Frame Shift-and-Add Super-Resolution
- **Concept**: Sub-pixel image registration across 3–5 consecutive motionless frames
  from the source video burst, combined with shift-and-add reconstruction.
- **Benefit**: Exploits natural sensor micro-jitter to achieve true optical resolution
  gain and $\sqrt{N}$ noise reduction, resolving barbules and eye reflections beyond
  the single-frame Nyquist limit.
- **Status**: Planned (Step 6).

### Technique 7: Synthetic Bokeh / Background Defocus
- **Concept**: Isolate the bird subject from harsh backyard clutter (railings, siding,
  chains) by applying a realistic synthetic depth-of-field blur to the background.
- **Mechanism**: Uses bird segmentation mask (or depth estimation) with edge-feathering
  and a circular disc/lens blur kernel on the background region.
- **Status**: Planned (Step 7).

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
   environment variables.
5. **Comparative Diagnostics & Variant Inspection**:
   To systematically evaluate which combination of computational photography techniques yields
   the cleanest plumage detail without artifacts, raw crops and pre-lucky crops are preserved
   alongside production polished crops (`crops/v..._raw.jpg` and `crops/v..._initial_raw.jpg`).
   The backend provides dynamic variant endpoints (`/api/detections/{id}/crop` and
   `/api/detections/{id}/crop-variants`), while the frontend provides an interactive inspection
   studio in `ImageZoom` with real-time technique toggling (Chroma, CLAHE, Sharpen, Lucky),
   preset buttons (`Full Polish`, `Raw`, `Chroma Only`, `CLAHE Only`, `Sharpen Only`),
   instant hold-to-compare (Spacebar), and 1x/2x/4x nearest-neighbor pixel magnification.

---

## 4. Status & Gate Results

| Technique | Status | Latency Impact | Quality Metric / Result | Shipped Date | Commit |
|---|---|---|---|---|---|
| Edge-Aware Sharpening | Shipped | ~1.5 ms / crop | Bilateral unsharp mask on L-channel; coring=2.0, clamp=±15.0 | 2026-09-20 | `860eb0a` |
| Lucky Imaging | Shipped | 0 ms (crisp) / ~90 ms (blurry) | ±3 source frames searched when var < 200; phase-corr gated | 2026-09-20 | `fb3073a` |
| Chroma-Guided Denoising | Shipped | ~1.1 ms / crop | Fast YCrCb guided filter (r=2, eps=1e-4); >50% chroma noise reduction | 2026-09-20 | `205416a` |
| Lightweight SISR (ONNX) | Planned | < 40 ms target | 2× upscaling on crops < 180 px | - | - |
| Mertens Exposure Fusion | Planned | < 10 ms target | 3-exposure multiscale blending for backlit crops | - | - |
| Shift-and-Add Super-Res | Planned | < 50 ms target | Sub-pixel alignment over 3-5 burst frames | - | - |
| Synthetic Bokeh | Planned | < 15 ms target | Subject isolation blur outside bird mask | - | - |
