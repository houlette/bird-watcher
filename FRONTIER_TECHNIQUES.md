# Frontier Techniques: Computational Photography & Backyard Ornithology

This document captures advanced image processing, micro-video, and behavioral ornithology techniques designed for the **BirdWatcher** system (4K Reolink camera vision + Haikubox acoustic monitoring), established September 2026.

It builds directly on the seven shipped techniques in `COMPUTATIONAL_PHOTOGRAPHY_PLAN.md` and the behavioral foundations established in Phase 4 of the Backyard Distribution Roadmap.

Like all BirdWatcher pipelines, every technique here is defined with explicit ornithological rationale, physical constraints, latency budgets, and feasibility bounds for our production environment (CAX21: 4-core ARM, 4 GB RAM).

---

## 1. Physical, Optical & Biological Context

1. **The Micro-Subject Scale:**
   - Small songbirds (Chickadees, Finches, Kinglets) occupy only $80 \times 80$ to $180 \times 180$ pixels in the 4K sensor frame.
   - At this scale, key avian diagnostic features—the corneal eye glint, iris ring, bill serrations, and individual feather barbules—are compressed to sub-10-pixel structures vulnerable to H.264/H.265 compression artifacts and 4:2:0 chroma subsampling.

2. **The Temporal Disconnect:**
   - Video clips are heavy and cumbersome to scrub in a rapid feed, yet static crops strip away the captivating micro-behavior of wild birds: saccadic head turns, nictitating membrane eye blinks, throat pulses during calls, and plumage breathing.

3. **Human vs. Avian Sensory Asymmetry:**
   - Humans are trichromatic ($RGB$). Birds are **tetrachromatic** (possessing a 4th ultraviolet/violet cone: UVS/VS).
   - Plumage that appears uniform to human vision (e.g. Blue Jay crests, House Sparrow throat bibs, Starling spangles) contains high-contrast ultraviolet signals used in avian sexual selection, flock hierarchy, and mate recognition.

4. **Optimal Foraging & Bioenergetics:**
   - Feeder visits are not arbitrary durations; they represent an active trade-off governed by Optimal Foraging Theory (Charnov's marginal value theorem): caloric intake rate (head-down pecking and seed hulling) versus predation risk (head-up 360° vigilance scans).

---

## 2. Catalog of Techniques

---

### Category A: Computational Photography & Micro-Video

#### Technique 1: Corneal Catchlight Restorer (Specular Eye Glint Reconstruction)
* **Concept:** In professional avian portraiture, the defining "spark of life" is the **catchlight**—the pinpoint specular reflection of the sky or sun on the curved, wet surface of the cornea. On wide-angle 4K security streams, this reflection is smeared into a dull, muddy grey cluster of pixels, making the bird appear taxidermied or lifeless.
* **Mechanism:**
  1. *Eye Region Localization:* Use the head sub-box from the YOLO crop (or dark circular radial symmetry / Hough transform) to isolate the candidate eye region ($\sim 16 \times 16$ px).
  2. *Pupil / Iris Core Segmentation:* Segment the deep black pupil disc in the luminance channel.
  3. *Sub-Pixel Glint Extraction:* Find the peak local gradient within the upper-anterior quadrant of the pupil.
  4. *Point Spread Function (PSF) Re-concentration:* Replace the blurred, low-contrast grey halo with an optically crisp sub-pixel Gaussian highlight disc ($\sigma \approx 0.6\text{–}0.9$ px, peak $L \ge 245$), while subtly deepening the surrounding pupil bed ($L \le 12$).
* **Latency Budget:** $< 1.0\text{ ms}$ per crop on 1 CPU thread (pure OpenCV / NumPy).
* **Failure Safety:** If no distinct specular gradient exists (e.g. bird facing away or closed eye), the filter bypasses the crop at 0 ms with zero modification.
* **Status:** Designed / Backlogged.

---

#### Technique 2: Avian Micro-Movement Cinemagraphs ("Living Portraits")
* **Concept:** Convert static feed thumbnails into lightweight, infinitely looping animated portraits where the bird subtly breathes, blinks, and shifts its head against a tack-sharp, frozen background.
* **Mechanism:**
  1. *Anchor Micro-Pause:* Identify the 1.2–1.8 second stationary window from the cached source MP4 clip (centered on the frame chosen by the Lucky Imaging engine).
  2. *Sub-Pixel Perch Stabilization:* Compute phase-correlation affine translation vectors across the 25 fps burst frames to lock the feeder perch and bird torso to sub-pixel coordinates.
  3. *Motion Mask Isolation:* Difference the stabilized frames against the anchor frame to isolate active micro-motion regions (eye blinks, throat flutters, chest breathing, wind through crown feathers).
  4. *Seamless Loop Synthesis:* Apply a temporal ping-pong or Poisson seam blend across the micro-motion mask, keeping the background and perch 100% frozen.
  5. *Lightweight Export:* Encode to an ultra-compact animated WebP or AVIF ($\sim 60\text{–}120\text{ KB}$), playable on card hover or auto-looping in the feed.
* **Latency Budget:** $\sim 25\text{–}40\text{ ms}$ per visit on worker thread during clip ingest.
* **Status:** Designed / Backlogged.

---

#### Technique 3: Tetrachromatic "AvianVision" (Ultraviolet Plumage Simulation)
* **Concept:** Provide an interactive toggle in the feed and ImageZoom studio (*Human Vision vs. Avian Vision*) that simulates how birds perceive each other, revealing hidden ultraviolet plumage patterns.
* **Mechanism:**
  1. *Ornithological Spectral Prior:* Utilize empirical avian spectrophotometry curves (e.g. Stoddard & Prum, Cornell Lab of Ornithology) for the detected species.
  2. *UV Channel Synthesis:* Estimate the 370 nm UV reflectance layer from the blue-channel gradient, feather structural angle, and species-specific plumage tracts (e.g. crown/crest for Corvids/Parids, throat bib for House Sparrows).
  3. *False-Color Tetrachromatic Mapping:* Re-map the synthesized 4-channel signal $(R, G, B, UV)$ into a perceptual human false-color gamut (mapping UV into electric magenta/violet luminescence while shifting background greenery to muted tones).
  4. *Client-Side Shader / Canvas Filter:* Implemented via a fast WebGL shader or 2D canvas color matrix filter for instant zero-backend-load toggling.
* **Latency Budget:** $< 2\text{ ms}$ client-side rendering.
* **Status:** Designed / Backlogged.

---

#### Technique 4: Trajectory-Guided Directional Motion Deconvolution
* **Concept:** Rapid head saccades and wing flutters cause directional motion blur at standard 1/60s to 1/120s camera shutter speeds.
* **Mechanism:**
  1. Extract instantaneous velocity vectors $(v_x, v_y)$ from frame-to-frame YOLO bounding box tracks (`track_bboxes` and `track_frames`).
  2. Synthesize a directional line Point Spread Function (PSF) matching the travel angle and velocity.
  3. Apply edge-preserved Richardson-Lucy deconvolution along the motion vector to restore crisp leading edges on bills and primary wing feathers.
* **Latency Budget:** $< 5\text{ ms}$ per crop.
* **Status:** Concept.

---

#### Technique 5: Physically-Based Solar Spectrum Correction
* **Concept:** Surveillance camera Auto White Balance (AWB) frequently casts a cold, greenish-blue tint over dawn visits or blows out bright plumage in direct midday sun.
* **Mechanism:**
  1. Calculate exact solar elevation and azimuth from timestamp and feeder GPS coordinates.
  2. Estimate atmospheric Rayleigh scattering and solar color temperature ($T_{sun}$ in Kelvin).
  3. Apply an analytical chromatic adaptation transform (von Kries / Bradford matrix) to restore natural golden-hour warmth and plumage fidelity.
* **Latency Budget:** $< 0.5\text{ ms}$ per crop.
* **Status:** Concept.

---

### Category B: Ornithology & Behavioral Bioenergetics

#### Technique 6: Foraging Kinematics & The Vigilance Index
* **Concept:** Deconstruct feeder visits into bioenergetic behavior: distinguishing high-alert predator surveillance from relaxed feeding sessions.
* **Mechanism:**
  1. *Kinematic Extraction:* Across all 3 fps track frames (`track_bboxes`), calculate the bounding box aspect ratio ($h/w$) and vertical centroid oscillation.
  2. *Head-Down (Foraging):* Squashed aspect ratio, lower centroid, steady position $\to$ active seed selection, hulling, pecking.
  3. *Head-Up (Vigilance):* Elongated vertical aspect ratio, high centroid, rapid horizontal saccadic yaw $\to$ 360° predator scanning.
  4. *Metrics Computed:*
     - **Vigilance Index (%):** Ratio of visit time spent in head-up alert posture.
     - **Peck Cadence:** Estimated pecks / second based on cyclic centroid dips.
  5. *Feed Surfacing:* Behavioral badges on feed cards:
     - 🛡️ *High Vigilance (82% alert)* — "Scanning cautiously; raptor alert"
     - 🌾 *Deep Foraging (14% alert)* — "Relaxed feeding session, ~16 pecks"
     - ⚡ *Snatch & Stash (0.9s)* — "Rapid seed caching run"
* **Latency Budget:** $< 0.2\text{ ms}$ per visit (pure vector math on existing database columns).
* **Status:** Designed / High Priority.

---

#### Technique 7: Plumage Biometric Re-ID (Individual Bird Fingerprinting)
* **Concept:** Differentiate between multiple individuals of the same common species (e.g. tracking whether your yard is visited by 3 resident Chickadees 15 times a day, or 40 transient flock members).
* **Mechanism:**
  1. *Diagnostic Tract Extraction:* Focus on species-specific high-entropy phenotypic markers:
     - **House Sparrows / Chickadees:** Contour perimeter irregularity, area, and asymmetry of the black melanin bib.
     - **Woodpeckers:** White spot dot-dash geometry on the primary flight feathers.
     - **Cardinals:** Facial mask margin curvature and beak color blemishes.
     - **Song Sparrows:** Breast spot centroid and streaking density.
  2. *Orientation Alignment:* Standardize crop rotation using eye-to-bill or torso angle.
  3. *Structural Embedding:* Extract a compact feature vector or contour distance profile.
  4. *Temporal Clustering:* Group visits across days into persistent **Individual Profiles** (*"Chickadee #1 - Jagged Bib"*, *"Nuthatch #2 - Split Brow"*).
* **Latency Budget:** $\sim 5\text{–}10\text{ ms}$ per visit on worker thread.
* **Status:** Designed / High Priority.

---

#### Technique 8: Feather Phenology & Molt Cycle Progression
* **Concept:** Track seasonal plumage transitions (pre-basic and pre-alternate molts), feather wear, and carotenoid dietary saturation drift across months.
* **Mechanism:**
  1. Monitor weekly rolling color histograms in CIE Lab on verified male American Goldfinches (winter olive $\to$ breeding canary yellow) and Northern Cardinals (dietary carotenoid depth).
  2. Measure feather fringe edge raggedness (high-frequency Laplacian variance on mantle contours) to detect worn, abraded flight feathers vs. fresh post-molt plumage.
  3. Surface milestone notifications and Insights cards: *"Spring Molt Underway: Goldfinches at 85% breeding plumage"*.
* **Latency Budget:** Periodic background batch query ($< 1\text{s}$ weekly).
* **Status:** Designed / Backlogged.

---

## 3. Prioritized Implementation Roadmap

| Priority | Technique | Category | Implementation Cost | Latency on CAX21 | Visual / Scientific Impact |
|:---:|---|---|:---:|:---:|:---:|
| **1** | **Corneal Catchlight Restorer** | Computational Photography | Low (pure OpenCV) | $< 1.0\text{ ms}$ / crop | ⭐⭐⭐⭐⭐ (transforms crops into vivid portraits) |
| **2** | **Foraging Vigilance Kinematics** | Behavioral Ornithology | Low (track bbox math) | $< 0.2\text{ ms}$ / visit | ⭐⭐⭐⭐ (instant behavioral depth in feed) |
| **3** | **AvianVision (UV Simulation)** | Science / Visualization | Low (frontend shader/LUT) | $< 2.0\text{ ms}$ (client) | ⭐⭐⭐⭐ (interactive scientific discovery) |
| **4** | **Avian Micro-Cinemagraphs** | Micro-Video | Medium (stabilization + WebP) | $\sim 25\text{ ms}$ / visit | ⭐⭐⭐⭐⭐ (magical living feed experience) |
| **5** | **Plumage Biometric Re-ID** | Ornithology / Computer Vision | High (patch alignment & clustering)| $\sim 8\text{ ms}$ / visit | ⭐⭐⭐⭐⭐ (names individual yard visitors) |
| **6** | **Molt & Phenology Tracking** | Ornithological Analytics | Low (aggregate metrics) | Batch query | ⭐⭐⭐ (seasonal storytelling) |

---

## 4. Status Tracking

| Technique | Status | Gate / Verification Criteria | Date | Commit |
|---|---|---|---|---|
| Corneal Catchlight Restorer | Planned | Sub-pixel PSF highlight; bypass on occluded/facing-away eyes | — | — |
| Foraging Vigilance Kinematics | Planned | Validated against manually scored video clips of feeding vs scanning | — | — |
| AvianVision (UV Mode) | Planned | Interactive toggle in ImageZoom and feed cards with false-color LUT | — | — |
| Avian Micro-Cinemagraphs | Planned | WebP $\le 120\text{ KB}$, phase-stabilized perch, zero loop seams | — | — |
| Plumage Biometric Re-ID | Planned | Silhouette / bib contour matching on held-out multi-visit sequences | — | — |
| Molt & Phenology Tracking | Planned | Verified seasonal color shifts on Goldfinches and Cardinals | — | — |
