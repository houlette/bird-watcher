# Creative Projects: Beyond the Bird Feed

This document tracks creative applications, games, and generative art installations designed to run on top of the **BirdWatcher** multi-modal data stream (4K Reolink camera vision + Haikubox acoustic monitoring).

---

## 1. Technical Data Foundation & Architecture Unlocks

Each project below leverages a combination of data sources already present in the BirdWatcher architecture, alongside newly reverse-engineered capabilities:

```
┌───────────────────────────────────────────────────────────┐
│                    DATA STREAM INPUTS                     │
├─────────────────────────────┬─────────────────────────────┤
│   Reolink 4K Feeder Camera  │     Haikubox Acoustic Box   │
├─────────────────────────────┼─────────────────────────────┤
│ • Motion MP4 clips & JPGs   │ • REST API: species, time,  │
│ • YOLO11 track bounding     │   BirdNET score             │
│   boxes [x, y, w, h] (4K)   │ • AppSync GraphQL:          │
│ • Temporal tracking splines │   - specSum (energy metric) │
│ • Calibrated size priors    │   - detectionArray (offsets)│
│ • Multi-frame sharpness &   │   - getAudioUrl -> .flac    │
│   brightness metrics        │     presigned audio stream  │
│ • User corrections & labels │ • yard_priors.json stats    │
└─────────────────────────────┴─────────────────────────────┘
```

### Reverse-Engineered Haikubox API Unlocks
* **GraphQL Endpoint:** `https://<appsync-id>.appsync-api.us-east-1.amazonaws.com/graphql`
* **Authentication:** AWS Cognito User Pool, via a pool id and app client id.
* **Where the values live:** the endpoint id, pool id and client id are
  account and vendor specific, so they stay out of this repo. Recover them
  from the Haikubox web app's own JS bundle when Phase 3 is built, and put
  them in `backend/.env` next to `HAIKUBOX_API_KEY` and `HAIKUBOX_SERIAL`,
  which is where every other credential in this project already lives.
* **Key Queries:**
  * `HitBySerialByDate`: returns `wav` (file path in Backblaze/S3), `score`, and `specSum` (spectral energy/intensity).
  * `GetAudioUrl(audioPath)`: resolves `wav` to a presigned URL streaming the raw **`audio/flac`** recording.
* **Client-side FFT:** Audio files can be decoded via Web Audio API (`decodeAudioData`) to generate raw waveforms and high-resolution frequency spectrograms.

---

## 2. Project Catalog

---

### Project 1: *The Perch & Flagon* (Cozy Feeder Tavern / Idle Sim)

> **Vibe:** *Rusty’s Retirement* meets *Travellers Rest* and *Neko Atsume*. A delightful, low-stress companion app that sits on a secondary monitor or docks at the bottom of the screen while you work.

#### Core Concept
Your bird feeder is an enchanted forest tavern. Real backyard birds visiting the feeder physically enter the tavern as whimsical patron travelers seeking food, warmth, and rest.

#### Live Data Integration
* **Real-time Arrivals:** When a camera detection fires (or Haikubox hears an arrival nearby), a patron enters through the tavern door.
* **Patron Personalities based on Species:**
  * **Chickadees & Titmice:** Hurried messengers and scouts. They dash in, order quick snacks (sunflower seeds, thistle crumbles), tip swiftly, and dart out.
  * **Blue Jays & Woodpeckers:** Boisterous adventurers and mercenaries. They command the large corner booth, order hearty suet platters, and have long visit durations.
  * **Mourning Doves & House Finches:** Regular locals who lounge on the benches and chat peacefully.
* **Audio Atmosphere:**
  * When Haikubox hears a bird singing in the trees, a tavern bard plays a flute or lute melody matching the bird’s pitch.
  * If Haikubox streams the raw `.flac` clip, the actual bird chirp plays softly with tavern reverb.
* **"Not a bird" Mishaps:** Wind spinners, squirrels, or shadows generate funny tavern events (a stray breeze extinguishing candles, a squirrel sneaking into the pantry).

#### Progression & Mechanics
* **Seed Shillings (Currency):** Real visits generate currency based on visit duration and rarity.
* **Tavern Customization:** Spend shillings to unlock cozy tavern upgrades (brass lanterns, miniature suet hearths, oak tables, acoustic perches).
* **Guestbook & Lore:** Complete visitor profile entries with procedurally generated avian dialogue and visitor backstories.

#### Tech Stack
* **Engine:** React + Pixi.js (or HTML5 Canvas / Phaser).
* **Backend:** Polling `/api/detections` or Server-Sent Events (SSE) from the BirdWatcher API.
* **Platform:** Web app with a compact "bottom-dock / mini-player" layout mode.

---

### Project 2: *Feederbound* (Live-Data Deckbuilder & Roguelike RPG)

> **Vibe:** *Slay the Spire* meets *Backpack Hero* and *Wingspan*, where your run’s party, relics, and deck are fueled by real yard activity.

#### Core Concept
An expedition roguelike where you lead a guild of avian heroes into woodland dungeons. Your card pool, relic drops, and temporary party buffs fluctuate dynamically based on what birds are currently active in your backyard.

#### Avian RPG Classes & Stats
Derived from real-world size priors, flight velocity, and feeding behavior:
* **The Scout / Rogue (Chickadee, Tufted Titmouse):**
  * *High Agility / Evasion.* Low base HP, rapid multi-hit peck attacks, ability to dodge incoming strikes and retreat into the trees.
* **The Berserker / Knight (Blue Jay):**
  * *High Physical Might / Disruption.* Employs high size-prior mass. Screech ability stuns enemy frontlines and sunders armor.
* **The Lancer / Sunderer (Downy / Red-bellied Woodpecker):**
  * *Piercing DPS.* Attacks ignore armor shields by drilling directly into enemy defenses.
* **The Guardian / Paladin (Mourning Dove):**
  * *High Tank Bulk / Passive Healing.* Slow, steadfast presence with high HP pool that absorbs damage and grants tranquility buffs to allies.
* **The Sorcerer / Mystic (Goldfinch, Warblers):**
  * *Elemental Swarm & Multi-cast.* Harnesses seed magic and rapid flocking strikes.

#### Live Yard Modifiers
* **Yard Presence Buffs:** If a species has visited your yard in the last 2 hours, that hero class receives a 20% morale/stat boost.
* **Audio Blessing:** Real-time Haikubox detections trigger instant combat boons (*"Song of the Canopy: +2 Energy this turn"*).
* **Feeder Boss Encounters:** Boss fights themed around feeder threats (The Neighborhood Cat, The Raccoon at Dusk, The Dominant Cooper’s Hawk).

#### Tech Stack
* **Engine:** React + Canvas / Phaser / WebGL.
* **State Management:** Zustand / Redux storing deck state, relic inventory, and dungeon rooms.
* **Backend Hook:** BirdWatcher detections SQLite DB + Haikubox audio triggers.

---

### Project 3: *Territory: Feeder Wars* (Dominance & Territory Control Sim)

> **Vibe:** Asynchronous ecological strategy game where real feeder pecking orders translate into live territory capture.

#### Core Concept
Your feeder camera frame is mapped into physical control zones:
* `Zone A`: The Suet Cage
* `Zone B`: Left Perch
* `Zone C`: Right Perch
* `Zone D`: Ground Tray / Drop Zone

Real-life birds landing, feeding, and displacing each other trigger automated territory battles between avian factions.

#### Faction Pecking Order Engine
Using YOLO frame tracking (`track_bboxes`):
1. **Displacement Detection:** When Bird A occupies Zone A, and Bird B (e.g., a Blue Jay) lands on Zone A within 1.0s causing Bird A to flee, the engine records an explicit **Pecking Order Displacement**.
2. **Zone Hold Time:** Factions accumulate influence points proportional to time spent feeding in each zone.
3. **Faction Rivalries:**
   * *The Blue Jay Syndicate* (Heavyweight aggressors)
   * *The Finch Coalition* (High-frequency swarms)
   * *The Picidae Woodpecker Guild* (Suet specialists)
   * *The Paridae Scouts* (Titmice & Chickadees darting in between shifts)

#### Player Role: Feeder Deity
* Players don't control the birds directly; instead, you act as the Sanctuary Steward.
* Place virtual feeder enhancements, set faction treaties, or bet virtual seed currency on daily territory control percentages.
* Daily newspaper / scoreboard summarizing turf battles: *"Jays seize Suet Block; Finches mount late afternoon counter-offensive on the Tray."*

#### Tech Stack
* **Backend:** Python background worker analyzing spatial coordinates from `track_bboxes` and visit timestamps.
* **Frontend:** SVG / Canvas interactive feeder diagram with real-time zone control heatmaps and territorial banners.

---

### Project 4: *Avian Flightlines & Kinetic Sand* (Generative Particle & Vector Art)

> **Vibe:** Ambient, fine-art visualization suitable for wall displays, ambient screensavers, or generative art prints.

#### Core Concept
Every bird visit is a trajectory through 4K coordinate space across time. By extracting velocity vectors, landing points, hops, and departure angles from `track_bboxes`, the system renders elegant, continuous abstract art.

#### Visual Formats

##### A. Calligraphic Flowfields (Sumi-e Ink Style)
* Bird trajectories act as dynamic force vectors pushing particles through a simulated 2D fluid grid.
* Fast swooping entries generate bold, sweeping ink brush strokes.
* Hesitant perching and feeding hops produce delicate stippling and ink blooms.
* Plumage color palettes (Jay cobalt, Goldfinch yellow, Cardinal vermilion, Titmouse slate) tint the flowlines.

##### B. Daily Feeder Celestial Tapestry / Mandala
* At the end of each day (or continuously looping), generate a circular astronomical chart or mandala:
  * Concentric rings represent time of day (dawn to dusk).
  * Orbits represent flight paths across the feeder.
  * Line thickness corresponds to bird mass/size prior.
  * Node clusters mark feeding coordinates.
  * Radiating ripples mark Haikubox audio confirmation events.

##### C. Kinetic Sand / Topographic Contours
* 3D generative contour map: every second a bird spends on a feeder coordinate deposits a virtual grain of sand.
* Renders a hypnotic wooden relief or sand topography showing the invisible physical geography created by wild visitors.

#### Tech Stack
* **Renderer:** HTML5 Canvas, p5.js, or Three.js / WebGL shaders.
* **Export:** High-res PNG / SVG export for physical printing or wallpaper rotation.

---

### Project 5: *Chrono-Chirps: Spectrogram Biome* (Living Audio Garden)

> **Vibe:** Zen generative terrarium / botanical ecosystem that slowly blooms, branches, and changes according to the acoustic landscape of your yard.

#### Core Concept
An ambient digital terrarium running in the background. Unlike traditional visualizers that react to simple microphone volume, this biome uses **real acoustic data from Haikubox**:
* Spectral energy (`specSum`)
* Call durations and frequency distribution
* Presigned `.flac` audio stream processed via Web Audio API FFT

#### Algorithmic Botany (Acoustic L-Systems)
* **Plant Seeding:** Each detected species spawns a distinct species of procedural plant (fractal L-system, botanical spline tree, or flowering vine).
* **Growth Parameters Driven by Sound:**
  * **Peak Frequency (kHz):** Sets petal hue and bloom geometry (high-pitch Chickadee whistles yield delicate blue/violet florets; lower Crow caws grow deep indigo moss and sturdy root stems).
  * **Spectral Energy (`specSum`):** Controls plant vitality and growth spurts. High-energy calls cause branches to sprout new leaves.
  * **Call Duration & Offsets (`detectionArray`):** Determines branch length and node bifurcations.
* **Visual Camera Cross-Pollination:**
  * When a bird is simultaneously confirmed on the camera (`audio_confirmed = True`), a glowing golden pollinator flutters to that plant, unlocking a full flowering state.

#### Ambient Audio Experience
* The terrarium can optionally play periodic, spatialized ambient playback of the real `.flac` audio clips captured during the day, accompanied by gentle ripple animations on the corresponding plants.

#### Tech Stack
* **Audio Engine:** Web Audio API (`decodeAudioData`, `AnalyserNode` FFT) streaming from Haikubox presigned URLs.
* **Graphics Engine:** Three.js (for a 3D isometric terrarium) or 2D HTML5 Canvas procedural L-systems.
* **Backend:** Haikubox AppSync GraphQL connector caching recent `.flac` links and `specSum` metrics.

---

## 3. Project Comparison & Recommended Roadmap

| # | Project | Primary Data Driver | Tech Stack | Estimated Effort | Target Experience |
| :- | :--- | :--- | :--- | :-: | :--- |
| **1** | **The Perch & Flagon** | Camera Detections + Audio Badges | React / Pixi.js | Moderate | Cozy desktop idle companion |
| **2** | **Feederbound** | Camera Species + Audio Buffs | React / Phaser | High | Engaging roguelike deckbuilder |
| **3** | **Territory: Feeder Wars** | 4K YOLO `track_bboxes` | Python + React/SVG | Moderate | Ecological turf-war strategy |
| **4** | **Avian Flightlines** | 4K Track Splines & Velocities | Canvas / p5.js / WebGL | Low–Moderate | Mesmerizing ambient art & prints |
| **5** | **Chrono-Chirps: Biome** | Haikubox GraphQL + FLAC Audio | Web Audio API + Three.js | Moderate | Zen living terrarium installation |

### Suggested Phasing

1. **Phase 1 (Quick Win & Visual Impact): Project 4 (*Avian Flightlines*)** — built, on the Art tab.
   * *Why first:* Complete tracking bboxes (`track_bboxes`) are already stored in the SQLite database for hundreds of visits. A standalone Canvas/WebGL view can be added to the frontend immediately without external auth or complex game engines.
   * *As built:* `/api/art` plus `ArtCanvas.tsx`, in three modes (flightlines, celestial mandala, topography). A flight is one detection, not one visit. Most rows predate per-frame tracking, so the page says which paths are observed and which are synthesised around a real perch.
2. **Phase 2 (Cozy Companion): Project 1 (*The Perch & Flagon*)** — built, on the Tavern tab.
   * *Why second:* Reuses existing frontend React components and `/api/detections` feed to create an immediate, daily-playable idle tavern.
   * *As built:* `/api/tavern` plus `TavernCanvas.tsx`. Each detection is a guest whose seat comes from a species archetype, whose colour and size come from the plumage palette Phase 1 introduced, and who pays Seed Shillings toward thirteen upgrades to the house. Detections the classifier would not name are cloaked strangers who pay one shilling; "Not a bird" corrections come back as tavern mishaps.
   * *Dwell:* a guest stays for their archetype's dwell, nudged by the real visit duration, at ninety wall-clock seconds a beat. The newest three never time out and neither does the quiet company, so the room is never empty on a yard that logs a few dozen birds a day.
   * *Open question:* the founding purse is capped at 1,200 shillings, so a long archive is a good start rather than an instant win. Whether that is the right figure is a guess, and a week of use is what would settle it.
3. **Phase 3 (Acoustic Living Art): Project 5 (*Chrono-Chirps*)**
   * *Why third:* Implement the reverse-engineered Haikubox GraphQL client to fetch raw audio and render the procedural botanical garden.
4. **Phase 4 (Deep Gameplay): Project 3 (*Feeder Wars*) & Project 2 (*Feederbound*)**
   * Build out deeper strategic territorial simulation and deckbuilding systems once the spatial and audio modules are proven.
