"""Named configs we'll replay and the sweep grids that vary one knob at a time.

A `Config` is a flat dict where keys are knob names known to
`replay.apply_config()`. Anything left out of a config keeps the value
currently in the source tree.

The two reference configs (PRE_REGRESSION and CURRENT_PROD) approximate
the parameter settings live ~3 days ago vs. now. They're how we answer
the regression-triage question: replay both on the same dataset and
diff the metrics.
"""
from __future__ import annotations


# Pre-regression config: parameter settings that were live ~3 days ago,
# before NMM, scene mask, daylight gate, the 30% padding bump, and the
# 10s clip duration cap. Our "control" arm for the A/B.
PRE_REGRESSION: dict = {
    "yolo_model_name": "yolo11s.pt",
    "target_fps": 3.0,
    "MAX_PROCESS_DURATION_SECONDS": 1e9,        # effectively unlimited
    "BIRD_CONFIDENCE_THRESHOLD": 0.20,
    "TILE_PX": 1024,
    "TILE_OVERLAP_PX": int(1024 * 0.20),
    "NMS_IOU": 0.50,
    "crop_padding": 0.15,
    "IN_RANGE_THRESHOLD": 0.10,
    "scene_mask_enabled": False,
    "nmm_enabled": False,                       # falls back to plain NMS
}

# Current production config snapshot (what's live now). Pinned explicitly
# so a future code change can't accidentally rewrite history of this A/B.
# Note: CURRENT_PROD's yolo_model_name was updated from yolo11s.pt to
# yolo11n.pt after the model-swap sweep showed YOLO11n produces ~9× fewer
# detections at known-FP locations on tail-FP scenes. Subsequent OFAT
# sweeps for non-model knobs measure their effect with YOLO11n as the
# baseline, since that's what's actually in production now.
CURRENT_PROD: dict = {
    "yolo_model_name": "yolo11n.pt",
    "target_fps": 3.0,
    "MAX_PROCESS_DURATION_SECONDS": 10.0,
    "BIRD_CONFIDENCE_THRESHOLD": 0.35,
    "TILE_PX": 1024,
    "TILE_OVERLAP_PX": int(1024 * 0.20),
    "NMS_IOU": 0.50,
    "TILE_SEAM_GAP_PX": 20,
    "TILE_SEAM_OVERLAP_FRAC": 0.5,
    "crop_padding": 0.30,
    "IN_RANGE_THRESHOLD": 0.10,
    "GRID_PX": 100,
    "MIN_NABS_PER_CELL": 10,
    "OVERRIDE_YOLO_CONFIDENCE": 0.65,
    "LOOKBACK_DAYS": 14,
    "MATCH_IOU_THRESHOLD": 0.30,
    "MAX_MISSED_FRAMES": 3,
    "scene_mask_enabled": True,
    "nmm_enabled": True,
}


# OFAT sweep grid. Each key names a knob; the list is values to try while
# all other knobs are held at CURRENT_PROD. Hand-tuned to bracket the
# current value on both sides so we can see the response curve.
OFAT_GRID: dict[str, list] = {
    # YOLO model swap — biggest potential CPU win if accuracy holds. Tested
    # alongside the threshold knobs so we can see whether nano is within a
    # few % of small on FP/TP metrics. (yolo11m intentionally omitted from
    # this iteration — it's ~4× slower than 11s and "smaller, not bigger"
    # is the actual question.)
    "yolo_model_name": ["yolo11n.pt", "yolo11s.pt"],
    # YOLO-stage knobs (each value triggers a YOLO re-run):
    "target_fps": [2.0, 3.0, 4.0, 5.0],
    "BIRD_CONFIDENCE_THRESHOLD": [0.15, 0.25, 0.35],   # bracket current 0.20 sparsely
    "TILE_OVERLAP_PX": [int(1024 * 0.20), int(1024 * 0.30), int(1024 * 0.40)],
    "NMS_IOU": [0.40, 0.50, 0.60],
    "TILE_SEAM_GAP_PX": [10, 20, 30],
    "MAX_PROCESS_DURATION_SECONDS": [5.0, 10.0, 20.0, 1e9],
    # Downstream-stage knobs (can in principle replay from cached YOLO output):
    "IN_RANGE_THRESHOLD": [0.05, 0.10, 0.15, 0.20],
    "OVERRIDE_YOLO_CONFIDENCE": [0.55, 0.65, 0.75, 0.85, 2.0],   # 2.0 ≈ off
    "MIN_NABS_PER_CELL": [5, 10, 20, 50],
    "GRID_PX": [75, 100, 150],
    "MATCH_IOU_THRESHOLD": [0.20, 0.30, 0.40],
}
