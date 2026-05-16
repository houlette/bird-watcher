"""Read-only species directory used by the active-learning picker.

Sources, in order:
  1. The yard_priors.json calibration file (preferred — the ~157 species
     that have actually been heard in this yard).
  2. The hand-coded NA_BACKYARD_ALLOWLIST in pipeline/classify.py, normalized
     to the eBird display form.

Either way, the picker shows the user a relevant short list rather than
525 globally-trained classes. Total counts (when available) are returned
so the UI can highlight common species or sort them by familiarity.
"""
from __future__ import annotations

from fastapi import APIRouter

from pipeline import calibration
from pipeline.classify import NA_BACKYARD_ALLOWLIST, _normalize_for_display

router = APIRouter()


@router.get("")
async def list_species() -> dict:
    cal = calibration._load_calibration()  # noqa: SLF001 — module-internal but stable
    if cal and isinstance(cal.get("species"), dict):
        items = [
            {"name": name, "total": info.get("total", 0)}
            for name, info in cal["species"].items()
            if isinstance(info, dict) and info.get("total", 0) >= calibration.MIN_DETECTIONS_FOR_ALLOWLIST
        ]
        items.sort(key=lambda r: r["total"], reverse=True)
        return {"source": "calibration", "species": items}

    items = [{"name": _normalize_for_display(name), "total": 0} for name in NA_BACKYARD_ALLOWLIST]
    items.sort(key=lambda r: r["name"])
    return {"source": "fallback", "species": items}
