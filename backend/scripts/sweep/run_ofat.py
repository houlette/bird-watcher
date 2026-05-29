"""Phase 6: One-Factor-At-A-Time (OFAT) optimization sweep.

For each tunable knob, hold every other knob at CURRENT_PROD's value and
sweep this knob across the OFAT_GRID range. Score each value of the knob
on the same dataset of replays. Write per-knob JSON to results/.

This is expensive: ~15 visits × (sum of grid sizes ≈ 40) replays ≈ 600
runs, each ~70 s on Apple Silicon → ~12 hours wall time worst-case.
Use --knobs to scope down (only sweep a few knobs) and --limit to
restrict the dataset for faster iteration.

Run from backend/:
    python -m scripts.sweep.run_ofat --knobs BIRD_CONFIDENCE_THRESHOLD IN_RANGE_THRESHOLD
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_SWEEP_DIR = Path(__file__).resolve().parent
_SWEEP_DATA = _SWEEP_DIR / "data"
_RESULTS = _SWEEP_DIR / "results"

# Reuse the symlink bootstrap from run_regression.
from scripts.sweep.run_regression import _bootstrap_environment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knobs", nargs="+", default=None,
                    help="restrict to these knobs (defaults to all in OFAT_GRID)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap visits per config (smoke-test mode)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("ofat")

    _bootstrap_environment()

    from db.session import SessionLocal, init_db
    from db.models import Visit
    from scripts.sweep.configs import CURRENT_PROD, OFAT_GRID
    from scripts.sweep.replay import apply_config, replay_visit
    from scripts.sweep import score

    init_db()

    # Load ground truth & playable visits.
    manifest = json.loads((_SWEEP_DATA / "manifest.json").read_text())
    gt_by_visit: dict[int, list[dict]] = {}
    for gt in manifest["ground_truth"]:
        gt_by_visit.setdefault(gt["visit_id"], []).append(gt)
    db = SessionLocal()
    try:
        all_visits = db.query(Visit).filter(Visit.id.in_(gt_by_visit.keys())).all()
    finally:
        db.close()
    playable = [v for v in all_visits if (_SWEEP_DATA / v.clip_path).exists()]
    if args.limit:
        playable = playable[: args.limit]
    log.info("Playable visits: %d", len(playable))

    knobs_to_sweep = args.knobs if args.knobs else list(OFAT_GRID.keys())
    _RESULTS.mkdir(exist_ok=True)

    for knob in knobs_to_sweep:
        if knob not in OFAT_GRID:
            log.warning("Unknown knob %s — skipping", knob)
            continue
        log.info("=" * 60)
        log.info("Sweeping knob %s across %d values", knob, len(OFAT_GRID[knob]))

        out_path = _RESULTS / f"ofat_{knob}.json"
        # Resume from existing partial results if present.
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            per_value_results = existing.get("values", [])
            done_values = {repr(v["value"]) for v in per_value_results}
            log.info("Resuming: %d values already complete in %s", len(done_values), out_path.name)
        else:
            per_value_results = []
            done_values = set()

        for value in OFAT_GRID[knob]:
            if repr(value) in done_values:
                continue
            cfg = dict(CURRENT_PROD)
            cfg[knob] = value
            label = f"{knob}={value}"
            log.info("--- %s ---", label)

            metrics_per_visit = []
            with apply_config(cfg):
                for i, v in enumerate(playable, 1):
                    db = SessionLocal()
                    try:
                        result = replay_visit(
                            clip_path=v.clip_path,
                            ground_truth_visit_id=v.id,
                            started_at=v.started_at,
                            db=db,
                            config_label=label,
                        )
                    finally:
                        db.close()
                    if result.error:
                        log.warning("  visit %d FAILED: %s", v.id, result.error)
                        continue
                    m = score.score_visit(result, gt_by_visit.get(v.id, []))
                    metrics_per_visit.append(m)
                    if i % 5 == 0:
                        log.info("  [%d/%d]", i, len(playable))

            agg = score.aggregate(metrics_per_visit, label)
            log.info("→ FP leak=%.1f%%  TP-real=%.1f%%  TP-any=%.1f%%  cls-hit=%.1f%%",
                     100*agg.fp_leak_rate, 100*agg.tp_preservation_rate_real,
                     100*agg.tp_preservation_rate_any, 100*agg.classifier_hit_rate)
            per_value_results.append({
                "value": value,
                "fp_leaked": agg.total_fp_leaked,
                "gt_nabs": agg.total_gt_nabs,
                "fp_leak_rate": agg.fp_leak_rate,
                "tps_preserved_real": agg.total_tps_preserved_real,
                "gt_real": agg.total_gt_real,
                "tp_preservation_real": agg.tp_preservation_rate_real,
                "tp_preservation_any": agg.tp_preservation_rate_any,
                "novel_detections": agg.total_novel,
                "classifier_hits": agg.total_classifier_hits,
                "classifier_misses": agg.total_classifier_misses,
                "classifier_rejected": agg.total_classifier_rejected,
                "classifier_hit_rate": agg.classifier_hit_rate,
            })
            # Persist after every value so an interrupted sweep doesn't lose
            # work — the next run picks up where this one left off via the
            # done_values check above.
            out_path.write_text(json.dumps({
                "knob": knob,
                "current_prod_value": CURRENT_PROD.get(knob),
                "values": per_value_results,
            }, indent=2))
            log.info("Persisted %d/%d values to %s",
                     len(per_value_results), len(OFAT_GRID[knob]), out_path.name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
