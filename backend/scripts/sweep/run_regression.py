"""Phase 5: A/B regression triage.

Replay every surviving labeled visit under two configurations
(PRE_REGRESSION and CURRENT_PROD) and report which metrics flipped.

Run from the backend/ directory:
    python -m scripts.sweep.run_regression
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path

# Backend dir is the import root.
_BACKEND = Path(__file__).resolve().parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_SWEEP_DIR = Path(__file__).resolve().parent
_SWEEP_DATA = _SWEEP_DIR / "data"
_RESULTS = _SWEEP_DIR / "results"


def _bootstrap_environment():
    """Point the pipeline at our local sweep data instead of backend/data.

    Critical: must happen BEFORE any `from pipeline...` import that
    captures DATA_DIR or DB_PATH at import time. We do this by symlinking
    backend/data → scripts/sweep/data BEFORE pipeline modules load."""
    real_data = _BACKEND / "data"
    sweep_data = _SWEEP_DATA
    if real_data.exists() and not real_data.is_symlink():
        # Preserve a backup so we don't nuke production data on local dev.
        backup = real_data.with_name("data.real-backup")
        if not backup.exists():
            real_data.rename(backup)
        else:
            shutil.rmtree(real_data)
    if real_data.is_symlink() or real_data.exists():
        real_data.unlink()
    real_data.symlink_to(sweep_data, target_is_directory=True)
    # Ensure crops/frames subdirs exist so process.py mkdir-on-import is happy.
    (sweep_data / "crops").mkdir(exist_ok=True)
    (sweep_data / "frames").mkdir(exist_ok=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap visits per config (smoke-test mode)")
    ap.add_argument("--configs", nargs="+", default=["PRE_REGRESSION", "CURRENT_PROD"],
                    help="which configs to replay (PRE_REGRESSION, CURRENT_PROD)")
    args = ap.parse_args()

    # Unbuffer stdout/stderr so progress is visible when piped through tee.
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("sweep")

    _bootstrap_environment()

    # Now safe to import pipeline modules.
    from db.session import SessionLocal, init_db
    from db.models import Detection, Species, Visit
    from scripts.sweep.configs import CURRENT_PROD, PRE_REGRESSION
    from scripts.sweep.replay import apply_config, replay_visit
    from scripts.sweep import score

    init_db()

    # ── Load ground truth ────────────────────────────────────────────────
    manifest = json.loads((_SWEEP_DATA / "manifest.json").read_text())
    gt_by_visit: dict[int, list[dict]] = {}
    for gt in manifest["ground_truth"]:
        gt_by_visit.setdefault(gt["visit_id"], []).append(gt)
    log.info("Ground truth: %d visits with labeled detections", len(gt_by_visit))

    # ── Find playable visits: those whose clip file is on local disk ────
    db = SessionLocal()
    try:
        all_visits = db.query(Visit).filter(Visit.id.in_(gt_by_visit.keys())).all()
    finally:
        db.close()
    playable = []
    for v in all_visits:
        local_clip = _SWEEP_DATA / v.clip_path
        if local_clip.exists():
            playable.append(v)
    log.info("Playable visits with surviving clip: %d / %d", len(playable), len(all_visits))

    if not playable:
        log.error("No surviving clips to replay. Pull more data and retry.")
        return 1

    if args.limit:
        playable = playable[: args.limit]
        log.info("--limit %d → restricting to first %d visit(s)", args.limit, len(playable))

    _RESULTS.mkdir(exist_ok=True)
    report_lines = []
    detailed_per_config: dict[str, list[dict]] = {}

    available_configs = {"PRE_REGRESSION": PRE_REGRESSION, "CURRENT_PROD": CURRENT_PROD}
    for config_label in args.configs:
        cfg = available_configs[config_label]
        log.info("=" * 60)
        log.info("Replaying %d visits under config %s …", len(playable), config_label)
        metrics_per_visit = []
        details = []

        # Each replay opens its own session; apply_config wraps all of them.
        with apply_config(cfg):
            for i, v in enumerate(playable, 1):
                db = SessionLocal()
                try:
                    result = replay_visit(
                        clip_path=v.clip_path,    # relative; process.py prefixes DATA_DIR
                        ground_truth_visit_id=v.id,
                        started_at=v.started_at,
                        db=db,
                        config_label=config_label,
                    )
                finally:
                    db.close()

                if result.error:
                    log.warning("  [%d/%d] visit %d FAILED: %s", i, len(playable), v.id, result.error)
                    continue
                gt_rows = gt_by_visit.get(v.id, [])
                m = score.score_visit(result, gt_rows)
                metrics_per_visit.append(m)
                details.append({
                    "visit_id": v.id,
                    "clip_path": v.clip_path,
                    "n_replay_detections": len(result.detections),
                    "n_gt_nabs": m.n_gt_nabs,
                    "n_gt_real": m.n_gt_real,
                    "fp_leaked": m.fp_leaked,
                    "tps_preserved_real": m.tps_preserved_real,
                    "classifier_hits": m.classifier_hits,
                })
                if i % 5 == 0 or i == len(playable):
                    log.info("  [%d/%d] visit %d: %d dets, fp_leaked=%d, tp_real=%d/%d",
                             i, len(playable), v.id, len(result.detections),
                             m.fp_leaked, m.tps_preserved_real, m.n_gt_real)

        agg = score.aggregate(metrics_per_visit, config_label)
        detailed_per_config[config_label] = details
        report = score.format_report(agg)
        log.info("\n" + report)
        report_lines.append(report)
        (_RESULTS / f"{config_label.lower()}_agg.json").write_text(
            json.dumps({
                "config": cfg,
                "metrics": {k: getattr(agg, k) for k in
                            ("n_visits", "total_gt_nabs", "total_gt_real", "total_gt_unknown",
                             "total_fp_leaked", "total_tps_preserved_real",
                             "total_tps_preserved_unknown", "total_classifier_hits",
                             "total_classifier_misses", "total_classifier_rejected", "total_novel")},
                "rates": {"fp_leak_rate": agg.fp_leak_rate,
                          "tp_preservation_real": agg.tp_preservation_rate_real,
                          "tp_preservation_any": agg.tp_preservation_rate_any,
                          "classifier_hit_rate": agg.classifier_hit_rate},
                "per_visit": details,
            }, indent=2),
        )

    # Final A/B summary.
    summary = "\n\n".join(report_lines)
    summary += "\n\n=== A/B headline ===\n"
    (_RESULTS / "regression_summary.txt").write_text(summary)
    log.info("Wrote results/regression_summary.txt and per-config agg JSON.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
