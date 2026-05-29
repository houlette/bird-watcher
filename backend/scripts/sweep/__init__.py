"""Parameter-sweep harness for the BirdWatcher pipeline.

Usage:
    cd backend
    python -m scripts.sweep.run_regression    # Phase 5: A/B regression triage
    python -m scripts.sweep.run_ofat          # Phase 6: one-factor-at-a-time sweep
    python -m scripts.sweep.report            # Phase 7: combined Markdown report

All output goes under `backend/scripts/sweep/results/`. The harness reads
the local copy of production data under `backend/scripts/sweep/data/`
(see Phase 1 of the plan for how that was populated).
"""
