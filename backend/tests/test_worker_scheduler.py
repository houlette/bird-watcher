"""The pipeline job must keep its own one-thread pool.

If it shares a pool, it runs on whichever thread is free, each of those
threads gets its own libgomp team, and torch's worker threads stop spinning
between operations: YOLO measured 0.416 s per tile that way against about
0.34 s with one team. See PIPELINE_EXECUTOR in pipeline/worker.py.
"""
from __future__ import annotations

from pipeline import worker


def test_pipeline_job_runs_alone_on_a_single_thread_pool():
    scheduler = worker._build_scheduler()

    job = scheduler.get_job("process_pending_visits")
    assert job.executor == worker.PIPELINE_EXECUTOR
    assert scheduler._lookup_executor(worker.PIPELINE_EXECUTOR)._pool._max_workers == 1
    assert [j.id for j in scheduler.get_jobs() if j.executor == worker.PIPELINE_EXECUTOR] == [
        "process_pending_visits"
    ]
