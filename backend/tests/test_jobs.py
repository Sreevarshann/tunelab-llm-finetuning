from tunelab import jobs
from tunelab.db import Job, session


def make_dataset():
    from tunelab.datasets import create_synthetic

    return create_synthetic(train=20, valid=8, test=8, test_heldout=8)


def test_completed_run_gets_tuned_and_single_baseline_evaluations(data_dir):
    ds = make_dataset()
    run = jobs.enqueue_train(ds.id, {"iters": 5})
    with session() as s:
        s.get(Job, run.id).status = "completed"
        s.commit()
    first = jobs.ensure_evaluations(run.id)
    assert len(first) == 4  # tuned + baseline for test and test_heldout
    run2 = jobs.enqueue_train(ds.id, {"iters": 5})
    second = jobs.ensure_evaluations(run2.id)
    assert len(second) == 2  # baselines are shared per dataset/split/base revision
    with session() as s:
        evals = [s.get(Job, i) for i in first + second]
    assert sorted(e.split for e in evals if e.adapter_job_id is None) == ["test", "test_heldout"]


def test_queue_runs_one_job_at_a_time_and_recovers_dead_workers(data_dir):
    ds = make_dataset()
    a = jobs.enqueue_train(ds.id, {"iters": 5})
    b = jobs.enqueue_train(ds.id, {"iters": 5})
    with session() as s:
        s.get(Job, a.id).status, s.get(Job, a.id).pid = "running", 999_999  # pid that does not exist
        s.commit()
    assert jobs.start_next() is None  # something is "running"
    jobs.recover_stale()
    with session() as s:
        assert s.get(Job, a.id).status == "failed" and "unexpectedly" in s.get(Job, a.id).error
        assert s.get(Job, b.id).status == "queued"


def test_cancel_running_sets_flag(data_dir):
    ds = make_dataset()
    a = jobs.enqueue_train(ds.id, {})
    with session() as s:
        s.get(Job, a.id).status = "running"
        s.commit()
    assert jobs.request_cancel(a.id).cancel_requested is True


def test_evaluations_are_dispatched_before_queued_training(data_dir, monkeypatch):
    ds = make_dataset()
    train = jobs.enqueue_train(ds.id, {"iters": 5})
    ev = jobs.enqueue_eval(ds.id, "test", None)
    started = []

    class FakeProc:
        pid = 4242

        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: started.append(a[0][-1]) or FakeProc())
    monkeypatch.setattr(jobs.threading, "Thread", lambda *a, **k: type("T", (), {"start": lambda self: None})())
    assert jobs.start_next() == ev.id
    assert started == [ev.id]
    with session() as s:
        assert s.get(Job, train.id).status == "queued"
