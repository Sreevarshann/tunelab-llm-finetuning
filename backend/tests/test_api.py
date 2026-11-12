import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(data_dir):
    from tunelab.main import app

    with TestClient(app) as c:
        yield c


def synthetic(client, **kw):
    r = client.post("/api/datasets/synthetic", json={"train": 40, "valid": 8, "test": 10, "test_heldout": 10, **kw})
    assert r.status_code == 201, r.text
    return r.json()


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["base_model"].startswith("mlx-community/") and h["base_revision"]


def test_synthetic_dataset_is_labelled_and_has_real_token_stats(client):
    d = synthetic(client)
    assert d["synthetic"] is True and "seed=7" in d["source"]
    assert {k: v["count"] for k, v in d["splits"].items()} == {"train": 40, "valid": 8, "test": 10, "test_heldout": 10}
    assert 100 < d["splits"]["train"]["tokens"]["mean"] < 400
    assert d["scoring"]["mode"] == "json_fields"
    rows = client.get(f"/api/datasets/{d['id']}/rows", params={"split": "test_heldout", "limit": 3}).json()
    assert rows["total"] == 10 and len(rows["rows"]) == 3
    assert client.get(f"/api/datasets/{d['id']}/rows", params={"split": "nope"}).status_code == 404
    assert client.get("/api/datasets/ds_missing").status_code == 404


def line(q, a):
    return json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})


def test_upload_single_file_auto_splits_and_detects_text_scoring(client):
    body = "\n".join(line(f"translate {i}", f"answer number {i}") for i in range(120))
    r = client.post("/api/datasets/upload", data={"name": "mine"}, files={"file": ("d.jsonl", body)})
    assert r.status_code == 201, r.text
    d = r.json()
    assert d["synthetic"] is False and d["scoring"] == {"mode": "text"}
    assert sum(v["count"] for v in d["splits"].values()) == 120


def test_upload_rejects_invalid_rows_with_details(client):
    body = line("q", "a") + "\n" + json.dumps({"messages": [{"role": "user", "content": "only user"}]}) + "\n{bad"
    r = client.post("/api/datasets/upload", data={"name": "bad"}, files={"file": ("d.jsonl", body)})
    assert r.status_code == 422
    assert "line 3: invalid JSON" in json.dumps(r.json())
    r = client.post("/api/datasets/upload", data={"name": "bad"}, files={"train": ("t.jsonl", line("q", "a"))})
    assert r.status_code == 422  # explicit splits need train + valid
    r = client.post("/api/datasets/upload", data={"name": "bin"}, files={"file": ("d.jsonl", b"\xff\xfe")})
    assert r.status_code == 422


def test_runs_validate_config_queue_and_cancel(client):
    d = synthetic(client)
    assert client.post("/api/runs", json={"dataset_id": d["id"], "config": {"rank": 0}}).status_code == 422
    assert client.post("/api/runs", json={"dataset_id": d["id"], "config": {"learning_rate": 1}}).status_code == 422
    assert client.post("/api/runs", json={"dataset_id": "ds_nope"}).status_code == 404
    run = client.post("/api/runs", json={"dataset_id": d["id"], "name": "t", "config": {"iters": 20}}).json()
    assert run["status"] == "queued" and run["config"]["iters"] == 20 and run["config"]["rank"] == 8
    detail = client.get(f"/api/runs/{run['id']}").json()
    assert detail["dataset"]["id"] == d["id"] and detail["evaluations"] == []
    assert client.post(f"/api/runs/{run['id']}/evaluate").status_code == 409
    assert client.get(f"/api/runs/{run['id']}/adapter").status_code == 409
    assert client.post(f"/api/runs/{run['id']}/cancel").json()["status"] == "cancelled"
    events = client.get(f"/api/jobs/{run['id']}/events").text
    assert "event: status" in events and '"cancelled"' in events and "event: end" in events


def test_playground_rejects_unfinished_runs(client):
    d = synthetic(client)
    run = client.post("/api/runs", json={"dataset_id": d["id"]}).json()
    r = client.post("/api/playground", json={"messages": [{"role": "user", "content": "hi"}], "run_ids": [run["id"]]})
    assert r.status_code == 409
    assert client.post("/api/playground", json={"messages": []}).status_code == 422


def test_compare_input_is_validated(client):
    assert client.get("/api/compare", params={"run_ids": "a;drop"}).status_code == 422
    assert client.get("/api/compare", params={"run_ids": "run_missing"}).status_code == 404


def test_timestamps_are_utc_with_offset(client):
    d = synthetic(client)
    run = client.post("/api/runs", json={"dataset_id": d["id"]}).json()
    assert d["created_at"].endswith("+00:00") and run["created_at"].endswith("+00:00")
    client.post(f"/api/runs/{run['id']}/cancel")
    events = client.get(f"/api/jobs/{run['id']}/events").text
    assert "+00:00" in events


def test_empty_base_revision_means_none(monkeypatch):
    from tunelab.config import Settings

    monkeypatch.setenv("TUNELAB_BASE_REVISION", "")
    assert Settings().base_revision is None
