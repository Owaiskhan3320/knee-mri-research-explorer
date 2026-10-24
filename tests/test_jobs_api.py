from dataclasses import replace
import threading
import time
import shutil
import pytest
from fastapi.testclient import TestClient
from app.settings import Settings
from app.jobs import BusyError, JobManager
from app.main import create_app
from app.settings import ROOT


def manager_fixture(tmp_path):
    manager = JobManager(Settings(root=tmp_path, queue_size=1, ttl_seconds=1))
    manager.settings.runtime.mkdir(parents=True)
    manager.worker_ready.set()
    return manager


def test_queue_bound_and_job_capability(tmp_path):
    m = manager_fixture(tmp_path)
    job = m.new_job("synthetic")
    with pytest.raises(KeyError):
        m.get(job.id, "wrong-token")
    with pytest.raises(BusyError):
        m.new_job("synthetic")
    assert m.get(job.id, job.token) is job
    m.cancel(job.id, job.token)
    assert not job.directory.exists()


def test_expired_result_is_removed(tmp_path):
    m = manager_fixture(tmp_path)
    job = m.new_job("synthetic")
    job.state, job.finished = "completed", time.monotonic() - 2
    with pytest.raises(KeyError):
        m.get(job.id, job.token)
    assert not job.directory.exists()


def test_api_does_not_expose_another_jobs_result(tmp_path, monkeypatch):
    m = manager_fixture(tmp_path)
    monkeypatch.setattr(m, "start", lambda: None)
    monkeypatch.setattr(m, "close", lambda: None)
    app = create_app(m.settings, manager=m, mount_ui=False)
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        response = client.post("/api/v1/jobs", data={"kind": "synthetic"})
        assert response.status_code == 202
        value = response.json()
        path = "/api/v1/jobs/" + value["id"]
        assert client.get(path).status_code == 404
        assert client.get(path + "/result").status_code == 404
        headers = {"Authorization": "Bearer " + value["access_token"]}
        assert client.get(path, headers=headers).status_code == 200
        assert client.get(path + "/result", headers=headers).status_code == 409
        assert client.delete(path, headers=headers).status_code == 200
        assert client.get(path, headers=headers).status_code == 404


def test_api_upload_is_disabled_when_configured(tmp_path, monkeypatch):
    m = manager_fixture(tmp_path)
    m.settings = replace(m.settings, allow_uploads=False)
    monkeypatch.setattr(m, "start", lambda: None)
    monkeypatch.setattr(m, "close", lambda: None)
    with TestClient(create_app(m.settings, m, mount_ui=False)) as client:
        assert client.post("/api/v1/jobs", data={"kind": "upload"}).status_code == 403


@pytest.mark.requires_model
def test_timeout_restarts_real_worker_and_next_job_succeeds(tmp_path):
    # Separate job storage: never let a test clean the live application's jobs.
    (tmp_path / "models").mkdir()
    (tmp_path / "app").mkdir()
    for name in ("deploy.pt", "manifest.json"):
        shutil.copyfile(ROOT / "models" / name, tmp_path / "models" / name)
    for name in ("model.py", "raw_preprocess.py"):
        shutil.copyfile(ROOT / "app" / name, tmp_path / "app" / name)
    m = JobManager(Settings(root=tmp_path, timeout_seconds=1))
    try:
        m.start()
        assert m.worker_ready.wait(60), m.startup_error
        original_pid = m.process.pid
        job = m.new_job("synthetic")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if job.state == "failed" and m.worker_ready.is_set() and m.process.pid != original_pid:
                break
            time.sleep(.1)
        assert job.state == "failed" and job.result is None
        assert not job.directory.exists()
        assert m.worker_ready.is_set() and m.process.pid != original_pid
        m.settings = replace(m.settings, timeout_seconds=60)
        second = m.new_job("synthetic")
        deadline = time.monotonic() + 60
        while second.state not in {"completed", "failed"} and time.monotonic() < deadline:
            time.sleep(.1)
        assert second.state == "completed", second.error
        assert len(second.result["valid_windows"]) == 40
    finally:
        m.close()
