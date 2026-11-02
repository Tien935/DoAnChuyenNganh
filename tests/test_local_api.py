import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from fastapi.testclient import TestClient

import backend.app as api
from scripts.evaluate import check_names, metric_summary


@pytest.fixture
def client(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "models").mkdir(parents=True)
    (root / "models/best.pt").write_bytes(b"test-only weights; never loaded")
    output = root / "local_runs"
    output.mkdir()
    monkeypatch.setattr(api, "ROOT", root)
    monkeypatch.setattr(api, "OUTPUT", output)
    monkeypatch.setattr(api, "JOBS", {})
    monkeypatch.setattr(api, "SIGNALS", {})
    monkeypatch.setattr(api, "POOL", SimpleNamespace(submit=lambda *args: None))
    return TestClient(api.app)


def upload(client, filename="photo.jpg", content=b"image", **options):
    return client.post("/api/jobs", data={"model": "models/best.pt", **options}, files={"file": (filename, content)})


@pytest.mark.parametrize("filename", ["payload.exe", "report.txt", "file.pt"])
def test_reject_unsupported_files(client, filename):
    assert upload(client, filename).status_code == 415
    assert client.get("/api/jobs").json() == []


def test_reject_missing_or_outside_checkpoint(client):
    assert upload(client, model="../best.pt").status_code == 400
    assert upload(client, model="models/missing.pt").status_code == 400


@pytest.mark.parametrize("options", [{"mode": "vlm"}, {"confidence": 1.5}, {"imgsz": 999}, {"device": "unavailable"}])
def test_reject_bad_options(client, options):
    assert upload(client, **options).status_code == 400


def test_empty_upload_is_recorded_as_failure_and_does_not_block_queue(client):
    assert upload(client, content=b"").status_code == 400
    assert client.get("/api/jobs").json()[0]["status"] == "failed"
    assert upload(client).status_code == 202


def test_one_job_at_a_time_and_cancellation(client):
    first = upload(client)
    assert first.status_code == 202
    ident = first.json()["id"]
    assert upload(client).status_code == 409
    assert client.post(f"/api/jobs/{ident}/cancel").json()["status"] == "cancelling"
    assert api.SIGNALS[ident].is_set()


def test_image_uses_prediction_when_video_tracking_option_is_selected(client):
    job = upload(client, mode="track").json()
    assert job["mode"] == "predict"
    assert job["requested_mode"] == "track"


def test_pause_and_resume_preserve_job(client):
    job = upload(client, "fixture.mp4").json()
    ident = job["id"]
    api.JOBS[ident]["status"] = "running"
    assert client.post(f"/api/jobs/{ident}/pause").json()["status"] == "paused"
    assert client.post(f"/api/jobs/{ident}/resume").json()["status"] == "running"
    assert client.post(f"/api/jobs/{ident}/resume").status_code == 400


def test_restart_marks_unfinished_run_as_interrupted(client):
    ident = upload(client).json()["id"]
    api.JOBS.clear()
    job = client.get(f"/api/jobs/{ident}").json()
    assert job["status"] == "interrupted"


def test_private_inputs_and_arbitrary_paths_are_not_served(client):
    ident = upload(client).json()["id"]
    assert client.get(f"/outputs/{ident}/input.jpg").status_code == 404
    assert client.get("/outputs/not-a-job/job.json").status_code == 404


def test_output_supports_video_range_requests(client):
    ident = upload(client).json()["id"]
    (api.OUTPUT / ident / "result.mp4").write_bytes(b"0123456789")
    response = client.get(f"/outputs/{ident}/result.mp4", headers={"Range": "bytes=2-5"})
    assert response.status_code == 206
    assert response.content == b"2345"


def test_report_and_csv_do_not_invent_accuracy(client):
    upload(client, content=b"")
    report = client.get("/api/report.md")
    assert report.status_code == 200
    assert "OpenCLIP chưa" in report.text
    assert "sau khi chạy scripts/evaluate.py" in report.text
    assert client.get("/api/benchmark.csv").content.startswith(b"\xef\xbb\xbf")


def test_evaluation_rejects_coco_and_reordered_classes():
    with pytest.raises(ValueError, match="không khớp"):
        check_names({0: "person", 1: "bottle"}, ["Battery", "Paper"])
    with pytest.raises(ValueError):
        check_names({0: "Paper", 1: "Battery"}, ["Battery", "Paper"])


def test_precision_is_separate_from_map():
    summary = metric_summary(SimpleNamespace(mp=.8, mr=.6, map50=.7, map=.4))
    assert summary["precision"] == .8
    assert summary["map50_95"] == .4
    assert summary["f1_from_mean_precision_recall"] == pytest.approx(2*.8*.6/1.4)


def test_windows_preview_read_lock_does_not_abort_processing(tmp_path, monkeypatch):
    previous = tmp_path / "preview.jpg"
    previous.write_bytes(b"previous preview remains available")
    def locked_replace(self, target):
        raise PermissionError("WinError 5: file is being streamed")
    monkeypatch.setattr(Path, "replace", locked_replace)
    api.save_preview(tmp_path, np.zeros((16,16,3), dtype=np.uint8))
    assert previous.read_bytes() == b"previous preview remains available"
