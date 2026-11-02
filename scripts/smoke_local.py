"""Exercise the running local API with real inference, not mocked detections."""
from pathlib import Path
import json
import sys
import time
import cv2
import numpy as np
import requests
import subprocess
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"


def wait(job):
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        response = requests.get(f"{BASE}/api/jobs/{job['id']}", timeout=20)
        response.raise_for_status()
        job = response.json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(.4)
    raise TimeoutError(job["id"])


def submit(path, mode="track", device="cpu"):
    with path.open("rb") as stream:
        response = requests.post(f"{BASE}/api/jobs", data={"model":"models/yolov12n-seg.pt", "mode":mode,
                                                          "device":device, "imgsz":320}, files={"file":(path.name,stream)}, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    samples = requests.get(BASE + "/api/samples", timeout=20).json()
    fixture = ROOT / "local_runs/fixtures"
    fixture.mkdir(exist_ok=True)
    picture = fixture / "sample_pet.jpg"
    response = requests.get(BASE + samples[0]["url"], timeout=20)
    response.raise_for_status()
    picture.write_bytes(response.content)
    source = cv2.imdecode(np.frombuffer(response.content,dtype=np.uint8),cv2.IMREAD_COLOR)
    source = cv2.resize(source,(320,320))
    clip = fixture / "technical_fixture.mp4"
    writer = cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*"mp4v"),12,(320,320))
    assert writer.isOpened()
    for frame in range(24):
        writer.write(cv2.warpAffine(source,np.float32([[1,0,frame/2],[0,1,0]]),(320,320)))
    writer.release()
    # Browser preview needs H.264; OpenCV's mp4v codec is not universally supported.
    browser_clip = fixture / "technical_fixture_h264.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(clip),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(browser_clip)],
                   check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    browser_clip.replace(clip)
    results = []
    image_job = wait(submit(picture,mode="predict"))
    assert image_job["status"] == "completed", image_job
    assert image_job["frames"] == 1 and image_job["waste_model"] is False
    results.append({"case":"real_image_cpu", "status":"PASS", "job_id":image_job["id"]})
    video_job = submit(clip)
    # Pause as soon as the job is running, then resume. Warmup may still be active.
    deadline = time.monotonic()+60
    while time.monotonic()<deadline:
        running = requests.get(f"{BASE}/api/jobs/{video_job['id']}",timeout=20).json()
        if running["status"] == "running":
            paused = requests.post(f"{BASE}/api/jobs/{video_job['id']}/pause",timeout=20)
            assert paused.status_code == 200, paused.text
            resumed = requests.post(f"{BASE}/api/jobs/{video_job['id']}/resume",timeout=20)
            assert resumed.status_code == 200, resumed.text
            break
        if running["status"] == "failed": raise RuntimeError(running)
        time.sleep(.1)
    video_job = wait(video_job)
    assert video_job["status"] == "completed", video_job
    assert video_job["frames"] == 24, video_job
    output = ROOT / "local_runs" / video_job["id"] / "result.mp4"
    cap = cv2.VideoCapture(str(output))
    actual = 0
    while cap.read()[0]: actual += 1
    cap.release()
    assert actual == 24, actual
    results.append({"case":"video_yolo_bytetrack_pause_resume_export", "status":"PASS", "job_id":video_job["id"], "decoded_output_frames":actual})
    after_track = wait(submit(picture,mode="track"))
    assert after_track["status"] == "completed", after_track
    assert after_track["mode"] == "predict"
    assert all(d["track_id"] is None for d in after_track["detections"]), after_track
    results.append({"case":"image_after_tracked_video_has_no_stale_tracker", "status":"PASS", "job_id":after_track["id"]})
    corrupt = fixture / "corrupt.jpg"
    corrupt.write_bytes(b"this is not an image")
    bad = wait(submit(corrupt))
    assert bad["status"] == "failed", bad
    results.append({"case":"corrupt_image_real_worker", "status":"PASS", "job_id":bad["id"]})
    cancelled = submit(clip)
    requests.post(f"{BASE}/api/jobs/{cancelled['id']}/cancel",timeout=20).raise_for_status()
    cancelled = wait(cancelled)
    assert cancelled["status"] == "cancelled", cancelled
    results.append({"case":"cancel_real_worker", "status":"PASS", "job_id":cancelled["id"]})
    hardware = requests.get(BASE + "/api/status",timeout=20).json()["machine"]
    if hardware["cuda_available"]:
        gpu_job = wait(submit(picture,mode="predict",device="0"))
        assert gpu_job["status"] == "completed", gpu_job
        results.append({"case":"real_image_cuda", "status":"PASS", "job_id":gpu_job["id"]})
    for url in ["/api/benchmark.csv", "/api/benchmark.png", "/api/report.md"]:
        response = requests.get(BASE + url,timeout=30)
        response.raise_for_status()
        (fixture / Path(url).name).write_bytes(response.content)
    report = {"fixture_note":"Video kỹ thuật 24 khung được tạo từ một ảnh test, không phải video thực địa và không đánh giá chất lượng tracking/đếm.","results":results}
    target = ROOT / "docs/week8/smoke_results.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
