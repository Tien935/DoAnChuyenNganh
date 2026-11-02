"""Run with: venv/Scripts/python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / "local_runs" / "config"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "local_runs"
OUTPUT.mkdir(exist_ok=True)
EXPECTED_NAMES = yaml.safe_load((ROOT / "config/waste.yaml").read_text(encoding="utf-8"))["names"]
app = FastAPI(title="EcoVision Local", docs_url="/api/docs")
POOL = ThreadPoolExecutor(max_workers=1)
LOCK = threading.RLock()
JOBS: dict[str, dict] = {}
SIGNALS: dict[str, threading.Event] = {}
MODELS: dict[str, object] = {}
MAX_UPLOAD = 512 * 1024 * 1024
CSV_FIELDS = ["frame", "class_id", "class_name", "confidence", "track_id", "x1", "y1", "x2", "y2"]
BENCH_FIELDS = ["id", "created_at", "filename", "status", "model", "weights_sha256", "waste_model", "mode", "device", "imgsz", "confidence", "width", "height", "frames", "source_fps", "source_duration_s", "model_load_s", "warmup_s", "inference_s", "inference_fps", "processing_s", "pipeline_fps", "encoding_s", "output_bytes", "unique_tracks", "error"]


def now():
    return datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")


def machine():
    return {"platform": platform.platform(), "cpu": platform.processor(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if torch.cuda.is_available() else None}


def local_model(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or path.suffix.lower() != ".pt" or not path.is_file():
        raise HTTPException(400, "Trọng số phải là file .pt có sẵn trong thư mục dự án. Chép best.pt vào models/ rồi tải lại danh sách.")
    return path


def available_models():
    paths = list((ROOT / "models").glob("*.pt")) + list((ROOT / "runs").glob("**/best.pt"))
    return [str(p.relative_to(ROOT)).replace("\\", "/") for p in sorted(paths, key=lambda p: ("best" not in p.name.lower(), str(p)))]


def persist(job):
    # Atomic replacement permits polling and reading completed reports while writing.
    folder = OUTPUT / job["id"]
    folder.mkdir(exist_ok=True)
    temp = folder / "job.tmp"
    temp.write_text(json.dumps(job, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(folder / "job.json")


def update(job, **values):
    with LOCK:
        job.update(values)
        persist(job)


def all_jobs():
    items = {}
    with LOCK:
        for path in OUTPUT.glob("*/job.json"):
            if path.parent.name in JOBS:
                continue
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if item["status"] in {"queued", "running", "paused", "cancelling"}:
                    item.update(status="interrupted", error="Máy chủ đã dừng trước khi hoàn tất. Hãy chạy lại tệp.")
                items[item["id"]] = item
            except (OSError, ValueError, KeyError):
                continue
        items.update({key: dict(value) for key, value in JOBS.items()})
    return sorted(items.values(), key=lambda j: j["created_at"], reverse=True)


@app.get("/api/status")
def status():
    return {"ready": bool(available_models()), "models": available_models(), "machine": machine(), "expected_classes": EXPECTED_NAMES,
            "notice": "YOLO + ByteTrack chạy thật. OpenCLIP chưa tích hợp. Trọng số COCO chỉ dùng kiểm tra kỹ thuật; cần best.pt khớp 16 lớp để đánh giá rác thải.",
            "default_device": "cpu", "default_imgsz": 320}


@app.get("/api/samples")
def samples():
    chosen = []
    labels = ROOT / "dataset/test/labels"
    for name in ["PET Bottle", "Battery", "Mask", "Paper", "Glass-Bottle", "syringe"]:
        class_id = EXPECTED_NAMES.index(name)
        for label in sorted(labels.glob("*.txt")):
            if any(line.split() and line.split()[0] == str(class_id) for line in label.read_text().splitlines()):
                picture = ROOT / "dataset/test/images" / (label.stem + ".jpg")
                if picture.exists():
                    chosen.append({"name": name, "url": "/samples/" + picture.name})
                    break
    return chosen


@app.get("/samples/{filename}")
def sample_file(filename: str):
    path = (ROOT / "dataset/test/images" / filename).resolve()
    if path.parent != (ROOT / "dataset/test/images").resolve() or path.suffix.lower() not in {".jpg", ".png", ".jpeg"} or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/fixtures")
def fixtures():
    path = OUTPUT / "fixtures/technical_fixture.mp4"
    return [{"name": "Video kỹ thuật 2 giây", "url": "/fixtures/technical_fixture.mp4"}] if path.is_file() else []


@app.get("/fixtures/technical_fixture.mp4")
def fixture_video():
    path = OUTPUT / "fixtures/technical_fixture.mp4"
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/jobs")
def list_jobs():
    return all_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    for job in all_jobs():
        if job["id"] == job_id:
            return job
    raise HTTPException(404, "Không tìm thấy lần chạy.")


@app.post("/api/jobs", status_code=202)
async def submit(file: UploadFile = File(...), model: str = Form(...), mode: str = Form("track"),
                 device: str = Form("cpu"), imgsz: int = Form(320), confidence: float = Form(0.25)):
    path = local_model(model)
    if mode not in {"predict", "track"} or device not in {"cpu", "0"} or imgsz not in {320, 480, 640} or not (0.01 <= confidence <= 0.99):
        raise HTTPException(400, "Cấu hình không hợp lệ.")
    if device == "0" and not torch.cuda.is_available():
        raise HTTPException(400, "CUDA không khả dụng; hãy chọn CPU.")
    extension = Path(file.filename or "").suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".mp4", ".webm", ".avi"}:
        raise HTTPException(415, "Chỉ hỗ trợ JPG, PNG, MP4, WebM, AVI.")
    with LOCK:
        if any(j["status"] in {"queued", "running", "paused", "cancelling"} for j in JOBS.values()):
            raise HTTPException(409, "Đang có tác vụ khác. Hãy chờ hoàn tất hoặc hủy tác vụ trước.")
        ident = uuid.uuid4().hex[:16]
        kind = "image" if extension in {".jpg", ".jpeg", ".png"} else "video"
        job = {"id": ident, "created_at": now(), "filename": Path(file.filename.replace("\\", "/")).name,
               "status": "queued", "progress": 0, "model": path.relative_to(ROOT).as_posix(),
               "mode": mode if kind == "video" else "predict", "requested_mode": mode,
               "device": device, "imgsz": imgsz, "confidence": confidence,
               "kind": kind, "frames": 0,
               "detections": [], "error": None, "vlm_enabled": False, "vlm_calls": 0}
        JOBS[ident] = job
        SIGNALS[ident] = threading.Event()
        persist(job)
    upload = OUTPUT / ident / ("input" + extension)
    size = 0
    try:
        with upload.open("wb") as stream:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "Tệp vượt quá 512 MB.")
                stream.write(chunk)
        if size == 0:
            raise HTTPException(400, "Tệp rỗng.")
        POOL.submit(run_job, job, upload, path)
    except Exception as exc:
        update(job, status="failed", error=str(getattr(exc, "detail", exc)))
        raise
    finally:
        await file.close()
    return dict(job)


@app.post("/api/jobs/{job_id}/{action}")
def control(job_id: str, action: str):
    with LOCK:
        job = JOBS.get(job_id)
        if not job or job["status"] not in {"queued", "running", "paused", "cancelling"}:
            raise HTTPException(409, "Tác vụ không còn hoạt động.")
        if action == "cancel":
            SIGNALS[job_id].set()
            update(job, status="cancelling")
        elif action == "pause" and job["status"] == "running" and job["kind"] == "video":
            update(job, status="paused")
        elif action == "resume" and job["status"] == "paused":
            update(job, status="running")
        else:
            raise HTTPException(400, "Thao tác không phù hợp với trạng thái hiện tại.")
    return dict(job)


def checkpoint(job):
    paused = 0.0
    while job["status"] == "paused":
        began = time.perf_counter()
        if SIGNALS[job["id"]].wait(0.1):
            return False, paused
        paused += time.perf_counter() - began
    return not SIGNALS[job["id"]].is_set(), paused


def reset_tracker(model):
    predictor = getattr(model, "predictor", None)
    if predictor and hasattr(predictor, "trackers"):
        for tracker in predictor.trackers:
            tracker.reset()


def save_preview(folder: Path, annotated):
    preview = folder / "preview.tmp"
    cv2.imencode(".jpg", annotated)[1].tofile(preview)
    try:
        preview.replace(folder / "preview.jpg")
    except PermissionError:
        # FileResponse may hold the old preview open on Windows. Keep that frame;
        # a refresh failure must never abort video encoding.
        pass


def run_job(job, source: Path, weights: Path):
    cap = None
    encoder = None
    encoder_log = None
    started = time.perf_counter()
    try:
        from ultralytics import YOLO
        with LOCK:
            if SIGNALS[job["id"]].is_set():
                update(job, status="cancelled")
                return
            update(job, status="running")
        fingerprint = hashlib.sha256(weights.read_bytes()).hexdigest()
        # model.track registers persistent prediction callbacks; do not reuse that
        # instance for plain prediction jobs, including images after a tracked video.
        key = f"{weights}:{fingerprint}:{job['device']}:{job['mode']}"
        began = time.perf_counter()
        if key not in MODELS:
            MODELS.clear()  # Keep only one model in RAM/VRAM on this laptop.
            MODELS[key] = YOLO(str(weights))
        model = MODELS[key]
        load_s = time.perf_counter() - began
        reset_tracker(model)
        names = [model.names[i] for i in range(len(model.names))]
        waste_model = names == EXPECTED_NAMES
        update(job, weights_sha256=fingerprint, classes=names, waste_model=waste_model, task=model.task,
               machine=machine(), model_load_s=load_s,
               model_note="Đúng bộ tên và thứ tự 16 lớp rác; cần xác nhận nguồn checkpoint với nhóm." if waste_model else "Trọng số chưa khớp 16 lớp rác: chỉ dùng thử luồng xử lý, không dùng báo cáo độ chính xác rác.")
        options = dict(imgsz=job["imgsz"], conf=job["confidence"], device=job["device"], verbose=False, half=False)
        began = time.perf_counter()
        model.predict(np.zeros((job["imgsz"], job["imgsz"], 3), dtype=np.uint8), **options)
        update(job, warmup_s=time.perf_counter() - began)
        reset_tracker(model)
        if job["kind"] == "image":
            frame = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Không đọc được ảnh. Tệp có thể bị hỏng hoặc không phải ảnh.")
            height, width = frame.shape[:2]
            total, fps = 1, 0.0
        else:
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise ValueError("Không đọc được video hoặc codec không được hỗ trợ.")
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            fps = fps if math.isfinite(fps) and fps > 0 else 25.0
            if width <= 0 or height <= 0:
                raise ValueError("Kích thước video không hợp lệ.")
        update(job, width=width, height=height, total_frames=total, source_fps=fps,
               source_duration_s=total / fps if fps else 0)
        folder = OUTPUT / job["id"]
        target = folder / ("result.png" if job["kind"] == "image" else "result.mp4")
        if job["kind"] == "video":
            import imageio_ffmpeg
            encoder_log = (folder / "encoder.log").open("wb")
            command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
                       "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                       "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encoder_log,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        inference_s = encoding_s = paused_s = 0.0
        frames = 0
        unique = set()
        processing_start = time.perf_counter()
        with (folder / "detections.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            while True:
                proceed, pause = checkpoint(job)
                paused_s += pause
                if not proceed:
                    update(job, status="cancelled")
                    return
                if cap:
                    success, frame = cap.read()
                    if not success:
                        break
                began = time.perf_counter()
                if job["mode"] == "track" and cap:
                    result = model.track(frame, persist=True, tracker="bytetrack.yaml", **options)[0]
                else:
                    result = model.predict(frame, **options)[0]
                if job["device"] == "0":
                    torch.cuda.synchronize()
                inference_s += time.perf_counter() - began
                frames += 1
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.cpu()
                    for index in range(len(boxes)):
                        cls = int(boxes.cls[index])
                        track_id = int(boxes.id[index]) if boxes.id is not None else None
                        bbox = boxes.xyxy[index].tolist()
                        detections.append({"class_id": cls, "class_name": model.names[cls], "confidence": float(boxes.conf[index]), "track_id": track_id, "bbox": bbox})
                        if track_id is not None:
                            unique.add(track_id)
                        writer.writerow(dict(zip(CSV_FIELDS, [frames, cls, model.names[cls], float(boxes.conf[index]), track_id, *bbox])))
                annotated = result.plot()
                began = time.perf_counter()
                if encoder:
                    encoder.stdin.write(annotated.tobytes())
                else:
                    cv2.imencode(".png", annotated)[1].tofile(target)
                encoding_s += time.perf_counter() - began
                if frames == 1 or frames % 5 == 0 or frames == total:
                    # Atomic preview updates prevent browser reads of half-written JPEGs.
                    save_preview(folder, annotated)
                    update(job, frames=frames, progress=min(99, round(frames * 100 / max(total, frames))), detections=detections,
                           preview_url=f"/outputs/{job['id']}/preview.jpg", inference_fps=frames / max(inference_s, 1e-9))
                if not cap:
                    break
        if not frames:
            raise ValueError("Video không có khung hình giải mã được.")
        if cap and total > 0 and frames < total - 1:
            raise ValueError(f"Video kết thúc sớm: đọc {frames}/{total} khung. Kiểm tra tệp đầu vào.")
        if encoder:
            began = time.perf_counter()
            encoder.stdin.close()
            code = encoder.wait(timeout=120)
            encoding_s += time.perf_counter() - began
            if code:
                raise ValueError("Xuất MP4 thất bại; xem encoder.log của lần chạy.")
        elapsed = time.perf_counter() - processing_start - paused_s
        with LOCK:
            cancelled = SIGNALS[job["id"]].is_set()
            update(job, status="cancelled" if cancelled else "completed", progress=100 if not cancelled else job["progress"],
                   frames=frames, detections=detections, inference_s=inference_s, inference_fps=frames / inference_s,
                   processing_s=elapsed, pipeline_fps=frames / elapsed, encoding_s=encoding_s,
                   encoding_note="Tổng thời gian ghi/đợi encoder và hoàn tất tệp; encoder chạy song song, không phải CPU time thuần.",
                   paused_s=paused_s, total_wall_s=time.perf_counter() - started, unique_tracks=len(unique) if cap and job["mode"] == "track" else None,
                   count_note="Số ID track duy nhất trong toàn video; chưa phải đếm qua vạch và chưa có nhãn tham chiếu.",
                   output_url=f"/outputs/{job['id']}/{target.name}" if not cancelled else None, output_bytes=target.stat().st_size,
                   csv_url=f"/outputs/{job['id']}/detections.csv", json_url=f"/outputs/{job['id']}/job.json")
    except Exception as exc:
        update(job, status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if cap:
            cap.release()
        if encoder and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        if encoder_log:
            encoder_log.close()


def benchmark_csv(jobs):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=BENCH_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(jobs)
    return "\ufeff" + stream.getvalue()


@app.get("/api/benchmark.csv")
def benchmark():
    return Response(benchmark_csv(all_jobs()), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="week8_benchmark.csv"'})


@app.get("/api/benchmark.png")
def chart():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    jobs = [j for j in all_jobs() if j["status"] == "completed"][:12][::-1]
    if not jobs:
        raise HTTPException(404, "Chưa có lần chạy hoàn tất để vẽ biểu đồ.")
    fig = Figure(figsize=(11, 5), layout="constrained")
    ax = fig.subplots()
    indices = np.arange(len(jobs))
    ax.bar(indices - .18, [j["inference_fps"] for j in jobs], .36, label="Inference + tracker FPS")
    ax.bar(indices + .18, [j["pipeline_fps"] for j in jobs], .36, label="Pipeline FPS (decode, infer, draw, encode)")
    ax.set_xticks(indices, [j["id"][:6] for j in jobs], rotation=30)
    ax.set(ylabel="Frames / second", xlabel="Run ID (configuration in benchmark.csv)", title="Measured local runs — includes technical tests with COCO weights")
    ax.legend(fontsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/report.md")
def report():
    jobs = all_jobs()
    completed = [j for j in jobs if j["status"] == "completed"]
    lines = ["# Báo cáo công việc STT 5 — Tuần 8 (02–08/11/2026)", "", f"Xuất lúc: {now()}", "", "## Phạm vi", "",
             "Tích hợp frontend với API YOLO và ByteTrack; xử lý ảnh/video, xuất MP4, lưu thông tin từng lần chạy và đo tốc độ trên laptop.",
             "OpenCLIP chưa được tích hợp. Độ tin cậy dự đoán không phải Precision. ID track duy nhất không phải sai số đếm.", "",
             "## Phần cứng và môi trường", "", "```json", json.dumps(machine(), ensure_ascii=False, indent=2), "```", "",
             "## Các lần chạy thực tế", "", "| Run | Tệp | Model | Chế độ | Thiết bị | Khớp 16 lớp | Khung | FPS suy luận + tracker | FPS pipeline | Xử lý (s) | Ghi/hoàn tất tệp (s) |",
             "|---|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for j in completed:
        safe_name = j["filename"].replace("|", "_").replace("\n", " ")
        lines.append(f"| {j['id']} | {safe_name} | {j['model']} | {j['mode']} | {j['device']} | {j.get('waste_model')} | {j['frames']} | {j['inference_fps']:.2f} | {j['pipeline_fps']:.2f} | {j['processing_s']:.3f} | {j['encoding_s']:.3f} |")
    lines += ["", "FPS pipeline gồm đọc khung, suy luận/tracking, vẽ và ghi/xuất; loại thời gian nạp model, warmup, upload và tạm dừng. Thời gian encoder là tổng thời gian ghi/đợi và hoàn tất tệp, không phải CPU time thuần. MP4 không giữ âm thanh.",
              "", "## Độ chính xác và confusion matrix", "",
              "Chỉ báo cáo P/R/F1/mAP và confusion matrix sau khi chạy scripts/evaluate.py với checkpoint rác đúng 16 lớp và tập test đã khóa. Không suy ra từ độ tin cậy hiển thị hoặc số liệu COCO.", "",
              "Chưa tự động kết luận mô hình tốt hơn bài báo; cần cùng dữ liệu, loại chỉ số Box/Mask và điều kiện so sánh.", "",
              "## Kiểm thử hồi quy", "", "Xem docs/week8/regression_tests.csv và kết quả kiểm thử tự động trong docs/week8/. Các mục kiểm thử thủ công chưa có bằng chứng phải để Chưa chạy.",
              "", "## Bằng chứng", "", "- local_runs/<run-id>/job.json: cấu hình, SHA256 trọng số, phần cứng, thời gian và trạng thái.", "- local_runs/<run-id>/detections.csv: dự đoán theo khung hình.", "- local_runs/<run-id>/result.png hoặc result.mp4: đầu ra chú thích.", "- /api/benchmark.csv và /api/benchmark.png: bảng và biểu đồ tốc độ thực đo.", "",
              f"Tổng tác vụ: {len(jobs)}; hoàn tất: {len(completed)}; các lần lỗi/hủy không tính vào kết quả tốc độ hoàn tất."]
    return Response("\n".join(lines), media_type="text/markdown; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="bao_cao_stt5_tuan8.md"'})


@app.get("/outputs/{job_id}/{filename}")
def output_file(job_id: str, filename: str):
    if len(job_id) != 16 or any(c not in "0123456789abcdef" for c in job_id) or filename not in {"result.png", "result.mp4", "preview.jpg", "detections.csv", "job.json"}:
        raise HTTPException(404)
    path = OUTPUT / job_id / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "no-store"}, filename=filename if filename.endswith((".csv", ".json")) else None)


app.mount("/", StaticFiles(directory=ROOT / "CTK-Frontend", html=True), name="frontend")
