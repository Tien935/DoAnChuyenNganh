"""Nhan dang rac bang checkpoint YOLO detection/segmentation (.pt).

Chay trong MOI TRUONG DA TRAIN best.pt de giu dung phien ban YOLO/PyTorch.
Khong tu nang cap ultralytics neu checkpoint dung mot fork YOLOv12.
Neu thieu OpenCV GUI: pip install opencv-python (khong dung headless).

Vi du Windows (PowerShell/CMD, moi lenh tren mot dong):
python waste_detector.py --model "D:\\DoAnChuyenNganh\\runs\\train_100_epochs\\weights\\best.pt"
python waste_detector.py --model "D:\\DoAnChuyenNganh\\runs\\train_100_epochs\\weights\\best.pt" --device cpu --imgsz 416
python waste_detector.py --model best.pt --device 0 --half --save demo.mp4
python waste_detector.py --model best.pt --source input.mp4 --save output.mp4

Q/Esc: thoat. S: luu anh vao snapshots/. Khong dem so vat the duy nhat:
Objects on frame chi la so box trong khung hien tai, khong phai tracking.
Webcam bo khung cu de giam tre. Video file xu ly tuan tu khong bo khung.
Video webcam ghi CFR theo thoi gian thuc, lap hinh khi inference cham;
video file giu FPS nguon. Khong ghi am thanh. Khong ghi de file co san.
"""

import argparse
from collections import deque
from datetime import datetime
import logging
import math
from pathlib import Path
import sys
import threading
import time

LOG = logging.getLogger("waste")


def arguments():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--source", default="0", help="Chi so webcam hoac duong dan video")
    p.add_argument("--device", default="auto", help="auto, cpu, 0, 1...")
    p.add_argument("--half", action="store_true", help="FP16: chi bat tren CUDA")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--width", type=int, default=640, help="Do rong webcam yeu cau")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--camera-fps", type=float, default=30)
    p.add_argument("--save", type=Path, help="Ghi video MP4; mac dinh khong ghi")
    p.add_argument("--record-fps", type=float, default=20, help="FPS file ghi webcam")
    p.add_argument("--no-display", action="store_true")
    a = p.parse_args()
    if not a.model.is_file() or a.model.suffix.lower() != ".pt":
        p.error("--model phai tro den checkpoint .pt co that")
    if not (0 < a.conf <= 1 and 0 < a.iou <= 1):
        p.error("conf va iou phai nam trong (0, 1]")
    values = (a.imgsz, a.max_det, a.width, a.height, a.camera_fps, a.record_fps)
    if any(not math.isfinite(x) or x <= 0 for x in values):
        p.error("Kich thuoc, FPS va max-det phai duong, huu han")
    if a.save and (a.save.suffix.lower() != ".mp4" or a.save.exists()):
        p.error("--save phai la duong dan .mp4 moi, chua ton tai")
    if not a.source.isdecimal():
        src = Path(a.source)
        if not src.is_file():
            p.error("--source phai la chi so webcam hoac video co that")
        if a.save and a.save.resolve() == src.resolve():
            p.error("File dau ra khong duoc trung dau vao")
    return a


class LatestCamera:
    """Mot producer doc camera; consumer chi nhan frame moi nhat.

    Khong co hang doi tang dan. Model va GUI chi chay trong main thread.
    """
    def __init__(self, cap):
        self.cap = cap
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.frame = None
        self.timestamp = 0.0
        self.sequence = 0
        self.done = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    break
                with self.condition:
                    self.frame = frame
                    self.timestamp = time.perf_counter()
                    self.sequence += 1
                    self.condition.notify_all()
        finally:
            with self.condition:
                self.done = True
                self.condition.notify_all()

    def read(self, last_sequence):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.sequence != last_sequence or self.done, timeout=5)
            if not ready:
                raise RuntimeError("Camera khong tra frame moi trong 5 giay")
            if self.sequence == last_sequence:
                raise RuntimeError("Camera ngung tra khung hinh")
            # Producer thay tham chieu, khong sua buffer dang xu ly.
            return self.sequence, self.frame, self.timestamp

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            # Thu mo khoa backend dang ket trong read().
            self.cap.release()
            self.thread.join(timeout=1)
            if self.thread.is_alive():
                LOG.warning("Backend camera chua dung; thread da duoc danh dau daemon")


class RealtimeWriter:
    """Ghi CFR theo thoi gian thuc, khong tua nhanh khi bo khung camera."""
    def __init__(self, writer, fps):
        self.writer = writer
        self.fps = fps
        self.start = None
        self.index = 0
        self.previous = None

    def push(self, frame, timestamp):
        if self.start is None:
            self.start = timestamp
            self.writer.write(frame)
            self.index = 1
        else:
            self.flush(timestamp)
        self.previous = frame

    def flush(self, timestamp):
        if self.previous is not None:
            while self.start + self.index / self.fps <= timestamp:
                self.writer.write(self.previous)
                self.index += 1


def run(a):
    # Lazy imports: --help van chay duoc khi chua cai thu vien ML.
    import cv2
    import torch
    from ultralytics import YOLO

    device = ("0" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    is_cuda = device != "cpu"
    if is_cuda:
        if not device.isdecimal() or not torch.cuda.is_available():
            raise ValueError("Device phai la cpu hoac chi so CUDA hop le; thu --device cpu")
        if int(device) >= torch.cuda.device_count():
            raise ValueError("Chi so GPU vuot so GPU hien co")
    if a.half and not is_cuda:
        LOG.warning("Bo qua --half vi dang chay CPU")

    LOG.info("Tai model: %s | device=%s", a.model, device)
    model = YOLO(str(a.model))
    if model.task not in {"detect", "segment"}:
        raise ValueError("Can checkpoint detection/segmentation, khong phai classification")
    LOG.info("Task=%s | Nhan trong checkpoint: %s", model.task, model.names)
    predict_args = dict(imgsz=a.imgsz, conf=a.conf, iou=a.iou, device=device,
                        half=a.half and is_cuda, max_det=a.max_det, verbose=False)
    webcam = a.source.isdecimal()
    source = int(a.source) if webcam else str(Path(a.source))
    cap = None
    camera = None
    writer = None
    realtime_writer = None
    frames = 0
    durations = deque(maxlen=30)
    processing_started = None
    window = "Waste Detection | Q/Esc: quit | S: snapshot"
    try:
        # DirectShow thuong huu ich tren Windows; fallback neu khong mo duoc.
        backend = cv2.CAP_DSHOW if webcam and sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(source, backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY:
            cap.release()
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError("Khong mo duoc camera/video. Kiem tra source va quyen camera")
        if webcam:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
            cap.set(cv2.CAP_PROP_FPS, a.camera_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Backend co the bo qua yeu cau.
            camera = LatestCamera(cap)
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
            LOG.warning("Khong doc duoc FPS nguon; dung 30 FPS cho video file")
        sequence = 0
        if not a.no_display:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        LOG.info("Q/Esc de thoat; S de luu anh. Ctrl+C cung dung chuong trinh")
        while True:
            if camera:
                sequence, frame, captured_at = camera.read(sequence)
            else:
                ok, frame = cap.read()
                if not ok:
                    if frames == 0:
                        raise RuntimeError("Video khong co khung hinh doc duoc")
                    LOG.info("Ket thuc video (hoac decoder ngung doc)")
                    break
                captured_at = time.perf_counter()

            if processing_started is None:
                LOG.info("Warm-up model voi kich thuoc frame thuc te...")
                with torch.inference_mode():
                    model.predict(source=frame, **predict_args)
                # Warm-up khong duoc tinh vao FPS/ghi hinh.
                processing_started = time.perf_counter()
                if camera:
                    continue  # Lay frame moi sau warm-up, tranh hien thi frame cu.
            loop_started = time.perf_counter()
            with torch.inference_mode():
                result = model.predict(source=frame, **predict_args)[0]
                annotated = result.plot(line_width=2, font_size=14)
            completed_at = time.perf_counter()
            model_ms = (result.speed or {}).get("inference")
            model_text = f"Inference: {model_ms:.1f} ms" if model_ms is not None else "Inference: N/A"
            fps_text = f"Pipeline FPS: {len(durations) / sum(durations):.1f}" if durations else "Pipeline FPS: warming up"
            count = len(result.boxes) if result.boxes is not None else 0
            lines = [fps_text, model_text, f"Objects on frame: {count}", f"Device: {device}"]
            if webcam:
                lines.append(f"Capture-to-result: {(completed_at - captured_at) * 1000:.0f} ms")
            for i, text in enumerate(lines):
                y = 24 + i * 24
                #cv2.putText(annotated, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            #0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(annotated, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 1, cv2.LINE_AA)

            if a.save:
                if writer is None:
                    a.save.parent.mkdir(parents=True, exist_ok=True)
                    h, w = annotated.shape[:2]
                    output_fps = a.record_fps if webcam else source_fps
                    writer = cv2.VideoWriter(str(a.save), cv2.VideoWriter_fourcc(*"mp4v"),
                                             output_fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("Khong mo duoc VideoWriter; kiem tra codec/quyen ghi")
                    if webcam:
                        realtime_writer = RealtimeWriter(writer, output_fps)
                if realtime_writer:
                    realtime_writer.push(annotated, time.perf_counter())
                else:
                    writer.write(annotated)
            if not a.no_display:
                cv2.imshow(window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("s"), ord("S")):
                    path = Path("snapshots") / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(path), annotated):
                        raise RuntimeError("Khong luu duoc anh")
                    LOG.info("Anh: %s", path.resolve())
                if key in (ord("q"), ord("Q"), 27):
                    frames += 1
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    frames += 1
                    break
            durations.append(max(time.perf_counter() - loop_started, 1e-9))
            frames += 1
    finally:
        # Thu tu cleanup van duoc thuc hien neu buoc truoc gap loi.
        try:
            if realtime_writer:
                realtime_writer.flush(time.perf_counter())
        finally:
            if writer is not None:
                writer.release()
            if camera:
                camera.close()
            if cap is not None:
                cap.release()
            if not a.no_display:
                cv2.destroyAllWindows()
        LOG.info("Da xu ly %d khung hinh", frames)
        if a.save and frames and a.save.is_file() and a.save.stat().st_size > 0:
            LOG.info("Video: %s", a.save.resolve())


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    a = arguments()
    try:
        run(a)
    except KeyboardInterrupt:
        LOG.info("Da dung theo yeu cau")
        return 0
    except Exception:
        LOG.exception("Chuong trinh gap loi; xem traceback de xac dinh nguyen nhan")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
