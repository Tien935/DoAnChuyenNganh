"""Pipeline YOLOv12s-seg + OpenCLIP nhận dạng & đề xuất nhãn rác thải.

Quy trình:
1. YOLOv12-seg phát hiện vật thể (bounding box + segmentation mask).
2. Cắt vùng vật thể (loại nền bằng mask, pad vuông) rồi đưa vào OpenCLIP.
3. So khớp Cosine Similarity với các câu mô tả (prompt ensembling) -> nhãn đề xuất.

Cách chạy:
    Menu   :  python yolo_openclip_pipeline.py        (chọn 1 = Webcam, 2 = Ảnh chụp)
    Webcam :  python yolo_openclip_pipeline.py --source 0
    Video  :  python yolo_openclip_pipeline.py --source video.mp4 --save out/result.mp4
    Ảnh    :  python yolo_openclip_pipeline.py --source anh.jpg            (lưu results/anh_result.jpg)
    Thư mục:  python yolo_openclip_pipeline.py --source thu_muc_anh/ --no-show
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# key (trùng tên class YOLO) -> (tên hiển thị ASCII, mô tả cho CLIP)
# Lưu ý: cv2.putText KHÔNG vẽ được tiếng Việt có dấu, nên tên hiển thị dùng ASCII.
CLASSES = {
    "Battery": ("Battery", "an electric battery or cylindrical battery cell"),
    "BiologicalWaste": ("Biological Waste", "organic biological food waste or compost scrap"),
    "Cardboard Paper": ("Cardboard", "a brown cardboard box or corrugated carton"),
    "Glass": ("Glass", "a broken glass shard or glass material"),
    "Glass-Bottle": ("Glass Bottle", "a glass bottle or glass jar"),
    "Glove": ("Glove", "a rubber glove or protective latex glove"),
    "HDPE Plastic": ("HDPE Plastic", "a thick HDPE plastic bottle, shampoo bottle or detergent jug"),
    "Mask": ("Mask", "a disposable blue medical face mask"),
    "Medicine": ("Medicine", "a medical pill blister pack or medicine tablets"),
    "Newspaper": ("Newspaper", "a printed paper newspaper with text and articles"),
    "PET Bottle": ("PET Bottle", "a transparent plastic drink bottle or PET water bottle"),
    "Paper": ("Paper", "crumpled waste paper or scrap paper"),
    "Single-Use-Plastic": ("Plastic Bag", "a thin plastic shopping bag or flimsy grocery bag"),
    "Single-layer Plastic": ("Plastic Film", "single-layer plastic film, clear food wrap or wrapper"),
    "Squeeze Tube": ("Squeeze Tube", "a flexible plastic squeeze tube for cream or toothpaste"),
    "syringe": ("Syringe", "a plastic medical syringe or needle"),
    # Lớp bổ sung ngoài dataset (zero-shot)
    "Metal Can": ("Metal Can", "an aluminum beverage soda can, Coca-Cola tin can, or metal beer can"),
    "Clean White Paper": ("White Paper A4", "a flat clean white A4 paper sheet or document"),
    # Lớp "không phải rác" giúp loại các phát hiện sai của YOLO
    "Background": ("Not Waste", "a human hand, a table, a floor or an empty background"),
}
BACKGROUND_KEY = "Background"
TEMPLATES = ["a photo of {}.", "a close-up photo of {}.", "a cropped image of {}."]

GREEN = (80, 220, 80)
ORANGE = (0, 165, 255)
GRAY = (180, 180, 180)


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv12-seg + OpenCLIP Waste Recognition")
    p.add_argument("--model", type=str,
                   default=r"d:\DoAnChuyenNganh\runs\train_100_epochs\weights\best.pt",
                   help="Đường dẫn trọng số best.pt của YOLOv12")
    p.add_argument("--source", type=str, default=None,
                   help="Bỏ trống để hiện menu | 0 (webcam) | file ảnh | thư mục ảnh | file video")
    p.add_argument("--clip-model", type=str, default="ViT-B-32")
    p.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k")
    p.add_argument("--conf", type=float, default=0.3, help="Ngưỡng tin cậy YOLO")
    p.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh suy luận của YOLO")
    p.add_argument("--clip-thr", type=float, default=0.5,
                   help="Dưới ngưỡng này, nhãn cuối cùng sẽ lấy theo YOLO")
    p.add_argument("--clip-every", type=int, default=10,
                   help="Webcam/video: chạy lại CLIP cho mỗi vật thể sau N khung (giảm tải)")
    p.add_argument("--save", type=str, default=None,
                   help="Video: file .mp4 | Ảnh: file ảnh (1 ảnh) hoặc thư mục. Ảnh mặc định lưu vào results/")
    p.add_argument("--no-show", action="store_true", help="Không mở cửa sổ hiển thị")
    p.add_argument("--width", type=int, default=1280, help="Độ rộng webcam")
    p.add_argument("--height", type=int, default=720, help="Độ cao webcam")
    p.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Tiện ích vẽ
# ----------------------------------------------------------------------------
def draw_tag(img, text, x, y, color, above=True, scale=0.5):
    """Vẽ nhãn có nền đen, tự giữ trong khung hình."""
    h, w = img.shape[:2]
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    bw, bh = tw + 8, th + base + 6
    x = int(min(max(0, x), max(0, w - bw)))
    y = y - bh if above else y
    if y < 0:
        y = 0
    y = int(min(y, h - bh))
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 4, y + th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def crop_object(frame, box, polygon, pad_ratio=0.05):
    """Cắt vật thể, loại nền bằng mask (nếu có) và pad thành ảnh vuông nền xám."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    if polygon is not None and len(polygon) >= 3:
        mask = np.zeros(crop.shape[:2], np.uint8)
        poly = (polygon - np.array([x1, y1])).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)
        crop = np.where(mask[..., None] > 0, crop, np.uint8(127))

    ch, cw = crop.shape[:2]
    if ch < 15 or cw < 15:
        return None
    side = max(ch, cw)  # pad vuông để CLIP (center-crop) không cắt mất rìa vật thể
    canvas = np.full((side, side, 3), 127, np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop
    return canvas


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
class YoloOpenClipPipeline:
    def __init__(self, yolo_path, clip_model_name, clip_pretrained, device,
                 imgsz=640, clip_thr=0.5, clip_every=10):
        self.yolo_device = device  # dạng '0' / 'cpu' cho Ultralytics
        use_cuda = device != "cpu" and torch.cuda.is_available()
        self.device = f"cuda:{device}" if (use_cuda and device.isdigit()) else ("cuda" if use_cuda else "cpu")
        self.dtype = torch.float16 if use_cuda else torch.float32
        self.imgsz, self.clip_thr, self.clip_every = imgsz, clip_thr, max(1, clip_every)

        print(f"[*] Nạp YOLO: {yolo_path} (device={self.device})")
        self.yolo = YOLO(yolo_path)

        print(f"[*] Nạp OpenCLIP: {clip_model_name} / {clip_pretrained}")
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained, device=self.device,
            precision="fp16" if use_cuda else "fp32")
        self.clip.eval()
        tokenizer = open_clip.get_tokenizer(clip_model_name)

        # Prompt ensembling: trung bình embedding của nhiều mẫu câu cho mỗi lớp
        self.keys = list(CLASSES.keys())
        self.display = {k: v[0] for k, v in CLASSES.items()}
        feats = []
        with torch.inference_mode():
            for k in self.keys:
                tokens = tokenizer([t.format(CLASSES[k][1]) for t in TEMPLATES]).to(self.device)
                f = self.clip.encode_text(tokens)
                f = f / f.norm(dim=-1, keepdim=True)
                f = f.mean(dim=0)
                feats.append(f / f.norm())
        self.text_features = torch.stack(feats)
        self.logit_scale = self.clip.logit_scale.exp().float().item()
        self.yolo_to_key = {k.lower(): k for k in self.keys}

        self.cache = {}  # track_id -> (key, prob, frame_idx)
        self.frame_idx = 0
        print(f"[+] Sẵn sàng với {len(self.keys)} danh mục.\n")

    @torch.inference_mode()
    def classify_batch(self, crops_bgr):
        """Phân loại nhiều crop trong 1 lần forward -> list[(key, prob)]."""
        tensors = [self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                   for c in crops_bgr]
        batch = torch.stack(tensors).to(self.device, dtype=self.dtype)
        img_f = self.clip.encode_image(batch)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        probs = (self.logit_scale * img_f.float() @ self.text_features.float().T).softmax(dim=-1)
        top_p, top_i = probs.max(dim=-1)
        return [(self.keys[i], p) for i, p in zip(top_i.tolist(), top_p.tolist())]

    def process(self, frame, conf=0.3, track=False):
        """Xử lý 1 khung hình, trả về (ảnh đã vẽ, danh sách kết quả)."""
        self.frame_idx += 1
        kwargs = dict(conf=conf, imgsz=self.imgsz, device=self.yolo_device, verbose=False)
        r = (self.yolo.track(frame, persist=True, **kwargs) if track
             else self.yolo.predict(frame, **kwargs))[0]

        annotated = r.plot(labels=False, conf=False, line_width=2)  # chỉ vẽ mask + box
        if r.boxes is None or len(r.boxes) == 0:
            return annotated, []

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        ids = (r.boxes.id.cpu().numpy().astype(int)
               if (track and r.boxes.id is not None) else [None] * len(boxes))
        polys = r.masks.xy if r.masks is not None else [None] * len(boxes)

        # Xác định vật thể nào cần chạy CLIP (cache theo track id để tăng FPS)
        clip_out, todo, todo_crops = {}, [], []
        for i, (box, tid) in enumerate(zip(boxes, ids)):
            cached = self.cache.get(tid) if tid is not None else None
            if cached and self.frame_idx - cached[2] < self.clip_every:
                clip_out[i] = cached[:2]
                continue
            crop = crop_object(frame, box, polys[i])
            if crop is None:
                clip_out[i] = None
            else:
                todo.append(i)
                todo_crops.append(crop)

        if todo_crops:
            for i, res in zip(todo, self.classify_batch(todo_crops)):
                clip_out[i] = res
                if ids[i] is not None:
                    self.cache[ids[i]] = (res[0], res[1], self.frame_idx)
        if len(self.cache) > 300:  # dọn cache cũ
            self.cache = {k: v for k, v in self.cache.items() if self.frame_idx - v[2] < 300}

        detections = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            y_name = r.names.get(int(cls_ids[i]), str(cls_ids[i]))
            y_key = self.yolo_to_key.get(y_name.lower(), y_name)
            res = clip_out.get(i)

            draw_tag(annotated, f"YOLO: {y_name} {confs[i]:.2f}", x1, y1, GREEN, above=True)
            if res is not None:
                c_key, c_prob = res
                agree = c_key == y_key
                if c_key == BACKGROUND_KEY:
                    final = f"{y_name} (?)"
                elif c_prob >= self.clip_thr:
                    final = self.display[c_key]
                else:
                    final = y_name
                draw_tag(annotated, f"CLIP: {self.display[c_key]} {c_prob * 100:.0f}%",
                         x1, y2, GREEN if agree else ORANGE, above=False)
            else:
                c_key, c_prob, final = None, 0.0, y_name

            detections.append({"box": (x1, y1, x2, y2), "yolo": y_name, "yolo_conf": float(confs[i]),
                               "clip": c_key, "clip_prob": float(c_prob), "final": final})
        return annotated, detections


# ----------------------------------------------------------------------------
# Chế độ ảnh
# ----------------------------------------------------------------------------
def imread_unicode(path):
    """Đọc ảnh an toàn với đường dẫn có dấu/tiếng Việt trên Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))
    return ok


def run_images(pipe, args, paths):
    save_arg = Path(args.save) if args.save else None
    single_file_out = save_arg is not None and save_arg.suffix.lower() in IMAGE_EXTS and len(paths) == 1
    out_dir = Path("results") if save_arg is None else (save_arg.parent if single_file_out else save_arg)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in paths:
        img = imread_unicode(p)
        if img is None:
            print(f"[!] Không đọc được ảnh: {p}")
            continue
        t0 = time.time()
        annotated, dets = pipe.process(img, conf=args.conf, track=False)
        dt = (time.time() - t0) * 1000

        out_path = save_arg if single_file_out else out_dir / f"{p.stem}_result.jpg"
        imwrite_unicode(out_path, annotated)

        print(f"\n[{p.name}] {len(dets)} vật thể - {dt:.0f} ms -> {out_path}")
        for d in dets:
            print(f"   - YOLO: {d['yolo']:<20} {d['yolo_conf']:.2f} | "
                  f"CLIP: {d['clip']} {d['clip_prob'] * 100:.0f}% | Kết luận: {d['final']}")

        if not args.no_show:
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Result", annotated)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# Chế độ webcam / video
# ----------------------------------------------------------------------------
def run_stream(pipe, args, is_webcam):
    src = int(args.source) if is_webcam else args.source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if (is_webcam and sys.platform == "win32") \
        else cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[!] Không thể mở nguồn '{args.source}'")
        return
    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    print("=" * 60)
    print(" Xanh lá: YOLO | Nhãn CLIP xanh = đồng ý, cam = khác YOLO")
    print(" Nhấn 'Q' hoặc 'Esc' để thoát, 'S' để chụp ảnh kết quả")
    print("=" * 60)

    fps_ema, prev, n = 0.0, time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out, _ = pipe.process(frame, conf=args.conf, track=True)

        now = time.time()
        inst = 1.0 / max(1e-5, now - prev)
        prev = now
        fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
        txt = f"FPS: {fps_ema:.1f}"
        cv2.putText(out, txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

        if args.save:
            if writer is None:  # lấy kích thước từ khung thực tế
                Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps_src if not is_webcam else 25.0, (w, h))
            writer.write(out)

        n += 1
        if args.no_show:
            continue
        cv2.imshow("YOLOv12 + OpenCLIP Waste Pipeline", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("s"), ord("S")):
            Path("results").mkdir(exist_ok=True)
            name = Path("results") / f"capture_{int(time.time())}.jpg"
            imwrite_unicode(name, out)
            print(f"[+] Đã chụp: {name}")

    cap.release()
    if writer:
        writer.release()
        print(f"[+] Đã lưu video: {args.save}")
    cv2.destroyAllWindows()
    print(f"[+] Xử lý xong {n} khung hình.")


def choose_image_path():
    """Mở hộp thoại chọn ảnh (tkinter); nếu không có thì nhập đường dẫn tay."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Chọn ảnh cần nhận dạng",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("Tất cả", "*.*")])
        root.destroy()
        return Path(path) if path else None
    except Exception:
        s = input("Nhập đường dẫn ảnh: ").strip().strip('"')
        return Path(s) if s else None


def interactive_menu(pipe, args):
    while True:
        print("\n" + "=" * 40)
        print(" CHỌN CHẾ ĐỘ")
        print("  1. Webcam (nhận dạng trực tiếp)")
        print("  2. Ảnh chụp (chọn ảnh -> trả ra ảnh kết quả)")
        print("  0. Thoát")
        print("=" * 40)
        choice = input("Nhập lựa chọn: ").strip()
        if choice == "1":
            args.source = "0"
            run_stream(pipe, args, is_webcam=True)
        elif choice == "2":
            path = choose_image_path()
            if path is None or not path.exists():
                print("[!] Chưa chọn ảnh hợp lệ.")
                continue
            run_images(pipe, args, [path])
        elif choice == "0":
            break
        else:
            print("[!] Lựa chọn không hợp lệ.")


def main():
    args = parse_args()
    pipe = YoloOpenClipPipeline(args.model, args.clip_model, args.clip_pretrained, args.device,
                                imgsz=args.imgsz, clip_thr=args.clip_thr, clip_every=args.clip_every)

    if args.source is None:
        interactive_menu(pipe, args)
        return

    src = Path(args.source)
    if args.source.isdigit():
        run_stream(pipe, args, is_webcam=True)
    elif src.is_dir():
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            sys.exit(f"[!] Thư mục không có ảnh: {src}")
        run_images(pipe, args, paths)
    elif src.suffix.lower() in IMAGE_EXTS:
        if not src.exists():
            sys.exit(f"[!] Không tìm thấy ảnh: {src}")
        run_images(pipe, args, [src])
    else:
        run_stream(pipe, args, is_webcam=False)


if __name__ == "__main__":
    main()
