"""Pipeline kết hợp YOLOv12s-seg và OpenCLIP (Vision-Language) để nhận dạng và đề xuất nhãn rác thải.

Quy trình:
1. YOLOv12 phát hiện vị trí vật thể (Bounding Box & Segmentation Mask).
2. Tự động cắt (crop) vùng ảnh vật thể.
3. Đưa vào OpenCLIP so khớp tương đồng Cosine (Cosine Similarity) với các câu mô tả ngữ nghĩa (Text Prompts).
4. Đề xuất nhãn phù hợp, đặc biệt hỗ trợ Zero-shot cho các vật thể nằm ngoài 16 class của YOLO (như Lon Coca / Kim loại, Giấy A4 trắng).
"""

# import argparse
# import sys
# import time
# from pathlib import Path
# import cv2
# import numpy as np
# import torch
# from PIL import Image
# import open_clip
# from ultralytics import YOLO

# # Danh sách nhãn và câu prompt mô tả chi tiết cho OpenCLIP
# PROMPT_DICT = {
#     # 16 class gốc trong dataset
#     "Battery": "a photo of an electric battery or cylindrical battery cell",
#     "BiologicalWaste": "a photo of organic biological food waste or compost scrap",
#     "Cardboard Paper": "a photo of a brown cardboard box or corrugated carton",
#     "Glass": "a photo of a broken glass shard or glass material",
#     "Glass-Bottle": "a photo of a glass bottle or glass jar",
#     "Glove": "a photo of a rubber glove or protective latex glove",
#     "HDPE Plastic": "a photo of a thick HDPE plastic bottle, shampoo bottle or detergent jug",
#     "Mask": "a photo of a disposable blue medical face mask",
#     "Medicine": "a photo of medical pill blister pack or medicine tablets",
#     "Newspaper": "a photo of a printed paper newspaper with text and articles",
#     "PET Bottle": "a photo of a transparent plastic drink bottle or PET water bottle",
#     "Paper": "a photo of crumpled waste paper or scrap paper",
#     "Single-Use-Plastic": "a photo of a thin plastic shopping bag or flimsy grocery bag",
#     "Single-layer Plastic": "a photo of single-layer plastic film, clear food wrap or wrapper",
#     "Squeeze Tube": "a photo of a flexible plastic squeeze tube for cream or toothpaste",
#     "syringe": "a photo of a plastic medical syringe or needle",
    
#     # Các lớp bổ sung ngoài dataset (Zero-shot Detection)
#     "Metal Can (Lon nhôm/kim loại)": "a photo of an aluminum beverage soda can, Coca-Cola tin can, or metal beer can",
#     "Clean White Paper (Giấy A4 phẳng)": "a photo of a flat clean white A4 paper sheet or document"
# }


# def parse_args():
#     parser = argparse.ArgumentParser(description="YOLOv12 + OpenCLIP Waste Recognition Pipeline")
#     parser.add_argument("--model", type=str, default=r"d:\DoAnChuyenNganh\runs\train_100_epochs\weights\best.pt",
#                         help="Đường dẫn tới file trọng số best.pt của YOLOv12")
#     parser.add_argument("--source", type=str, default="0",
#                         help="Nguồn đầu vào: 0 (webcam), đường dẫn file ảnh (.jpg/.png) hoặc video (.mp4)")
#     parser.add_argument("--clip-model", type=str, default="ViT-B-32",
#                         help="Kiến trúc OpenCLIP (mặc định ViT-B-32 nhẹ và chuẩn)")
#     parser.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k",
#                         help="Bộ trọng số pre-trained cho OpenCLIP")
#     parser.add_argument("--conf", type=float, default=0.3,
#                         help="Ngưỡng tin cậy của YOLO để trích xuất ứng viên (0.3)")
#     parser.add_argument("--save", type=str, default=None,
#                         help="Đường dẫn lưu video hoặc ảnh kết quả (nếu cần)")
#     parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu",
#                         help="Thiết bị chạy (0, cuda, cpu)")
#     return parser.parse_args()


# class YoloOpenClipPipeline:
#     def __init__(self, yolo_path, clip_model_name, clip_pretrained, device="cuda:0"):
#         # Chuẩn hóa device string cho PyTorch (ví dụ '0' thành 'cuda:0')
#         if isinstance(device, str) and device.isdigit():
#             self.device = f"cuda:{device}"
#         else:
#             self.device = device
#         print(f"[*] Đang nạp mô hình YOLOv12 từ: {yolo_path} trên thiết bị: {device}...")
#         self.yolo = YOLO(yolo_path)

#         print(f"[*] Đang nạp mô hình OpenCLIP ({clip_model_name} - {clip_pretrained})...")
#         self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
#             clip_model_name, pretrained=clip_pretrained, device=self.device
#         )
#         self.clip_model.eval()
#         self.tokenizer = open_clip.get_tokenizer(clip_model_name)

#         # Biên dịch sẵn Text Embeddings để tăng tốc độ xử lý realtime
#         print("[*] Đang mã hóa các câu mô tả văn bản (Text Prompts)...")
#         self.class_names = list(PROMPT_DICT.keys())
#         self.prompt_texts = list(PROMPT_DICT.values())

#         text_tokens = self.tokenizer(self.prompt_texts).to(self.device)
#         with torch.no_grad():
#             self.text_features = self.clip_model.encode_text(text_tokens)
#             self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
#         print(f"[+] Khởi tạo thành công! Sẵn sàng so khớp {len(self.class_names)} danh mục rác thải.\n")

#     def classify_crop(self, crop_bgr):
#         """Đưa vùng ảnh cắt vào OpenCLIP để so sánh với câu mô tả văn bản."""
#         h, w = crop_bgr.shape[:2]
#         if h < 15 or w < 15:
#             return None, 0.0

#         crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(crop_rgb)
#         image_input = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             image_features = self.clip_model.encode_image(image_input)
#             image_features /= image_features.norm(dim=-1, keepdim=True)

#             # Tính Cosine Similarity và Softmax để ra % xác suất
#             similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
#             top_prob, top_idx = similarity[0].topk(1)

#         best_class = self.class_names[top_idx[0].item()]
#         best_prob = top_prob[0].item()
#         return best_class, best_prob

#     def process_frame(self, frame, conf_thresh=0.3):
#         """Xử lý 1 khung hình: YOLOv12 phát hiện -> OpenCLIP thẩm định."""
#         h_frame, w_frame = frame.shape[:2]
        
#         # 1. Dự đoán với YOLOv12
#         results = self.yolo.predict(source=frame, conf=conf_thresh, device=self.device, verbose=False)[0]
        
#         # Vẽ viền mask và bounding box cơ bản từ YOLO
#         annotated = results.plot(line_width=2, font_size=12)

#         # 2. Xử lý từng vật thể tìm thấy
#         if results.boxes is not None and len(results.boxes) > 0:
#             for i, box in enumerate(results.boxes):
#                 xyxy = box.xyxy[0].cpu().numpy().astype(int)
#                 x1, y1, x2, y2 = xyxy
#                 # Đảm bảo tọa độ hợp lệ
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(w_frame, x2), min(h_frame, y2)

#                 yolo_cls_id = int(box.cls[0].item())
#                 yolo_cls_name = results.names.get(yolo_cls_id, str(yolo_cls_id))
#                 yolo_conf = float(box.conf[0].item())

#                 # Cắt vùng ảnh vật thể
#                 crop = frame[y1:y2, x1:x2]
#                 clip_label, clip_conf = self.classify_crop(crop)

#                 # Vẽ nhãn đề xuất từ OpenCLIP ngay bên dưới bounding box
#                 if clip_label is not None:
#                     # Màu vàng cam nổi bật cho nhãn OpenCLIP
#                     label_text = f"CLIP: {clip_label} ({clip_conf*100:.1f}%)"
#                     text_y = min(h_frame - 10, y2 + 20)
                    
#                     # Nền đen cho chữ
#                     (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#                     cv2.rectangle(annotated, (x1, text_y - th - 4), (x1 + tw + 6, text_y + 4), (0, 0, 0), -1)
#                     cv2.putText(annotated, label_text, (x1 + 3, text_y - 2),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

#         return annotated


# def main():
#     args = parse_args()
#     pipeline = YoloOpenClipPipeline(
#         yolo_path=args.model,
#         clip_model_name=args.clip_model,
#         clip_pretrained=args.clip_pretrained,
#         device=args.device
#     )

#     is_webcam = args.source.isdigit()
#     src = int(args.source) if is_webcam else args.source

#     cap = cv2.VideoCapture(src)
#     if not cap.isOpened():
#         print(f"[!] Lỗi: Không thể mở nguồn đầu vào '{args.source}'")
#         sys.exit(1)

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

#     writer = None
#     if args.save:
#         save_path = Path(args.save)
#         save_path.parent.mkdir(parents=True, exist_ok=True)
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
#         print(f"[*] Sẽ lưu kết quả vào: {save_path}")

#     print("=" * 60)
#     print(" HỆ THỐNG YOLOv12 + OpenCLIP ĐANG HOẠT ĐỘNG")
#     print(" - Nhãn xanh lá: Kết quả từ YOLOv12 (Instance Segmentation)")
#     print(" - Nhãn vàng cam: Kết quả đề xuất ngữ nghĩa từ OpenCLIP")
#     print(" - Nhấn 'Q' hoặc 'Esc' để dừng chương trình.")
#     print("=" * 60)

#     prev_time = time.time()
#     frame_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         processed_frame = pipeline.process_frame(frame, conf_thresh=args.conf)

#         # Tính toán FPS
#         curr_time = time.time()
#         fps_calc = 1.0 / max(1e-5, (curr_time - prev_time))
#         prev_time = curr_time

#         # Hiển thị thông số FPS góc trên
#         cv2.putText(processed_frame, f"YOLOv12 + OpenCLIP | FPS: {fps_calc:.1f}", (15, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
#         cv2.putText(processed_frame, f"YOLOv12 + OpenCLIP | FPS: {fps_calc:.1f}", (15, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)

#         if writer:
#             writer.write(processed_frame)

#         cv2.imshow("YOLOv12 + OpenCLIP Waste Pipeline", processed_frame)
#         frame_count += 1

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q') or key == ord('Q') or key == 27:
#             break

#     cap.release()
#     if writer:
#         writer.release()
#         print(f"[+] Đã lưu file video thành công: {args.save}")
#     cv2.destroyAllWindows()
#     print(f"[+] Đã hoàn thành xử lý {frame_count} khung hình.")


# if __name__ == "__main__":
#     main()
"""Pipeline YOLOv12s-seg + OpenCLIP nhận dạng & đề xuất nhãn rác thải.
 
Quy trình:
1. YOLOv12-seg phát hiện vật thể (bounding box + segmentation mask).
2. Cắt vùng vật thể (loại nền bằng mask, pad vuông) rồi đưa vào OpenCLIP.
3. So khớp Cosine Similarity với các câu mô tả (prompt ensembling) -> nhãn đề xuất.
 
Cách chạy:
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
    p.add_argument("--source", type=str, default="0",
                   help="0 (webcam) | file ảnh | thư mục ảnh | file video")
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
        sys.exit(1)
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
 
 
def main():
    args = parse_args()
    pipe = YoloOpenClipPipeline(args.model, args.clip_model, args.clip_pretrained, args.device,
                                imgsz=args.imgsz, clip_thr=args.clip_thr, clip_every=args.clip_every)
 
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
 