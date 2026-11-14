import base64
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).parent / ".yolo_config")
os.environ["MPLCONFIGDIR"] = str(Path(__file__).parent / ".mpl_config")
os.environ["GRADIO_CHECK_UPDATE"] = "False"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import cv2
import gradio as gr
import imageio
import numpy as np
import open_clip
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Khởi tạo FastAPI server làm backend tích hợp
server_app = FastAPI(title="YOLOv12 Waste Classification System")

# ----------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN & THIẾT BỊ
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
BEST_WEIGHTS = ROOT_DIR / "runs" / "train_100_epochs" / "weights" / "best.pt"
FALLBACK_WEIGHTS = ROOT_DIR / "yolov12s-seg.pt"
MODEL_PATH = BEST_WEIGHTS if BEST_WEIGHTS.is_file() else FALLBACK_WEIGHTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ----------------------------------------------------------------------------
# 2. BẢNG TRA CỨU 16 LỚP + ZERO-SHOT & QUY CHUẨN PHÂN LOẠI RÁC
# ----------------------------------------------------------------------------
WASTE_DATA = {
    # 16 class gốc của tập huấn luyện YOLOv12
    "Battery": {
        "vi": "Pin điện tử",
        "category": "Rác Nguy Hại",
        "bin": "Thùng Đỏ / Cam (Nguy Hại)",
        "color": (230, 40, 40),      # Đỏ
        "clip_desc": "an electric battery or cylindrical battery cell",
        "action": "Thu gom riêng biệt, nộp điểm thu pin cũ, tuyệt đối không đốt hoặc chôn."
    },
    "BiologicalWaste": {
        "vi": "Rác thực phẩm / Hữu cơ",
        "category": "Rác Hữu Cơ",
        "bin": "Thùng Xanh Lá (Hữu Cơ)",
        "color": (46, 139, 87),      # Xanh lá cây
        "clip_desc": "organic biological food waste or compost scrap",
        "action": "Ủ làm phân bón hữu cơ hoặc thức ăn chăn nuôi gia súc."
    },
    "Cardboard Paper": {
        "vi": "Bìa Carton",
        "category": "Rác Tái Chế",
        "bin": "Thùng Xanh Dương / Trắng (Tái Chế)",
        "color": (210, 105, 30),     # Nâu cam
        "clip_desc": "a brown cardboard box or corrugated carton",
        "action": "Gấp gọn phẳng, giữ khô ráo để chuyển về nhà máy tái chế giấy."
    },
    "Glass": {
        "vi": "Mảnh thủy tinh vỡ",
        "category": "Rác Tái Chế",
        "bin": "Thùng Xanh Dương (Tái Chế)",
        "color": (0, 180, 216),      # Xanh ngọc
        "clip_desc": "a broken glass shard or glass material",
        "action": "Bọc cẩn thận trong giấy/hộp để tránh gây đứt tay người gom rác."
    },
    "Glass-Bottle": {
        "vi": "Chai thủy tinh",
        "category": "Rác Tái Chế",
        "bin": "Thùng Xanh Dương (Tái Chế)",
        "color": (0, 150, 199),      # Xanh dương đậm
        "clip_desc": "a glass bottle or glass jar",
        "action": "Tráng sạch cặn nước bên trong, tái sử dụng hoặc tái chế 100%."
    },
    "Glove": {
        "vi": "Găng tay cao su / Y tế",
        "category": "Rác Nguy Hại",
        "bin": "Thùng Vàng / Cam (Rác Y Tế)",
        "color": (255, 140, 0),      # Cam
        "clip_desc": "a rubber glove or protective latex glove",
        "action": "Rác thải có nguy cơ lây nhiễm vi khuẩn, bỏ thùng rác kín có nắp đậy."
    },
    "HDPE Plastic": {
        "vi": "Chai nhựa HDPE (Can/dầu gội)",
        "category": "Rác Tái Chế",
        "bin": "Thùng Xanh Dương (Tái Chế)",
        "color": (72, 149, 239),     # Xanh lam
        "clip_desc": "a thick HDPE plastic bottle, shampoo bottle or detergent jug",
        "action": "Đổ sạch dung dịch thừa, tháo nắp, ép bẹp chuyển cơ sở tái chế."
    },
    "Mask": {
        "vi": "Khẩu trang y tế",
        "category": "Rác Nguy Hại",
        "bin": "Thùng Vàng / Cam (Rác Y Tế)",
        "color": (255, 75, 75),      # Đỏ cam
        "clip_desc": "a disposable blue medical face mask",
        "action": "Cắt quai đeo để bảo vệ chim muông hoang dã, vứt thùng rác y tế."
    },
    "Medicine": {
        "vi": "Thuốc men y tế",
        "category": "Rác Nguy Hại",
        "bin": "Thùng Đỏ / Cam (Nguy Hại)",
        "color": (220, 20, 60),      # Đỏ mận
        "clip_desc": "a medical pill blister pack or medicine tablets",
        "action": "Dược phẩm hết hạn/dư thừa, không xả vào bồn cầu gây ô nhiễm nguồn nước."
    },
    "Newspaper": {
        "vi": "Báo chí / Tạp chí",
        "category": "Rác Tái Chế",
        "bin": "Thùng Xanh Dương / Trắng (Tái Chế)",
        "color": (67, 97, 238),      # Xanh đậm
        "clip_desc": "a printed paper newspaper with text and articles",
        "action": "Buộc thành bó phẳng, tránh để ngấm nước bẩn, chuyển đi tái chế."
    },
    "PET Bottle": {
        "vi": "Chai nhựa PET (Nước uống)",
        "category": "Rác Tái Chế",
        "bin": "Thùng Xanh Dương (Tái Chế)",
        "color": (63, 55, 201),      # Tím xanh
        "clip_desc": "a transparent plastic drink bottle or PET water bottle",
        "action": "Đổ sạch nước, tháo nhãn màng nilon và vặn chặt nắp/ép dẹp."
    },
    "Paper": {
        "vi": "Giấy vụn / Giấy in",
        "category": "Rác Tái Chế",
        "bin": "Thùng Xanh Dương / Trắng (Tái Chế)",
        "color": (76, 201, 240),     # Xanh da trời
        "clip_desc": "crumpled waste paper or scrap paper",
        "action": "Thu gom tái chế bột giấy mới, tiết kiệm tài nguyên rừng."
    },
    "Single-Use-Plastic": {
        "vi": "Túi nilon dùng 1 lần",
        "category": "Rác Vô Cơ / Khác",
        "bin": "Thùng Xám / Đen (Rác Còn Lại)",
        "color": (140, 140, 140),    # Xám
        "clip_desc": "a thin plastic shopping bag or flimsy grocery bag",
        "action": "Hạn chế sử dụng, nếu dính bẩn bỏ vào rác chôn lấp/xử lý nhiệt."
    },
    "Single-layer Plastic": {
        "vi": "Màng nhựa bọc thực phẩm",
        "category": "Rác Vô Cơ / Khác",
        "bin": "Thùng Xám / Đen (Rác Còn Lại)",
        "color": (160, 160, 160),    # Xám nhạt
        "clip_desc": "single-layer plastic film, clear food wrap or wrapper",
        "action": "Không thể tái chế kinh tế, phân loại vào rác vô cơ khó phân hủy."
    },
    "Squeeze Tube": {
        "vi": "Tuýp bóp (Kem/Mỹ phẩm)",
        "category": "Rác Vô Cơ / Khác",
        "bin": "Thùng Xám / Đen (Rác Còn Lại)",
        "color": (115, 115, 115),    # Xám chì
        "clip_desc": "a flexible plastic squeeze tube for cream or toothpaste",
        "action": "Nhựa phức hợp nhiều lớp, phân loại vào rác vô cơ chôn lấp."
    },
    "syringe": {
        "vi": "Bơm kim tiêm",
        "category": "Rác Nguy Hại",
        "bin": "Hộp Kháng Thủng Y Tế (Nguy Hại)",
        "color": (255, 0, 0),        # Đỏ tươi
        "clip_desc": "a plastic medical syringe or needle",
        "action": "Vật sắc nhọn truyền nhiễm cực kỳ nguy hiểm, phải bỏ vào hộp kháng thủng."
    },
    "Clean White Paper": {
        "vi": "Giấy A4 trắng sạch",
        "category": "Rác Tái Chế",
        "bin": "Thùng Xanh Dương / Trắng (Tái Chế)",
        "color": (144, 224, 239),    # Lam nhạt
        "clip_desc": "a flat clean white A4 paper sheet or document",
        "action": "Giấy chất lượng cao, ưu tiên hàng đầu cho việc tái chế văn phòng phẩm."
    },
    "Background": {
        "vi": "Không phải rác / Nền",
        "category": "—",
        "bin": "Không xác định",
        "color": (200, 200, 200),
        "clip_desc": "a human hand, a table, a floor or an empty background",
        "action": "Bỏ qua phát hiện."
    }
}

PROMPT_TEMPLATES = [
    "a photo of {}.",
    "a close-up photo of {}.",
    "a cropped image of {}."
]

# ----------------------------------------------------------------------------
# 3. QUẢN LÝ TẢI MÔ HÌNH (LAZY LOAD & CACHE)
# ----------------------------------------------------------------------------
YOLO_MODEL = None
CLIP_MODEL = None
CLIP_PREPROCESS = None
CLIP_TEXT_FEATURES = None
CLIP_KEYS = list(WASTE_DATA.keys())
FONT = None


def load_font(size=18):
    global FONT
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def get_yolo():
    global YOLO_MODEL
    if YOLO_MODEL is None:
        print(f"[*] Nạp mô hình YOLOv12: {MODEL_PATH} trên {DEVICE}...")
        YOLO_MODEL = YOLO(str(MODEL_PATH))
    return YOLO_MODEL


def get_clip():
    global CLIP_MODEL, CLIP_PREPROCESS, CLIP_TEXT_FEATURES
    if CLIP_MODEL is None:
        print(f"[*] Nạp mô hình OpenCLIP ViT-B/32 trên {CLIP_DEVICE}...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=CLIP_DEVICE,
            precision="fp16" if torch.cuda.is_available() else "fp32"
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        feats = []
        with torch.inference_mode():
            for k in CLIP_KEYS:
                desc = WASTE_DATA[k]["clip_desc"]
                tokens = tokenizer([t.format(desc) for t in PROMPT_TEMPLATES]).to(CLIP_DEVICE)
                f = model.encode_text(tokens)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.mean(dim=0) / f.mean(dim=0).norm())

        CLIP_MODEL = model
        CLIP_PREPROCESS = preprocess
        CLIP_TEXT_FEATURES = torch.stack(feats)
        print("[+] OpenCLIP đã sẵn sàng trích xuất embedding.")
    return CLIP_MODEL, CLIP_PREPROCESS, CLIP_TEXT_FEATURES


# ----------------------------------------------------------------------------
# 4. TRÍCH XUẤT ẢNH CẮT & PHÂN LOẠI CLIP
# ----------------------------------------------------------------------------
def crop_and_pad(img_rgb, box, polygon=None, pad_ratio=0.05):
    """Cắt vùng vật thể và tạo ảnh vuông đệm nền xám cho CLIP."""
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    crop = img_rgb[y1:y2, x1:x2].copy()
    if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
        return None

    if polygon is not None and len(polygon) >= 3:
        mask = np.zeros(crop.shape[:2], np.uint8)
        poly = (polygon - np.array([x1, y1])).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)
        crop = np.where(mask[..., None] > 0, crop, np.uint8(127))

    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    canvas = np.full((side, side, 3), 127, np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop
    return canvas


@torch.inference_mode()
def classify_with_clip(crop_list_rgb):
    """Đưa danh sách crop qua OpenCLIP tính xác suất softmax."""
    model, preprocess, text_feats = get_clip()
    tensors = [preprocess(Image.fromarray(c)) for c in crop_list_rgb]
    batch = torch.stack(tensors).to(CLIP_DEVICE, dtype=CLIP_DTYPE)

    img_f = model.encode_image(batch)
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp().float().item()

    probs = (logit_scale * img_f.float() @ text_feats.float().T).softmax(dim=-1)
    top_p, top_i = probs.max(dim=-1)
    return [(CLIP_KEYS[i], p) for i, p in zip(top_i.tolist(), top_p.tolist())]


# ----------------------------------------------------------------------------
# 5. VẼ KẾT QUẢ ĐỒ HỌA ĐẸP MẮT (BOX, MASK, CHỮ TIẾNG VIỆT CÓ DẤU)
# ----------------------------------------------------------------------------
def draw_results(image_rgb, detections):
    """Vẽ bounding boxes, segmentation masks trong suốt và nhãn tiếng Việt."""
    overlay = image_rgb.copy()
    h, w = image_rgb.shape[:2]

    # 1. Vẽ Mask bán trong suốt (Alpha Blending)
    for det in detections:
        poly = det.get("poly")
        color = det["color"]
        if poly is not None and len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], color)

    blended = cv2.addWeighted(overlay, 0.35, image_rgb, 0.65, 0)

    # 2. Vẽ viền đa giác và viền Bounding Box
    for det in detections:
        color = det["color"]
        poly = det.get("poly")
        if poly is not None and len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(blended, [pts], isClosed=True, color=color, thickness=2)

        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)

    # 3. Vẽ nhãn bằng PIL để hỗ trợ tiếng Việt có dấu rõ nét
    pil_img = Image.fromarray(blended)
    draw = ImageDraw.Draw(pil_img)
    font = load_font(15)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        tag_main = f"{det['label_vi']} ({det['conf']:.0%})"
        tag_sub = f"{det['category']} | {det['clip_badge']}" if det.get("clip_badge") else det["category"]

        # Tính kích thước khung chứa nhãn
        bbox1 = draw.textbbox((0, 0), tag_main, font=font)
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]

        bbox2 = draw.textbbox((0, 0), tag_sub, font=font)
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]

        tag_w = max(w1, w2) + 12
        tag_h = h1 + h2 + 10

        # Tọa độ nhãn: ưu tiên đặt phía trên BBox, nếu chật thì đưa xuống dưới
        tx = max(0, min(x1, w - tag_w))
        ty = y1 - tag_h - 4
        if ty < 0:
            ty = min(y1 + 4, h - tag_h)

        # Vẽ nền đen bán trong suốt cho nhãn
        draw.rectangle([tx, ty, tx + tag_w, ty + tag_h], fill=(20, 20, 20))
        # Viền màu phân loại cho nhãn
        draw.rectangle([tx, ty, tx + tag_w, ty + tag_h], outline=det["color"], width=1)
        # Chữ dòng 1 (Tên vật thể + Độ tin cậy)
        draw.text((tx + 6, ty + 3), tag_main, font=font, fill=(255, 255, 255))
        # Chữ dòng 2 (Nhóm phân loại & Thùng rác)
        draw.text((tx + 6, ty + h1 + 5), tag_sub, font=font, fill=det["color"])

    return np.array(pil_img)


# ----------------------------------------------------------------------------
# 6. HÀM XỬ LÝ SUY LUẬN CHÍNH (ẢNH, VIDEO, WEBCAM)
# ----------------------------------------------------------------------------
def process_single_frame(frame_rgb, conf_thr=0.35, iou_thr=0.5, use_clip=True, clip_thr=0.5, render=True):
    """Hàm lõi suy luận cho 1 frame ảnh RGB."""
    yolo = get_yolo()
    results = yolo.predict(
        source=np.ascontiguousarray(frame_rgb[..., ::-1]),  # Ultralytics nhận numpy theo thứ tự BGR
        conf=conf_thr,
        iou=iou_thr,
        imgsz=640,
        device=DEVICE,
        half=(DEVICE == "cuda"),
        verbose=False
    )[0]

    detections = []
    crops_to_clip = []
    crop_indices = []

    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        polys = results.masks.xy if results.masks is not None else [None] * len(boxes)

        for i, box in enumerate(boxes):
            y_cls_name = results.names.get(int(cls_ids[i]), str(cls_ids[i]))
            matched_key = y_cls_name if y_cls_name in WASTE_DATA else "Background"
            meta = WASTE_DATA.get(matched_key, WASTE_DATA["Background"])

            det = {
                "box": box,
                "poly": polys[i],
                "yolo_class": y_cls_name,
                "conf": float(confs[i]),
                "label_vi": meta["vi"],
                "category": meta["category"],
                "bin": meta["bin"],
                "action": meta["action"],
                "color": meta["color"],
                "clip_label": None,
                "clip_prob": 0.0,
                "clip_badge": None,
                "final_label": meta["vi"]
            }
            detections.append(det)

            if use_clip:
                crop = crop_and_pad(frame_rgb, box, polys[i])
                if crop is not None:
                    crops_to_clip.append(crop)
                    crop_indices.append(i)

        # Chạy OpenCLIP theo lô (Batch)
        if use_clip and crops_to_clip:
            try:
                clip_results = classify_with_clip(crops_to_clip)
                for idx, (c_key, c_prob) in zip(crop_indices, clip_results):
                    c_meta = WASTE_DATA.get(c_key, WASTE_DATA["Background"])
                    detections[idx]["clip_label"] = c_meta["vi"]
                    detections[idx]["clip_prob"] = float(c_prob)
                    detections[idx]["clip_badge"] = f"CLIP: {c_meta['vi']} ({c_prob:.0%})"

                    # Nếu CLIP có độ tự tin cao và phát hiện ra nhãn Zero-shot (Lon kim loại, Giấy A4)
                    if c_key in ["Metal Can", "Clean White Paper"] and c_prob >= clip_thr:
                        detections[idx]["label_vi"] = c_meta["vi"]
                        detections[idx]["category"] = c_meta["category"]
                        detections[idx]["bin"] = c_meta["bin"]
                        detections[idx]["action"] = c_meta["action"]
                        detections[idx]["color"] = c_meta["color"]
                        detections[idx]["final_label"] = f"{c_meta['vi']} (Zero-shot)"
            except Exception as e:
                print(f"[!] Lỗi khi chạy OpenCLIP: {e}")

    annotated = draw_results(frame_rgb, detections) if render else frame_rgb
    return annotated, detections


def run_image_inference(input_image, conf_thr, iou_thr, use_clip, clip_thr):
    """Callback xử lý tab Ảnh."""
    if input_image is None:
        return None, pd.DataFrame(), "Vui lòng tải lên một hình ảnh để phân tích!"

    t0 = time.time()
    annotated_img, detections = process_single_frame(
        input_image, conf_thr, iou_thr, use_clip, clip_thr
    )
    elapsed = (time.time() - t0) * 1000

    if not detections:
        summary_md = f"""### Kết quả phân tích:
- **Số lượng rác phát hiện:** 0 vật thể.
- **Thời gian xử lý:** `{elapsed:.1f} ms`
- *(Gợi ý: Thử hạ thanh trượt 'Ngưỡng tin cậy YOLO' nếu vật thể mờ hoặc nhỏ)*
"""
        return annotated_img, pd.DataFrame(), summary_md

    # Lập bảng kết quả chi tiết
    table_rows = []
    cat_counts = {}
    for i, d in enumerate(detections, 1):
        cat = d["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

        clip_str = f"{d['clip_label']} ({d['clip_prob']:.0%})" if d["clip_label"] else "Tắt"
        table_rows.append({
            "STT": i,
            "Tên Vật Thể": d["label_vi"],
            "Nhóm Rác": d["category"],
            "Độ tin cậy YOLO": f"{d['conf']:.1%}",
            "Xác thực OpenCLIP": clip_str,
            "Thùng rác khuyến nghị": d["bin"],
            "Hướng dẫn xử lý": d["action"]
        })

    df = pd.DataFrame(table_rows)

    # Thống kê tổng hợp
    breakdown_text = " | ".join([f"**{k}:** {v}" for k, v in cat_counts.items()])
    summary_md = f"""### Tổng kết nhận dạng:
- **Tổng số rác phát hiện:** **{len(detections)}** vật thể.
- **Phân bổ danh mục:** {breakdown_text}
- **Thời gian suy luận:** `{elapsed:.1f} ms` (~`{1000/max(elapsed, 1):.1f} FPS`)
- **Mô hình kích hoạt:** `YOLOv12s-seg` {'+ `OpenCLIP Zero-shot`' if use_clip else '(Thuần YOLO)'}
"""
    return annotated_img, df, summary_md


def run_video_inference(video_path, conf_thr, iou_thr, use_clip, clip_thr, progress=gr.Progress()):
    """Callback xử lý tab Video (H.264 MP4 export)."""
    if not video_path:
        return None, pd.DataFrame(), "Vui lòng tải lên một file video!"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, pd.DataFrame(), "Không thể đọc file video!"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = ROOT_DIR / "runs" / "output_video.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Chuẩn hóa kích thước video nếu quá lớn để đảm bảo tốc độ
    target_w, target_h = width, height
    if max(width, height) > 1280:
        scale = 1280.0 / max(width, height)
        target_w, target_h = int(width * scale), int(height * scale)

    writer = imageio.get_writer(
        str(out_path),
        format="ffmpeg",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p"
    )

    frame_count = 0
    all_detections_count = {}
    cat_summary = {}
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            if target_w != width:
                frame = cv2.resize(frame, (target_w, target_h))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Xử lý frame (Đối với video dài, chạy CLIP mỗi 5 frame để giữ tốc độ)
            do_clip_this_frame = use_clip and (frame_count % 3 == 0)
            annotated_frame, dets = process_single_frame(
                frame_rgb, conf_thr, iou_thr, do_clip_this_frame, clip_thr
            )

            writer.append_data(annotated_frame)

            for d in dets:
                name = d["label_vi"]
                all_detections_count[name] = all_detections_count.get(name, 0) + 1
                cat = d["category"]
                cat_summary[cat] = cat_summary.get(cat, 0) + 1

            if total_frames > 0:
                progress(frame_count / total_frames, desc=f"Đang xử lý frame {frame_count}/{total_frames}...")

    finally:
        cap.release()
        writer.close()

    total_time = time.time() - t0
    avg_fps = frame_count / max(total_time, 0.001)

    # Lập bảng tổng hợp
    table_rows = []
    for name, cnt in sorted(all_detections_count.items(), key=lambda x: x[1], reverse=True):
        meta = next((v for v in WASTE_DATA.values() if v["vi"] == name), WASTE_DATA["Background"])
        table_rows.append({
            "Loại Rác": name,
            "Nhóm Rác": meta["category"],
            "Tổng lượt xuất hiện (khung hình)": cnt,
            "Thùng rác khuyến nghị": meta["bin"]
        })
    df = pd.DataFrame(table_rows)

    summary_md = f"""### 🎬 Hoàn tất xử lý video:
- **Tổng số khung hình đã duyệt:** `{frame_count}` frames
- **Thời gian xử lý:** `{total_time:.2f} s` (Tốc độ trung bình: `{avg_fps:.1f} FPS`)
- **Phân bổ theo nhóm rác:** {', '.join([f"**{k}**: {v}" for k, v in cat_summary.items()])}
"""
    return str(out_path), df, summary_md


# ----------------------------------------------------------------------------
# 6b. WEBCAM TRỰC TIẾP: <video> gốc + <canvas> overlay (1 khung hình duy nhất)
# ----------------------------------------------------------------------------
# Kiến trúc:
#   - Trình duyệt phát webcam bằng thẻ <video> ở tốc độ gốc (30+ FPS, không qua server)
#   - Mỗi lượt chỉ gửi 1 frame (JPEG nhỏ) lên /api/webcam, nhận JSON kết quả, vẽ lên <canvas>
#   - Luôn gửi frame MỚI NHẤT, không xếp hàng đợi => không tích lũy độ trễ
CAM_PORT = int(os.environ.get("PORT", 7860)) + 1
CLIP_EVERY = 3            # OpenCLIP chạy mỗi N frame, các frame còn lại dùng lại kết quả theo IoU
_CAM_LOCK = threading.Lock()
_CAM_STATE = {"n": 0, "prev": []}


def _iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _poly_norm(poly, w, h):
    """Đơn giản hóa đa giác và chuẩn hóa về [0, 1] để canvas vẽ ở mọi kích thước."""
    if poly is None or len(poly) < 3:
        return None
    p = np.ascontiguousarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    p = cv2.approxPolyDP(p, 1.5, True).reshape(-1, 2)
    if len(p) < 3:
        return None
    return [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in p]


def _cam_apply_clip(d, key, prob, clip_thr):
    meta = WASTE_DATA.get(key, WASTE_DATA["Background"])
    d["clip_key"], d["clip_prob"] = key, float(prob)
    d["clip"] = f"{meta['vi']} ({prob:.0%})"
    d["zero_shot"] = False
    if key in ("Metal Can", "Clean White Paper") and prob >= clip_thr:
        d["label"], d["category"] = meta["vi"], meta["category"]
        d["bin"], d["action"], d["color"] = meta["bin"], meta["action"], list(meta["color"])
        d["zero_shot"] = True


def webcam_detect(jpeg_bytes, conf_thr, iou_thr, use_clip, clip_thr):
    t0 = time.perf_counter()
    bgr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return {"error": "frame_khong_hop_le", "dets": []}
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with _CAM_LOCK:
        yolo = get_yolo()
        # Ultralytics coi mảng numpy là BGR => truyền thẳng BGR
        res = yolo.predict(source=bgr, conf=conf_thr, iou=iou_thr, imgsz=640,
                           device=DEVICE, half=(DEVICE == "cuda"), verbose=False)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            polys = res.masks.xy if res.masks is not None else [None] * len(boxes)
            for i, box in enumerate(boxes):
                name = res.names.get(int(cls_ids[i]), str(cls_ids[i]))
                meta = WASTE_DATA.get(name, WASTE_DATA["Background"])
                dets.append({
                    "_box": box, "_poly_px": polys[i], "yolo_class": name,
                    "conf": float(confs[i]), "label": meta["vi"], "category": meta["category"],
                    "bin": meta["bin"], "action": meta["action"], "color": list(meta["color"]),
                    "clip": None, "clip_key": None, "clip_prob": 0.0, "zero_shot": False,
                })

        # ---- OpenCLIP: chạy mỗi CLIP_EVERY frame, frame khác dùng lại kết quả frame trước ----
        if use_clip and dets:
            run_clip = (_CAM_STATE["n"] % CLIP_EVERY == 0)
            need = []
            for i, d in enumerate(dets):
                reused = False
                if not run_clip:
                    best = max(
                        ((_iou_xyxy(d["_box"], pb), pk, pp) for pb, pc, pk, pp in _CAM_STATE["prev"]
                         if pc == d["yolo_class"] and pk is not None),
                        default=(0.0, None, 0.0), key=lambda t: t[0])
                    if best[0] > 0.3:
                        _cam_apply_clip(d, best[1], best[2], clip_thr)
                        reused = True
                if not reused:
                    crop = crop_and_pad(rgb, d["_box"], d["_poly_px"])
                    if crop is not None:
                        need.append((i, crop))
            if need:
                try:
                    out = classify_with_clip([c for _, c in need])
                    for (i, _), (key, prob) in zip(need, out):
                        _cam_apply_clip(dets[i], key, prob, clip_thr)
                except Exception as e:
                    print(f"[!] Lỗi OpenCLIP (webcam): {e}")

        _CAM_STATE["n"] += 1
        _CAM_STATE["prev"] = [(d["_box"], d["yolo_class"], d["clip_key"], d["clip_prob"]) for d in dets]

    payload = []
    for d in dets:
        x1, y1, x2, y2 = d["_box"]
        payload.append({
            "box": [x1 / w, y1 / h, x2 / w, y2 / h],
            "poly": _poly_norm(d["_poly_px"], w, h),
            "label": d["label"], "category": d["category"], "bin": d["bin"], "action": d["action"],
            "color": d["color"], "conf": d["conf"], "clip": d["clip"], "zero_shot": d["zero_shot"],
        })
    return {"w": w, "h": h, "ms": round((time.perf_counter() - t0) * 1000, 1), "dets": payload}


WEBCAM_PAGE = r"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Webcam Trực Tiếp</title>
<style>
  :root{--bg:#0b1220;--card:#111a2e;--line:#22304d;--txt:#e5edf9;--mut:#8da2c4;--ok:#10b981;--warn:#f59e0b;--bad:#ef4444}
  *{box-sizing:border-box} html,body{margin:0;background:transparent;color:var(--txt);font-family:"Segoe UI",system-ui,Arial,sans-serif}
  .wrap{max-width:880px;margin:0 auto;padding:4px 4px 16px}
  #stage{position:relative;width:100%;aspect-ratio:16/9;background:#0f172a;border:1px solid var(--line);border-radius:14px;overflow:hidden}
  #video,#overlay{position:absolute;inset:0;width:100%;height:100%}
  #video{object-fit:contain;background:#000} #overlay{pointer-events:none}
  #idle{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;background:#0f172a;cursor:pointer;z-index:3}
  #idle .big{font-size:1.25rem;font-weight:700;color:var(--ok)} #idle .sub{color:var(--mut);font-size:.95rem}
  #idle .icon{font-size:3rem}
  #badge{position:absolute;top:10px;left:10px;z-index:2;padding:4px 10px;border-radius:999px;font-size:.8rem;font-weight:600;background:rgba(0,0,0,.55);color:#fff;display:none}
  #badge.live::before{content:"● ";color:var(--bad)} #badge.pause{color:var(--warn)}
  .bar{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
  button{font:inherit;font-weight:600;border:0;border-radius:10px;padding:10px 16px;cursor:pointer;color:#fff;background:#334155}
  button:disabled{opacity:.4;cursor:not-allowed}
  #bStart{background:var(--ok)} #bPause{background:var(--warn);color:#1a1200} #bStop{background:var(--bad)}
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-bottom:12px}
  .stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px}
  .stat b{display:block;font-size:1.25rem} .stat span{color:var(--mut);font-size:.78rem}
  .cfg{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:12px;font-size:.88rem}
  .cfg label{display:flex;flex-direction:column;gap:4px;color:var(--mut)} .cfg input[type=range]{width:100%}
  .cfg .chk{flex-direction:row;align-items:center;gap:8px;color:var(--txt)}
  #msg{color:var(--warn);font-size:.9rem;min-height:1.2em;margin-bottom:8px}
  table{width:100%;border-collapse:collapse;font-size:.85rem;background:var(--card);border-radius:10px;overflow:hidden}
  th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top} th{color:var(--mut);font-weight:600;background:#0e1628}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
</style></head><body>
<div class="wrap">
  <div id="stage">
    <video id="video" muted playsinline></video>
    <canvas id="overlay"></canvas>
    <div id="badge"></div>
    <div id="idle"><div class="big">Nhấp để bật Webcam</div><div class="sub">Trình duyệt sẽ hỏi quyền truy cập camera, hãy chọn “Cho phép”</div></div>
  </div>
  <div class="bar">
    <button id="bStart">▶ Bật Webcam</button>
    <button id="bPause" disabled>⏸ Tạm dừng</button>
    <button id="bStop" disabled>⏹ Tắt Webcam</button>
  </div>
  <div id="msg"></div>
  <div class="stats">
    <div class="stat"><b id="sFps">–</b><span>Tốc độ nhận diện (FPS)</span></div>
    <div class="stat"><b id="sMs">–</b><span>Thời gian xử lý / frame</span></div>
    <div class="stat"><b id="sCnt">0</b><span>Vật thể trong khung hình</span></div>
  </div>
  <div class="cfg">
    <label>Ngưỡng tin cậy YOLO: <b id="vConf">0.35</b><input id="conf" type="range" min="0.1" max="0.9" step="0.05" value="0.35"></label>
    <label>Ngưỡng IoU NMS: <b id="vIou">0.50</b><input id="iou" type="range" min="0.1" max="0.9" step="0.05" value="0.5"></label>
    <label class="chk"><input id="clip" type="checkbox" checked> Bật OpenCLIP (xác thực + Zero-shot)</label>
    <label>Ngưỡng CLIP Zero-shot: <b id="vClipThr">0.50</b><input id="clipThr" type="range" min="0.3" max="0.9" step="0.05" value="0.5"></label>
  </div>
  <table><thead><tr><th>#</th><th>Vật thể</th><th>Nhóm</th><th>YOLO</th><th>OpenCLIP</th><th>Thùng rác</th></tr></thead><tbody id="tbody"></tbody></table>
</div>
<script>
const $ = id => document.getElementById(id);
const video=$('video'), canvas=$('overlay'), ctx=canvas.getContext('2d'), stage=$('stage');
const cap=document.createElement('canvas'), capCtx=cap.getContext('2d');
let stream=null, running=false, state='idle', loopId=0, times=[];

function setState(s){
  state=s;
  $('idle').style.display = s==='idle' ? 'flex' : 'none';
  $('bStart').disabled = s!=='idle';
  $('bPause').disabled = s==='idle';
  $('bStop').disabled  = s==='idle';
  $('bPause').textContent = s==='paused' ? '▶ Tiếp tục' : '⏸ Tạm dừng';
  const b=$('badge'); b.style.display = s==='idle' ? 'none':'block';
  b.className = s==='live' ? 'live' : 'pause';
  b.textContent = s==='live' ? 'ĐANG PHÁT TRỰC TIẾP' : 'ĐÃ TẠM DỪNG';
}
function msg(t){ $('msg').textContent = t || ''; }
function clearOverlay(){ ctx.clearRect(0,0,canvas.width,canvas.height); }
function resetStats(){ $('sFps').textContent='–'; $('sMs').textContent='–'; $('sCnt').textContent='0'; $('tbody').innerHTML=''; times=[]; }

async function startCam(){
  msg('');
  if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
    msg('Trình duyệt không hỗ trợ webcam hoặc trang chưa chạy trên localhost/HTTPS.'); return;
  }
  try{
    stream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}},audio:false});
  }catch(e){
    msg('Không mở được webcam: ' + (e.name==='NotAllowedError' ? 'bạn chưa cấp quyền camera.' : e.message)); return;
  }
  video.srcObject = stream;
  await video.play();
  await new Promise(r => { if(video.videoWidth) r(); else video.onloadedmetadata = () => r(); });
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  stage.style.aspectRatio = video.videoWidth + '/' + video.videoHeight;
  setState('live'); startLoop();
}
function pauseResume(){
  if(state==='live'){ running=false; video.pause(); setState('paused'); }
  else if(state==='paused'){ video.play(); setState('live'); startLoop(); }
}
function stopCam(){
  running=false; loopId++;
  if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; }
  video.srcObject=null; clearOverlay(); resetStats(); msg(''); setState('idle');
}

function startLoop(){ running=true; const id=++loopId; loop(id); }
async function loop(id){
  let fails=0;
  while(running && id===loopId){
    if(!video.videoWidth){ await new Promise(r=>requestAnimationFrame(r)); continue; }
    const s = Math.min(1, 640/Math.max(video.videoWidth, video.videoHeight));
    cap.width = Math.round(video.videoWidth*s); cap.height = Math.round(video.videoHeight*s);
    capCtx.drawImage(video,0,0,cap.width,cap.height);
    const blob = await new Promise(r=>cap.toBlob(r,'image/jpeg',0.65));
    const q = new URLSearchParams({conf:$('conf').value, iou:$('iou').value, clip:$('clip').checked?1:0, clip_thr:$('clipThr').value});
    const t0=performance.now();
    try{
      const r = await fetch('/api/webcam?'+q, {method:'POST', body:blob, headers:{'Content-Type':'image/jpeg'}});
      if(!r.ok) throw new Error('HTTP '+r.status);
      const data = await r.json();
      if(!(running && id===loopId)) break;
      fails=0; msg(''); render(data, performance.now()-t0);
    }catch(e){
      fails++; clearOverlay(); msg('Mất kết nối tới bộ nhận diện, đang thử lại… ('+e.message+')');
      await new Promise(r=>setTimeout(r, Math.min(1500, 200*fails)));
    }
  }
}

function render(data, rtt){
  const now=performance.now(); times.push(now); while(times.length && now-times[0]>2000) times.shift();
  const fps = times.length>1 ? (times.length-1)/((times[times.length-1]-times[0])/1000) : 0;
  $('sFps').textContent = fps.toFixed(1);
  $('sMs').textContent = Math.round(rtt)+' ms';
  $('sCnt').textContent = data.dets.length;
  drawDets(data.dets); fillTable(data.dets);
}

function drawDets(dets){
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);
  const fs=Math.max(13, Math.round(W/55));
  ctx.font='600 '+fs+'px "Segoe UI",Arial,sans-serif'; ctx.textBaseline='top'; ctx.lineJoin='round';
  for(const d of dets){
    const [r,g,b]=d.color, col='rgb('+r+','+g+','+b+')';
    if(d.poly){
      ctx.beginPath();
      d.poly.forEach(([x,y],i)=> i ? ctx.lineTo(x*W,y*H) : ctx.moveTo(x*W,y*H));
      ctx.closePath(); ctx.fillStyle='rgba('+r+','+g+','+b+',0.35)'; ctx.fill();
      ctx.lineWidth=2; ctx.strokeStyle=col; ctx.stroke();
    }
    const x1=d.box[0]*W, y1=d.box[1]*H, x2=d.box[2]*W, y2=d.box[3]*H;
    ctx.lineWidth=2; ctx.strokeStyle=col; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    const l1 = d.label+' ('+Math.round(d.conf*100)+'%)';
    const l2 = d.category + (d.clip ? ' | CLIP: '+d.clip : '') + (d.zero_shot ? ' (Zero-shot)' : '');
    const w1=ctx.measureText(l1).width, w2=ctx.measureText(l2).width;
    const tw=Math.max(w1,w2)+12, th=fs*2+10;
    let tx=Math.max(0, Math.min(x1, W-tw)), ty=y1-th-3; if(ty<0) ty=Math.min(y1+3, H-th);
    ctx.fillStyle='rgba(20,20,20,0.88)'; ctx.fillRect(tx,ty,tw,th);
    ctx.lineWidth=1; ctx.strokeStyle=col; ctx.strokeRect(tx,ty,tw,th);
    ctx.fillStyle='#fff'; ctx.fillText(l1, tx+6, ty+3);
    ctx.fillStyle=col;   ctx.fillText(l2, tx+6, ty+fs+6);
  }
}

function fillTable(dets){
  const tb=$('tbody'); tb.innerHTML='';
  dets.forEach((d,i)=>{
    const tr=document.createElement('tr');
    const cells=[i+1, null, d.category, Math.round(d.conf*100)+'%', d.clip||'Tắt', d.bin];
    cells.forEach((v,j)=>{
      const td=document.createElement('td');
      if(j===1){ const dot=document.createElement('span'); dot.className='dot'; dot.style.background='rgb('+d.color.join(',')+')'; td.appendChild(dot); td.appendChild(document.createTextNode(d.label)); }
      else td.textContent=v;
      tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
}

$('idle').onclick=startCam; $('bStart').onclick=startCam; $('bPause').onclick=pauseResume; $('bStop').onclick=stopCam;
[['conf','vConf'],['iou','vIou'],['clipThr','vClipThr']].forEach(([a,b])=>{ $(a).oninput=()=>{ $(b).textContent=(+$(a).value).toFixed(2); }; });
window.addEventListener('beforeunload', ()=>{ if(stream) stream.getTracks().forEach(t=>t.stop()); });
setState('idle');
</script></body></html>
"""


@server_app.get("/webcam")
async def webcam_page():
    return HTMLResponse(WEBCAM_PAGE, headers={"Cache-Control": "no-store"})


@server_app.post("/api/webcam")
async def webcam_api(request: Request, conf: float = 0.35, iou: float = 0.5,
                     clip: int = 1, clip_thr: float = 0.5):
    data = await request.body()
    result = await run_in_threadpool(webcam_detect, data, conf, iou, bool(clip), clip_thr)
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


def start_webcam_server():
    """Chạy backend webcam ở cổng riêng (luồng nền), độc lập với Gradio."""
    server = uvicorn.Server(uvicorn.Config(server_app, host="127.0.0.1", port=CAM_PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()


def warmup_models():
    """Nạp & chạy thử mô hình trước khi demo để frame đầu tiên không bị đơ."""
    print("[*] Làm nóng mô hình (YOLO + OpenCLIP)...")
    dummy = np.full((480, 640, 3), 114, np.uint8)
    get_yolo().predict(source=dummy, imgsz=640, device=DEVICE, half=(DEVICE == "cuda"), verbose=False)
    get_clip()
    classify_with_clip([np.full((224, 224, 3), 127, np.uint8)])
    print("[+] Mô hình đã sẵn sàng.")


# ----------------------------------------------------------------------------
# 7. THIẾT KẾ GIAO DIỆN GRADIO HIỆN ĐẠI
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}
.hero-box {
    text-align: center;
    padding: 24px;
    border-radius: 12px;
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}
.hero-box h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: #4ade80;
}
.hero-box p {
    font-size: 1.05rem;
    opacity: 0.9;
    margin-bottom: 0;
}
.badge-tag {
    display: inline-block;
    padding: 4px 12px;
    margin: 4px;
    border-radius: 20px;
    font-size: 0.85rem;
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.25);
}
.instruction-card {
    border-left: 4px solid #10b981;
    padding: 12px;
    background: #f0fdf4;
    border-radius: 4px;
    margin-bottom: 12px;
}

/* ------------------------------------------------------------- */
/* VIỆT HÓA HOÀN TOÀN KHU VỰC UPLOAD (ẢNH, VIDEO, WEBCAM)        */
/* ------------------------------------------------------------- */

/* 1. Xóa sạch mọi text tiếng Anh mặc định trong tất cả các khung upload */
[data-testid="upload-text"] {
    font-size: 0 !important;
    line-height: 1 !important;
}
[data-testid="upload-text"] * {
    font-size: 0 !important;
}
[data-testid="upload-text"] .or {
    display: none !important;
}
[data-testid="upload-text"] [data-testid="upload-icon"],
[data-testid="upload-text"] [data-testid="upload-icon"] * {
    font-size: initial !important;
    display: inline-block !important;
}

/* 2. Hiển thị Tiếng Việt cho khung Upload Ảnh */
#img-input [data-testid="upload-text"]::before {
    content: "Kéo thả Hình ảnh vào đây" !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    display: block !important;
    margin-top: 10px !important;
    color: #10b981 !important;
}
#img-input [data-testid="upload-text"]::after {
    content: "— hoặc nhấp để chọn ảnh từ máy tính —" !important;
    font-size: 0.95rem !important;
    display: block !important;
    margin-top: 5px !important;
    color: #0284c7 !important;
    cursor: pointer !important;
}

/* 3. Hiển thị Tiếng Việt cho khung Upload Video */
#vid-input [data-testid="upload-text"]::before {
    content: "Kéo thả Video vào đây" !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    display: block !important;
    margin-top: 10px !important;
    color: #10b981 !important;
}
#vid-input [data-testid="upload-text"]::after {
    content: "— hoặc nhấp để chọn video từ máy tính —" !important;
    font-size: 0.95rem !important;
    display: block !important;
    margin-top: 5px !important;
    color: #0284c7 !important;
    cursor: pointer !important;
}

/* 4. Hiển thị Tiếng Việt cho khung Webcam */
#cam-input [data-testid="upload-text"]::before {
    content: "Kéo thả ảnh hoặc Bật Webcam" !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    display: block !important;
    margin-top: 10px !important;
    color: #10b981 !important;
}
#cam-input [data-testid="upload-text"]::after {
    content: "— nhấp vào biểu tượng Máy ảnh để chụp —" !important;
    font-size: 0.95rem !important;
    display: block !important;
    margin-top: 5px !important;
    color: #0284c7 !important;
    cursor: pointer !important;
}

/* 5. Quy tắc áp dụng cho mọi khung upload tổng quát */
.wrap[data-testid="upload-text"]::before {
    content: "Kéo thả tệp tin vào đây" !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    display: block !important;
    margin-top: 10px !important;
    color: #10b981 !important;
}
.wrap[data-testid="upload-text"]::after {
    content: "— hoặc nhấp để tải lên từ thiết bị —" !important;
    font-size: 0.95rem !important;
    display: block !important;
    margin-top: 5px !important;
    color: #0284c7 !important;
    cursor: pointer !important;
}

/* 6. Việt hóa nút Bật Webcam ("Click to Access WebCam") */
[title="grant webcam access"] button div.wrap,
#cam-input button div.wrap {
    font-size: 0 !important;
}
[title="grant webcam access"] button div.wrap::after,
#cam-input button div.wrap::after {
    content: "Nhấp để Bật Webcam / Máy ảnh" !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #10b981 !important;
    display: block !important;
    margin-top: 10px !important;
    cursor: pointer !important;
}
"""

# Cấu hình Việt hóa (i18n) cho Gradio
VI_DICT = {
    "upload_text": {
        "drop_image": "Kéo thả Hình ảnh vào đây",
        "drop_video": "Kéo thả Video vào đây",
        "drop_file": "Kéo thả Tệp tin vào đây",
        "click_to_upload": "hoặc Nhấp để chọn tệp",
        "paste_clipboard": "Dán từ bộ nhớ tạm",
        "drop_gallery": "Kéo thả tệp vào đây"
    },
    "common": {
        "or": "hoặc",
        "clear": "Xóa",
        "record": "Bắt đầu ghi hình",
        "stop_recording": "Dừng ghi hình",
        "submit": "Gửi",
        "download": "Tải về"
    },
    "video": {
        "drop_to_upload": "Kéo thả tệp video vào đây để tải lên"
    },
    "image": {
        "drop_to_upload": "Kéo thả tệp ảnh vào đây để tải lên",
        "allow_webcam_access": "Vui lòng cấp quyền truy cập webcam"
    },
    "image_editor": {
        "upload": "Tải tệp lên",
        "upload_image": "Tải ảnh lên",
        "webcam": "Máy ảnh (Webcam)"
    }
}
i18n_config = gr.I18n(en=VI_DICT, vi=VI_DICT)

with gr.Blocks(title="Hệ Thống Phân Loại Rác Thông Minh") as demo:

    # Header banner
    gr.HTML("""
    <div class="hero-box">
        <h1>🌿 HỆ THỐNG PHÂN LOẠI RÁC THẢI THÔNG MINH</h1>
        <p>Ứng dụng Mô hình Học sâu <b>YOLOv12</b> kết hợp Mô hình Thị giác - Ngôn ngữ <b>OpenCLIP</b></p>
        <div style="margin-top: 12px;">
            <span class="badge-tag"> YOLOv12s-seg (mAP@0.5: 80.6%)</span>
            <span class="badge-tag"> OpenCLIP ViT-B/32 Zero-shot</span>
            <span class="badge-tag"> Chuẩn Phân Loại 4 Nhóm Rác</span>
        </div>
    </div>
    """)

    with gr.Tabs():
        # ====================================================================
        # TAB 1: PHÂN TÍCH HÌNH ẢNH
        # ====================================================================
        with gr.TabItem(" Phân Tích Hình Ảnh"):
            with gr.Row():
                with gr.Column(scale=5):
                    img_input = gr.Image(type="numpy", label="Ảnh Đầu Vào (Kéo thả hoặc dán ảnh)", elem_id="img-input")

                    with gr.Accordion("⚙️ Tùy Chỉnh Tham Số Nhận Dạng", open=True):
                        conf_slider = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.35, step=0.05,
                            label="Ngưỡng Tin Cậy YOLO (Confidence Threshold)"
                        )
                        iou_slider = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.5, step=0.05,
                            label="Ngưỡng IoU Non-Max Suppression"
                        )
                        use_clip_chk = gr.Checkbox(
                            value=True,
                            label="Kích hoạt OpenCLIP (Xác thực ngữ nghĩa & Zero-shot các vật thể lạ)"
                        )
                        clip_thr_slider = gr.Slider(
                            minimum=0.3, maximum=0.9, value=0.5, step=0.05,
                            label="Ngưỡng Quyết Định Zero-shot CLIP"
                        )

                    btn_detect_img = gr.Button(" Phân Tích Rác Thải", variant="primary", size="lg")

                with gr.Column(scale=7):
                    img_output = gr.Image(type="numpy", label="Ảnh Kết Quả (Phân Vùng Instance Segmentation)")
                    img_summary = gr.Markdown("###  Kết quả phân tích sẽ hiển thị tại đây")

            gr.Markdown("### 📋 Bảng Chi Tiết Vật Thể & Hướng Dẫn Phân Loại Rác Chuẩn")
            img_table = gr.Dataframe(
                headers=["STT", "Tên Vật Thể", "Nhóm Rác", "Độ tin cậy YOLO", "Xác thực OpenCLIP", "Thùng rác khuyến nghị", "Hướng dẫn xử lý"],
                interactive=False,
                wrap=True
            )

            # Ảnh mẫu có sẵn trong thư mục sample_images
            sample_dir = ROOT_DIR / "sample_images"
            if sample_dir.is_dir():
                samples = [str(p) for p in sample_dir.glob("*.jpg")]
                if samples:
                    gr.Markdown("####  Ảnh Mẫu Thử Nghiệm Nhanh:")
                    gr.Examples(
                        examples=samples,
                        inputs=img_input,
                        label="Nhấp vào ảnh để thử nghiệm"
                    )

            btn_detect_img.click(
                fn=run_image_inference,
                inputs=[img_input, conf_slider, iou_slider, use_clip_chk, clip_thr_slider],
                outputs=[img_output, img_table, img_summary]
            )

        # ====================================================================
        # TAB 2: PHÂN TÍCH VIDEO
        # ====================================================================
        with gr.TabItem("Phân Tích Video"):
            with gr.Row():
                with gr.Column(scale=5):
                    vid_input = gr.Video(label="Tải lên Video Rác Thải (.mp4, .avi, .mov)", elem_id="vid-input")
                    with gr.Accordion("Cấu Hình Video", open=False):
                        vid_conf = gr.Slider(0.1, 1.0, value=0.35, step=0.05, label="Ngưỡng YOLO Conf")
                        vid_iou = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Ngưỡng IoU")
                        vid_clip = gr.Checkbox(value=False, label="Bật OpenCLIP (Tắt để tăng tốc độ xuất video)")
                        vid_clip_thr = gr.Slider(0.3, 0.9, value=0.5, step=0.05, label="Ngưỡng CLIP")

                    btn_detect_vid = gr.Button("Bắt Đầu Xử Lý Video", variant="primary", size="lg")

                with gr.Column(scale=7):
                    vid_output = gr.Video(label="Video Kết Quả (H.264 Playback)")
                    vid_summary = gr.Markdown("### Thống kê video sẽ hiển thị sau khi hoàn tất xử lý")

            gr.Markdown("### Thống Kê Tần Suất Xuất Hiện Các Loại Rác Trong Video")
            vid_table = gr.Dataframe(
                headers=["Loại Rác", "Nhóm Rác", "Tổng lượt xuất hiện (khung hình)", "Thùng rác khuyến nghị"],
                interactive=False
            )

            btn_detect_vid.click(
                fn=run_video_inference,
                inputs=[vid_input, vid_conf, vid_iou, vid_clip, vid_clip_thr],
                outputs=[vid_output, vid_table, vid_summary]
            )

        # ====================================================================
        # TAB 3: WEBCAM TRỰC TIẾP
        # ====================================================================
        with gr.TabItem("📹 Webcam Trực Tiếp"):
            gr.Markdown(f"""
            ### 📹 Nhận Diện Trực Tiếp Trên Luồng Webcam (1 khung hình duy nhất)
            - Nhấp **Bật Webcam**, cho phép camera. Kết quả (mặt nạ, khung, nhãn) được vẽ **ngay trên luồng video**.
            - Dùng **Tạm dừng / Tiếp tục** để đóng băng khung hình, **Tắt Webcam** để tắt hẳn camera.
            - Nếu khung không hiện camera, mở trực tiếp: [http://127.0.0.1:{CAM_PORT}/webcam](http://127.0.0.1:{CAM_PORT}/webcam)
            """)
            gr.HTML(
                f'<iframe src="http://127.0.0.1:{CAM_PORT}/webcam" allow="camera; autoplay" '
                f'style="width:100%;height:1050px;border:0;border-radius:12px;"></iframe>'
            )

        # ====================================================================
        # TAB 4: KẾT QUẢ HUẤN LUYỆN MÔ HÌNH
        # ====================================================================
        with gr.TabItem("📈 Kết Quả Huấn Luyện Mô Hình"):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #134e4a, #065f46); color: white; padding: 22px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <h2 style="color: #6ee7b7; margin-top: 0; margin-bottom: 6px; font-size: 1.5rem;">📊 KẾT QUẢ THỰC NGHIỆM HUẤN LUYỆN YOLOV12s-SEG (100 EPOCHS)</h2>
                <p style="margin: 0; opacity: 0.9; font-size: 0.95rem;">Đánh giá toàn diện trên tập dữ liệu kiểm thử (Test Split - 1.085 ảnh) và tập thẩm định (Validation Split - 1.082 ảnh)</p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 16px;">
                    <div style="background: rgba(255,255,255,0.12); padding: 14px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.15);">
                        <div style="font-size: 0.85rem; opacity: 0.9;">mAP @ 0.5 (Mask)</div>
                        <div style="font-size: 1.85rem; font-weight: bold; color: #4ade80;">80.6%</div>
                        <div style="font-size: 0.75rem; opacity: 0.8; margin-top: 2px;">Vượt bài báo +1.4%</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.12); padding: 14px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.15);">
                        <div style="font-size: 0.85rem; opacity: 0.9;">Precision (Độ chuẩn xác)</div>
                        <div style="font-size: 1.85rem; font-weight: bold; color: #38bdf8;">83.5%</div>
                        <div style="font-size: 0.75rem; opacity: 0.8; margin-top: 2px;">Vượt bài báo +2.5%</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.12); padding: 14px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.15);">
                        <div style="font-size: 0.85rem; opacity: 0.9;">Recall (Độ bao phủ)</div>
                        <div style="font-size: 1.85rem; font-weight: bold; color: #fbbf24;">76.9%</div>
                        <div style="font-size: 0.75rem; opacity: 0.8; margin-top: 2px;">Vượt bài báo +2.4%</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.12); padding: 14px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.15);">
                        <div style="font-size: 0.85rem; opacity: 0.9;">F1-Score</div>
                        <div style="font-size: 1.85rem; font-weight: bold; color: #f472b6;">0.801</div>
                        <div style="font-size: 0.75rem; opacity: 0.8; margin-top: 2px;">Cân bằng tối ưu P & R</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.12); padding: 14px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.15);">
                        <div style="font-size: 0.85rem; opacity: 0.9;">Tốc độ (Inference)</div>
                        <div style="font-size: 1.85rem; font-weight: bold; color: #c084fc;">~51 FPS</div>
                        <div style="font-size: 0.75rem; opacity: 0.8; margin-top: 2px;">~19.6 ms / ảnh</div>
                    </div>
                </div>
            </div>
            """)

            gr.Markdown("### 1. Bảng So Sánh Chỉ Số Thực Nghiệm Với Bài Báo Tham Chiếu")
            comp_df = pd.DataFrame([
                {"Chỉ số đánh giá": "Loại bài toán", "Bài báo gốc (Baseline)": "Object Detection (Bounding Box)", "Mô hình đồ án (YOLOv12s-seg)": "Instance Segmentation (Polygon Mask)", "Cải tiến / Chênh lệch": "Nâng cấp bài toán phân vùng"},
                {"Chỉ số đánh giá": "Số lượng lớp (Classes)", "Bài báo gốc (Baseline)": "8 lớp", "Mô hình đồ án (YOLOv12s-seg)": "16 lớp (+ 2 Zero-shot)", "Cải tiến / Chênh lệch": "Chi tiết gấp đôi danh mục rác"},
                {"Chỉ số đánh giá": "Precision (Độ chuẩn xác)", "Bài báo gốc (Baseline)": "81.0%", "Mô hình đồ án (YOLOv12s-seg)": "83.5%", "Cải tiến / Chênh lệch": "+ 2.5%"},
                {"Chỉ số đánh giá": "Recall (Độ bao phủ)", "Bài báo gốc (Baseline)": "74.5%", "Mô hình đồ án (YOLOv12s-seg)": "76.9%", "Cải tiến / Chênh lệch": "+ 2.4%"},
                {"Chỉ số đánh giá": "mAP @ 0.5", "Bài báo gốc (Baseline)": "79.2%", "Mô hình đồ án (YOLOv12s-seg)": "80.6%", "Cải tiến / Chênh lệch": "+ 1.4% mAP"},
                {"Chỉ số đánh giá": "F1-Score", "Bài báo gốc (Baseline)": "0.776", "Mô hình đồ án (YOLOv12s-seg)": "0.801", "Cải tiến / Chênh lệch": "+ 0.025"},
                {"Chỉ số đánh giá": "Tốc độ xử lý (Inference)", "Bài báo gốc (Baseline)": "~ 55 FPS (Tesla T4)", "Mô hình đồ án (YOLOv12s-seg)": "~ 51 FPS (RTX 4050)", "Cải tiến / Chênh lệch": "Đạt chuẩn thời gian thực"}
            ])
            gr.Dataframe(value=comp_df, interactive=False, wrap=True)

            train_run_dir = ROOT_DIR / "runs" / "train_100_epochs"
            results_img_path = train_run_dir / "results.png"
            if results_img_path.is_file():
                gr.Markdown("### 2. Biểu Đồ Tiến Trình Huấn Luyện 100 Epochs (Learning Curves)")
                gr.Image(value=str(results_img_path), label="Đồ thị suy giảm Loss (Box, Mask seg, Cls, DFL) và tăng trưởng mAP qua 100 Epochs", interactive=False)

            pr_path = train_run_dir / "MaskPR_curve.png"
            f1_path = train_run_dir / "MaskF1_curve.png"
            if pr_path.is_file() and f1_path.is_file():
                gr.Markdown("### 3. Đường Cong Đánh Giá Phân Vùng Điểm Ảnh (Segmentation Curves)")
                with gr.Row():
                    gr.Image(value=str(pr_path), label="Mask Precision-Recall Curve (Đường cong P-R theo từng lớp)", interactive=False)
                    gr.Image(value=str(f1_path), label="Mask F1-Confidence Curve (Đường cong F1 tối ưu theo ngưỡng tin cậy)", interactive=False)

            val0_labels = train_run_dir / "val_batch0_labels.jpg"
            val0_pred = train_run_dir / "val_batch0_pred.jpg"
            if val0_labels.is_file() and val0_pred.is_file():
                gr.Markdown("### 4. So Sánh Nhãn Thực Tế (Ground Truth) vs Dự Đoán Của Mô Hình (Validation)")
                with gr.Row():
                    gr.Image(value=str(val0_labels), label="Nhãn Đa Giác Thực Tế (Ground Truth - Batch 0)", interactive=False)
                    gr.Image(value=str(val0_pred), label="YOLOv12s-seg Phân Vùng Dự Đoán (Prediction - Batch 0)", interactive=False)

            val1_labels = train_run_dir / "val_batch1_labels.jpg"
            val1_pred = train_run_dir / "val_batch1_pred.jpg"
            if val1_labels.is_file() and val1_pred.is_file():
                with gr.Row():
                    gr.Image(value=str(val1_labels), label="Nhãn Đa Giác Thực Tế (Ground Truth - Batch 1)", interactive=False)
                    gr.Image(value=str(val1_pred), label="YOLOv12s-seg Phân Vùng Dự Đoán (Prediction - Batch 1)", interactive=False)



# ----------------------------------------------------------------------------
# 8. ĐIỂM VÀO CHƯƠNG TRÌNH
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    start_webcam_server()
    warmup_models()
    print(f"[*] Đang khởi động Giao diện Web Gradio tại http://127.0.0.1:{port} ...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        theme=gr.themes.Soft(primary_hue="emerald"),
        css=CUSTOM_CSS,
        i18n=i18n_config,
        allowed_paths=[str(ROOT_DIR)],
        share=False,
        inbrowser=True
    )
