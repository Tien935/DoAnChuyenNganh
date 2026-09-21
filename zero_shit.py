import torch
import open_clip
from PIL import Image
import os

# Tên 6 loại rác
class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Prompt cho OpenCLIP
prompts = [
    "a photo of cardboard waste",
    "a photo of glass waste",
    "a photo of metal waste",
    "a photo of paper waste",
    "a photo of plastic waste",
    "a photo of trash"
]

# Chọn thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load mô hình
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

model = model.to(device)

# Tokenizer
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Chuyển prompt thành Text Embedding
text = tokenizer(prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)

# Chuẩn hóa Text Embedding
text_features /= text_features.norm(
    dim=-1,
    keepdim=True
)

# Thư mục chứa ảnh
image_folder = "images"

# Lấy danh sách ảnh
image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
]

# Phân loại từng ảnh
for filename in image_files:

    print("\n================================")
    print("Ảnh:", filename)
    print("================================")

    # Đường dẫn ảnh
    image_path = os.path.join(
        image_folder,
        filename
    )

    # Đọc và xử lý ảnh
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    # Tạo Image Embedding
    with torch.no_grad():
        image_features = model.encode_image(image)

    # Chuẩn hóa Image Embedding
    image_features /= image_features.norm(
        dim=-1,
        keepdim=True
    )

    # Tính độ tương đồng giữa ảnh và text
    similarity = image_features @ text_features.T

    # Chuyển thành xác suất
    probabilities = similarity.softmax(dim=-1)[0]

    # Tạo danh sách kết quả
    results = []

    for label, probability in zip(
        class_names,
        probabilities
    ):
        results.append(
            (label, probability.item() * 100)
        )

    # Sắp xếp từ cao xuống thấp
    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Hiển thị kết quả
    for label, probability in results:
        print(f"{label:<12}: {probability:.2f}%")

    # Kết quả dự đoán cao nhất
    print("\n>>> Dự đoán:", results[0][0])
