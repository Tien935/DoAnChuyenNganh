import torch
import open_clip

print("OpenCLIP đã được import thành công!")

# Chọn thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Thiết bị sử dụng:", device)

# Load OpenCLIP với QuickGELU đúng với OpenAI weights
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu",
    pretrained="openai"
)

model = model.to(device)
model.eval()

# Tokenizer
tokenizer = open_clip.get_tokenizer(
    "ViT-B-32-quickgelu"
)

print("Đã tải mô hình OpenCLIP thành công!")