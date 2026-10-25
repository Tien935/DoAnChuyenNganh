import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image

OOD_DIR = Path(r"d:\DACN\week7\data\ood_images")
OOD_DIR.mkdir(parents=True, exist_ok=True)

# 1. Cac mau anh tong hop (Noise, Blank, Gradient, Texture)
def create_synthetic_ood():
    print("Tao cac anh tong hop...")
    Image.new("RGB", (256, 256), (0, 0, 0)).save(OOD_DIR / "syn_pure_black.jpg")
    Image.new("RGB", (256, 256), (255, 255, 255)).save(OOD_DIR / "syn_pure_white.jpg")
    Image.new("RGB", (256, 256), (128, 128, 128)).save(OOD_DIR / "syn_pure_grey.jpg")
    
    np.random.seed(42)
    noise_arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(noise_arr).save(OOD_DIR / "syn_gaussian_noise.jpg")

    grad = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        grad[i, :, 0] = int(255 * i / 256)
        grad[:, i, 1] = int(255 * (256 - i) / 256)
        grad[i, :, 2] = 160
    Image.fromarray(grad).save(OOD_DIR / "syn_gradient.jpg")

    check = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(0, 256, 32):
        for x in range(0, 256, 32):
            if ((x // 32) + (y // 32)) % 2 == 0:
                check[y:y+32, x:x+32] = 255
    Image.fromarray(check).save(OOD_DIR / "syn_checkerboard.jpg")
    print("Tao thanh cong 6 anh tong hop.")

# 2. Cac anh doi song that ngoai pham vi (Picsum seed real-world photos)
REAL_SEEDS = [
    "laptop", "smartphone", "office_desk", "computer_keyboard", "wrist_watch",
    "cat", "dog", "horse", "bird", "flower",
    "forest_tree", "mountain", "ocean_beach", "night_sky", "city_street",
    "sports_car", "bicycle", "airplane", "building_facade", "living_room",
    "human_portrait", "child_playing", "coffee_cup_clean", "wooden_table", "bookshelf",
    "apple_fresh", "banana_fresh", "running_shoe", "t_shirt_clean", "backpack"
]

def download_real_ood():
    print(f"Dang tai {len(REAL_SEEDS)} anh chup thuc te ngoai pham vi rac...")
    headers = {"User-Agent": "Mozilla/5.0"}
    success = 0
    for seed in REAL_SEEDS:
        dest = OOD_DIR / f"real_{seed}.jpg"
        url = f"https://picsum.photos/seed/{seed}/300/300"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            with Image.open(dest) as img:
                img.convert("RGB").save(dest, "JPEG")
            success += 1
        except Exception as e:
            print(f"Loi tai {seed}: {e}")
    print(f"Da tai thanh cong {success}/{len(REAL_SEEDS)} anh thuc te.")

if __name__ == "__main__":
    create_synthetic_ood()
    download_real_ood()
    total = len(list(OOD_DIR.glob('*.jpg')))
    print(f"Tong so anh OOD san sang: {total}")
