
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import open_clip
from PIL import Image


# ============================================================
# 1. CAU HINH MO HINH
# ============================================================

PROJECT_DIR = Path(r"D:\DACN")

MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

# Giu nguyen prompt ensemble tu week4_yoloclip.py
PROMPTS = {
    "cardboard": [
        "a photo of cardboard waste",
        "a photo of a cardboard box",
        "an image of discarded cardboard",
        "a piece of cardboard waste",
        "cardboard recyclable material",
    ],
    "glass": [
        "a photo of glass waste",
        "a photo of a glass bottle",
        "an image of discarded glass",
        "a piece of glass waste",
        "glass recyclable material",
    ],
    "metal": [
        "a photo of metal waste",
        "a photo of a metal can",
        "an image of discarded metal",
        "a piece of metal waste",
        "metal recyclable material",
    ],
    "paper": [
        "a photo of paper waste",
        "a photo of discarded paper",
        "an image of paper waste",
        "a piece of paper waste",
        "paper recyclable material",
    ],
    "plastic": [
        "a photo of plastic waste",
        "a photo of a plastic bottle",
        "an image of discarded plastic",
        "a piece of plastic waste",
        "plastic recyclable material",
    ],
    "trash": [
        "a photo of trash",
        "a photo of general waste",
        "an image of discarded trash",
        "a piece of miscellaneous waste",
        "a pile of general garbage",
    ],
}

# Nguong ban dau, can kiem thu them.
# Softmax score khong phai confidence da duoc hieu chuan.
DEFAULT_UNKNOWN_THRESHOLD = 0.60


# ============================================================
# 2. LOP SUY LUAN OPENCLIP
# ============================================================

class VLMInference:
    """Module suy luan OpenCLIP cho anh crop cua vat the."""

    def __init__(
        self,
        threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
        device: Optional[str] = None,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold phai nam trong khoang 0.0 den 1.0"
            )

        self.threshold = threshold

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"Khoi tao OpenCLIP tren thiet bi: {self.device}")

        # Load mo hinh OpenCLIP
        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                MODEL_NAME,
                pretrained=PRETRAINED,
            )
        )

        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Tao dac trung van ban mot lan khi khoi tao mo hinh
        self.text_features = self._build_text_features()

        print("Da khoi tao OpenCLIP thanh cong!")

    def _build_text_features(self) -> torch.Tensor:
        """
        Tao dac trung van ban cho tung lop.
        Moi lop su dung 5 prompt, sau do lay trung binh dac trung.
        """

        features_by_class = {}

        with torch.no_grad():
            for class_name in CLASSES:
                prompts = PROMPTS[class_name]

                tokens = self.tokenizer(prompts).to(self.device)

                text_features = self.model.encode_text(tokens)

                # Chuan hoa dac trung tung prompt
                text_features = text_features / text_features.norm(
                    dim=-1,
                    keepdim=True,
                )

                # Lay trung binh dac trung cua cac prompt
                class_feature = text_features.mean(dim=0)

                # Chuan hoa dac trung cua lop
                class_feature = class_feature / class_feature.norm()

                features_by_class[class_name] = class_feature

        # Shape: [6, embedding_dimension]
        return torch.stack(
            [features_by_class[name] for name in CLASSES]
        )

    def predict_vlm(
        self,
        image_path: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Phan loai mot anh crop.

        Ket qua gom:
        - predicted_class: nhan sau khi ap dung nguong
        - top_class: nhan co diem cao nhat truoc nguong
        - top_score: softmax score cao nhat
        - scores: diem so cua tung lop
        - status: CONFIDENT hoac UNKNOWN
        """

        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Khong tim thay anh: {image_path}"
            )

        active_threshold = (
            self.threshold if threshold is None else threshold
        )

        if not 0.0 <= active_threshold <= 1.0:
            raise ValueError(
                "threshold phai nam trong khoang 0.0 den 1.0"
            )

        # Doc anh va chuyen sang RGB
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")

        # Tien xu ly anh theo yeu cau cua OpenCLIP
        image_input = self.preprocess(image).unsqueeze(0)
        image_input = image_input.to(self.device)

        with torch.no_grad():
            # Trich xuat dac trung anh
            image_features = self.model.encode_image(image_input)

            # Chuan hoa dac trung anh
            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True,
            )

            # Do tuong dong giua anh va 6 lop van ban
            similarities = (
                    100.0 * image_features @ self.text_features.T
            )[0]

            # Chuyen diem tuong dong thanh softmax score
            probabilities = torch.softmax(similarities, dim=0)

        # Lop co diem cao nhat
        best_index = int(probabilities.argmax().item())

        top_class = CLASSES[best_index]
        top_score = float(probabilities[best_index].item())

        # Diem so cua tung lop
        scores = {
            class_name: float(probabilities[i].item())
            for i, class_name in enumerate(CLASSES)
        }

        # Ap dung nguong unknown
        if top_score >= active_threshold:
            predicted_class = top_class
            status = "CONFIDENT"
        else:
            predicted_class = "unknown"
            status = "UNKNOWN"

        # Tra ve ket qua co cau truc thong nhat
        result = {
            "image_path": str(image_path),
            "predicted_class": predicted_class,
            "top_class": top_class,
            "top_score": top_score,
            "threshold": active_threshold,
            "status": status,
            "scores": scores,
        }

        return result


# ============================================================
# 3. CAC HAM GOI MODULE
# ============================================================

_default_vlm = None


def load_vlm(
    threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
    device: Optional[str] = None,
) -> VLMInference:
    """
    Khoi tao mo hinh.
    Nen goi mot lan va tai su dung cho nhieu anh.
    """

    return VLMInference(
        threshold=threshold,
        device=device,
    )


def predict_vlm(
    image_path: str,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Ham tien loi de goi truc tiep tu module khac.

    Mo hinh chi duoc load mot lan trong cac lan goi ham nay.
    """

    global _default_vlm

    if _default_vlm is None:
        _default_vlm = load_vlm()

    return _default_vlm.predict_vlm(
        image_path,
        threshold=threshold,
    )


# ============================================================
# 4. CHAY KIEM TRA DOC LAP
# ============================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Kiem tra suy luan OpenCLIP tren mot anh."
    )

    parser.add_argument(
        "image_path",
        help="Duong dan den anh crop can phan loai",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_UNKNOWN_THRESHOLD,
        help="Nguong unknown, mac dinh 0.50",
    )

    args = parser.parse_args()

    # Load mo hinh
    vlm = load_vlm(threshold=args.threshold)

    # Du doan anh
    result = vlm.predict_vlm(args.image_path)

    # In ket qua JSON
    print("\n===== KET QUA OPENCLIP =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))
