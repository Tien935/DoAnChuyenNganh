from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import open_clip
from PIL import Image

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

# Standard 5-prompt ensemble from Week 5/6
DEFAULT_PROMPTS = {
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


class VLMEvaluator:
    """Core OpenCLIP evaluator supporting custom prompt sets, top-1/top-2 margin analysis."""

    def __init__(self, prompts: Optional[Dict[str, List[str]]] = None, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.prompts = prompts or DEFAULT_PROMPTS

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.text_features = self.build_text_features(self.prompts)

    def set_prompts(self, prompts: Dict[str, List[str]]):
        self.prompts = prompts
        self.text_features = self.build_text_features(self.prompts)

    def build_text_features(self, prompts: Dict[str, List[str]]) -> torch.Tensor:
        features_by_class = {}
        with torch.no_grad():
            for c in CLASSES:
                prompt_list = prompts.get(c, [c])
                tokens = self.tokenizer(prompt_list).to(self.device)
                tf = self.model.encode_text(tokens)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                class_feat = tf.mean(dim=0)
                class_feat = class_feat / class_feat.norm()
                features_by_class[c] = class_feat
        return torch.stack([features_by_class[c] for c in CLASSES])

    def predict_image(self, image_path: Path, threshold: float = 0.50) -> Dict[str, Any]:
        with Image.open(image_path) as src:
            image = src.convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feat = self.model.encode_image(image_input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            similarities = (100.0 * img_feat @ self.text_features.T)[0]
            probs = torch.softmax(similarities, dim=0)

        # Sort probabilities to get top-1 and top-2
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        top1_idx = int(sorted_indices[0].item())
        top2_idx = int(sorted_indices[1].item())

        top1_class = CLASSES[top1_idx]
        top1_score = float(sorted_probs[0].item())
        top2_class = CLASSES[top2_idx]
        top2_score = float(sorted_probs[1].item())
        margin = top1_score - top2_score

        is_unknown = top1_score < threshold
        predicted_class = "unknown" if is_unknown else top1_class
        status = "UNKNOWN" if is_unknown else "CONFIDENT"

        scores_dict = {CLASSES[i]: float(probs[i].item()) for i in range(len(CLASSES))}

        return {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "predicted_class": predicted_class,
            "top1_class": top1_class,
            "top1_score": round(top1_score, 4),
            "top2_class": top2_class,
            "top2_score": round(top2_score, 4),
            "margin": round(margin, 4),
            "threshold": threshold,
            "status": status,
            "scores": scores_dict,
        }
