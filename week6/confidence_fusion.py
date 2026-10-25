from typing import Dict, Any, Optional


def combine_confidence(yolo_conf: float, vlm_conf: float, alpha: float = 0.5) -> float:
    """
    Công thức kết hợp confidence đề xuất:
    C_final = alpha * C_yolo + (1 - alpha) * C_vlm

    Parameters
    ----------
    yolo_conf : float
        Độ tin cậy của mô hình YOLO, trong đoạn [0, 1].
    vlm_conf : float
        Độ tin cậy của mô hình OpenCLIP (Softmax score), trong đoạn [0, 1].
    alpha : float, optional
        Trọng số ưu tiên YOLO, mặc định là 0.5 (cân bằng).

    Returns
    -------
    float
        Điểm tin cậy kết hợp C_final.
    """
    for name, value in [
        ("yolo_conf", yolo_conf),
        ("vlm_conf", vlm_conf),
        ("alpha", alpha),
    ]:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} phải nằm trong khoảng 0 đến 1 (nhận được: {value})")

    return alpha * yolo_conf + (1.0 - alpha) * vlm_conf


def fuse_prediction(
    yolo_class: str,
    yolo_conf: float,
    vlm_result: Dict[str, Any],
    alpha: float = 0.5,
    conflict_strategy: str = "yolo_priority",
) -> Dict[str, Any]:
    """
    Quy tắc hợp nhất dự đoán giữa YOLO và OpenCLIP (VLM).

    Parameters
    ----------
    yolo_class : str
        Nhãn dự đoán từ YOLO (cardboard, glass, metal, paper, plastic, trash).
    yolo_conf : float
        Confidence từ YOLO [0, 1].
    vlm_result : Dict[str, Any]
        Kết quả trả về từ VLMInference.predict_vlm(...).
        Gồm: top_class, top_score, status ('CONFIDENT' hoặc 'UNKNOWN').
    alpha : float, optional
        Trọng số hợp nhất khi hai mô hình đồng thuận, mặc định 0.5.
    conflict_strategy : str, optional
        Chiến lược khi bất đồng:
        - 'yolo_priority' (mặc định): ưu tiên YOLO làm baseline phát hiện.
        - 'higher_confidence': chọn mô hình có confidence cao hơn.

    Returns
    -------
    Dict[str, Any]
        Dictionary chứa:
        - final_class: Nhãn chốt cuối cùng
        - final_confidence: Điểm tin cậy cuối cùng
        - fusion_reason: Lý do ra quyết định ('AGREEMENT', 'DISAGREEMENT_USE_YOLO',
                         'DISAGREEMENT_USE_VLM', 'VLM_UNKNOWN_USE_YOLO')
        - is_agreement: bool
        - yolo_class, yolo_confidence
        - vlm_class, vlm_confidence, vlm_status
        - alpha
    """
    vlm_top_class = str(vlm_result.get("top_class", "")).strip().lower()
    vlm_status = str(vlm_result.get("status", "CONFIDENT")).strip().upper()
    vlm_conf = float(vlm_result.get("top_score", 0.0))

    y_cls = str(yolo_class).strip().lower()
    is_agreement = (y_cls == vlm_top_class)

    # 1. Trường hợp VLM không chắc chắn (UNKNOWN) -> Fallback về YOLO
    if vlm_status == "UNKNOWN":
        return {
            "final_class": y_cls,
            "final_confidence": round(yolo_conf, 4),
            "fusion_reason": "VLM_UNKNOWN_USE_YOLO",
            "is_agreement": is_agreement,
            "yolo_class": y_cls,
            "yolo_confidence": round(yolo_conf, 4),
            "vlm_class": vlm_top_class,
            "vlm_confidence": round(vlm_conf, 4),
            "vlm_status": vlm_status,
            "alpha": alpha,
        }

    # 2. Trường hợp hai mô hình đồng thuận (AGREEMENT)
    if is_agreement:
        combined = combine_confidence(yolo_conf, vlm_conf, alpha=alpha)
        return {
            "final_class": y_cls,
            "final_confidence": round(combined, 4),
            "fusion_reason": "AGREEMENT",
            "is_agreement": True,
            "yolo_class": y_cls,
            "yolo_confidence": round(yolo_conf, 4),
            "vlm_class": vlm_top_class,
            "vlm_confidence": round(vlm_conf, 4),
            "vlm_status": vlm_status,
            "alpha": alpha,
        }

    # 3. Trường hợp hai mô hình bất đồng (DISAGREEMENT)
    if conflict_strategy == "higher_confidence":
        if vlm_conf > yolo_conf:
            final_class = vlm_top_class
            final_conf = vlm_conf
            reason = "DISAGREEMENT_USE_VLM"
        else:
            final_class = y_cls
            final_conf = yolo_conf
            reason = "DISAGREEMENT_USE_YOLO"
    else:
        # Mặc định theo quy ước tuần 6: ưu tiên YOLO
        final_class = y_cls
        final_conf = yolo_conf
        reason = "DISAGREEMENT_USE_YOLO"

    return {
        "final_class": final_class,
        "final_confidence": round(final_conf, 4),
        "fusion_reason": reason,
        "is_agreement": False,
        "yolo_class": y_cls,
        "yolo_confidence": round(yolo_conf, 4),
        "vlm_class": vlm_top_class,
        "vlm_confidence": round(vlm_conf, 4),
        "vlm_status": vlm_status,
        "alpha": alpha,
    }


if __name__ == "__main__":
    # Minh họa thử nghiệm
    print("===== DEMO CONFIDENCE FUSION MODULE =====")
    yolo_cls = "glass"
    yolo_c = 0.82

    # Case 1: Đồng thuận
    vlm_sample_agree = {
        "top_class": "glass",
        "top_score": 0.88,
        "status": "CONFIDENT",
    }
    res_agree = fuse_prediction(yolo_cls, yolo_c, vlm_sample_agree, alpha=0.5)
    print("\nCase 1 (Đồng thuận):", res_agree)

    # Case 2: Bất đồng (YOLO ưu tiên)
    vlm_sample_disagree = {
        "top_class": "plastic",
        "top_score": 0.75,
        "status": "CONFIDENT",
    }
    res_disagree = fuse_prediction(yolo_cls, yolo_c, vlm_sample_disagree, alpha=0.5)
    print("\nCase 2 (Bất đồng):", res_disagree)

    # Case 3: VLM Unknown
    vlm_sample_unk = {
        "top_class": "metal",
        "top_score": 0.42,
        "status": "UNKNOWN",
    }
    res_unk = fuse_prediction(yolo_cls, yolo_c, vlm_sample_unk, alpha=0.5)
    print("\nCase 3 (VLM Unknown):", res_unk)
