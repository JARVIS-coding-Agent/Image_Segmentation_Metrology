"""分割與眼睛定位評估（spec §3、FR-011/FR-012）。

- IoU  = |A∩B| / |A∪B|
- Dice = 2|A∩B| / (|A|+|B|)
- NME  = (1/N) Σ ||p_i - p̂_i||₂ / d_norm
純 numpy 實作以利單元測試。
"""

from __future__ import annotations

import math

import numpy as np

from src.models import EvaluationResult


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """兩布林遮罩的 IoU。聯集為 0 時回傳 0。"""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """兩布林遮罩的 Dice 係數。面積和為 0 時回傳 0。"""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    if total == 0:
        return 0.0
    return 2.0 * float(inter) / float(total)


def nme(
    predicted: list[tuple[float, float]],
    ground_truth: list[tuple[float, float]],
    d_norm: float,
) -> float:
    """Normalized Mean Error：平均點位誤差除以正規化尺度 d_norm。

    predicted 與 ground_truth 為等長的 (x, y) 清單；d_norm 常用眼間距或頭框尺寸。
    """
    if len(predicted) != len(ground_truth):
        raise ValueError("predicted 與 ground_truth 長度需相同")
    if len(predicted) == 0:
        return 0.0
    if d_norm <= 0:
        raise ValueError("d_norm 必須 > 0")
    total = 0.0
    for (px, py), (gx, gy) in zip(predicted, ground_truth):
        total += math.hypot(px - gx, py - gy)
    return (total / len(predicted)) / d_norm


def evaluate_eye_localization(
    predicted: list[tuple[float, float]],
    ground_truth: list[tuple[float, float]],
    d_norm: float,
    gt_source: str = "manual",
) -> EvaluationResult:
    """以 ground truth 對照眼睛定位，回傳 NME 評估結果（FR-012）。"""
    value = nme(predicted, ground_truth, d_norm)
    return EvaluationResult(
        metric="NME",
        scope="eye_localization",
        value=value,
        gt_source=gt_source,
        sample_size=len(predicted),
    )
