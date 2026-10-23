"""evaluate 單元測試：IoU / Dice / NME（純 numpy，已知輸入）。"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import (
    dice,
    evaluate_eye_localization,
    evaluate_segmentation,
    iou,
    nme,
)


def test_iou_identical_masks():
    a = np.array([[1, 1], [0, 0]], dtype=bool)
    assert iou(a, a) == pytest.approx(1.0)


def test_iou_half_overlap():
    a = np.array([[1, 1], [0, 0]], dtype=bool)
    b = np.array([[1, 0], [1, 0]], dtype=bool)
    # 交集=1 (左上), 聯集=3 -> 1/3
    assert iou(a, b) == pytest.approx(1.0 / 3.0)


def test_iou_empty_union_zero():
    a = np.zeros((2, 2), dtype=bool)
    assert iou(a, a) == 0.0


def test_dice_identical():
    a = np.array([[1, 1], [1, 0]], dtype=bool)
    assert dice(a, a) == pytest.approx(1.0)


def test_dice_half_overlap():
    a = np.array([[1, 1], [0, 0]], dtype=bool)
    b = np.array([[1, 0], [1, 0]], dtype=bool)
    # 2*交集(1) / (面積和 2+2=4) = 0.5
    assert dice(a, b) == pytest.approx(0.5)


def test_dice_empty_zero():
    a = np.zeros((2, 2), dtype=bool)
    assert dice(a, a) == 0.0


def test_nme_zero_when_perfect():
    pred = [(0.0, 0.0), (10.0, 0.0)]
    gt = [(0.0, 0.0), (10.0, 0.0)]
    assert nme(pred, gt, d_norm=10.0) == pytest.approx(0.0)


def test_nme_known_value():
    pred = [(3.0, 4.0)]   # 誤差 5
    gt = [(0.0, 0.0)]
    assert nme(pred, gt, d_norm=10.0) == pytest.approx(0.5)


def test_nme_length_mismatch_raises():
    with pytest.raises(ValueError):
        nme([(0.0, 0.0)], [], d_norm=1.0)


def test_nme_bad_dnorm_raises():
    with pytest.raises(ValueError):
        nme([(0.0, 0.0)], [(1.0, 1.0)], d_norm=0.0)


def test_evaluate_eye_localization_result():
    res = evaluate_eye_localization(
        [(3.0, 4.0)], [(0.0, 0.0)], d_norm=10.0, gt_source="manual"
    )
    assert res.metric == "NME"
    assert res.scope == "eye_localization"
    assert res.value == pytest.approx(0.5)
    assert res.sample_size == 1



def test_evaluate_segmentation_perfect_match():
    m = np.array([[1, 1], [0, 0]], dtype=bool)
    results = evaluate_segmentation([m], [m])
    assert len(results) == 2
    by_metric = {r.metric: r for r in results}
    assert by_metric["IoU"].value == pytest.approx(1.0)
    assert by_metric["Dice"].value == pytest.approx(1.0)
    assert by_metric["IoU"].scope == "segmentation"
    assert by_metric["IoU"].sample_size == 1


def test_evaluate_segmentation_averages_over_samples():
    a = np.array([[1, 1], [0, 0]], dtype=bool)       # 與 b 半重疊
    b = np.array([[1, 0], [1, 0]], dtype=bool)
    # 第一組完全相同(IoU=1)、第二組半重疊(IoU=1/3) → 平均 2/3
    results = evaluate_segmentation([a, a], [a, b])
    by_metric = {r.metric: r for r in results}
    assert by_metric["IoU"].value == pytest.approx((1.0 + 1.0 / 3.0) / 2)
    assert by_metric["IoU"].sample_size == 2


def test_evaluate_segmentation_empty_returns_empty():
    assert evaluate_segmentation([], []) == []


def test_evaluate_segmentation_length_mismatch_raises():
    """長度不等代表 pred/gt 配對錯亂，應 ValueError 而非 min() 靜默截斷。"""
    m = np.array([[1, 0], [0, 1]], dtype=bool)
    with pytest.raises(ValueError, match="長度需相同"):
        evaluate_segmentation([m, m], [m])


def test_evaluate_segmentation_shape_mismatch_raises():
    """同 index 但 shape 不符，提早丟清楚錯誤（不拖到 iou() 才炸）。"""
    pred = np.zeros((2, 2), dtype=bool)
    gt = np.zeros((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_segmentation([pred], [gt])
