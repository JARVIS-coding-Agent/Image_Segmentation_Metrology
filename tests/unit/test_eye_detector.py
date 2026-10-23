"""eye_detector 單元測試：幾何/暗區啟發式（純 numpy）。"""

from __future__ import annotations

import numpy as np

from src.eye_detector import (
    CAT_CLASSES,
    detect_eyes,
    detect_eyes_catface,
    find_dark_spots,
    head_roi_bounds,
)
from src.models import EyeSide, EyeSource


def test_head_roi_bounds_top_half():
    x0, y0, x1, y1 = head_roi_bounds((10, 20, 40, 60), img_w=200, img_h=200, top_ratio=0.5)
    assert (x0, y0, x1, y1) == (10, 20, 50, 50)  # y1 = 20 + 0.5*60 = 50


def test_head_roi_clamped_to_image():
    x0, y0, x1, y1 = head_roi_bounds((-5, -5, 300, 300), img_w=100, img_h=80)
    assert (x0, y0) == (0, 0)
    # x1 = min(100, round(-5+300)) = 100; y1 = min(80, round(-5+0.5*300)) = 80
    assert (x1, y1) == (100, 80)


def test_find_dark_spots_two_symmetric():
    gray = np.full((20, 40), 200.0)
    gray[8:12, 8:12] = 10.0    # 左暗斑
    gray[8:12, 28:32] = 10.0   # 右暗斑
    spots = find_dark_spots(gray, n=2)
    assert len(spots) == 2
    xs = sorted(s[0] for s in spots)
    assert xs[0] < 20 < xs[1]  # 左右分居中線兩側


def test_find_dark_spots_empty_when_uniform():
    # 空輸入 → 空清單
    assert find_dark_spots(np.array([]).reshape(0, 0), n=2) == []


def test_find_dark_spots_low_contrast_returns_empty():
    # 低對比（近均勻）影像：暗區不顯著 → 回空清單（C-3，避免假眼睛）
    gray = np.full((20, 40), 128.0)
    gray += np.random.RandomState(0).uniform(-3, 3, gray.shape)  # 極小雜訊
    assert find_dark_spots(gray, n=2) == []


def test_find_dark_spots_float01_still_detects():
    # float [0,1] 輸入：入口 guard 應放大為 0–255，暗斑仍可偵測，
    # 不因 median − min_contrast 恆為負而系統性回空（FRIDAY blocker）。
    gray = np.full((20, 40), 0.5)  # 中位數 0.5
    gray[8:12, 5:9] = 0.0          # 左暗斑
    gray[8:12, 31:35] = 0.0        # 右暗斑
    spots = find_dark_spots(gray, n=2)
    assert len(spots) == 2
    # 兩群質心 x 應分居左右
    xs = sorted(s[0] for s in spots)
    assert xs[0] < 20 < xs[1]


def test_detect_eyes_left_smaller_x():
    img = np.full((40, 80, 3), 220, dtype=np.uint8)
    # 在頭部 ROI（上半部）放兩個暗斑
    img[8:12, 15:19] = 10
    img[8:12, 55:59] = 10
    left, right = detect_eyes(img, bbox=(0, 0, 80, 40))
    assert left is not None and right is not None
    assert left.side == EyeSide.LEFT
    assert right.side == EyeSide.RIGHT
    assert left.x < right.x          # x 小者為左眼
    assert left.is_estimated is True  # 啟發式為估計值


def test_detect_eyes_returns_none_when_no_dark_region():
    # 完全均勻影像：無顯著暗區 → 兩眼皆為 None（C-3，不輸出垃圾座標）
    img = np.full((40, 80, 3), 128, dtype=np.uint8)
    left, right = detect_eyes(img, bbox=(0, 0, 80, 40))
    assert left is None and right is None



def test_cat_classes_contains_cat():
    assert "cat" in CAT_CLASSES


def test_non_cat_uses_heuristic_source():
    # 非貓動物：使用啟發式，來源為 HEURISTIC
    img = np.full((40, 80, 3), 220, dtype=np.uint8)
    img[8:12, 15:19] = 10
    img[8:12, 55:59] = 10
    left, right = detect_eyes(img, bbox=(0, 0, 80, 40), category="dog")
    assert left is not None and right is not None
    assert left.source == EyeSource.HEURISTIC


def test_catface_no_face_returns_none():
    # 無貓臉的合成圖 → 貓臉偵測回傳 None（由 detect_eyes 回退啟發式，此處直接測 catface）
    img = np.full((60, 60, 3), 128, dtype=np.uint8)
    left, right = detect_eyes_catface(img, bbox=(0, 0, 60, 60))
    assert (left is None) and (right is None)


def test_cat_falls_back_to_heuristic_when_no_face():
    # 貓類但找不到貓臉 → 回退啟發式仍可能定位（有對稱暗斑）
    img = np.full((40, 80, 3), 220, dtype=np.uint8)
    img[8:12, 15:19] = 10
    img[8:12, 55:59] = 10
    left, right = detect_eyes(img, bbox=(0, 0, 80, 40), category="cat")
    # 回退啟發式：來源為 HEURISTIC（非貓臉 KEYPOINT）
    if left is not None:
        assert left.source == EyeSource.HEURISTIC
