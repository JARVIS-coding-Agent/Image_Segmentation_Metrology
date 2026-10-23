"""mask_to_original 共用對齊函式測試（P0-B / IoU 對齊正確性）。

重點：驗證 mask→原圖的轉換在「非方形影像」下仍對齊，不會像先前的等距抽樣
把 letterbox padding 一起拉伸導致錯位。疊圖與 IoU/Dice 共用此函式。
"""

import numpy as np

from src.segmentation import mask_to_original


def test_none_mask_returns_none():
    assert mask_to_original(None, 10, 10) is None


def test_empty_or_bad_dims_return_none():
    assert mask_to_original(np.zeros((0, 0)), 10, 10) is None
    assert mask_to_original(np.zeros((5,)), 10, 10) is None  # 非 2D


def test_already_aligned_mask_passthrough():
    m = np.zeros((6, 4), dtype=bool)
    m[1:3, 1:3] = True
    out = mask_to_original(m, 6, 4)
    assert out.shape == (6, 4)
    assert out.dtype == bool
    np.testing.assert_array_equal(out, m)


def test_float_mask_thresholded():
    m = np.array([[0.9, 0.1], [0.6, 0.0]])
    out = mask_to_original(m, 2, 2)
    np.testing.assert_array_equal(out, np.array([[True, False], [True, False]]))


def test_non_square_resize_preserves_orientation():
    """非方形原圖：mask 需正確 resize 到 (H, W) 且保留左右/上下方向。

    以一個「左半部為 True」的小 mask，resize 到較大的非方形尺寸後，
    左半部仍應為 True、右半部為 False（若用錯誤的抽樣/轉置會失敗）。
    """
    small = np.zeros((2, 2), dtype=bool)
    small[:, 0] = True  # 左半 True
    out = mask_to_original(small, 8, 20)  # 非方形：高 8、寬 20
    assert out.shape == (8, 20)
    # 左半（前 10 欄）應大多為 True，右半應為 False
    assert out[:, :10].all()
    assert not out[:, 10:].any()


def test_non_square_resize_vertical_orientation():
    """上下方向同樣需保留：上半 True → resize 後上半仍 True。"""
    small = np.zeros((2, 2), dtype=bool)
    small[0, :] = True  # 上半 True
    out = mask_to_original(small, 20, 8)  # 非方形：高 20、寬 8
    assert out.shape == (20, 8)
    assert out[:10, :].all()
    assert not out[10:, :].any()
