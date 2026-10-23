"""mask_to_original 共用對齊函式測試（P0-B / IoU 對齊正確性）。

重點：驗證 mask→原圖轉換在 retina_masks 正常路徑（shape 已對齊）直接放行；
而 shape 不符原圖時「拒絕猜測」回傳 None（來源座標系不明，可能含 letterbox
padding，盲目 resize 會重演錯位）。疊圖與 IoU/Dice 共用此函式。
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


def test_shape_mismatch_rejected_returns_none():
    """shape 不符原圖時拒絕猜測（來源座標系不明，可能含 letterbox padding）。

    與其盲目 resize 一個來源不明的 mask（可能悄悄重演座標錯位、產出看似正常的
    錯誤 IoU），不如回傳 None 讓呼叫端跳過該樣本。正常路徑應走 retina_masks，
    mask 一開始就對齊原圖、不會進到這個分支。
    """
    small = np.zeros((2, 2), dtype=bool)
    small[:, 0] = True
    assert mask_to_original(small, 8, 20) is None  # 非方形、尺寸不符 → 拒絕
    assert mask_to_original(small, 20, 8) is None


def test_mismatch_but_exact_shape_still_passes():
    """尺寸恰等於原圖才放行（retina_masks 正常路徑）。"""
    m = np.zeros((20, 8), dtype=bool)
    m[0, :] = True
    out = mask_to_original(m, 20, 8)
    assert out.shape == (20, 8)
    assert out[0, :].all()
    assert not out[1:, :].any()
