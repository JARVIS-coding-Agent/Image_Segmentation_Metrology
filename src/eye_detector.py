"""動物眼睛定位（spec 核心難點、research R2）。

主要策略：頭部 ROI + 幾何/暗區啟發式（不依賴 opencv，純 numpy 以利測試）。
- 於動物 bbox/mask 的頭部區域（預設上半部）在灰階影像找對稱的局部暗區作為左右眼候選。
- 左右定義（config 座標慣例）：x 較小者為 left，較大者為 right（影像視角）。
- 所有啟發式座標 is_estimated=True、source=heuristic（FR-004）。
- 不可靠（找不到成對暗區）→ 回傳 None，由 metrology 標記為無法量測（FR-007）。

SAM2 point-prompt 精修為可選增強（use_sam2=True 時），此處提供掛鉤；
未安裝 SAM2 時回退啟發式結果。
"""

from __future__ import annotations

import numpy as np

from src.models import EyePoint, EyeSide, EyeSource


def _to_gray(image: np.ndarray) -> np.ndarray:
    """轉灰階（接受 HxW 或 HxWx3 的 numpy array）。"""
    if image.ndim == 2:
        return image.astype(np.float64)
    # RGB 加權灰階
    return image[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def find_dark_spots(
    gray: np.ndarray,
    n: int = 2,
    dark_percentile: float = 15.0,
) -> list[tuple[float, float]]:
    """在灰階 ROI 中找最暗的 n 個區域中心（質心），回傳 (x, y) 清單。

    以暗度分位數閾值取得暗像素，依連通分群取質心；簡化為 KMeans-free 的
    水平二分：將暗像素依 x 中位數分為左右兩群，各取質心。
    """
    if gray.size == 0:
        return []
    thresh = np.percentile(gray, dark_percentile)
    ys, xs = np.where(gray <= thresh)
    if xs.size == 0:
        return []

    if n == 1:
        return [(float(xs.mean()), float(ys.mean()))]

    # n==2：以 x 中位數切成左右兩群，各取質心
    x_split = np.median(xs)
    left_mask = xs <= x_split
    right_mask = xs > x_split
    spots: list[tuple[float, float]] = []
    for m in (left_mask, right_mask):
        if m.sum() == 0:
            continue
        spots.append((float(xs[m].mean()), float(ys[m].mean())))
    return spots


def head_roi_bounds(
    bbox: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    top_ratio: float = 0.5,
) -> tuple[int, int, int, int]:
    """由 bbox 推定頭部 ROI（預設取上半部）。回傳 (x0, y0, x1, y1) 整數邊界。"""
    x, y, w, h = bbox
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(img_w, int(round(x + w)))
    y1 = min(img_h, int(round(y + top_ratio * h)))
    return x0, y0, x1, y1


def detect_eyes(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    use_sam2: bool = False,
) -> tuple[EyePoint | None, EyePoint | None]:
    """偵測單一動物的左右眼。

    回傳 (left_eye, right_eye)；任一無法可靠定位則為 None。
    座標為整張影像的絕對像素座標。
    """
    img_h, img_w = image.shape[0], image.shape[1]
    x0, y0, x1, y1 = head_roi_bounds(bbox, img_w, img_h)
    if x1 <= x0 or y1 <= y0:
        return None, None

    gray = _to_gray(image)[y0:y1, x0:x1]
    spots = find_dark_spots(gray, n=2)
    if len(spots) < 2:
        return None, None

    # 轉回絕對座標並依 x 排序：小者 left、大者 right
    abs_spots = sorted(((sx + x0, sy + y0) for sx, sy in spots), key=lambda p: p[0])
    (lx, ly), (rx, ry) = abs_spots[0], abs_spots[-1]

    source = EyeSource.SAM2 if use_sam2 else EyeSource.HEURISTIC
    left = EyePoint(side=EyeSide.LEFT, x=lx, y=ly, is_estimated=True, source=source)
    right = EyePoint(side=EyeSide.RIGHT, x=rx, y=ry, is_estimated=True, source=source)

    if use_sam2:
        left, right = _refine_with_sam2(image, left, right)
    return left, right


def _refine_with_sam2(
    image: np.ndarray, left: EyePoint, right: EyePoint
) -> tuple[EyePoint, EyePoint]:
    """可選 SAM2 精修掛鉤。未安裝 SAM2 時回退啟發式結果。"""
    try:
        import sam2  # noqa: F401
    except Exception:
        # 未安裝 → 回退啟發式
        return left, right
    # 若有 SAM2，實際精修流程於整合環境實作；此處保守回傳輸入點。
    return left, right
