"""動物眼睛定位（spec 核心難點、research R2）。

分流策略（依動物類別）：
- 貓（cat）：先以 OpenCV Haar frontalcatface 偵測貓臉，於臉框上半部左右各找暗點
  作為雙眼（source=keypoint，偵測器輔助）；失敗則回退下列啟發式。
- 其他動物：頭部 ROI + 幾何/暗區啟發式（純 numpy）。
  於動物 bbox 頭部區域（預設上半部）在灰階影像找對稱的局部暗區作為左右眼候選。

共通規則：
- 左右定義（config 座標慣例）：x 較小者為 left，較大者為 right（影像視角）。
- 所有座標 is_estimated=True（COCO 無動物眼睛標註，皆為估計，FR-004）。
- 不可靠（找不到成對暗區 / 貓臉）→ 回傳 None，由 metrology 標記為無法量測（FR-007）。

SAM2 point-prompt 精修為可選增強（use_sam2=True 時）；未安裝 SAM2 時回退。
"""

from __future__ import annotations

import numpy as np

from src.models import EyePoint, EyeSide, EyeSource

# 視為「貓」的類別（Haar frontalcatface 僅對貓有效）
CAT_CLASSES = {"cat"}

_CAT_CASCADE = None


def _get_cat_cascade():
    """延遲載入 OpenCV Haar frontalcatface cascade（未裝 opencv 則回傳 None）。"""
    global _CAT_CASCADE
    if _CAT_CASCADE is None:
        try:
            import os

            import cv2

            path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalcatface.xml")
            _CAT_CASCADE = cv2.CascadeClassifier(path)
        except Exception:
            _CAT_CASCADE = False  # 標記為不可用
    return _CAT_CASCADE or None


def _to_gray(image: np.ndarray) -> np.ndarray:
    """轉灰階（接受 HxW 或 HxWx3 的 numpy array）。"""
    if image.ndim == 2:
        return image.astype(np.float64)
    # RGB 加權灰階
    return image[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def find_dark_spots(
    gray: np.ndarray,
    n: int = 2,
    min_contrast: float = 25.0,
    min_gap_ratio: float = 0.15,
) -> list[tuple[float, float]]:
    """在灰階 ROI 中找最暗的 n 個區域中心（質心），回傳 (x, y) 清單。

    閾值策略（C-3，避免靜默 false positive）：
    - 以「中位數亮度 − ``min_contrast``」為暗像素門檻，只選出**明顯比整體暗**的像素；
      這與固定百分位不同，暗斑佔比很小時也不會把背景誤選進來。
    - 若沒有任何像素低於門檻（例如低對比 / 均勻影像），回傳空清單，
      由上層標記為無法量測，而非輸出垃圾座標。

    n==2 的「成對」驗證（避免假眼睛，ULTRON P1）：
    - **不**以 x 中位數硬切兩群；單一暗斑被中位數劈半會產生一對緊貼的假眼睛。
    - 改以 x 方向的**最大間隙**找真正的分離點：唯有左右兩團之間存在
      顯著空隙（間隙 ≥ ROI 寬的 ``min_gap_ratio``）才視為兩隻眼。
    - 否則（單一連續暗斑、或兩團貼太近無法區分）回傳空清單，
      由上層標記為無法量測。

    輸入前提：灰階值域為 0–255。若偵測到 float [0,1] 影像（最大值 ≤ 1），
    在此自動放大為 0–255，避免 ``median − min_contrast`` 恆為負導致系統性拒絕量測
    （防禦縱深；根本修正在 pipeline 讀取層統一正規化）。
    """
    if gray.size == 0:
        return []

    # 防禦縱深：float [0,1] 影像放大到 0–255，與 min_contrast 的尺度對齊
    if float(gray.max()) <= 1.0:
        gray = gray * 255.0

    median_level = float(np.median(gray))
    thresh = median_level - min_contrast
    ys, xs = np.where(gray <= thresh)
    if xs.size == 0:
        return []

    if n == 1:
        return [(float(xs.mean()), float(ys.mean()))]

    # n==2：以 x 方向最大間隙分群，並驗證左右兩團確實分離（ULTRON P1）
    roi_w = float(gray.shape[1])
    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    ux = np.unique(xs_sorted)
    if ux.size < 2:
        # 只占一條 x（退化的單一暗斑）→ 無法成對
        return []

    # 找相鄰唯一 x 之間的最大間隙作為左右分界
    gaps = np.diff(ux)
    gi = int(np.argmax(gaps))
    max_gap = float(gaps[gi])
    # 間隙需達 ROI 寬的一定比例，否則視為單一連續暗斑 → 不成對
    if max_gap < max(2.0, roi_w * min_gap_ratio):
        return []

    split_x = (ux[gi] + ux[gi + 1]) / 2.0
    left_mask = xs_sorted <= split_x
    right_mask = ~left_mask
    spots: list[tuple[float, float]] = []
    for m in (left_mask, right_mask):
        if m.sum() == 0:
            # 理論上不會發生（間隙保證兩側皆有點），保守處理
            return []
        spots.append((float(xs_sorted[m].mean()), float(ys_sorted[m].mean())))
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


def detect_eyes_catface(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[EyePoint | None, EyePoint | None]:
    """以 Haar frontalcatface 偵測貓臉，於臉框上半部左右各找暗點作為雙眼。

    回傳 (left_eye, right_eye)，座標為整張影像絕對像素；無法定位則 (None, None)。
    眼睛座標仍屬估計值（is_estimated=True），來源標記為偵測器輔助（KEYPOINT）。
    """
    cascade = _get_cat_cascade()
    if cascade is None:
        return None, None

    try:
        import cv2
    except Exception:
        return None, None

    img_h, img_w = image.shape[0], image.shape[1]
    bx, by, bw, bh = bbox
    # 限制在動物 bbox 範圍內搜尋貓臉
    x0 = max(0, int(bx))
    y0 = max(0, int(by))
    x1 = min(img_w, int(bx + bw))
    y1 = min(img_h, int(by + bh))
    if x1 <= x0 or y1 <= y0:
        return None, None

    sub = image[y0:y1, x0:x1]
    gray = sub if sub.ndim == 2 else cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)

    faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
    if len(faces) == 0:
        return None, None

    # 取最大的一張臉（對應此動物）
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

    # 眼睛搜尋區：臉框上半偏中（y: 0.15~0.55）
    ey0 = fy + int(0.15 * fh)
    ey1 = fy + int(0.55 * fh)
    roi = gray[ey0:ey1, fx:fx + fw].astype(np.float64)
    if roi.size == 0 or roi.shape[1] < 4:
        return None, None
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    h, w = roi.shape

    def darkest_in(half: np.ndarray, offx: int) -> tuple[float, float]:
        my, mx = np.unravel_index(int(np.argmin(half)), half.shape)
        # 轉回整張影像絕對座標（sub 原點 x0,y0；臉框 fx；眼睛區 ey0）
        return float(x0 + fx + offx + mx), float(y0 + ey0 + my)

    lx, ly = darkest_in(roi[:, : w // 2], 0)
    rx, ry = darkest_in(roi[:, w // 2 :], w // 2)

    left = EyePoint(side=EyeSide.LEFT, x=lx, y=ly, is_estimated=True, source=EyeSource.KEYPOINT)
    right = EyePoint(side=EyeSide.RIGHT, x=rx, y=ry, is_estimated=True, source=EyeSource.KEYPOINT)
    return left, right


def detect_eyes(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    use_sam2: bool = False,
    category: str | None = None,
) -> tuple[EyePoint | None, EyePoint | None]:
    """偵測單一動物的左右眼。

    分流策略：
    - 貓（category in CAT_CLASSES）：先用 Haar 貓臉偵測 + 臉內眼睛定位；
      失敗則回退幾何/暗區啟發式。
    - 其他動物：直接使用幾何/暗區啟發式。

    回傳 (left_eye, right_eye)；任一無法可靠定位則為 None。座標為整張影像絕對像素。
    """
    if category in CAT_CLASSES:
        left, right = detect_eyes_catface(image, bbox)
        if left is not None and right is not None:
            if use_sam2:
                left, right = _refine_with_sam2(image, left, right)
            return left, right
        # 貓臉偵測失敗 → 回退啟發式

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

    # 先以啟發式來源建點；是否升級為 SAM2 由 _refine_with_sam2 實際結果決定（C-4：不造假）
    left = EyePoint(side=EyeSide.LEFT, x=lx, y=ly, is_estimated=True, source=EyeSource.HEURISTIC)
    right = EyePoint(side=EyeSide.RIGHT, x=rx, y=ry, is_estimated=True, source=EyeSource.HEURISTIC)

    if use_sam2:
        left, right = _refine_with_sam2(image, left, right)
    return left, right


def _refine_with_sam2(
    image: np.ndarray, left: EyePoint, right: EyePoint
) -> tuple[EyePoint, EyePoint]:
    """可選 SAM2 精修掛鉤。

    僅在 SAM2 實際可用且成功精修時，才將座標來源標記為 SAM2；
    未安裝或未精修時，保留輸入點的原始來源（不造假 source，C-4）。
    """
    try:
        import sam2  # noqa: F401
    except Exception:
        # 未安裝 → 回退，維持原始來源（不標記為 sam2）
        return left, right
    # 若有 SAM2，實際精修流程於整合環境實作；此處保守回傳輸入點，
    # 並且因為未進行實質精修，同樣不竄改來源為 sam2。
    return left, right
