"""YOLO11-seg 動物實例分割（research R1、FR-002/FR-013）。

延遲載入 ultralytics，使本模組在未安裝時仍可匯入（供其他模組與測試）。
輸出每隻動物的 category / bbox / mask / confidence，並標記 low_confidence。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import Config


@dataclass
class Detection:
    """單一動物偵測結果。"""

    category: str
    bbox: tuple[float, float, float, float]  # x, y, w, h（像素）
    confidence: float
    mask: np.ndarray | None  # HxW 布林/0-1 遮罩（可為 None）
    low_confidence: bool


class AnimalSegmenter:
    """封裝 YOLO11-seg，僅保留動物類別偵測。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # 延遲載入

            self._model = YOLO(self.cfg.yolo_weights)
        return self._model

    def segment(self, image: np.ndarray) -> list[Detection]:
        """對單張影像推論，回傳動物偵測清單。

        以 retina_masks=True 推論，讓 YOLO 回傳「已去除 letterbox padding、
        且對齊原圖解析度」的 mask。這樣 mask 與 boxes.xyxy 共用同一個原圖座標系，
        下游疊圖與 IoU/Dice 不需再對 mask 做任何縮放（避免把 padding 一起拉伸
        導致非方形影像錯位）。
        """
        model = self._load()
        results = model(image, verbose=False, retina_masks=True)
        detections: list[Detection] = []
        animal_set = set(self.cfg.animal_classes)

        for r in results:
            names = r.names
            boxes = getattr(r, "boxes", None)
            masks = getattr(r, "masks", None)
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                category = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                if category not in animal_set:
                    continue
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
                bbox = (x1, y1, x2 - x1, y2 - y1)
                mask = None
                if masks is not None and masks.data is not None and i < len(masks.data):
                    mask = masks.data[i].cpu().numpy()
                detections.append(
                    Detection(
                        category=category,
                        bbox=bbox,
                        confidence=conf,
                        mask=mask,
                        low_confidence=conf < self.cfg.conf_threshold,
                    )
                )
        return detections


def mask_to_original(
    mask: np.ndarray | None, img_h: int, img_w: int
) -> np.ndarray | None:
    """將 YOLO mask 正規化為對齊原圖的 (img_h, img_w) 布林遮罩。

    疊圖（visualize）與 IoU/Dice（pipeline）兩個 consumer 共用本函式，確保兩邊
    用「完全相同」的 mask→原圖轉換，不會各自引入不一致的對齊誤差。

    前提：segment() 以 retina_masks=True 推論，mask 已去除 letterbox padding 且
    對齊原圖解析度，因此正常情況 shape 已等於 (img_h, img_w)，直接回傳。

    若 shape 不符原圖：本函式「拒絕猜測」——回傳 None 並記 warning。原因是我們
    無從判斷這個不符尺寸的 mask 是「乾淨的等比縮放」還是「含 letterbox padding
    的 padded mask」；對後者盲目 resize 會悄悄重演座標錯位，還產出看似正常的錯誤
    IoU。要支援 padded mask 必須帶入原始 letterbox 的 scale/pad metadata 反投影，
    那屬另一條路徑（見 segment() 的 retina_masks）。回傳 None 表示無可信 mask。
    """
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim != 2 or m.size == 0:
        return None
    binary = m > 0.5
    if binary.shape == (img_h, img_w):
        return binary
    # shape 不符 → 來源座標系不明（可能含 letterbox padding），拒絕猜測
    from src import logging_utils as _log

    _log.warn(
        f"mask shape {binary.shape} != image ({img_h}, {img_w})；"
        "來源座標系不明（可能含 letterbox padding），拒絕盲目 resize。"
        "請確認 segment() 以 retina_masks=True 推論。"
    )
    return None
