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
        """對單張影像推論，回傳動物偵測清單。"""
        model = self._load()
        results = model(image, verbose=False)
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
