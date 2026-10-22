"""全域設定：動物類別、閾值、路徑、種子、單位與座標慣例。

座標慣例（constitution I / spec FR-010）：
- 像素座標，原點位於影像左上角，x 向右遞增、y 向下遞增。
- 左右眼以「影像視角」的水平位置區分：x 較小者為左眼 (left)，較大者為右眼 (right)。
- 距離預設單位為像素 (pixel)。
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np

# COCO 動物類別（動物任選，涵蓋常見動物類別）
COCO_ANIMAL_CLASSES: tuple[str, ...] = (
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
)


@dataclass
class Config:
    """執行設定。"""

    coco_dir: str = field(default_factory=lambda: os.getenv("COCO_DIR", "data/coco"))
    yolo_weights: str = field(default_factory=lambda: os.getenv("YOLO_WEIGHTS", "yolo11n-seg.pt"))
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", "outputs"))
    min_animals: int = field(default_factory=lambda: int(os.getenv("MIN_ANIMALS", "2")))
    conf_threshold: float = field(default_factory=lambda: float(os.getenv("CONF_THRESHOLD", "0.25")))
    seed: int = field(default_factory=lambda: int(os.getenv("SEED", "42")))
    use_sam2: bool = field(default_factory=lambda: os.getenv("USE_SAM2", "false").lower() == "true")
    scale: float | None = None
    animal_classes: tuple[str, ...] = COCO_ANIMAL_CLASSES

    def __post_init__(self) -> None:
        scale_env = os.getenv("SCALE", "").strip()
        if self.scale is None and scale_env:
            self.scale = float(scale_env)

    @property
    def distance_unit(self) -> str:
        """距離單位：有比例尺時為 real，否則 pixel。"""
        return "real" if self.scale else "pixel"


def set_deterministic(seed: int) -> None:
    """固定隨機種子以確保決定性（spec FR-014）。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
