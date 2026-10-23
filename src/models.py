"""pydantic 資料模型，對應 data-model.md 的實體與驗證規則。

座標慣例見 config.py。所有座標為像素，原點左上。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageStatus(str, Enum):
    OK = "ok"
    SKIPPED_INSUFFICIENT_ANIMALS = "skipped_insufficient_animals"
    ERROR = "error"


class EyeSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class EyeSource(str, Enum):
    HEURISTIC = "heuristic"
    SAM2 = "sam2"
    KEYPOINT = "keypoint"


class RecordType(str, Enum):
    EYE_DISTANCE = "eye_distance"  # 測量 A
    INTER_ANIMAL_RIGHT_EYE = "inter_animal_right_eye"  # 測量 B


class DistanceUnit(str, Enum):
    PIXEL = "pixel"
    REAL = "real"
    NORMALIZED = "normalized"


class EyePoint(BaseModel):
    """某動物的一隻眼睛座標（data-model: EyePoint）。"""

    side: EyeSide
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    is_estimated: bool = True  # 啟發式產生預設為估計值（FR-004）
    source: EyeSource = EyeSource.HEURISTIC
    confidence: float | None = None


class AnimalInstance(BaseModel):
    """圖片中單一隻被偵測的動物（data-model: AnimalInstance）。"""

    instance_id: str
    image_id: str
    category: str
    bbox: tuple[float, float, float, float]  # x, y, w, h
    mask_ref: str | None = None
    mask: Any = None  # HxW numpy 遮罩（0/1 或布林）；不序列化，供疊圖使用（P0-B）
    confidence: float = Field(ge=0.0, le=1.0)
    low_confidence: bool = False
    left_eye: EyePoint | None = None
    right_eye: EyePoint | None = None
    measurable: bool = False
    unmeasurable_reason: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _check_bbox(self) -> "AnimalInstance":
        _, _, w, h = self.bbox
        if w <= 0 or h <= 0:
            raise ValueError("bbox width/height 必須 > 0")
        return self


class Image(BaseModel):
    """輸入圖片（data-model: Image）。"""

    image_id: str
    file_name: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    animal_count: int = Field(ge=0)
    status: ImageStatus = ImageStatus.OK
    reason: str | None = None


class MeasurementRecord(BaseModel):
    """一筆量測結果（data-model: MeasurementRecord）。"""

    record_type: RecordType
    image_id: str
    subject_instance_id: str
    peer_instance_id: str | None = None
    distance: float | None = None
    unit: DistanceUnit = DistanceUnit.PIXEL
    measurable: bool = True
    reason: str | None = None
    low_confidence: bool = False

    @model_validator(mode="after")
    def _check_measurable(self) -> "MeasurementRecord":
        # 不可量測時不得輸出臆測距離（FR-007）
        if not self.measurable and self.distance is not None:
            raise ValueError("measurable=False 時 distance 必須為 None")
        return self


class EvaluationResult(BaseModel):
    """分割或眼睛定位的品質度量（data-model: EvaluationResult）。"""

    metric: str  # IoU / Dice / NME
    scope: str  # segmentation / eye_localization
    value: float
    image_id: str | None = None
    gt_source: str = ""
    sample_size: int = 0
