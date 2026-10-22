"""距離測量公式（spec §4、FR-005/FR-006）。

座標為像素，原點左上。距離為歐氏距離。
- 測量 A：同一隻動物的左眼↔右眼距離。
- 測量 B：兩隻動物的右眼↔右眼距離（於 US2 填入）。
"""

from __future__ import annotations

import math

from src.models import (
    AnimalInstance,
    DistanceUnit,
    EyePoint,
    MeasurementRecord,
    RecordType,
)


def euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    """兩點歐氏距離 sqrt((x1-x2)^2 + (y1-y2)^2)。"""
    return math.hypot(x1 - x2, y1 - y2)


def _apply_scale(distance_px: float, scale: float | None) -> tuple[float, DistanceUnit]:
    """依比例尺換算距離與單位。無比例尺則回傳像素。"""
    if scale:
        return distance_px * scale, DistanceUnit.REAL
    return distance_px, DistanceUnit.PIXEL


def eye_distance(animal: AnimalInstance, scale: float | None = None) -> MeasurementRecord:
    """測量 A：單一動物雙眼距離（FR-005、FR-007）。

    左右眼任一缺失 → measurable=False、不輸出臆測距離。
    """
    left: EyePoint | None = animal.left_eye
    right: EyePoint | None = animal.right_eye

    if left is None or right is None:
        missing = "left" if left is None else "right"
        return MeasurementRecord(
            record_type=RecordType.EYE_DISTANCE,
            image_id=animal.image_id,
            subject_instance_id=animal.instance_id,
            distance=None,
            measurable=False,
            reason=f"missing_{missing}_eye",
            low_confidence=animal.low_confidence,
        )

    dist_px = euclidean(left.x, left.y, right.x, right.y)
    dist, unit = _apply_scale(dist_px, scale)
    return MeasurementRecord(
        record_type=RecordType.EYE_DISTANCE,
        image_id=animal.image_id,
        subject_instance_id=animal.instance_id,
        distance=dist,
        unit=unit,
        measurable=True,
        low_confidence=animal.low_confidence,
    )
