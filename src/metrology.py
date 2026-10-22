"""距離測量公式（spec §4、FR-005/FR-006）。

座標為像素，原點左上。距離為歐氏距離。
- 測量 A：同一隻動物的左眼↔右眼距離。
- 測量 B：兩隻動物的右眼↔右眼距離（於 US2 填入）。
"""

from __future__ import annotations

import math
from itertools import combinations

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



def inter_animal_right_eye(
    animals: list[AnimalInstance],
    scale: float | None = None,
) -> list[MeasurementRecord]:
    """測量 B：兩隻動物右眼距離（FR-006）。

    配對規則：
    - 已定位右眼的動物恰 2 隻 → 1 筆。
    - 已定位右眼的動物恰 ≥3 隻 → 所有兩兩配對 C(n,2) 筆。
    - 僅 1 隻（或 0 隻）有右眼 → 回傳 1 筆「不適用」紀錄（measurable=False）。
    """
    with_right = [a for a in animals if a.right_eye is not None]

    if len(with_right) < 2:
        subject = animals[0].instance_id if animals else "unknown"
        image_id = animals[0].image_id if animals else "unknown"
        return [
            MeasurementRecord(
                record_type=RecordType.INTER_ANIMAL_RIGHT_EYE,
                image_id=image_id,
                subject_instance_id=subject,
                peer_instance_id=None,
                distance=None,
                measurable=False,
                reason="not_applicable_need_two_right_eyes",
            )
        ]

    records: list[MeasurementRecord] = []
    for a, b in combinations(with_right, 2):
        assert a.right_eye is not None and b.right_eye is not None
        dist_px = euclidean(a.right_eye.x, a.right_eye.y, b.right_eye.x, b.right_eye.y)
        dist, unit = _apply_scale(dist_px, scale)
        records.append(
            MeasurementRecord(
                record_type=RecordType.INTER_ANIMAL_RIGHT_EYE,
                image_id=a.image_id,
                subject_instance_id=a.instance_id,
                peer_instance_id=b.instance_id,
                distance=dist,
                unit=unit,
                measurable=True,
                low_confidence=a.low_confidence or b.low_confidence,
            )
        )
    return records
