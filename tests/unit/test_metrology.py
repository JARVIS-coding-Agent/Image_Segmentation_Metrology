"""metrology 單元測試：測量 A（雙眼距離）。測量 B 測試於 US2 補上。"""

from __future__ import annotations

import math

import pytest

from src.metrology import euclidean, eye_distance
from src.models import AnimalInstance, DistanceUnit, EyePoint, RecordType


def _animal(left: EyePoint | None, right: EyePoint | None, **kw) -> AnimalInstance:
    return AnimalInstance(
        instance_id=kw.get("instance_id", "img1#0"),
        image_id=kw.get("image_id", "img1"),
        category=kw.get("category", "cat"),
        bbox=kw.get("bbox", (0.0, 0.0, 100.0, 100.0)),
        confidence=kw.get("confidence", 0.9),
        low_confidence=kw.get("low_confidence", False),
        left_eye=left,
        right_eye=right,
    )


def test_euclidean_3_4_5():
    assert euclidean(0, 0, 3, 4) == pytest.approx(5.0)


def test_euclidean_coincident_is_zero():
    assert euclidean(10, 10, 10, 10) == 0.0


def test_eye_distance_known_coords():
    a = _animal(
        EyePoint(side="left", x=0, y=0),
        EyePoint(side="right", x=3, y=4),
    )
    rec = eye_distance(a)
    assert rec.record_type == RecordType.EYE_DISTANCE
    assert rec.measurable is True
    assert rec.distance == pytest.approx(5.0)
    assert rec.unit == DistanceUnit.PIXEL
    assert rec.subject_instance_id == "img1#0"


def test_eye_distance_coincident_zero():
    a = _animal(
        EyePoint(side="left", x=50, y=50),
        EyePoint(side="right", x=50, y=50),
    )
    rec = eye_distance(a)
    assert rec.measurable is True
    assert rec.distance == pytest.approx(0.0)


def test_eye_distance_missing_right_eye_unmeasurable():
    a = _animal(EyePoint(side="left", x=10, y=10), None)
    rec = eye_distance(a)
    assert rec.measurable is False
    assert rec.distance is None
    assert rec.reason == "missing_right_eye"


def test_eye_distance_missing_left_eye_unmeasurable():
    a = _animal(None, EyePoint(side="right", x=10, y=10))
    rec = eye_distance(a)
    assert rec.measurable is False
    assert rec.distance is None
    assert rec.reason == "missing_left_eye"


def test_eye_distance_with_scale_real_unit():
    a = _animal(
        EyePoint(side="left", x=0, y=0),
        EyePoint(side="right", x=10, y=0),
    )
    rec = eye_distance(a, scale=0.5)  # 1 px = 0.5 單位
    assert rec.unit == DistanceUnit.REAL
    assert rec.distance == pytest.approx(5.0)


def test_eye_distance_propagates_low_confidence():
    a = _animal(
        EyePoint(side="left", x=0, y=0),
        EyePoint(side="right", x=3, y=4),
        low_confidence=True,
    )
    rec = eye_distance(a)
    assert rec.low_confidence is True
