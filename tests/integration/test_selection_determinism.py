"""US3 整合測試：篩選輸出可重現 + 量測決定性。"""

from __future__ import annotations

import json
import os

import numpy as np

from src.coco_selector import write_selection
from src.config import Config
from src.pipeline import process_image
from src.segmentation import Detection


def _image() -> np.ndarray:
    img = np.full((120, 240, 3), 220, dtype=np.uint8)
    img[20:26, 30:36] = 10
    img[20:26, 70:76] = 10
    img[20:26, 150:156] = 10
    img[20:26, 190:196] = 10
    return img


def _dets() -> list[Detection]:
    return [
        Detection(category="cat", bbox=(10, 10, 100, 100), confidence=0.9, mask=None, low_confidence=False),
        Detection(category="dog", bbox=(130, 10, 100, 100), confidence=0.9, mask=None, low_confidence=False),
    ]


def test_write_selection_is_sorted_and_reproducible(tmp_path):
    unsorted = [
        {"image_id": 30, "file_name": "c.jpg", "animal_count": 2},
        {"image_id": 10, "file_name": "a.jpg", "animal_count": 3},
        {"image_id": 20, "file_name": "b.jpg", "animal_count": 2},
    ]
    p1 = write_selection(unsorted, str(tmp_path / "out1"))
    p2 = write_selection(list(reversed(unsorted)), str(tmp_path / "out2"))
    data1 = json.load(open(p1, encoding="utf-8"))
    data2 = json.load(open(p2, encoding="utf-8"))
    assert [d["image_id"] for d in data1] == [10, 20, 30]  # 依 id 排序
    assert data1 == data2  # 輸入順序不同但輸出相同（可重現）


def test_measurement_deterministic(tmp_path):
    cfg = Config()
    cfg.output_dir = str(tmp_path)
    img = _image()
    _, animals_a, records_a = process_image(img, "d1", "d1.png", _dets(), cfg, cfg.output_dir)
    _, animals_b, records_b = process_image(img, "d1", "d1.png", _dets(), cfg, cfg.output_dir)

    dists_a = [r.distance for r in records_a]
    dists_b = [r.distance for r in records_b]
    assert dists_a == dists_b  # 相同輸入 → 完全一致（FR-014 / SC-005）
