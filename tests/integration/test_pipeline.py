"""US1 整合測試：process_image 端到端（合成影像 + 假偵測，不依賴 YOLO）。"""

from __future__ import annotations

import os

import numpy as np

from src.config import Config
from src.csv_writer import record_to_row, write_csv
from src.models import RecordType
from src.pipeline import process_image
from src.segmentation import Detection


def _image_with_two_animals() -> np.ndarray:
    """合成 120x240 影像，左右各一隻動物，各含兩個暗斑（眼睛）。"""
    img = np.full((120, 240, 3), 220, dtype=np.uint8)
    # 動物 A（左半）頭部暗斑
    img[20:26, 30:36] = 10
    img[20:26, 70:76] = 10
    # 動物 B（右半）頭部暗斑
    img[20:26, 150:156] = 10
    img[20:26, 190:196] = 10
    return img


def _two_detections() -> list[Detection]:
    return [
        Detection(category="cat", bbox=(10, 10, 100, 100), confidence=0.9, mask=None, low_confidence=False),
        Detection(category="cat", bbox=(130, 10, 100, 100), confidence=0.9, mask=None, low_confidence=False),
    ]


def test_process_image_measures_two_animals(tmp_path):
    cfg = Config()
    cfg.output_dir = str(tmp_path)
    img = _image_with_two_animals()
    image_obj, animals, records = process_image(
        img, "t1", "t1.png", _two_detections(), cfg, cfg.output_dir
    )

    assert image_obj.status.value == "ok"
    assert len(animals) == 2
    eye_records = [r for r in records if r.record_type == RecordType.EYE_DISTANCE]
    assert len(eye_records) == 2
    # 至少一隻可量測並有距離
    measurable = [r for r in eye_records if r.measurable]
    assert len(measurable) >= 1
    for r in measurable:
        assert r.distance is not None and r.distance > 0

    # 疊圖檔已產生
    overlay = os.path.join(cfg.output_dir, "annotated", "t1.png")
    assert os.path.exists(overlay)


def test_process_image_skips_insufficient_animals(tmp_path):
    cfg = Config()
    cfg.output_dir = str(tmp_path)
    img = _image_with_two_animals()
    only_one = _two_detections()[:1]
    image_obj, animals, records = process_image(
        img, "t2", "t2.png", only_one, cfg, cfg.output_dir
    )
    assert image_obj.status.value == "skipped_insufficient_animals"
    assert animals == []
    assert records == []


def test_csv_roundtrip_eye_distance(tmp_path):
    cfg = Config()
    cfg.output_dir = str(tmp_path)
    img = _image_with_two_animals()
    _, animals, records = process_image(
        img, "t3", "t3.png", _two_detections(), cfg, cfg.output_dir
    )
    by_id = {a.instance_id: a for a in animals}
    rows = []
    for r in records:
        subj = by_id.get(r.subject_instance_id)
        peer = by_id.get(r.peer_instance_id) if r.peer_instance_id else None
        if subj is None:
            continue
        rows.append(record_to_row(r, subj, peer))
    csv_path = write_csv(rows, os.path.join(str(tmp_path), "measurements.csv"))
    assert os.path.exists(csv_path)
    content = open(csv_path, encoding="utf-8").read()
    assert "record_type" in content
    assert "eye_distance" in content


def test_process_image_produces_inter_animal_record(tmp_path):
    cfg = Config()
    cfg.output_dir = str(tmp_path)
    img = _image_with_two_animals()
    _, animals, records = process_image(
        img, "t4", "t4.png", _two_detections(), cfg, cfg.output_dir
    )
    inter = [r for r in records if r.record_type == RecordType.INTER_ANIMAL_RIGHT_EYE]
    # 兩隻動物皆定位右眼 → 恰 1 筆且可量測；若眼睛未定位則為不適用 1 筆
    assert len(inter) == 1
    if all(a.right_eye is not None for a in animals):
        assert inter[0].measurable is True
        assert inter[0].distance is not None and inter[0].distance > 0
