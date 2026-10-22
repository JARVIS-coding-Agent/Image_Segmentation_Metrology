"""量測結果 CSV 輸出（contracts/measurements-csv.md、FR-009）。

固定欄位 schema。使用標準庫 csv 以減少硬相依。
"""

from __future__ import annotations

import csv
import os

from src.models import AnimalInstance, MeasurementRecord, RecordType

CSV_FIELDS = [
    "image_id",
    "record_type",
    "subject_instance_id",
    "peer_instance_id",
    "category",
    "left_eye_x",
    "left_eye_y",
    "right_eye_x",
    "right_eye_y",
    "peer_right_eye_x",
    "peer_right_eye_y",
    "distance",
    "unit",
    "is_estimated",
    "low_confidence",
    "measurable",
    "reason",
]


def _fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.4g}" if isinstance(v, float) else str(v)


def record_to_row(
    rec: MeasurementRecord,
    subject: AnimalInstance,
    peer: AnimalInstance | None = None,
) -> dict[str, str]:
    """將一筆 MeasurementRecord 轉為 CSV 列（依 record_type 填相關欄位）。"""
    row = {k: "" for k in CSV_FIELDS}
    row["image_id"] = str(rec.image_id)
    row["record_type"] = rec.record_type.value
    row["subject_instance_id"] = rec.subject_instance_id
    row["peer_instance_id"] = rec.peer_instance_id or ""
    row["unit"] = rec.unit.value
    row["distance"] = _fmt(rec.distance)
    row["measurable"] = str(rec.measurable)
    row["low_confidence"] = str(rec.low_confidence)
    row["reason"] = rec.reason or ""

    if rec.record_type == RecordType.EYE_DISTANCE:
        row["category"] = subject.category
        if subject.left_eye:
            row["left_eye_x"] = _fmt(subject.left_eye.x)
            row["left_eye_y"] = _fmt(subject.left_eye.y)
        if subject.right_eye:
            row["right_eye_x"] = _fmt(subject.right_eye.x)
            row["right_eye_y"] = _fmt(subject.right_eye.y)
        row["is_estimated"] = str(
            bool(subject.left_eye and subject.left_eye.is_estimated)
        )
    else:  # INTER_ANIMAL_RIGHT_EYE
        cat = subject.category + ("/" + peer.category if peer else "")
        row["category"] = cat
        if subject.right_eye:
            row["right_eye_x"] = _fmt(subject.right_eye.x)
            row["right_eye_y"] = _fmt(subject.right_eye.y)
        if peer and peer.right_eye:
            row["peer_right_eye_x"] = _fmt(peer.right_eye.x)
            row["peer_right_eye_y"] = _fmt(peer.right_eye.y)
        est = bool(subject.right_eye and subject.right_eye.is_estimated)
        row["is_estimated"] = str(est)
    return row


def write_csv(rows: list[dict[str, str]], output_path: str) -> str:
    """寫入固定欄位 CSV，回傳路徑。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
