"""COCO 篩選：挑出含 ≥min_animals 隻動物的圖片（FR-001）。

延遲載入 pycocotools，使本模組在未安裝時仍可匯入。
輸出可重現的 selected_images.json（圖片依 id 排序）。
"""

from __future__ import annotations

import json
import os

from src.config import Config


def _animal_category_ids(coco, animal_classes: set[str]) -> set[int]:
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    return {c["id"] for c in cats if c["name"] in animal_classes}


def select_images(cfg: Config, annotation_file: str) -> list[dict]:
    """回傳含 ≥min_animals 隻動物的圖片清單（可重現，依 image_id 排序）。

    每筆：{"image_id", "file_name", "animal_count"}。
    """
    from pycocotools.coco import COCO  # 延遲載入

    coco = COCO(annotation_file)
    animal_cat_ids = _animal_category_ids(coco, set(cfg.animal_classes))

    selected: list[dict] = []
    for img_id in sorted(coco.getImgIds()):
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=list(animal_cat_ids), iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        animal_count = len(anns)
        if animal_count >= cfg.min_animals:
            info = coco.loadImgs([img_id])[0]
            selected.append(
                {
                    "image_id": img_id,
                    "file_name": info.get("file_name", ""),
                    "animal_count": animal_count,
                }
            )
    return selected


def write_selection(selected: list[dict], output_dir: str) -> str:
    """將篩選結果寫入 selected_images.json（排序後，確保可重現）。"""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "selected_images.json")
    ordered = sorted(selected, key=lambda d: d["image_id"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
    return out_path
