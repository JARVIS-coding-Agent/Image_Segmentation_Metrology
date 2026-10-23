"""COCO 篩選：挑出含 ≥min_animals 隻動物的圖片（FR-001）。

延遲載入 pycocotools，使本模組在未安裝時仍可匯入。
輸出可重現的 selected_images.json（圖片依 id 排序）。
"""

from __future__ import annotations

import json
import os

import numpy as np

from src.config import Config


def _animal_category_ids(coco, animal_classes: set[str]) -> set[int]:
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    return {c["id"] for c in cats if c["name"] in animal_classes}


class CocoGroundTruth:
    """以 pycocotools 讀取 COCO，提供動物 segmentation 的 ground truth 聯集 mask。

    供 `--eval` 計算 YOLO 預測 mask 與 COCO 標註 mask 的 IoU / Dice（P0-A）。
    延遲載入 pycocotools；未安裝則建構失敗，由呼叫端容錯。
    """

    def __init__(self, annotation_file: str, animal_classes: set[str]) -> None:
        from pycocotools.coco import COCO

        self._coco = COCO(annotation_file)
        self._animal_cat_ids = _animal_category_ids(self._coco, animal_classes)
        # file_name（去副檔名）→ image_id，供以檔名對應
        self._stem_to_id = {
            os.path.splitext(info["file_name"])[0]: info["id"]
            for info in self._coco.loadImgs(self._coco.getImgIds())
        }

    def union_animal_mask(self, image_stem: str) -> np.ndarray | None:
        """回傳該圖所有動物標註的聯集二值 mask（HxW）；查無則 None。"""
        img_id = self._stem_to_id.get(image_stem)
        if img_id is None:
            return None
        ann_ids = self._coco.getAnnIds(
            imgIds=[img_id], catIds=list(self._animal_cat_ids), iscrowd=None
        )
        anns = self._coco.loadAnns(ann_ids)
        if not anns:
            return None
        info = self._coco.loadImgs([img_id])[0]
        h, w = info["height"], info["width"]
        union = np.zeros((h, w), dtype=bool)
        for ann in anns:
            m = self._coco.annToMask(ann).astype(bool)
            if m.shape == union.shape:
                union |= m
            else:
                # annToMask 理論上與影像 h/w 一致；若不符代表 annotation 與 image
                # metadata 有落差，靜默跳過會讓 GT 少算、IoU 虛高。明確警告。
                from src import logging_utils as _log

                _log.warn(
                    f"coco gt mask shape mismatch image_id={img_id} "
                    f"ann={m.shape} img=({h}, {w}); skip this annotation"
                )
        return union


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
