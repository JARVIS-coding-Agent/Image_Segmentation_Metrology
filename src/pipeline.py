"""端到端 pipeline CLI 進入點（contracts/cli.md）。

退出碼：0 = 批次完成（即使部分圖片 skip/unmeasurable）；非 0 = 啟動期失敗。
US1：分割 → 眼睛定位 → 測量 A → 疊圖 → CSV。US2/US3 後續填入。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from src import csv_writer, eye_detector, metrology
from src import logging_utils as log
from src.config import Config, set_deterministic
from src.models import AnimalInstance, Image, ImageStatus, MeasurementRecord
from src.segmentation import AnimalSegmenter, Detection


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="COCO 動物分割 + 眼睛定位 + 雙眼/兩動物右眼距離量測 pipeline",
    )
    p.add_argument("--coco-dir", default=None, help="COCO 影像與 annotations 目錄")
    p.add_argument("--input", default=None, help="指定單張/目錄輸入；省略則用篩選結果")
    p.add_argument("--output-dir", default=None, help="輸出目錄")
    p.add_argument("--min-animals", type=int, default=None, help="篩選門檻（含以上）")
    p.add_argument("--conf-threshold", type=float, default=None, help="分割信心閾值")
    p.add_argument("--use-sam2", action="store_true", help="啟用 SAM2 眼睛精修（可選）")
    p.add_argument("--scale", type=float, default=None, help="比例尺（實際長度/像素）")
    p.add_argument("--seed", type=int, default=None, help="隨機種子（決定性）")
    p.add_argument("--eval", action="store_true", help="啟用評估（需 data/ground_truth）")
    p.add_argument("--limit", type=int, default=None, help="限制處理圖片數（除錯用）")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    """以 CLI 旗標覆寫環境變數預設值。"""
    cfg = Config()
    if args.coco_dir is not None:
        cfg.coco_dir = args.coco_dir
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.min_animals is not None:
        cfg.min_animals = args.min_animals
    if args.conf_threshold is not None:
        cfg.conf_threshold = args.conf_threshold
    if args.seed is not None:
        cfg.seed = args.seed
    if args.use_sam2:
        cfg.use_sam2 = True
    if args.scale is not None:
        cfg.scale = args.scale
    return cfg


def detections_to_animals(
    image: np.ndarray,
    image_id: str,
    detections: list[Detection],
    cfg: Config,
) -> list[AnimalInstance]:
    """將偵測結果轉為 AnimalInstance，並定位左右眼、判定可量測性。"""
    animals: list[AnimalInstance] = []
    for i, det in enumerate(detections):
        left, right = eye_detector.detect_eyes(image, det.bbox, use_sam2=cfg.use_sam2)
        measurable = left is not None and right is not None
        reason = None if measurable else "eye_not_reliable"
        animals.append(
            AnimalInstance(
                instance_id=f"{image_id}#{i}",
                image_id=image_id,
                category=det.category,
                bbox=det.bbox,
                confidence=det.confidence,
                low_confidence=det.low_confidence,
                left_eye=left,
                right_eye=right,
                measurable=measurable,
                unmeasurable_reason=reason,
            )
        )
    return animals


def process_image(
    image: np.ndarray,
    image_id: str,
    file_name: str,
    detections: list[Detection],
    cfg: Config,
    output_dir: str,
) -> tuple[Image, list[AnimalInstance], list[MeasurementRecord]]:
    """處理單張影像的 US1 流程：篩選門檻 → 眼睛定位 → 測量 A → 疊圖。"""
    h, w = image.shape[0], image.shape[1]
    animal_count = len(detections)

    if animal_count < cfg.min_animals:
        log.skip(f"image={image_id} reason=insufficient_animals (count={animal_count})")
        img = Image(
            image_id=image_id,
            file_name=file_name,
            width=w,
            height=h,
            animal_count=animal_count,
            status=ImageStatus.SKIPPED_INSUFFICIENT_ANIMALS,
            reason=f"count={animal_count} < min_animals={cfg.min_animals}",
        )
        return img, [], []

    animals = detections_to_animals(image, image_id, detections, cfg)
    records: list[MeasurementRecord] = []
    for a in animals:
        rec = metrology.eye_distance(a, scale=cfg.scale)
        records.append(rec)
        if not rec.measurable:
            log.warn(f"instance={a.instance_id} unmeasurable reason={rec.reason}")

    # 測量 B：兩動物右眼距離（FR-006）
    pair_records = metrology.inter_animal_right_eye(animals, scale=cfg.scale)
    records.extend(pair_records)

    # 疊圖（含測量 B 右眼配對連線）
    by_id = {a.instance_id: a for a in animals}
    right_eye_pairs = [
        (by_id[r.subject_instance_id], by_id[r.peer_instance_id], r.distance)
        for r in pair_records
        if r.measurable and r.peer_instance_id and r.distance is not None
    ]

    from src.visualize import render_overlay

    overlay_path = os.path.join(output_dir, "annotated", f"{image_id}.png")
    render_overlay(image, animals, overlay_path, right_eye_pairs=right_eye_pairs)

    measured = sum(1 for r in records if r.measurable and r.record_type.value == "eye_distance")
    pairs = sum(1 for r in pair_records if r.measurable)
    log.info(f"image={image_id} measured animals={measured}/{animal_count} pairs={pairs}")
    img = Image(
        image_id=image_id,
        file_name=file_name,
        width=w,
        height=h,
        animal_count=animal_count,
        status=ImageStatus.OK,
    )
    return img, animals, records


def run(args: argparse.Namespace) -> int:
    """執行 pipeline。回傳退出碼。"""
    cfg = config_from_args(args)
    set_deterministic(cfg.seed)
    log.info(
        f"pipeline start coco_dir={cfg.coco_dir} min_animals={cfg.min_animals} "
        f"seed={cfg.seed} use_sam2={cfg.use_sam2} eval={args.eval}"
    )

    # 收集輸入影像路徑
    inputs: list[str] = []
    if args.input and os.path.isdir(args.input):
        for fn in sorted(os.listdir(args.input)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                inputs.append(os.path.join(args.input, fn))
    elif args.input and os.path.isfile(args.input):
        inputs.append(args.input)

    if args.limit is not None:
        inputs = inputs[: args.limit]

    if not inputs:
        log.warn("no input images resolved（請用 --input 指定圖片或目錄）")
        log.info("pipeline done images=0")
        return 0

    try:
        segmenter = AnimalSegmenter(cfg)
    except Exception as exc:  # 啟動期失敗
        log.error(f"failed to init segmenter: {exc}")
        return 2

    rows: list[dict[str, str]] = []
    processed = 0
    for path in inputs:
        image_id = os.path.splitext(os.path.basename(path))[0]
        with log.isolate_failure(image_id):
            import matplotlib.image as mpimg

            image = (mpimg.imread(path) * 255).astype(np.uint8) if path.lower().endswith(
                (".png",)
            ) else mpimg.imread(path)
            detections = segmenter.segment(image)
            img, animals, records = process_image(
                image, image_id, os.path.basename(path), detections, cfg, cfg.output_dir
            )
            by_id = {a.instance_id: a for a in animals}
            for rec in records:
                subj = by_id.get(rec.subject_instance_id)
                peer = by_id.get(rec.peer_instance_id) if rec.peer_instance_id else None
                if subj is None:
                    continue
                rows.append(csv_writer.record_to_row(rec, subj, peer))
            processed += 1

    csv_path = os.path.join(cfg.output_dir, "measurements.csv")
    csv_writer.write_csv(rows, csv_path)
    log.info(f"pipeline done images={processed} records={len(rows)} csv={csv_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
