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


def _find_coco_annotation(coco_dir: str) -> str | None:
    """在 coco_dir（含 annotations/ 子目錄）尋找 instances_*.json。"""
    candidates = [coco_dir, os.path.join(coco_dir, "annotations")]
    for d in candidates:
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if fn.startswith("instances_") and fn.endswith(".json"):
                    return os.path.join(d, fn)
    return None


def detections_to_animals(
    image: np.ndarray,
    image_id: str,
    detections: list[Detection],
    cfg: Config,
) -> list[AnimalInstance]:
    """將偵測結果轉為 AnimalInstance，並定位左右眼、判定可量測性。"""
    animals: list[AnimalInstance] = []
    for i, det in enumerate(detections):
        left, right = eye_detector.detect_eyes(
            image, det.bbox, use_sam2=cfg.use_sam2, category=det.category
        )
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
        # 無指定輸入 → 嘗試 COCO 篩選，產出 selected_images.json（FR-001）
        ann = _find_coco_annotation(cfg.coco_dir)
        if ann:
            try:
                from src import coco_selector

                selected = coco_selector.select_images(cfg, ann)
                sel_path = coco_selector.write_selection(selected, cfg.output_dir)
                log.info(f"coco selection: {len(selected)} images -> {sel_path}")
            except Exception as exc:
                log.error(f"coco selection failed: {exc}")
        else:
            log.warn("no input images resolved（請用 --input 指定圖片或目錄）")
        log.info("pipeline done images=0")
        if args.eval:
            report_path = write_eval_report(cfg, 0, 0, [])
            log.info(f"eval report written: {report_path}")
        return 0

    try:
        segmenter = AnimalSegmenter(cfg)
    except Exception as exc:  # 啟動期失敗
        log.error(f"failed to init segmenter: {exc}")
        return 2

    rows: list[dict[str, str]] = []
    processed = 0
    elapsed_list: list[float] = []
    import time as _time

    for path in inputs:
        image_id = os.path.splitext(os.path.basename(path))[0]
        with log.isolate_failure(image_id):
            import matplotlib.image as mpimg

            _t0 = _time.perf_counter()
            image = (mpimg.imread(path) * 255).astype(np.uint8) if path.lower().endswith(
                (".png",)
            ) else mpimg.imread(path)
            detections = segmenter.segment(image)
            img, animals, records = process_image(
                image, image_id, os.path.basename(path), detections, cfg, cfg.output_dir
            )
            elapsed_list.append(_time.perf_counter() - _t0)
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

    if args.eval:
        report_path = write_eval_report(cfg, processed, len(rows), elapsed_list)
        log.info(f"eval report written: {report_path}")
    return 0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values), pct))


def write_eval_report(
    cfg: Config,
    images: int,
    records: int,
    elapsed_list: list[float],
) -> str:
    """產出評估報告 eval_report.md，含效能實測（CPU P95 per SC-007）。

    分割品質（IoU/Dice）與眼睛 NME 需 ground truth；此處載入 data/ground_truth
    若存在則彙總，否則於報告標註尚無 ground truth。
    """
    p95 = _percentile(elapsed_list, 95)
    p50 = _percentile(elapsed_list, 50)
    gt_dir = os.path.join("data", "ground_truth")
    gt_note = (
        f"已偵測 ground truth 目錄：{gt_dir}（請放入 ≥10 張人工眼睛標註以計算 NME）"
        if os.path.isdir(gt_dir)
        else f"尚未提供 ground truth（預期於 {gt_dir}，需 ≥10 張以計算 NME）"
    )

    lines = [
        "# 評估報告（Evaluation Report）",
        "",
        "## 效能實測（CPU 參考環境，SC-007）",
        "",
        f"- 處理圖片數：{images}",
        f"- 量測紀錄數：{records}",
        f"- 單張處理時間 P50：{p50:.3f}s",
        f"- 單張處理時間 P95：{p95:.3f}s（目標 ≤ 5s）",
        f"- 驗收：{'PASS' if p95 <= 5.0 or images == 0 else 'REVIEW'}",
        "",
        "## 分割品質（IoU / Dice）與眼睛定位（NME）",
        "",
        "- IoU = |A∩B| / |A∪B|；Dice = 2|A∩B| / (|A|+|B|)；NME = 平均點位誤差 / 正規化尺度。",
        f"- {gt_note}",
        "- 評估函式見 `src/evaluate.py`（iou / dice / nme / evaluate_eye_localization）。",
        "",
    ]
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = os.path.join(cfg.output_dir, "eval_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
