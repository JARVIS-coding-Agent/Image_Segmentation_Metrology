"""端到端 pipeline CLI 進入點（contracts/cli.md）。

退出碼：0 = 批次完成（即使部分圖片 skip/unmeasurable）；非 0 = 啟動期失敗。
後續階段（US1/US2/US3）會逐步填入 run() 的各處理步驟。
"""

from __future__ import annotations

import argparse
import sys

from src import logging_utils as log
from src.config import Config, set_deterministic


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


def run(args: argparse.Namespace) -> int:
    """執行 pipeline。回傳退出碼。"""
    cfg = config_from_args(args)
    set_deterministic(cfg.seed)
    log.info(
        f"pipeline start coco_dir={cfg.coco_dir} min_animals={cfg.min_animals} "
        f"seed={cfg.seed} use_sam2={cfg.use_sam2} eval={args.eval}"
    )
    # US1/US2/US3 階段將於此串接：篩選 → 分割 → 眼睛定位 → 測量 → 評估 → 視覺化 → CSV
    log.info("pipeline scaffold ready（處理步驟將於後續階段填入）")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
