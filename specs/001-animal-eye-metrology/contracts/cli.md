# 契約：CLI 介面

**Feature**: 001-animal-eye-metrology | **Date**: 2026-10-22

本專案對外介面為離線 CLI（`src/pipeline.py` 進入點）。無線上 API 與登入機制（README 標註「不適用」）。

## 指令

```bash
python -m src.pipeline [OPTIONS]
```

## 子流程 / 旗標

| 旗標 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `--coco-dir` | path | `data/coco` | COCO 影像與 annotations 目錄 |
| `--input` | path? | 無 | 指定單張/目錄輸入；省略則用篩選結果 |
| `--output-dir` | path | `outputs` | 疊圖、CSV、報告輸出目錄 |
| `--min-animals` | int | 2 | 篩選門檻（含以上） |
| `--conf-threshold` | float | 0.25 | 分割信心閾值；低於視為 low_confidence 參考 |
| `--use-sam2` | flag | false | 啟用 SAM2 眼睛精修（可選） |
| `--scale` | float? | 無 | 比例尺（實際長度/像素）；提供則輸出 real 單位 |
| `--seed` | int | 42 | 隨機種子（決定性） |
| `--eval` | flag | false | 啟用評估（需 `data/ground_truth`） |
| `--limit` | int? | 無 | 限制處理圖片數（除錯用） |

## 行為契約

- 退出碼 0：批次完成（即使部分圖片被 skip 或標記 unmeasurable）。
- 退出碼非 0：僅限無法啟動（缺模型權重、COCO 路徑不存在等啟動期錯誤）。
- 單張圖片處理失敗 → 記錄 error 並續跑，不影響退出碼（SC-001、FR-015）。
- 相同輸入 + 相同旗標 + 相同 seed → 相同輸出（FR-014）。

## 輸出產物

| 產物 | 路徑 | 說明 |
|------|------|------|
| 疊圖 | `outputs/annotated/<image_id>.png` | mask + 左右眼 + 連線 + 距離數值 |
| 量測 CSV | `outputs/measurements.csv` | 見 measurements-csv.md |
| 評估報告 | `outputs/eval_report.md` | IoU/Dice/NME + 效能實測（`--eval` 時） |
| 篩選清單 | `outputs/selected_images.json` | 可重現的合格圖片清單 |

## 標準輸出訊息（範例語意）

- `[skip] image=<id> reason=insufficient_animals (count=1)`
- `[warn] instance=<id> unmeasurable reason=eye_not_reliable`
- `[info] image=<id> measured animals=2 pairs=1 elapsed=1.8s`
