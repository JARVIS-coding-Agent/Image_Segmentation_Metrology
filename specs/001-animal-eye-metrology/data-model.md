# Phase 1 資料模型：動物眼睛量測 Pipeline

**Feature**: 001-animal-eye-metrology | **Date**: 2026-10-22

依 spec 的 Key Entities 與功能需求定義資料模型。實作以 pydantic 模型承載（`src/models.py`）。座標慣例：像素、原點左上、x 向右 y 向下；左右眼以影像視角 x 位置區分。

---

## 實體總覽與關係

```text
Image (1) ──< (N) AnimalInstance (1) ──< (0..2) EyePoint
Image (1) ──< (N) MeasurementRecord        # 測量 A：每 AnimalInstance 一筆；測量 B：每配對一筆
Image (1) ──< (N) EvaluationResult         # 可選，對照 ground truth
```

---

## Entity: Image（輸入圖片）

| 欄位 | 型別 | 說明 / 驗證 |
|------|------|-------------|
| image_id | str/int | COCO image id，唯一 |
| file_name | str | 影像檔名 |
| width | int | 影像寬（> 0） |
| height | int | 影像高（> 0） |
| animal_count | int | 偵測到的動物數；合格需 ≥ 2 |
| status | enum | `ok` / `skipped_insufficient_animals` / `error`；非 ok 須附 reason |
| reason | str? | status 非 ok 時的原因說明 |

**規則**：animal_count < 2 → status=`skipped_insufficient_animals`，不中斷整批（FR-001、FR-015、SC-001）。

## Entity: AnimalInstance（動物實例）

| 欄位 | 型別 | 說明 / 驗證 |
|------|------|-------------|
| instance_id | str | 於該圖唯一（如 `img123#0`） |
| image_id | str/int | 所屬 Image 外鍵 |
| category | str | 動物類別（COCO 類別名，如 cat/dog） |
| bbox | [x,y,w,h] | 像素座標，w/h > 0 |
| mask_ref | str? | instance mask 參照（RLE 或檔案路徑） |
| confidence | float | 偵測信心 [0,1] |
| low_confidence | bool | confidence < 閾值時為 true（FR-013） |
| left_eye | EyePoint? | 左眼（可為 None） |
| right_eye | EyePoint? | 右眼（可為 None） |
| measurable | bool | 是否可量測雙眼距離（需左右眼皆可靠） |
| unmeasurable_reason | str? | measurable=false 時的原因（FR-007） |

**規則**：左右眼任一缺失或不可靠 → measurable=false、填 reason、不輸出臆測距離（FR-007）。

## Entity: EyePoint（眼睛座標）

| 欄位 | 型別 | 說明 / 驗證 |
|------|------|-------------|
| side | enum | `left` / `right`（影像視角，x 小者為 left） |
| x | float | 像素 x，0 ≤ x < width |
| y | float | 像素 y，0 ≤ y < height |
| is_estimated | bool | 啟發式產生 → true（FR-004） |
| source | enum | `heuristic` / `sam2` / `keypoint` |
| confidence | float? | 定位信心（可選） |

## Entity: MeasurementRecord（量測紀錄）

| 欄位 | 型別 | 說明 / 驗證 |
|------|------|-------------|
| record_type | enum | `eye_distance`（測量 A）/ `inter_animal_right_eye`（測量 B） |
| image_id | str/int | 所屬 Image |
| subject_instance_id | str | 測量 A：該動物；測量 B：動物 A |
| peer_instance_id | str? | 測量 B：動物 B（測量 A 為 None） |
| distance | float? | 歐氏距離；不可量測時為 None |
| unit | enum | `pixel`（預設）/ `real`（提供比例尺時）/ `normalized` |
| measurable | bool | 是否成功量測 |
| reason | str? | measurable=false 時原因 |
| low_confidence | bool | 來源實例為低信心時標記 |

**規則**：
- 測量 A：每個 measurable AnimalInstance 一筆（FR-005）。
- 測量 B：恰兩隻動物 → 一筆；恰 ≥3 隻已定位右眼 → 所有兩兩配對 C(n,2) 筆（FR-006）。
- 僅一隻動物有右眼 → 測量 B 標記「不適用（需至少兩隻右眼）」。

## Entity: EvaluationResult（評估結果）

| 欄位 | 型別 | 說明 / 驗證 |
|------|------|-------------|
| metric | enum | `IoU` / `Dice` / `NME` |
| scope | enum | `segmentation` / `eye_localization` |
| value | float | 指標數值 |
| image_id | str/int? | 對應圖片（彙總時可為 None） |
| gt_source | str | ground truth 來源（人工標註檔參照） |
| sample_size | int | 評估樣本數（NME 須 ≥ 10） |

---

## 狀態轉移（Image 處理流程）

```text
received → segmented → eyes_localized → measured → evaluated(optional) → done
             │              │               │
             ▼              ▼               ▼
      skipped(<2 animals)  per-instance    per-instance
      / error(flag+skip)   unmeasurable    measurable=false
```

單張 error 以隔離方式記錄並續跑下一張（SC-001、FR-015）。
