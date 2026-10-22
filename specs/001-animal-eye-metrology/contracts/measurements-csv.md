# 契約：量測結果 CSV（measurements.csv）

**Feature**: 001-animal-eye-metrology | **Date**: 2026-10-22

固定欄位 schema，對應 FR-009 / data-model 的 MeasurementRecord。每列為一筆量測（測量 A 或測量 B）。

## 欄位定義

| 欄位 | 型別 | 必填 | 說明 |
|------|------|:----:|------|
| image_id | str/int | 是 | 所屬圖片 id |
| record_type | str | 是 | `eye_distance`（測量 A）/ `inter_animal_right_eye`（測量 B） |
| subject_instance_id | str | 是 | 測量 A：該動物；測量 B：動物 A |
| peer_instance_id | str | 否 | 測量 B：動物 B（測量 A 留空） |
| category | str | 是 | 動物類別（測量 B 可為兩類別以 `/` 連接或留 subject 類別） |
| left_eye_x | float | 否 | 測量 A：subject 左眼 x（測量 B 留空） |
| left_eye_y | float | 否 | 測量 A：subject 左眼 y |
| right_eye_x | float | 否 | subject 右眼 x |
| right_eye_y | float | 否 | subject 右眼 y |
| peer_right_eye_x | float | 否 | 測量 B：peer 右眼 x |
| peer_right_eye_y | float | 否 | 測量 B：peer 右眼 y |
| distance | float | 否 | 歐氏距離；unmeasurable 時留空 |
| unit | str | 是 | `pixel` / `real` / `normalized` |
| is_estimated | bool | 是 | 眼睛座標是否為啟發式估計 |
| low_confidence | bool | 是 | 來源實例是否低信心 |
| measurable | bool | 是 | 是否成功量測 |
| reason | str | 否 | measurable=false 時的原因 |

## 範例列（CSV）

```csv
image_id,record_type,subject_instance_id,peer_instance_id,category,left_eye_x,left_eye_y,right_eye_x,right_eye_y,peer_right_eye_x,peer_right_eye_y,distance,unit,is_estimated,low_confidence,measurable,reason
285,eye_distance,285#0,,cat,210.5,160.2,248.0,162.7,,,37.6,pixel,True,False,True,
285,eye_distance,285#1,,cat,402.1,180.0,441.3,181.5,,,39.2,pixel,True,False,True,
285,inter_animal_right_eye,285#0,285#1,cat/cat,,,248.0,162.7,441.3,181.5,194.2,pixel,True,False,True,
139,eye_distance,139#0,,dog,,,,,,,,pixel,True,False,False,eye_not_reliable
```

## 規則

- 欄位順序固定、標頭一致（Output Consistency，constitution III）。
- 布林以 `True`/`False` 表示；空值以空字串。
- 測量 A 與測量 B 共用同一 schema，以 `record_type` 區分。
- `distance` 僅在 `measurable=True` 時有值。
