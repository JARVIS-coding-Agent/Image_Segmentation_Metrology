# Quickstart：動物眼睛量測 Pipeline 驗證指南

**Feature**: 001-animal-eye-metrology | **Date**: 2026-10-22

本指南描述可執行的端到端驗證情境，用以證明功能運作。實作細節見 `tasks.md` 與各模組；資料契約見 `contracts/`，實體見 `data-model.md`。

## 先決條件

- Python 3.11
- 已安裝 `requirements.txt`（pinned 版本）
- 已於 `data/coco/` 放置 COCO 子集（影像 + annotations JSON）
- （可選）`data/ground_truth/` 放置至少 10 張人工眼睛標註，供 NME 驗證

## 環境建置

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 依需要調整
```

## 情境 1：篩選含 2+ 動物圖片（驗證 User Story 3 / FR-001）

```bash
python -m src.pipeline --coco-dir data/coco --min-animals 2 --limit 20
```

**預期**：`outputs/selected_images.json` 僅含動物數 ≥2 的圖片；重複執行內容一致（可重現）。

## 情境 2：測量 A — 每隻動物雙眼距離（驗證 User Story 1 / FR-005）

```bash
python -m src.pipeline --coco-dir data/coco --output-dir outputs
```

**預期**：
- `outputs/measurements.csv` 含 `record_type=eye_distance` 的列，附左右眼座標與 `distance`。
- `outputs/annotated/<id>.png` 可目視確認左右眼標記、連線與距離數值。
- 無法可靠定位眼睛的動物：`measurable=False` 且有 `reason`，無臆測 distance。

## 情境 3：測量 B — 兩動物右眼距離（驗證 User Story 2 / FR-006）

**預期**（同上次執行輸出）：
- 含 `record_type=inter_animal_right_eye` 的列。
- 恰兩隻動物 → 一筆；恰 ≥3 隻已定位右眼 → C(n,2) 筆。
- 僅一隻動物有右眼 → 標記「不適用」而非報錯。

## 情境 4：評估（驗證 FR-011/012、SC-004）

```bash
python -m src.pipeline --coco-dir data/coco --eval
```

**預期**：`outputs/eval_report.md` 含 IoU/Dice（分割）與 NME（眼睛，樣本 ≥10），並附單張 CPU P95 處理時間實測。

## 情境 5：決定性（驗證 FR-014 / SC-005）

```bash
python -m src.pipeline --coco-dir data/coco --seed 42 --output-dir out_a
python -m src.pipeline --coco-dir data/coco --seed 42 --output-dir out_b
# 比較兩次 measurements.csv 應完全一致
```

## 情境 6：單元測試（驗證 metrology/evaluate 數學正確性）

```bash
pytest tests/unit -q
pytest tests/integration -q
```

**預期**：距離公式對已知座標、重合點（零距離）、缺眼邊界皆通過；IoU/Dice/NME 對已知輸入回傳正確值。

## 情境 7：Docker 執行（驗證 SC-006 部署）

```bash
docker compose build
docker compose run --rm pipeline python -m src.pipeline --coco-dir data/coco
```

**預期**：容器內可離線完成 pipeline 並產出相同結構的輸出。

## 驗收對照

| 情境 | 對應需求 |
|------|---------|
| 1 | FR-001、US3 |
| 2 | FR-003/004/005/007/008、US1、SC-002 |
| 3 | FR-006、US2、SC-003 |
| 4 | FR-011/012、SC-004、SC-007 |
| 5 | FR-014、SC-005 |
| 6 | FR-005/006/011（數學正確性） |
| 7 | FR-016、SC-006 |
