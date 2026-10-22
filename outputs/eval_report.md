# 評估報告（Evaluation Report）

## 效能實測（CPU 參考環境，SC-007）

- 處理圖片數：1
- 量測紀錄數：3
- 單張處理時間 P50：7.552s
- 單張處理時間 P95：7.552s（目標 ≤ 5s）
- 驗收：REVIEW

## 分割品質（IoU / Dice）與眼睛定位（NME）

- IoU = |A∩B| / |A∪B|；Dice = 2|A∩B| / (|A|+|B|)；NME = 平均點位誤差 / 正規化尺度。
- 尚未提供 ground truth（預期於 data\ground_truth，需 ≥10 張以計算 NME）
- 評估函式見 `src/evaluate.py`（iou / dice / nme / evaluate_eye_localization）。
