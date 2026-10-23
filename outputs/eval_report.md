# 評估報告（Evaluation Report）

## 效能實測（CPU 參考環境，SC-007）

- 處理圖片數：1
- 量測紀錄數：3
- 單張平均處理時間：5.688s（**僅供參考，含模型載入**）
- **樣本數不足（n=1 < 5），不計 P95**；效能驗收請以多張樣本（建議 ≥20 張）執行 `--eval` 取得具統計意義的 P95。

## 分割品質（IoU / Dice）與眼睛定位（NME）

- IoU = |A∩B| / |A∪B|；Dice = 2|A∩B| / (|A|+|B|)；NME = 平均點位誤差 / 正規化尺度。
- 尚未提供 ground truth（預期於 data\ground_truth，需 ≥10 張以計算 NME）
- 評估函式見 `src/evaluate.py`（iou / dice / nme / evaluate_eye_localization）。
