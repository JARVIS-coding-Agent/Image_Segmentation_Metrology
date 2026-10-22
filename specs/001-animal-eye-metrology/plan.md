# 實作計畫：動物眼睛量測 Pipeline（Image Segmentation + Metrology）

**Branch**: `001-animal-eye-metrology` | **Date**: 2026-10-22 | **Spec**: [spec.md](./spec.md)

**Input**: 功能規格書位於 `specs/001-animal-eye-metrology/spec.md`

## Summary

本功能是一條離線影像處理 pipeline：從 COCO 資料集篩選「含兩隻（含）以上動物」的圖片，使用實例分割模型取得每隻動物的輪廓（mask / bbox / 類別），再以「頭部 mask + 幾何/暗區啟發式」為主要策略（SAM2 point-prompt 精修為可選）定位每隻動物的左右眼，接著計算兩項測量：(A) 每隻動物的雙眼歐氏距離、(B) 兩隻動物的右眼歐氏距離。最後輸出疊圖、固定欄位的量測 CSV，以及以 IoU/Dice/NME 為基礎的評估報告。無法可靠定位眼睛時標記「無法量測」並記錄原因，不輸出臆測值；相同輸入與設定須產生相同結果。

技術途徑：Python 離線 CLI pipeline，模組化拆分（COCO 篩選 → 分割 → 眼睛定位 → 測量 → 評估 → 視覺化 → 端到端串接），以 CPU 為參考環境（單張 P95 ≤ 5 秒），固定隨機種子與套件版本以確保決定性。

## Technical Context

**Language/Version**: Python 3.11

**Primary Dependencies**: ultralytics（YOLO11-seg，實例分割）、pycocotools（COCO 讀取與篩選）、opencv-python + numpy（影像處理與幾何計算）、matplotlib（視覺化疊圖）、pydantic（量測結果結構化與驗證）、pandas（CSV 輸出）；SAM2 / segment-anything 為**可選**（眼睛區域精修，預設關閉）

**Storage**: 檔案系統（COCO 影像與標註 JSON、輸入樣本圖、輸出疊圖 PNG、`measurements.csv`、評估報告）；無資料庫

**Testing**: pytest（距離公式單元測試、pipeline 整合測試、評估函式測試）

**Target Platform**: 跨平台 CLI（Linux / macOS / Windows），並提供 Docker 容器化執行；以 CPU 執行為預設

**Project Type**: 單一專案（離線 CLI / library pipeline）

**Performance Goals**: 單張圖片「分割 + 眼睛定位」P95 ≤ 5 秒（CPU 參考環境），並記錄實測值

**Constraints**: 離線可執行（模型權重可本地取得後離線推論）；決定性（固定種子、pinned 版本、相同輸入相同輸出）；單張失敗不得中斷整批；眼睛估計值須標記

**Scale/Scope**: 評估樣本至少 10 張合格圖片（含 ≥2 動物）；pipeline 模組約 8 個；以示範與驗證為目標，非大規模線上服務

## Constitution Check

*GATE: Phase 0 研究前必須通過，Phase 1 設計後重新檢查。*

| 原則 | 本計畫如何符合 | 狀態 |
|------|----------------|:----:|
| I. Code Quality | 模組單一職責（coco_selector / segmentation / eye_detector / metrology / evaluate / visualize / pipeline）；公開介面以 docstring 記錄用途、輸入輸出、座標慣例與單位；依賴以 pinned 版本宣告於 requirements.txt | PASS |
| II. Testing Standards | metrology 與 evaluate 以 pytest 單元測試覆蓋已知座標、邊界（重合點/零距離/缺眼）與數值正確性，目標覆蓋率 ≥ 80%；pipeline 以固定小樣本做整合測試；bug 先寫重現測試 | PASS |
| III. Output & Visualization Consistency | 疊圖以一致配色/標籤區分左右眼、畫 mask/連線/數值；CSV 採固定欄位 schema；座標單一慣例、距離單一單位（像素預設）；空/錯誤狀態輸出明確訊息 | PASS |
| IV. Performance & Robustness | CPU P95 ≤ 5s 並記錄實測；固定種子、pinned 權重/版本以求決定性；缺眼 → 依策略標記並續跑，單張失敗不中斷整批 | PASS |
| V. Language & Documentation | spec/plan/tasks/README 等交付物以繁體中文撰寫（本檔為繁中）；模型、評估公式（IoU/Dice/NME）、測量公式於文件與程式同步 | PASS |

**額外合規**：資料來源限 COCO 含 ≥2 動物、篩選可重現；眼睛定位誠實（啟發式估計須標記）；模型選擇與評估公式透明；低信心量測標記；授權與隱私（僅動物影像）。

**結論**：無違反項目，Complexity Tracking 留空。初次 gate 通過。

## Project Structure

### Documentation (this feature)

```text
specs/001-animal-eye-metrology/
├── plan.md              # 本檔（/speckit.plan 輸出）
├── spec.md              # 功能規格書
├── research.md          # Phase 0 輸出
├── data-model.md        # Phase 1 輸出
├── quickstart.md        # Phase 1 輸出
├── contracts/           # Phase 1 輸出（CLI 介面與資料契約）
│   ├── cli.md
│   └── measurements-csv.md
├── checklists/
│   └── requirements.md  # 規格品質檢查清單
└── tasks.md             # /speckit.tasks 輸出（本指令不建立）
```

### Source Code (repository root)

```text
src/
├── config.py            # 設定：動物類別、信心閾值、路徑、種子、單位
├── coco_selector.py     # 篩選含 2+ 動物的圖片（可重現）
├── segmentation.py      # YOLO11-seg 動物輪廓/實例分割
├── eye_detector.py      # 眼睛定位（頭部 mask + 幾何/暗區啟發式；SAM2 可選精修）
├── metrology.py         # 距離測量公式（測量 A / 測量 B）
├── evaluate.py          # IoU / Dice / NME 評估
├── visualize.py         # 疊圖與標註輸出
├── models.py            # pydantic 資料模型（實體定義）
└── pipeline.py          # 端到端串接 + CLI 進入點

tests/
├── unit/
│   ├── test_metrology.py    # 距離公式單元測試
│   └── test_evaluate.py     # IoU/Dice/NME 單元測試
└── integration/
    └── test_pipeline.py     # 端到端整合測試（固定小樣本）

data/
├── coco/                # COCO 子集（影像 + annotations JSON）
├── samples/             # 範例輸入圖
└── ground_truth/        # 人工標註眼睛 ground truth（≥10 張，供 NME 驗證）

outputs/
├── annotated/           # 疊圖結果
├── measurements.csv     # 量測結果（含範例）
└── eval_report.md       # 評估報告（IoU/Dice/NME + 效能實測）

docs/
└── architecture.md     # 系統架構圖（mermaid）

# 專案根
README.md                # 繁中：介紹/技術棧/運行/Docker/API 與測試帳號適用性
requirements.txt         # pinned 版本
.env.example             # 環境變數範例
Dockerfile
docker-compose.yml
```

**Structure Decision**: 採單一專案結構。此為離線 CLI / library pipeline，無前後端分離需求；以功能模組（selector/segmentation/eye_detector/metrology/evaluate/visualize/pipeline/models）對應 spec 的資料流各階段，測試分 unit 與 integration 兩層，符合 constitution 的單一職責與測試分層要求。

## Complexity Tracking

> 無 Constitution 違反項目，本節留空。
