# Tasks: 動物眼睛量測 Pipeline（Image Segmentation + Metrology）

**Input**: 設計文件位於 `specs/001-animal-eye-metrology/`

**Prerequisites**: plan.md、spec.md、research.md、data-model.md、contracts/（cli.md、measurements-csv.md）、quickstart.md

**Tests**: 本功能包含測試任務。理由：spec 的 FR-011/FR-012、SC-004/SC-005 與 constitution II（Testing Standards，metrology/evaluate 覆蓋率 ≥ 80%）明確要求測試。

**Organization**: 任務依使用者故事分組，使各故事可獨立實作與測試。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可平行（不同檔案、無未完成相依）
- **[Story]**: 對應使用者故事（US1/US2/US3）
- 描述含確切檔案路徑

## Path Conventions

單一專案：`src/`、`tests/` 位於 repo 根；資料 `data/`、輸出 `outputs/`、文件 `docs/`。

---

## Phase 1: Setup（共用基礎建設）

**Purpose**: 專案初始化與基本結構

- [X] T001 建立專案目錄結構（src/、tests/unit/、tests/integration/、data/{coco,samples,ground_truth}/、outputs/{annotated}/、docs/）per plan.md
- [X] T002 建立 `requirements.txt`，以 pinned 版本列出 python 3.11 相依：ultralytics、pycocotools、opencv-python、numpy、matplotlib、pydantic、pandas、pytest（SAM2/segment-anything 列為可選 extra）
- [X] T003 [P] 建立 `.env.example`，含 COCO 路徑、模型權重路徑、信心閾值、種子等環境變數範例
- [X] T004 [P] 設定 linting/formatting 工具（ruff + black）設定檔於 repo 根（pyproject.toml 或對應設定）
- [X] T005 [P] 建立 `pytest` 設定（pyproject.toml 或 pytest.ini），設定 tests/ 探索路徑與覆蓋率門檻（metrology/evaluate ≥ 80%）

---

## Phase 2: Foundational（阻斷性前置，所有故事開工前必須完成）

**Purpose**: 所有使用者故事共用的核心基礎

**⚠️ CRITICAL**: 本階段完成前，任何使用者故事不得開工

- [X] T006 [P] 在 `src/config.py` 實作設定：動物類別清單（COCO 動物類別）、分割信心閾值（預設 0.25）、路徑、隨機種子（預設 42）、距離單位（預設 pixel）、座標慣例常數
- [X] T007 在 `src/models.py` 以 pydantic 實作資料模型：`Image`、`AnimalInstance`、`EyePoint`、`MeasurementRecord`、`EvaluationResult`，欄位與驗證規則依 data-model.md（如 EyePoint.side ∈ {left,right}；x 滿足 0 ≤ x < width；y 滿足 0 ≤ y < height；is_estimated 布林；MeasurementRecord.record_type ∈ {eye_distance, inter_animal_right_eye}；unit ∈ {pixel, real, normalized}）
- [X] T008 [P] 在 `src/config.py` 或 `src/logging_utils.py` 設定錯誤處理與日誌基礎（單張失敗隔離續跑、標準輸出訊息格式 per contracts/cli.md，如 `[skip]/[warn]/[info]`）
- [X] T009 在 `src/pipeline.py` 建立 CLI 進入點骨架與旗標解析（per contracts/cli.md：--coco-dir/--input/--output-dir/--min-animals/--conf-threshold/--use-sam2/--scale/--seed/--eval/--limit），固定種子、退出碼 0=批次完成 / 非 0=啟動期失敗

**Checkpoint**: 基礎就緒——使用者故事可開始

---

## Phase 3: User Story 1 - 量測單一動物的雙眼距離 (Priority: P1) 🎯 MVP

**Goal**: 對含 ≥2 動物的圖片，分割動物、定位左右眼、計算每隻動物雙眼歐氏距離，並輸出疊圖與 CSV；無法可靠定位眼睛時標記「無法量測」不輸出臆測值。

**Independent Test**: 給定一張含兩隻貓的圖片執行 pipeline，`measurements.csv` 出現 `record_type=eye_distance` 列含左右眼座標與 distance，疊圖可目視確認左右眼標記與連線；缺眼動物 measurable=False 且有 reason。

### Tests for User Story 1 ⚠️（先寫並確認失敗）

- [ ] T010 [P] [US1] 在 `tests/unit/test_metrology.py` 寫雙眼距離（測量 A）單元測試：已知座標的歐氏距離、重合點零距離、缺眼回傳不可量測
- [ ] T011 [P] [US1] 在 `tests/integration/test_pipeline.py` 寫 US1 整合測試：固定小樣本圖片 → 產出含 eye_distance 列的 CSV 與疊圖檔

### Implementation for User Story 1

- [ ] T012 [P] [US1] 在 `src/segmentation.py` 實作 YOLO11-seg 分割：載入 COCO 預訓練權重、對影像推論、回傳每隻動物的 category/bbox/mask/confidence（low_confidence 標記 per FR-013）
- [ ] T013 [P] [US1] 在 `src/eye_detector.py` 實作眼睛定位主策略（頭部 mask + 幾何/暗區啟發式 per research R2）：於頭部 ROI 找對稱暗區作左右眼候選，x 小者為 left、大者為 right，產生 EyePoint（is_estimated=True, source=heuristic）；不可靠時回傳 None 並附原因
- [ ] T014 [US1] 在 `src/eye_detector.py` 加入可選 SAM2 point-prompt 精修（--use-sam2 時啟用，預設關閉，source=sam2），失敗則回退啟發式
- [ ] T015 [US1] 在 `src/metrology.py` 實作測量 A：`eye_distance(left, right) = sqrt((xL-xR)^2+(yL-yR)^2)`；左右眼任一為 None → 回傳 measurable=False 與 reason（FR-005、FR-007）
- [ ] T016 [US1] 在 `src/visualize.py` 實作疊圖：畫 mask、左右眼點（一致配色區分左右）、連線與距離數值，輸出 `outputs/annotated/<image_id>.png`（FR-008、constitution III）
- [ ] T017 [US1] 在 `src/pipeline.py` 串接 US1 流程（分割 → 眼睛定位 → 測量 A → 疊圖 → 寫入 measurements.csv 的 eye_distance 列 per contracts/measurements-csv.md），單張失敗隔離續跑

**Checkpoint**: User Story 1 可獨立運作並測試（MVP）

---

## Phase 4: User Story 2 - 量測兩隻動物之間的右眼距離 (Priority: P2)

**Goal**: 計算兩隻動物右眼歐氏距離；恰兩隻動物輸出一對，恰 ≥3 隻已定位右眼輸出所有兩兩配對 C(n,2)；僅一隻右眼時標記「不適用」。

**Independent Test**: 給定含兩隻以上且至少兩隻定位右眼的圖片，`measurements.csv` 出現 `record_type=inter_animal_right_eye` 列；僅一隻右眼時標記不適用而非報錯。

### Tests for User Story 2 ⚠️

- [ ] T018 [P] [US2] 在 `tests/unit/test_metrology.py` 寫測量 B 單元測試：兩動物右眼距離正確性、3 隻動物產生 C(3,2)=3 筆配對、僅一隻右眼回傳不適用

### Implementation for User Story 2

- [ ] T019 [US2] 在 `src/metrology.py` 實作測量 B：`inter_animal_right_eye(Ra, Rb)=||Ra-Rb||`；依 FR-006 產生配對（2 隻→1 筆；≥3 隻→所有兩兩配對）；僅一隻右眼 → 標記「不適用（需至少兩隻右眼）」
- [ ] T020 [US2] 在 `src/visualize.py` 擴充疊圖：標示兩動物右眼連線與距離（與 US1 配色/標籤一致）
- [ ] T021 [US2] 在 `src/pipeline.py` 串接 US2：於同批次輸出 inter_animal_right_eye 列至 measurements.csv（共用 schema，以 record_type 區分）

**Checkpoint**: User Story 1 與 2 皆可獨立運作

---

## Phase 5: User Story 3 - 篩選合格圖片並產出可驗證的交付物 (Priority: P3)

**Goal**: 從 COCO 自動篩選含 ≥2 動物圖片（可重現），並產出評估報告（IoU/Dice/NME）與完整交付物。

**Independent Test**: 對 COCO 子集執行篩選得可重現的合格清單；對樣本產出 measurements.csv 與疊圖；對 ≥10 張 ground truth 計算 NME/IoU。

### Tests for User Story 3 ⚠️

- [ ] T022 [P] [US3] 在 `tests/unit/test_evaluate.py` 寫評估單元測試：IoU=|A∩B|/|A∪B|、Dice=2|A∩B|/(|A|+|B|)、NME 對已知輸入的正確值
- [ ] T023 [P] [US3] 在 `tests/integration/test_pipeline.py` 寫 US3 整合測試：篩選回傳僅含 ≥2 動物圖片且兩次執行一致（決定性）

### Implementation for User Story 3

- [ ] T024 [P] [US3] 在 `src/coco_selector.py` 實作 COCO 篩選：以 pycocotools 讀取標註、挑出含 ≥min-animals 動物的圖片、輸出可重現的 `outputs/selected_images.json`（FR-001）
- [ ] T025 [P] [US3] 在 `src/evaluate.py` 實作 IoU/Dice（分割）與 NME（眼睛，以眼間距/頭框正規化）計算（FR-011）
- [ ] T026 [US3] 在 `src/evaluate.py` 實作 ground truth 對照：讀取 `data/ground_truth`（≥10 張）計算像素誤差/NME，彙總 sample_size（FR-012、SC-004）
- [ ] T027 [US3] 在 `src/pipeline.py` 串接 --eval：輸出 `outputs/eval_report.md`（IoU/Dice/NME + 單張 CPU P95 處理時間實測 per SC-007）
- [ ] T028 [US3] 在 `src/pipeline.py` 串接篩選流程（--coco-dir → selected_images.json → 後續處理）

**Checkpoint**: 三個使用者故事皆可獨立運作

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: 跨故事的交付物與收尾

- [ ] T029 [P] 撰寫 `README.md`（繁中）：專案介紹、技術棧、本地運行步驟、Docker 指令、API 文件與測試帳號「不適用」說明（FR-016、SC-006、constitution V）
- [ ] T030 [P] 建立 `Dockerfile` 與 `docker-compose.yml`，支援離線執行 pipeline（SC-006）
- [ ] T031 [P] 產出系統架構圖 `docs/architecture.png`（COCO→篩選→分割→眼睛→測量→評估→視覺化 資料流）
- [ ] T032 產生 `outputs/measurements.csv` 範例（至少一筆測量 A 與一筆測量 B，欄位 per contracts/measurements-csv.md）（交付項目 4）
- [ ] T033 [P] 補齊決定性驗證（固定種子，相同輸入兩次執行結果一致）測試於 `tests/integration/test_pipeline.py`（FR-014、SC-005）
- [ ] T034 確認 metrology 與 evaluate 模組覆蓋率 ≥ 80%（constitution II），不足則補單元測試
- [ ] T035 執行 `quickstart.md` 全部情境驗證，修正發現的問題
- [ ] T036 [P] 程式碼清理：移除死碼/除錯輸出、補 docstring（座標慣例/單位/輸入輸出 per constitution I）

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 無相依，可立即開始
- **Foundational (Phase 2)**: 依賴 Setup 完成，阻斷所有使用者故事
- **User Stories (Phase 3+)**: 皆依賴 Foundational 完成
  - US1（P1）可最先完成並作為 MVP
  - US2（P2）依賴 US1 的眼睛定位（右眼），邏輯上接續 US1
  - US3（P3）篩選可獨立，評估依賴 US1/US2 的量測輸出
- **Polish (Phase 6)**: 依賴所需使用者故事完成

### User Story Dependencies

- **US1 (P1)**: Foundational 後即可開始，無其他故事相依
- **US2 (P2)**: Foundational 後可開始；共用 eye_detector/metrology，建議於 US1 後進行以複用右眼定位
- **US3 (P3)**: 篩選（coco_selector）可與 US1 平行；評估（evaluate ground truth）需 US1/US2 量測結果

### Within Each User Story

- 測試先寫並確認失敗 → models → 服務/演算法模組 → pipeline 串接
- metrology/evaluate 數學正確性由單元測試保證

### Parallel Opportunities

- Setup 中 T003/T004/T005 可平行
- Foundational 中 T006/T008 可平行（T007 模型、T009 CLI 骨架有相依）
- US1 中 T010/T011 測試可平行；T012/T013 可平行（不同檔案）
- US3 中 T024（篩選）可與 US1 的實作平行開工
- Polish 中 T029/T030/T031/T036 可平行

---

## Parallel Example: User Story 1

```bash
# 先平行啟動 US1 測試（確認失敗）：
Task: "在 tests/unit/test_metrology.py 寫雙眼距離單元測試"
Task: "在 tests/integration/test_pipeline.py 寫 US1 整合測試"

# 平行啟動不同檔案的實作：
Task: "在 src/segmentation.py 實作 YOLO11-seg 分割"
Task: "在 src/eye_detector.py 實作眼睛定位主策略"
```

---

## Implementation Strategy

### MVP First（僅 User Story 1）

1. 完成 Phase 1 Setup
2. 完成 Phase 2 Foundational（阻斷所有故事）
3. 完成 Phase 3 User Story 1
4. **STOP 並驗證**：獨立測試 US1（雙眼距離 + 疊圖 + CSV）
5. 可展示 MVP

### Incremental Delivery

1. Setup + Foundational → 基礎就緒
2. US1 → 獨立測試 → 展示（MVP：測量 A）
3. US2 → 獨立測試 → 展示（測量 B）
4. US3 → 獨立測試 → 展示（篩選 + 評估）
5. Polish → 交付物齊備（README/Docker/架構圖/CSV 範例）

---

## Notes

- [P] = 不同檔案、無相依
- [Story] 標籤對應使用者故事以利追溯
- 各故事可獨立完成並測試
- 實作前先確認測試失敗
- 每個任務或邏輯群組後提交 commit
- 可於任一 checkpoint 停下獨立驗證
- 避免：模糊任務、同檔衝突、破壞獨立性的跨故事相依
