# 第一題：Image Segmentation + Metrology 專案架構與開發計畫

> 將圖像分割 (Image Segmentation) 模型的輸出，應用於實際測量學 (Metrology) 任務。

---

## 0. 交付文件清單（必繳，10/26 20:30 前回傳連結）

以下為題目要求的交付項目，本專案必須全部產出：

| # | 交付項目 | 對應檔案/位置 | 狀態 |
|---|----------|--------------|:----:|
| 1 | **完整的專案代碼（上傳至 GitHub）** | 整個 `Q1_Image_Segmentation_Metrology/` 專案 repo | ☐ |
| 2 | **README.md**（內容見下方子項） | `README.md` | ☐ |
| 3 | **.env.example 文件** | `.env.example` | ☐ |
| 4 | **至少一個測試資料的 CSV 範例** | `outputs/measurements.csv`（量測結果範例） | ☐ |
| 5 | **系統架構圖** | `docs/architecture.png` | ☐ |

### README.md 必須包含（但不限於）
- [ ] **專案介紹** — 題目背景（Segmentation → Metrology）、目標與成果摘要
- [ ] **技術棧說明** — Python、YOLO11-seg、pycocotools、OpenCV、SAM2（可選）等
- [ ] **本地運行步驟** — 環境建置、安裝依賴、下載 COCO 子集、執行 pipeline
- [ ] **Docker 部署指令** — 建置與執行容器的指令
- [ ] **API 文件連結** — 本題以離線 pipeline 為主；若無 API 需明確註明，或提供推論服務的文件連結
- [ ] **測試帳號資訊** — 本題若無登入機制需明確註明「不適用」；若有服務則提供測試帳號

> 註：第一題為影像處理 pipeline，若「API 文件連結」與「測試帳號資訊」不適用，README 需**明確標註不適用並說明原因**，以免被視為遺漏。

---

## 1. 題目需求拆解

| # | 需求 | 說明 |
|---|------|------|
| 1 | 資料來源 | 使用 **COCO 資料集**，挑出含 **2 隻（含）以上動物**的圖片，動物任選 |
| 2 | 分割任務 | 用 **Segmentation 模型** 個別框出：①動物輪廓、②動物眼睛 |
| 3 | 測量 A | 量測**每隻動物的雙眼距離**（同一隻動物左眼↔右眼） |
| 4 | 測量 B | 量測**任意兩隻動物的右眼距離**（動物 A 右眼 ↔ 動物 B 右眼） |
| 5 | 模型說明 | 解釋使用什麼 AI 模型、為什麼、評估標準公式 |
| 6 | 測量說明 | 解釋量測方式與公式、以及如何驗證 |

### 關鍵技術挑戰
- COCO 標註**只有動物的 instance mask / bbox，沒有「眼睛」的關鍵點或分割標註**。
  （COCO 僅 `person` 類別有 keypoints，動物沒有。）
- 因此「動物眼睛定位」需要額外策略，這是本題的核心難點。

---

## 2. 技術選型與理由

### 2.1 動物輪廓分割（Instance Segmentation）

| 候選模型 | 說明 | 選用 |
|---------|------|------|
| **Mask R-CNN** (Detectron2) | COCO 預訓練、instance mask 精準、業界標準 | 備選 |
| **YOLOv8-seg / YOLO11-seg** (Ultralytics) | COCO 預訓練、即時、API 簡潔、輸出 mask + bbox | ✅ 主選 |
| SAM / SAM2 | 零樣本分割品質極高，但需 prompt，非類別感知 | 輔助（精修輪廓） |

**主選 YOLO11-seg 理由：**
- COCO 80 類預訓練，直接涵蓋 `cat / dog / horse / sheep / cow / elephant / bear / zebra / giraffe / bird` 等動物類別。
- 一次推論同時取得 **類別、bbox、instance mask、信心分數**，方便後續測量。
- Python API 成熟、部署容易、推論快。

### 2.2 動物眼睛定位（本題核心）

COCO 無動物眼睛標註，採**分層策略**，由穩健到精細：

- **方案 A（主線）：動物臉部/眼睛關鍵點模型**
  - 使用針對動物的 landmark 模型（如貓臉關鍵點、或自訓練輕量 keypoint head）。
  - 或使用 **MediaPipe / 專用動物臉偵測** 取得眼睛點位。
- **方案 B（通用備援）：以分割輪廓 + 幾何啟發式推定眼睛位置**
  - 在動物頭部 mask 的上半部，用對比度/暗區偵測（眼睛通常為對稱暗斑）定位左右眼中心。
- **方案 C（高品質可選）：SAM2 point-prompt**
  - 以方案 B 得到的候選點作為 prompt，讓 SAM2 精修眼睛區域，取質心為眼睛座標。

> 開發時先以方案 A/C 為主、方案 B 為保底，確保一定能產出可量測的眼睛座標。

### 2.3 技術棧

```
Python 3.11
ultralytics (YOLO11-seg)         # 動物輪廓/instance 分割
pycocotools                      # COCO 讀取與篩選
opencv-python, numpy             # 影像處理與幾何計算
sam2 / segment-anything (optional) # 眼睛區域精修
matplotlib                       # 視覺化輸出
pydantic                         # 測量結果結構化
pytest                           # 單元測試/驗證
```

---

## 3. 評估標準公式

### 3.1 分割品質評估

- **IoU (Intersection over Union)**：
  $$ IoU = \frac{|A \cap B|}{|A \cup B|} $$
- **Dice 係數**：
  $$ Dice = \frac{2|A \cap B|}{|A| + |B|} $$
- **mAP@[.5:.95]**（instance segmentation 標準，沿用 COCO 評估協定）。

### 3.2 眼睛定位評估

- **NME (Normalized Mean Error)**，以眼間距或頭部框尺寸正規化：
  $$ NME = \frac{1}{N}\sum_{i=1}^{N}\frac{\lVert p_i - \hat{p}_i \rVert_2}{d_{norm}} $$

---

## 4. 測量方式與公式

### 4.1 座標定義
每隻動物 `k` 取得左眼 `L_k=(x,y)`、右眼 `R_k=(x,y)`（像素座標）。

### 4.2 測量 A — 每隻動物雙眼距離
$$ D_{eyes}^{(k)} = \lVert L_k - R_k \rVert_2 = \sqrt{(x_{L}-x_{R})^2 + (y_{L}-y_{R})^2} $$

### 4.3 測量 B — 兩隻動物右眼距離
$$ D_{R}^{(a,b)} = \lVert R_a - R_b \rVert_2 $$

### 4.4 像素 → 實際單位（可選）
若需真實世界距離，需比例尺 (scale)：
$$ D_{real} = D_{pixel} \times s \quad (s = \text{實際長度}/\text{像素長度}) $$
無標定物時，以**像素距離**或**以眼間距正規化的相對距離**表示並說明。

### 4.5 驗證方式
1. **視覺化疊圖**：在原圖畫出 mask、眼睛點、連線與距離數值，人工目視檢查。
2. **人工標註對照**：手動標少量眼睛點作 ground truth，計算 NME / 像素誤差。
3. **一致性檢查**：左右眼對稱性、雙眼距離與頭部框尺寸比例合理性。
4. **單元測試**：對已知座標驗證距離函式數學正確性（pytest）。

---

## 5. 專案資料夾結構

```
Q1_Image_Segmentation_Metrology/
├── README.md                     # 專案介紹/運行步驟/結果展示
├── PROJECT_PLAN.md               # 本文件
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── coco/                     # COCO 子集（含 2+ 動物的圖片）
│   └── samples/                  # 範例輸入圖
├── src/
│   ├── config.py                 # 設定（類別、閾值、路徑）
│   ├── coco_selector.py          # 篩選含 2+ 動物的圖片
│   ├── segmentation.py           # YOLO11-seg 輪廓分割
│   ├── eye_detector.py           # 眼睛定位（方案 A/B/C）
│   ├── metrology.py              # 距離測量公式
│   ├── evaluate.py               # IoU/Dice/NME 評估
│   ├── visualize.py              # 疊圖與標註輸出
│   └── pipeline.py               # 端到端串接
├── outputs/
│   ├── annotated/                # 疊圖結果
│   └── measurements.csv          # 測量結果輸出
├── tests/
│   ├── test_metrology.py         # 距離公式單元測試
│   └── test_pipeline.py
├── notebooks/
│   └── demo.ipynb                # 展示用
└── docs/
    └── architecture.png          # 系統架構圖
```

---

## 6. 系統資料流（Pipeline）

```
COCO 資料集
   │  coco_selector.py（篩選 2+ 動物圖片）
   ▼
輸入影像
   │  segmentation.py（YOLO11-seg → 動物 mask/bbox/類別）
   ▼
每隻動物 instance
   │  eye_detector.py（定位左右眼座標）
   ▼
眼睛座標 {L_k, R_k}
   │  metrology.py（計算雙眼距離 / 兩動物右眼距離）
   ▼
測量結果  ──►  evaluate.py（IoU/Dice/NME 驗證）
   │
   ▼
visualize.py（疊圖 + measurements.csv）
```

---

## 7. 開發計畫（里程碑）

| 階段 | 工作項目 | 產出 | 預估 |
|------|----------|------|------|
| M1 環境 | 建環境、裝套件、下載 COCO 子集 | requirements.txt、data/ | 0.5 天 |
| M2 篩選 | coco_selector：挑含 2+ 動物圖片 | 篩選後圖片清單 | 0.5 天 |
| M3 分割 | YOLO11-seg 動物輪廓分割 + 視覺化 | segmentation.py、疊圖 | 0.5 天 |
| M4 眼睛 | 眼睛定位（方案 A→C，B 保底） | eye_detector.py | 1 天 |
| M5 測量 | 雙眼距離、兩動物右眼距離 | metrology.py、measurements.csv | 0.5 天 |
| M6 評估 | IoU/Dice/NME + 人工對照驗證 | evaluate.py、報告 | 0.5 天 |
| M7 交付 | README、架構圖、Docker、測試 | 完整交付物 | 1 天 |

---

## 8. 交付物對照（題目要求）

| 題目要求 | 對應產出 |
|---------|---------|
| GitHub 完整代碼 | 整個 `Q1_*` 專案 |
| README.md | `README.md`（介紹/技術棧/運行/Docker/API/測試帳號 N/A 則註明） |
| .env.example | `.env.example` |
| 測試 CSV 範例 | `outputs/measurements.csv` 範例 |
| 系統架構圖 | `docs/architecture.png` |
| 模型/評估公式說明 | 本文件 §2、§3 + README |
| 測量方式/驗證說明 | 本文件 §4 + README |
