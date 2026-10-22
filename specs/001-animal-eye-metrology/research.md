# Phase 0 研究：動物眼睛量測 Pipeline

**Feature**: 001-animal-eye-metrology | **Date**: 2026-10-22

本文件整理技術選型決策，解決 plan.md Technical Context 中的關鍵選擇。格式：Decision（選擇）/ Rationale（理由）/ Alternatives（替代方案）。

---

## R1. 實例分割模型

- **Decision**: 使用 Ultralytics YOLO11-seg（COCO 預訓練）。
- **Rationale**:
  - COCO 80 類預訓練，直接涵蓋 cat/dog/horse/sheep/cow/elephant/bear/zebra/giraffe/bird 等動物類別，無需自行訓練。
  - 單次推論同時取得類別、bbox、instance mask、信心分數，直接餵給後續眼睛定位與測量。
  - Python API 成熟、CPU 可推論、部署容易，符合 CPU P95 ≤ 5 秒與離線需求。
- **Alternatives considered**:
  - Mask R-CNN（Detectron2）：mask 精度高，但相依較重、CPU 推論較慢、安裝複雜 → 列為備選。
  - SAM / SAM2 單獨作為分割主力：零樣本品質高但非類別感知、需 prompt，不適合直接產出「動物類別 + 實例」 → 僅作眼睛精修可選用途。

## R2. 動物眼睛定位策略（核心難點）

- **Decision**: 以「頭部區域推定 + 幾何/暗區啟發式」為主要來源，SAM2 point-prompt 精修為可選增強（預設關閉）。所有啟發式座標標記為估計值。
- **Rationale**:
  - COCO 無動物眼睛標註（僅 person 類別有 keypoints），無法直接取得眼睛座標。
  - 通用「動物臉關鍵點模型」取得與可靠度跨物種不穩定，作為硬相依風險高。
  - 幾何啟發式相依最少、最可重現：於動物 mask/bbox 的頭部區域（依朝向推定上/前段）用對比度/暗區偵測對稱暗斑作為左右眼候選，取質心為座標。
  - 符合 spec 澄清（Session 2026-10-22）與 constitution「量測誠實」原則；估計值標記可誠實反映品質。
- **Alternatives considered**:
  - 專用關鍵點模型為主：準確但物種受限、相依重 → 否決為主力。
  - SAM2 為主：品質佳但需權重與 prompt 流程、CPU 較慢 → 降為可選精修。
- **啟發式細節（供實作參考）**:
  - 頭部區域：以 bbox 長邊方向與 mask 幾何推定頭部 ROI（預設取 mask 上半/前段）。
  - 眼睛候選：於頭部 ROI 灰階影像找對稱的局部暗區（低亮度、近似圓形），以水平位置區分左右眼。
  - 左右定義：以影像座標 x 較小者為「左眼」、較大者為「右眼」（影像視角），於文件明確宣告此慣例。
  - 可靠度：若找不到成對對稱暗區或信心不足 → 標記「無法量測」。

## R3. 座標慣例與距離單位

- **Decision**: 像素座標，原點於影像左上角，x 向右、y 向下；距離預設單位為「像素」。左右眼以影像視角的 x 位置區分（x 小者為左眼）。
- **Rationale**: OpenCV/影像處理標準慣例，與 YOLO 輸出一致；無標定物時真實世界單位不可得，像素距離最可驗證。符合 FR-010 單一慣例/單一單位要求。
- **Alternatives considered**: 以動物解剖左右（動物自身視角）定義 → 需朝向判定、易混淆，否決；改以影像視角並明確宣告。
- **實際單位（可選）**: 若提供比例尺 s（實際長度/像素長度），輸出 D_real = D_pixel × s；無標定物時可另提供「以眼間距正規化的相對距離」。

## R4. 測量公式

- **Decision**:
  - 測量 A（雙眼距離）：`D_eyes(k) = ||L_k − R_k||₂ = sqrt((xL−xR)² + (yL−yR)²)`
  - 測量 B（兩動物右眼距離）：`D_R(a,b) = ||R_a − R_b||₂`；標準情境兩隻動物輸出一對，恰含 3 隻以上時輸出所有兩兩配對 C(n,2)。
- **Rationale**: 歐氏距離為像素平面上直線距離的標準定義，數學可單元測試驗證；配對規則來自 spec 澄清。
- **Alternatives considered**: 曼哈頓距離等 → 不符「距離」直覺，否決。

## R5. 評估指標

- **Decision**: 分割品質用 IoU 與 Dice；眼睛定位用 NME（以眼間距或頭框尺寸正規化）；分割整體可選報 mAP@[.5:.95]（沿用 COCO 協定）。
- **Rationale**:
  - IoU = |A∩B| / |A∪B|、Dice = 2|A∩B| / (|A|+|B|) 為分割標準度量，可用 mask 直接計算並單元測試。
  - NME = (1/N) Σ ||p_i − p̂_i||₂ / d_norm 對不同尺寸影像可比較，適合眼睛點位誤差。
- **Alternatives considered**: 僅用像素誤差 → 不同影像尺寸不可比，故以 NME 正規化為主、像素誤差為輔。

## R6. 驗證方式

- **Decision**: 四層驗證：(1) 視覺化疊圖人工目視；(2) 至少 10 張合格圖片的人工眼睛 ground truth 對照計算 NME/像素誤差；(3) 一致性檢查（左右眼對稱、雙眼距與頭框比例合理）；(4) metrology 函式 pytest 單元測試。
- **Rationale**: 對齊 spec SC-002/003/004 與 FR-011/012；量化（NME/IoU）與人工（疊圖）互補，數學正確性由單元測試保證。
- **Alternatives considered**: 僅人工目視 → 不可量化、不可回歸，否決。

## R7. 決定性與效能

- **Decision**: 固定隨機種子、pinned 模型權重與套件版本；以 CPU 為參考環境量測單張 P95，記錄實測於 eval_report。單張失敗以 try/except 隔離並記錄，不中斷整批。
- **Rationale**: 符合 constitution IV 與 FR-014；相同輸入相同輸出是 metrology 可信度前提。
- **Alternatives considered**: 不固定種子 → 結果不可重現，否決。

## R8. 技術棧與部署

- **Decision**: Python 3.11；核心 ultralytics / pycocotools / opencv-python / numpy / matplotlib / pydantic / pandas / pytest，皆 pinned；SAM2 為可選 extra。提供 Dockerfile 與 docker-compose 以容器化離線執行；README 標註 API 文件與測試帳號「不適用」及原因。
- **Rationale**: 對齊 PROJECT_PLAN 技術棧與交付物要求（FR-016、SC-006）；Docker 確保環境可重現。
- **Alternatives considered**: conda 環境 → 評審重現成本較高，以 requirements.txt + Docker 為主。

---

## NEEDS CLARIFICATION 解決狀態

Technical Context 中無殘留 NEEDS CLARIFICATION：語言、相依、測試、平台、效能目標、約束、規模皆已在 spec 澄清階段或本研究中確定。
