# 2025 T-Brain 股票飆股預測競賽

> **競賽**：[2025 永豐 AI GO – 股票飆股預測](https://tbrain.trendmicro.com.tw/)  
> **任務**：二元分類（飆股 / 非飆股）  
> **最終成績**：Private Leaderboard **86 / 868**  
> **評估指標**：F1-score

---

## 目錄

- [競賽背景](#競賽背景)
- [解法架構](#解法架構)
- [目錄結構](#目錄結構)
- [環境安裝](#環境安裝)
- [Pipeline 執行步驟](#pipeline-執行步驟)
- [模型演進說明](#模型演進說明)
- [技術亮點](#技術亮點)

---

## 競賽背景

本競賽由永豐金控主辦，目標是根據台灣上市股票的多維度特徵（籌碼面、價量面、基本面），預測個股未來一段時間內是否會成為「飆股」。

- **訓練資料**：~10,214 個欄位（含 ID、標籤 `y`、10,212 個特徵）
- **標籤**：`y = 1` 代表飆股（嚴重正負不平衡，正樣本佔比極低）
- **資料特性**：每筆資料代表單一股票在某一時間點的快照，不同筆資料間無時序關聯

特徵分類：
| 類別 | 代表欄位 |
|------|----------|
| 籌碼面 | 外資買/賣張、券商分點買超/賣超、三大法人庫存 |
| 價量面 | K(9)、D(9)、RSI、MACD、歷史波動率、報酬率 |
| 基本面 | 營收、淨值、EPS、成長率 |

---

## 解法架構

```
training.csv (10,214 欄)
      │
      ▼
01_correlation_analysis.py   ← 多次分層抽樣 + Pearson 相關係數
      │  高相關特徵對偵測 (|r| > 0.7)
      ▼
02_refactor_csv.py           ← 過濾冗餘特徵 (10,214 → ~2,911 欄)
      │
      ▼
03_merge_column.py           ← 外資/籌碼群組欄位融合
      │
      ▼
04_training_vN.py            ← 不平衡處理 + 集成訓練
      │  SMOTE / ADASYN
      │  CatBoost + XGBoost + LightGBM
      │  BalancedBagging + EasyEnsemble + Stacking
      ▼
05_inference_vN.py           ← 多模型投票推論
      │  Soft Voting / Hard Voting / Hybrid
      ▼
results/final_prediction_*.csv  ← 提交結果
```

---

## 目錄結構

```
.
├── 01_correlation_analysis.py    # 特徵相關性分析
├── 02_refactor_csv.py            # 高相關特徵過濾
├── 03_merge_column.py            # 欄位融合
├── 04_training_v1.py ~ v7.py     # 訓練腳本（7個演進版本）
├── 05_inference.py               # 推論（早期版）
├── 05_inference_v6.py            # 推論 v6
├── 05_inference_v7.py            # 推論 v7（支援軟/硬/混合投票）
├── 06_trans_test.py              # 測試集格式轉換
├── 00_step_record.ipynb          # 實驗過程記錄
├── GAN/                          # 聯邦學習 / 神經網路實驗
│   ├── classification_train.py
│   ├── classification_train_fed.py
│   ├── federated_train.py
│   └── ...
├── try/                          # 資料探索腳本
│   ├── clean_csv.py
│   ├── downsample.py
│   ├── openai4o_merge.py
│   └── ...
├── requirements.txt
└── README.md
```

> **注意**：`models/`、`results/`、`*.csv`、`*.pkl` 等大型檔案已透過 `.gitignore` 排除，不納入版本控制。

---

## 環境安裝

建議使用 Python 3.10+，並以虛擬環境隔離依賴：

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

主要依賴：

| 套件 | 用途 |
|------|------|
| `catboost` | 主力梯度提升模型 |
| `xgboost` | 集成模型之一 |
| `lightgbm` | 集成模型之一 |
| `imbalanced-learn` | SMOTE / ADASYN / EasyEnsemble |
| `scikit-learn` | Stacking / 評估指標 |
| `torch` | 聯邦學習神經網路 |

---

## Pipeline 執行步驟

> 執行前請先將競賽原始資料放至專案根目錄。

### 1. 特徵相關性分析
```bash
python 01_correlation_analysis.py
# 輸出：correlation_results_*/
```

### 2. 過濾高相關特徵
```bash
python 02_refactor_csv.py
# 輸入：training.csv
# 輸出：training_filtered.csv（特徵數大幅縮減）
```

### 3. 欄位融合
```bash
python 03_merge_column.py
# 輸入：training_filtered.csv
# 輸出：training_filtered_merged.csv
```

### 4. 訓練模型（以 v7 為例）
```bash
python 04_training_v7.py
# 輸入：training_filtered_merged.csv
# 輸出：models/ 目錄（各演算法 .pkl / .cbm 模型）
```

### 5. 推論輸出（以 v7 為例）
```bash
python 05_inference_v7.py \
  --public_data filtered_public_x_merged.csv \
  --private_data 38_Private_Test_Set.../private_x.csv
# 輸出：results/final_prediction_*.csv
```

---

## 模型演進說明

| 版本 | 主要技術 | 說明 |
|------|----------|------|
| v1 | CatBoost + SMOTE | 基線版本 |
| v2 | + XGBoost / LightGBM | 多模型加入 |
| v3 | + BalancedBagging / EasyEnsemble | 集成不平衡方法 |
| v4 | 特徵工程優化 | 欄位名稱標準化、缺值處理 |
| v5 | + Stacking | 以 Logistic Regression 做 meta-learner |
| v6 | + Hybrid Voting | 軟硬投票混合策略 |
| v7 | + ADASYN + GPU 加速 | 全面優化，最終提交版本 |

---

## 技術亮點

### 不平衡資料處理
- 採用 **ADASYN**（v7）取代 SMOTE，對難以分類的少數樣本生成更多合成樣本
- 同時使用 **EasyEnsembleClassifier** 於 Bagging 層做二次平衡

### 集成學習策略
- **Stacking**：CatBoost / XGBoost / LightGBM 作為 base learners，Logistic Regression 作為 meta-learner
- **Hybrid Voting**：結合軟投票（機率平均）與硬投票（多數決），動態調整閾值

### 聯邦學習實驗（GAN/ 目錄）
- 將訓練集切分為 100 個子集，模擬分散式客戶端
- 使用 **FedAvg** 演算法聚合全域模型
- 搭配 PyTorch 神經網路分類器（MLP）
- 此方向為實驗性探索，最終提交以梯度提升模型為主

---

## Q&A（競賽官方說明節錄）

**Q：資料欄位如何分類？**  
A：訓練資料 10,214 欄，含 ID、標籤 `y`（飆股）及 10,212 個特徵，分為籌碼面、價量面、基本面三大類。

**Q：`y` 代表過去還是未來？**  
A：`y` 代表**未來**一定時間內是否成為飆股。

**Q：不同 ID 之間有時序關係嗎？**  
A：沒有。每筆資料獨立代表某股票於某時間點的快照。

