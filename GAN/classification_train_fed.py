# ========== 匯入套件與參數 ==========
import os
import argparse
import copy
import gc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ========== 參數設定 ==========
parser = argparse.ArgumentParser()
parser.add_argument('--aggr', type=str, default='fedadam', choices=['fedadam', 'fedavg'], help='Aggregation method')
parser.add_argument('--fedadam_lr', type=float, default=0.001, help='FedAdam learning rate')
parser.add_argument('--fedadam_beta1', type=float, default=0.9, help='FedAdam beta1')
parser.add_argument('--fedadam_beta2', type=float, default=0.999, help='FedAdam beta2')
parser.add_argument('--fedadam_eps', type=float, default=1e-8, help='FedAdam epsilon')
args = parser.parse_args()
AGGREGATION_METHOD = args.aggr

# ========== 特徵與裝置 ==========
# feature_cols = [
#     '外資券商_mean','外資券商_std','外資券商_max','外資券商_min',
#     '主力券商_mean','主力券商_std','主力券商_max','主力券商_min',
#     '官股券商_mean','官股券商_std','官股券商_max','官股券商_min',
#     '個股券商分點籌碼分析_mean','個股券商分點籌碼分析_std','個股券商分點籌碼分析_max',
#     '個股券商分點籌碼分析_min','個股券商分點區域分析_mean','個股券商分點區域分析_std',
#     '個股券商分點區域分析_max','個股券商分點區域分析_min','個股主力買賣超統計_mean',
#     '個股主力買賣超統計_std','個股主力買賣超統計_max','個股主力買賣超統計_min',
#     '日外資_mean','日外資_std','日外資_max','日外資_min','日自營_mean','日自營_std',
#     '日自營_max','日自營_min','日投信_mean','日投信_std','日投信_max','日投信_min',
#     '技術指標_mean','技術指標_std','技術指標_max','技術指標_min','月營收_mean',
#     '月營收_std','月營收_max','月營收_min','季IFRS財報_mean','季IFRS財報_std',
#     '季IFRS財報_max','季IFRS財報_min','買超分點_mean','買超分點_std','買超分點_max',
#     '買超分點_min','賣超分點_mean','賣超分點_std','賣超分點_max','賣超分點_min',
#     '其他_mean','其他_std','其他_max','其他_min'
#     ]
feature_cols = [
"技術指標_週RSI(5)", "技術指標_週RSI(10)", "技術指標_週MACD", "技術指標_週K(9)",
"技術指標_週DIF-週MACD", "技術指標_週DIF", "技術指標_週-DI(14)", "技術指標_週D(9)",
"技術指標_週ADX(14)", "技術指標_週+DI(14)", "技術指標_相對強弱比(週)", "技術指標_相對強弱比(日)",
"技術指標_近六月歷史波動率(%)", "技術指標_近三月歷史波動率(%)", "技術指標_近二月歷史波動率(%)",
"技術指標_近九月歷史波動率(%)", "技術指標_近一年歷史波動率(%)", "技術指標_近一月歷史波動率(%)",
"技術指標_季RSI(5)", "技術指標_季RSI(10)", "技術指標_季MACD", "技術指標_季K(9)",
"技術指標_季DIF-季MACD", "技術指標_季DIF", "技術指標_季-DI(14)", "技術指標_季D(9)",
"技術指標_季ADX(14)", "技術指標_季+DI(14)", "技術指標_乖離率(60日)", "技術指標_乖離率(250日)",
"技術指標_乖離率(20日)", "技術指標_年化波動度(250D)", "技術指標_年化波動度(21D)",
"技術指標_月RSI(5)", "技術指標_月RSI(10)", "技術指標_月MACD", "技術指標_月K(9)",
"技術指標_月DIF-月MACD", "技術指標_月DIF", "技術指標_月-DI(14)", "技術指標_月D(9)",
"技術指標_月ADX(14)", "技術指標_月+DI(14)", "技術指標_W%R(5)", "技術指標_W%R(10)",
"技術指標_RSI(5)", "技術指標_RSI(10)", "技術指標_MACD", "技術指標_K(9)",
"技術指標_EWMA波動率(%)", "技術指標_DIF-MACD", "技術指標_DIF", "技術指標_+DI(14)",
"技術指標_-DI(14)", "技術指標_D(9)", "技術指標_Beta係數(65D)", "技術指標_Beta係數(250D)",
"技術指標_Beta係數(21D)", "技術指標_Alpha(250D)", "技術指標_ADX(14)", "技術指標_保力加通道–頂部(20)",
"技術指標_保力加通道–均線(20)", "技術指標_保力加通道–底部(20)", "技術指標_CM-VIX(%)",
"技術指標_SAR", "技術指標_TR(1)", "技術指標_ADXR(14)", "技術指標_+DM(14)",
"技術指標_-DM(14)", "技術指標_週TR(14)", "技術指標_週ADXR(14)", "技術指標_週+DM(14)",
"技術指標_週-DM(14)", "技術指標_月TR(14)", "技術指標_月ADXR(14)", "技術指標_月+DM(14)",
"技術指標_月-DM(14)", "技術指標_季TR(14)", "技術指標_季ADXR(14)", "技術指標_季+DM(14)",
"技術指標_季-DM(14)",
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 模型與資料集定義 ==========
class StockDataset(Dataset):
    def __init__(self, df):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(df[feature_cols].values.astype("float32"))
        self.y = df['飆股'].values.astype("int64")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # 增加神經元數量
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)  # 降低 Dropout 比例

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.fc4(x))
        return self.out(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ========== FedAvg / FedAdam 更新邏輯 ==========
def fedavg_update(global_model, local_models):
    for key in global_model.state_dict().keys():
        avg_param = torch.stack([m.state_dict()[key].float() for m in local_models], dim=0).mean(dim=0)
        global_model.state_dict()[key].copy_(avg_param)
    return global_model

def fedadam_update(global_model, local_models, m_t, v_t, beta1=0.9, beta2=0.999, epsilon=1e-8, lr=1e-2):
    delta = {}
    for key in global_model.state_dict().keys():
        if not torch.is_floating_point(global_model.state_dict()[key]):
            continue  # 跳過非浮點數型別
        delta[key] = torch.stack([global_model.state_dict()[key] - local.state_dict()[key] for local in local_models], dim=0).mean(dim=0)

    for key in delta.keys():  # 只更新浮點數權重
        m_t[key] = beta1 * m_t[key] + (1 - beta1) * delta[key]
        v_t[key] = beta2 * v_t[key] + (1 - beta2) * delta[key] ** 2
        update = lr * m_t[key] / (v_t[key].sqrt() + epsilon)
        global_model.state_dict()[key].sub_(update)
    return global_model, m_t, v_t

# ========== 資料載入 ==========
def load_dataset_split(filepath):
    df = pd.read_csv(filepath).dropna()
    indices = range(len(df))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=df['飆股'], random_state=42)
    return Subset(StockDataset(df), train_idx), Subset(StockDataset(df), val_idx)

# ========== 聯邦訓練主程式 ==========
def federated_train_classifier(dataset_paths, local_epochs=5, num_rounds=20, patience=5):
    local_models, train_loaders, val_loaders = [], [], []

    for path in dataset_paths:
        train_set, val_set = load_dataset_split(path)
        train_loaders.append(DataLoader(train_set, batch_size=32, shuffle=True))
        val_loaders.append(DataLoader(val_set, batch_size=128))
        model = Classifier(len(feature_cols)).to(device)
        local_models.append(model)

    global_model = copy.deepcopy(local_models[0])
    criterion = FocalLoss(alpha=0.5, gamma=1)  # 減小 gamma

    if AGGREGATION_METHOD == "fedadam":
        m_t = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
        v_t = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}

    best_f1 = -1
    f1_history = []
    patience_counter = 0

    for rnd in range(num_rounds):
        print(f"🔁 Federated Round {rnd+1}/{num_rounds}")
        new_local_models = []
        for i, model in enumerate(local_models):
            model = copy.deepcopy(global_model)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            for _ in range(local_epochs):
                for x, y in train_loaders[i]:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits, y)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
            new_local_models.append(model)
            # 本地驗證 F1
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for loader in val_loaders:
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        pred = model(x).argmax(dim=1).cpu().numpy()  # 使用本地模型
                        all_preds.extend(pred)
                        all_labels.extend(y.cpu().numpy())
                local_f1 = f1_score(all_labels, all_preds, average='macro')
            print(f"  ✅ Client {i+1}: Local Val F1={local_f1:.4f}")

        if AGGREGATION_METHOD == "fedadam":
            global_model, m_t, v_t = fedadam_update(
                global_model, new_local_models, m_t, v_t,
                beta1=args.fedadam_beta1,
                beta2=args.fedadam_beta2,
                epsilon=args.fedadam_eps,
                lr=args.fedadam_lr)
            print("  🔄 Global model updated using FedAdam")
        else:
            global_model = fedavg_update(global_model, new_local_models)
            print("  🔄 Global model updated using FedAvg")

        # 全域驗證
        global_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for loader in val_loaders:
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    pred = global_model(x).argmax(dim=1).cpu().numpy()  # 確保轉為 NumPy 陣列
                    all_preds.extend(pred)
                    all_labels.extend(y.cpu().numpy())  # 確保轉為 NumPy 陣列
        global_f1 = f1_score(all_labels, all_preds, average='macro')
        f1_history.append(global_f1)
        print(f"  🌍 Global Val F1: {global_f1:.4f}")

        if global_f1 > best_f1:
            best_f1 = global_f1
            patience_counter = 0
            torch.save(global_model.state_dict(), f"classifier_best_{AGGREGATION_METHOD}_onlySkill.pth")
            print("  ✅ Best model updated.")
        else:
            patience_counter += 1
            print(f"  ⚠️ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered. Training stopped.")
                break  # 提前結束訓練

    # 儲存與畫圖
    torch.save(global_model.state_dict(), f"classifier_final_{AGGREGATION_METHOD}_onlySkill.pth")
    plt.plot(f1_history, label='Global Val F1')
    plt.title('Federated Learning Classification F1 Curve')
    plt.xlabel('Round')
    plt.ylabel('F1')
    plt.grid(True)
    plt.savefig(f"classifier_f1_curve_{AGGREGATION_METHOD}_onlySkill.png")
    print("📈 F1 curve saved.")
# ========== 執行區 ==========
if __name__ == '__main__':
    # import os 
    # dataset_paths = []
    # for file in os.listdir("E:\\Tbrain_stock_analysis\\sample_pool"):
    #     file_path = os.path.join("E:\\Tbrain_stock_analysis\\sample_pool", file)
    #     if file.endswith(".csv"):
    #         dataset_paths.append(file_path)

    dataset_paths = [
        "E:\\Tbrain_stock_analysis\\training_onlySkill.csv",
        "E:\\Tbrain_stock_analysis\\training_onlySkill.csv",
        "E:\\Tbrain_stock_analysis\\training_onlySkill.csv",
    ]

    # dataset_paths = [
    #     "E:\\Tbrain_stock_analysis\\training_4o_cleaned.csv",
    #     "E:\\Tbrain_stock_analysis\\training_4o_cleaned.csv",
    #     "E:\\Tbrain_stock_analysis\\training_4o_cleaned.csv",
    #     "E:\\Tbrain_stock_analysis\\training_4o_cleaned.csv",
    #     "E:\\Tbrain_stock_analysis\\training_4o_cleaned.csv",
    # ]

    federated_train_classifier(dataset_paths, local_epochs=50, num_rounds=200, patience=10)
