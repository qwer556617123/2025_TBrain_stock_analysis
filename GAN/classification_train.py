import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import gc

# 設定 GPU 裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特徵欄位
feature_cols = [
    '外資_進出_5d_Avg', '外資_買賣力_5d_Avg', '外資_吃貨比_5d_Avg',
    '外資_出貨比_5d_Avg', '外資_成交力_Avg', '主力_成交力_10d_Avg',
    '法人買張合計', '法人賣張合計', '法人買賣超合計', '法人持股比率合計',
    '技術指標_波動率_短期vs長期', '技術指標_Beta_短期vs長期',
    '月營收_綜合成長率_Avg', 'IFRS_流動性_Avg', 'IFRS_獲利能力_Avg',
    'IFRS_經營效率_Avg', 'IFRS_成長性_Avg', 'Top5買超_買均張_Avg',
    'Top5賣超_賣均值_Avg'
]

# Dataset 定義
class StockDataset(Dataset):
    def __init__(self, df):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(df[feature_cols].values.astype("float32"))
        self.y = df['飆股'].values.astype("int64")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 簡單分類器模型
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.fc3(x))
        return self.out(x)

# 定義 FocalLoss
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

# 定義資料讀取函數
def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna()  # 確保資料中沒有缺失值
    return df

# 訓練函數
def train():
    df = load_data("E:\\Tbrain_stock_analysis\\training_filtered_merged.csv")
    train_idx, val_idx = train_test_split(range(len(df)), test_size=0.2, stratify=df['飆股'], random_state=42)
    train_set = StockDataset(df.iloc[train_idx])
    val_set = StockDataset(df.iloc[val_idx])

    # 計算類別權重
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=df['飆股'])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # 更新損失函數為 FocalLoss
    criterion = FocalLoss(alpha=0.75, gamma=2)

    # 定義 WeightedRandomSampler
    class_counts = df.iloc[train_idx]['飆股'].value_counts()
    class_weights = 1. / class_counts
    train_weights = df.iloc[train_idx]['飆股'].map(class_weights).values

    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    # 更新 DataLoader
    train_loader = DataLoader(train_set, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=128)

    model = Classifier(len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    num_epochs = 150
    history = {"loss": [], "val_f1": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if torch.isnan(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()  # 更新學習率

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_true.extend(y.numpy())
                y_pred.extend(pred)

        f1 = f1_score(y_true, y_pred, average='macro')  # 使用 macro 或 weighted
        history["loss"].append(total_loss)
        history["val_f1"].append(f1)
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.2f}, Val F1: {f1:.4f}")

        # 繪製混淆矩陣
        if epoch == num_epochs - 1:  # 最後一個 epoch
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title(f"Confusion Matrix (Epoch {epoch+1})")
            plt.savefig(f"confusion_matrix_epoch_{epoch+1}.png")
            plt.close()

    torch.save(model.state_dict(), "classifier_final.pth")
    print("✅ 模型已儲存：classifier_final.pth")

    plt.plot(history["val_f1"], label="F1-score")
    plt.plot(history["loss"], label="Loss")
    plt.legend()
    plt.title("F1-score / Loss")
    plt.savefig("classifier_curve.png")
    plt.close()

if __name__ == '__main__':
    train()
