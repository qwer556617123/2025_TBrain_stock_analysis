import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import gc

from models.generator import Generator
from models.discriminator import Discriminator

# ====== 定義特徵欄位（全域）======
# feature_cols = [
#     '外資_進出_5d_Avg', '外資_買賣力_5d_Avg', '外資_吃貨比_5d_Avg',
#     '外資_出貨比_5d_Avg', '外資_成交力_Avg', '主力_成交力_10d_Avg',
#     '法人買張合計', '法人賣張合計', '法人買賣超合計', '法人持股比率合計',
#     '技術指標_波動率_短期vs長期', '技術指標_Beta_短期vs長期',
#     '月營收_綜合成長率_Avg', 'IFRS_流動性_Avg', 'IFRS_獲利能力_Avg',
#     'IFRS_經營效率_Avg', 'IFRS_成長性_Avg', 'Top5買超_買均張_Avg',
#     'Top5賣超_賣均值_Avg'
# ]
feature_cols = [
    '外資券商_mean','外資券商_std','外資券商_max','外資券商_min',
    '主力券商_mean','主力券商_std','主力券商_max','主力券商_min',
    '官股券商_mean','官股券商_std','官股券商_max','官股券商_min',
    '個股券商分點籌碼分析_mean','個股券商分點籌碼分析_std','個股券商分點籌碼分析_max',
    '個股券商分點籌碼分析_min','個股券商分點區域分析_mean','個股券商分點區域分析_std',
    '個股券商分點區域分析_max','個股券商分點區域分析_min','個股主力買賣超統計_mean',
    '個股主力買賣超統計_std','個股主力買賣超統計_max','個股主力買賣超統計_min',
    '日外資_mean','日外資_std','日外資_max','日外資_min','日自營_mean','日自營_std',
    '日自營_max','日自營_min','日投信_mean','日投信_std','日投信_max','日投信_min',
    '技術指標_mean','技術指標_std','技術指標_max','技術指標_min','月營收_mean',
    '月營收_std','月營收_max','月營收_min','季IFRS財報_mean','季IFRS財報_std',
    '季IFRS財報_max','季IFRS財報_min','買超分點_mean','買超分點_std','買超分點_max',
    '買超分點_min','賣超分點_mean','賣超分點_std','賣超分點_max','賣超分點_min',
    '其他_mean','其他_std','其他_max','其他_min'
]

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, input, target):
        logpt = -self.ce(input, target)
        pt = torch.exp(logpt)
        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * logpt
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# ====== 資料集定義 ======
class StockDataset(Dataset):
    def __init__(self, df, feature_cols, target_col='飆股'):
        self.X = df[feature_cols].values.astype('float32')
        self.y = df[target_col].values.astype('int64')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def preprocess_data(file_path, chunk_size=50000):
    all_chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk = chunk.drop(columns=['ID'], errors='ignore')
        chunk = chunk.dropna(subset=feature_cols + ['飆股'])
        all_chunks.append(chunk)

    full_df = pd.concat(all_chunks, ignore_index=True)
    scaler = StandardScaler()
    full_df[feature_cols] = scaler.fit_transform(full_df[feature_cols])
    return StockDataset(full_df, feature_cols)

def sample_binary_labels(batch_size, p=0.5):
    return torch.bernoulli(torch.full((batch_size,), p)).long()

# ====== 超參數 ======
noise_dim = 100
hidden_dim = 64
seq_len = 20
batch_size = 16  # 適配 8GB GPU
num_epochs = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== 資料處理與模型建立 ======
csv_path = "E:\\Tbrain_stock_analysis\\try\\balanced_set_1.csv"
dataset = preprocess_data(csv_path)
input_dim = len(feature_cols)

G = Generator(noise_dim, input_dim, hidden_dim, seq_len).to(device)
D = Discriminator(input_dim=input_dim).to(device)

optim_G = torch.optim.Adam(G.parameters(), lr=2e-4)
optim_D = torch.optim.Adam(D.parameters(), lr=2e-4)
bce_loss = nn.BCELoss()
ce_loss = FocalLoss(gamma=2.0)

# ====== 切分資料集並建立 Loader ======
indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=dataset.y, random_state=42)
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ====== Weighted Loss ======
class_sample_counts = [sum(dataset.y == 0), sum(dataset.y == 1)]
weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float32)
class_weights = weights / weights.sum()
criterion_bce = nn.BCELoss(reduction='none')

def weighted_bce_loss(pred, target):
    weight = torch.tensor([class_weights[1] if t == 1 else class_weights[0] for t in target], device=pred.device)
    return (criterion_bce(pred, target.float()) * weight).mean()

# ====== 訓練主迴圈 ======
best_f1 = 0
patience = 3
wait = 0

torch.cuda.empty_cache()
gc.collect()

metrics = {"epoch": [], "loss_D": [], "loss_G": [], "val_f1": []}

for epoch in range(num_epochs):
    G.train(); D.train()
    total_loss_D = 0; total_loss_G = 0

    for real_seq, real_label in train_loader:
        real_seq, real_label = real_seq.to(device), real_label.to(device)

        optim_D.zero_grad()
        real_out, real_cls_out = D(real_seq)
        loss_D_real = bce_loss(real_out, torch.ones_like(real_out))
        loss_D_cls = ce_loss(real_cls_out, real_label)

        z = torch.randn(batch_size, noise_dim).to(device)
        fake_labels = sample_binary_labels(batch_size).to(device)
        fake_seq = G(z, fake_labels)
        fake_out, _ = D(fake_seq[:, -1, :].detach())
        loss_D_fake = bce_loss(fake_out, torch.zeros_like(fake_out))
        loss_D = loss_D_real + loss_D_fake + 0.5 * loss_D_cls
        loss_D.backward()
        optim_D.step()

        optim_G.zero_grad()
        z = torch.randn(batch_size, noise_dim).to(device)
        gen_labels = sample_binary_labels(batch_size).to(device)
        gen_seq = G(z, gen_labels)
        gen_out, gen_cls_out = D(gen_seq[:, -1, :])  # 使用最後一個時間步作為輸入
        loss_G_adv = bce_loss(gen_out, torch.ones_like(gen_out))
        # 檢查形狀是否正確
        assert gen_cls_out.dim() == 2 and gen_cls_out.shape[1] == 2, f"gen_cls_out shape is {gen_cls_out.shape}, should be [B, 2]"
        assert gen_labels.dim() == 1, f"gen_labels shape is {gen_labels.shape}, should be [B]"
        loss_G_cls = ce_loss(gen_cls_out, gen_labels.long())  # 保證是 logits 對 long labels
        loss_G = loss_G_adv + 0.5 * loss_G_cls
        loss_G.backward()
        optim_G.step()

        total_loss_D += loss_D.item()
        total_loss_G += loss_G.item()

    # === 驗證 ===
    G.eval(); D.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for val_seq, val_label in val_loader:
            val_seq, val_label = val_seq.to(device), val_label.to(device)
            _, cls_out = D(val_seq)
            pred = torch.argmax(torch.softmax(cls_out, dim=1), dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(val_label.cpu().numpy())

    val_f1 = f1_score(all_targets, all_preds)
    metrics["epoch"].append(epoch + 1)
    metrics["loss_D"].append(total_loss_D)
    metrics["loss_G"].append(total_loss_G)
    metrics["val_f1"].append(val_f1)

    print(f"[Epoch {epoch+1}] D Loss: {total_loss_D:.4f}, G Loss: {total_loss_G:.4f}, Val F1: {val_f1:.4f}")

    
if val_f1 > best_f1:
    best_f1 = val_f1
    torch.save(G.state_dict(), "GAN_generator_final.pth")
    torch.save(D.state_dict(), "GAN_discriminator_final.pth")


# 儲存學習歷史與圖表
df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv("GAN_training_metrics.csv", index=False)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(df_metrics["epoch"], df_metrics["loss_D"], label="Discriminator Loss")
plt.plot(df_metrics["epoch"], df_metrics["loss_G"], label="Generator Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df_metrics["epoch"], df_metrics["val_f1"], label="Validation F1", color="green")
plt.title("Validation F1 Score")
plt.xlabel("Epoch"); plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig("GAN_training_curve.png")
plt.close()
print("📊 已儲存 loss/F1 圖表與 GAN_training_metrics.csv")