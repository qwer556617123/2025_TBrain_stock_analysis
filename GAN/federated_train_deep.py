import os
import copy
import torch
import pandas as pd
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.generator_deep import Generator
from models.discriminator_deep import Discriminator
import torch.nn as nn

# ========== 特徵欄位與超參數 ==========
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
    '其他_mean','其他_std','其他_max','其他_min']

noise_dim = 100
hidden_dim = 64
seq_len = 20
batch_size = 32
local_epochs = 10
num_rounds = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== 資料處理 ==========
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

# ========== Focal Loss 實作 ==========
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

# ========== CLI 設定 ==========
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fedadam_lr', type=float, default=0.0005, help='FedAdam global learning rate')
parser.add_argument('--fedadam_beta1', type=float, default=0.9, help='FedAdam beta1')
parser.add_argument('--fedadam_beta2', type=float, default=0.999, help='FedAdam beta2')
parser.add_argument('--fedadam_eps', type=float, default=1e-8, help='FedAdam epsilon')
parser.add_argument('--aggr', type=str, default='fedadam', choices=['fedadam', 'fedavg'], help='Aggregation method')
args = parser.parse_args()
AGGREGATION_METHOD = args.aggr

# ========== FedAdam 更新 ==========
def fedadam_update(global_model, local_models, m_t, v_t, beta1=0.9, beta2=0.999, epsilon=1e-8, lr=1e-2):
    delta = {}
    for key in global_model.state_dict().keys():
        delta[key] = torch.stack([global_model.state_dict()[key] - local.state_dict()[key] for local in local_models], dim=0).mean(dim=0)

    for key in global_model.state_dict().keys():
        m_t[key] = beta1 * m_t[key] + (1 - beta1) * delta[key]
        v_t[key] = beta2 * v_t[key] + (1 - beta2) * (delta[key] ** 2)
        update = lr * m_t[key] / (v_t[key].sqrt() + epsilon)
        global_model.state_dict()[key].sub_(update)

    return global_model, m_t, v_t

# ========== FedAvg 更新 ==========
def fedavg_update(global_model, local_models):
    for key in global_model.state_dict().keys():
        avg_param = torch.stack([model.state_dict()[key].float() for model in local_models], dim=0).mean(dim=0)
        global_model.state_dict()[key].copy_(avg_param)
    return global_model

# ========== 平均模型權重（Deprecated by FedAvg/FedAdam） ==========
def average_weights(generators, discriminators):
    def average_model(models):
        global_model = copy.deepcopy(models[0])
        for key in global_model.state_dict().keys():
            avg_param = torch.stack([m.state_dict()[key].float() for m in models], dim=0).mean(dim=0)
            global_model.state_dict()[key].copy_(avg_param)
        return global_model
    return average_model(generators), average_model(discriminators)

# ========== 單一 Epoch 訓練 ==========
def train_one_epoch(G, D, loader, criterion_bce, criterion_cls, optim_G, optim_D, noise_dim, device):
    G.train(); D.train()
    total_loss_G = 0; total_loss_D = 0
    for real_seq, real_label in loader:
        real_seq, real_label = real_seq.to(device), real_label.to(device)
        real_label = real_label.view(-1)

        z = torch.randn(real_seq.size(0), noise_dim).to(device)
        gen_labels = torch.bernoulli(torch.full((real_seq.size(0),), 0.5)).long().to(device)

        # === 訓練 Discriminator ===
        fake_seq = G(z, gen_labels)
        optim_D.zero_grad()
        real_out, real_cls_out = D(real_seq)
        loss_D_real = criterion_bce(real_out, torch.ones_like(real_out))
        loss_D_cls = criterion_cls(real_cls_out, real_label)
        fake_out, _ = D(fake_seq.detach())
        loss_D_fake = criterion_bce(fake_out, torch.zeros_like(fake_out))
        loss_D = loss_D_real + loss_D_fake + 0.5 * loss_D_cls
        loss_D.backward()
        optim_D.step()

        # === 訓練 Generator ===
        optim_G.zero_grad()
        gen_seq = G(z, gen_labels)
        gen_out, gen_cls_out = D(gen_seq)

        # ✅ 檢查是否 batch_size 和 label 對齊，避免 1 vs 32 錯誤
        if gen_cls_out.shape[0] != gen_labels.shape[0]:
            gen_cls_out = gen_cls_out.expand(gen_labels.shape[0], -1)

        loss_G_adv = criterion_bce(gen_out, torch.ones_like(gen_out))
        loss_G_cls = criterion_cls(gen_cls_out, gen_labels)
        loss_G = loss_G_adv + 0.5 * loss_G_cls
        loss_G.backward()
        optim_G.step()

        total_loss_G += loss_G.item()
        total_loss_D += loss_D.item()
    return total_loss_G, total_loss_D

# ========== 驗證 ==========
def evaluate_model(D, val_loaders):
    D.eval()
    scores = []
    with torch.no_grad():
        for i, loader in enumerate(val_loaders):
            all_preds, all_labels = [], []
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                _, cls_out = D(x)
                pred = torch.argmax(torch.softmax(cls_out, dim=1), dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
            f1 = f1_score(all_labels, all_preds)
            scores.append(f1)
            print(f"    📊 Global Model Val F1 on Dataset {i+1}: {f1:.4f}")
    return scores

# ========== 聯邦訓練主程式 ==========
def federated_train(dataset_paths):
    local_generators, local_discriminators = [], []
    local_loaders, val_loaders = [], []

    for path in dataset_paths:
        dataset = preprocess_data(path)
        indices = list(range(len(dataset)))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=dataset.y, random_state=42)
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        local_loaders.append(DataLoader(train_subset, batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(val_subset, batch_size=batch_size, shuffle=False))

        G = Generator(noise_dim, len(feature_cols), hidden_dim, seq_len).to(device)
        D = Discriminator(input_dim=len(feature_cols)).to(device)
        local_generators.append(G)
        local_discriminators.append(D)


    criterion_bce = torch.nn.BCELoss()
    criterion_cls = FocalLoss(gamma=2.0)

    global_G = copy.deepcopy(local_generators[0])
    global_D = copy.deepcopy(local_discriminators[0])

    if AGGREGATION_METHOD == "fedadam":
        m_G = {k: torch.zeros_like(v) for k, v in global_G.state_dict().items()}
        v_G = {k: torch.zeros_like(v) for k, v in global_G.state_dict().items()}
        m_D = {k: torch.zeros_like(v) for k, v in global_D.state_dict().items()}
        v_D = {k: torch.zeros_like(v) for k, v in global_D.state_dict().items()}


    best_f1 = -1.0
    f1_history = []
    patience = 10
    patience_counter = 0

    for rnd in range(num_rounds):
        print(f"\n🔁 Federated Round {rnd+1}/{num_rounds}")
        for i in range(len(dataset_paths)):
            G, D = local_generators[i], local_discriminators[i]
            optim_G = torch.optim.Adam(G.parameters(), lr=2e-4)
            optim_D = torch.optim.Adam(D.parameters(), lr=2e-4)
            for _ in range(local_epochs):
                loss_G, loss_D = train_one_epoch(G, D, local_loaders[i], criterion_bce, criterion_cls, optim_G, optim_D, noise_dim, device)
            D.eval()
            val_loader = val_loaders[i]
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    _, cls_out = D(x)
                    pred = torch.argmax(torch.softmax(cls_out, dim=1), dim=1)
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
            val_f1 = f1_score(all_labels, all_preds)
            print(f"  ✅ Dataset {i+1}: G_loss={loss_G:.2f}, D_loss={loss_D:.2f}, Local Val F1={val_f1:.4f}")

        if AGGREGATION_METHOD == "fedadam":
            global_G, m_G, v_G = fedadam_update(global_G, local_generators, m_G, v_G, beta1=args.fedadam_beta1, beta2=args.fedadam_beta2, epsilon=args.fedadam_eps, lr=args.fedadam_lr)
            global_D, m_D, v_D = fedadam_update(global_D, local_discriminators, m_D, v_D, beta1=args.fedadam_beta1, beta2=args.fedadam_beta2, epsilon=args.fedadam_eps, lr=args.fedadam_lr)
            print("  🔄 Global model updated using FedAdam")
        else:
            global_G = fedavg_update(global_G, local_generators)
            global_D = fedavg_update(global_D, local_discriminators)
            print("  🔄 Global model updated using FedAvg")

        f1_scores = evaluate_model(global_D, val_loaders)
        avg_f1 = sum(f1_scores) / len(f1_scores)
        f1_history.append(avg_f1)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            patience_counter = 0
            torch.save(global_G.state_dict(), "best_federated_GAN_generator_" + AGGREGATION_METHOD + ".pth")
            torch.save(global_D.state_dict(), "best_federated_GAN_discriminator_" + AGGREGATION_METHOD + ".pth")
            print(f"  ✅ New best global F1: {best_f1:.4f}, model saved")
        else:
            patience_counter += 1
            print(f"  ⚠️ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

        for i in range(len(dataset_paths)):
            local_generators[i].load_state_dict(global_G.state_dict())
            local_discriminators[i].load_state_dict(global_D.state_dict())

    torch.save(global_G.state_dict(), "federated_GAN_generator_" + AGGREGATION_METHOD + ".pth")
    torch.save(global_D.state_dict(), "federated_GAN_discriminator_" + AGGREGATION_METHOD + ".pth")
    
    # 畫出每輪的平均 F1 曲線
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, len(f1_history)+1), f1_history, marker='o')
    plt.title("Global Average F1 per Round")
    plt.xlabel("Federated Round")
    plt.ylabel("Average F1 Score")
    plt.grid(True)
    plt.savefig("federated_avg_f1_curve_" + AGGREGATION_METHOD + ".png")
    print("📈 F1 curve saved as federated_avg_f1_curve.png")

    print("\n✅ 全局模型儲存完成")

# ========== 執行區 ==========
if __name__ == "__main__":
    import os 
    dataset_paths = []
    for file in os.listdir("E:\\Tbrain_stock_analysis\\sample_pool"):
        file_path = os.path.join("E:\\Tbrain_stock_analysis\\sample_pool", file)
        if file.endswith(".csv"):
            dataset_paths.append(file_path)
    # dataset_paths = [
    #     "E:\\Tbrain_stock_analysis\\try\\balanced_set_1.csv",
    #     "E:\\Tbrain_stock_analysis\\try\\balanced_set_2.csv",
    #     "E:\\Tbrain_stock_analysis\\try\\balanced_set_3.csv",
    #     "E:\\Tbrain_stock_analysis\\try\\balanced_set_4.csv",
    #     "E:\\Tbrain_stock_analysis\\try\\balanced_set_5.csv",
    #     "E:\\Tbrain_stock_analysis\\try\\balanced_set_6.csv",
    #     "E:\\Tbrain_stock_analysis\\try\\balanced_set_7.csv",
    #     "E:\\Tbrain_stock_analysis\\try\\balanced_set_8.csv",
    #     "E:\\Tbrain_stock_analysis\\try\\balanced_set_9.csv",
    #     "E:\\Tbrain_stock_analysis\\try\\balanced_set_10.csv",
    # ]
    federated_train(dataset_paths)
