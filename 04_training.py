# 04_training.py

import os
import gc
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# LightGBM / XGBoost / CatBoost
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# For stacking
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier

# 用於內存釋放
import gc

# 是否有需要的話可引入 psutil 監控 CPU, 記憶體，但這裡暫不示範
# import psutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================
# 新增：優化 learning curve
# ==========================
def plot_learning_curve_optimized(estimator, X, y,
                                  train_sizes=[0.2, 0.5, 1.0],
                                  cv=3,
                                  scoring='accuracy',
                                  n_jobs=1):
    """
    使用較少的 train_sizes，減少 cv 次數，一次性繪圖。
    n_jobs: 1 可避免同時開太多CPU執行緒導致卡頓。
    """
    print("[Info] 開始繪製學習曲線 (Optimized)")
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring,
        n_jobs=n_jobs
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)

    plt.figure()
    plt.title("Learning Curve (Optimized)")
    plt.xlabel("Training examples")
    plt.ylabel(scoring)

    # Plot training scores
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training score")
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1)

    # Plot validation scores
    plt.plot(train_sizes, valid_scores_mean, 'o-', label="Validation score")
    plt.fill_between(train_sizes,
                     valid_scores_mean - valid_scores_std,
                     valid_scores_mean + valid_scores_std, alpha=0.1)

    plt.legend(loc="best")
    plt.show()
    plt.close()


# ==========================
# 示例：對 DataFrame 作 SMOTE
# 可自行套用至 handle_imbalance 或 tune_hyperparameters 內
# ==========================
def apply_smote(X, y, ratio=0.5, random_state=42):
    """
    以較低比例 (ratio) 執行 SMOTE, ratio=0.5 => 少數類別會變成多數類別的一半。
    """
    print(f"[Info] Applying SMOTE with sampling_strategy={ratio}")
    sm = SMOTE(sampling_strategy=ratio, random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    print("After SMOTE, X shape:", X_res.shape)
    print("Class distribution:", y_res.value_counts())
    return X_res, y_res


# ==========================
# DataSet for PyTorch
# ==========================
class StockDataset(Dataset):
    def __init__(self, X_df, y_df):
        self.X = torch.tensor(X_df.values, dtype=torch.float32)
        self.y = torch.tensor(y_df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================
# Shallow NN
# ==========================
class ShallowNN(nn.Module):
    def __init__(self, input_dim):
        super(ShallowNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Output logits for binary classification
        )

    def forward(self, x):
        return self.fc(x).squeeze()


# ==========================
# Deep NN
# ==========================
class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Output logits for binary classification
        )

    def forward(self, x):
        return self.fc(x).squeeze()


# ==========================
# 模型評估函式
# ==========================
def evaluate_model(model, X_test, y_test, model_name=''):
    """
    統一的模型評估函式，可同時支援 sklearn 類與 PyTorch
    """
    if isinstance(model, nn.Module):
        # PyTorch
        model.eval()
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        del X_test_tensor, outputs
        gc.collect()
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
    else:
        # Sklearn / XGBoost / LightGBM / CatBoost
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)

    print(f"{model_name} 評估結果:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC Score: {auc:.4f}")

    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }


# ==========================
# CatBoost/XGB/LightGBM 調參或訓練函式
# 已改為限制執行緒與 subsample
# ==========================
def train_catboost(X_train, y_train, X_val, y_val,
                   n_estimators=600, max_depth=8,
                   subsample=0.8,
                   thread_count=4,
                   use_gpu=True):
    task_type = 'GPU' if use_gpu else 'CPU'
    print(f"\n=== 訓練 CatBoost (GPU={use_gpu}, thread_count={thread_count}, subsample={subsample}) ===")
    model = CatBoostClassifier(
        od_type='Iter',
        od_wait=1000,
        iterations=n_estimators,
        learning_rate=0.01,
        depth=max_depth,
        subsample=subsample,
        thread_count=thread_count,
        task_type=task_type,
        random_state=42,
        verbose= 1000,
        bootstrap_type='Poisson',
        eval_metric='F1'
    )
    model.fit(X_train, y_train,
              eval_set=(X_val, y_val),)
    return model


def train_xgboost(X_train, y_train, X_val, y_val,
                  n_estimators=600, max_depth=6,
                  subsample=0.8,
                  n_jobs=4):
    print(f"\n=== 訓練 XGBoost (n_jobs={n_jobs}, subsample={subsample}) ===")
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        learning_rate=0.01,
        random_state=42,
        n_jobs=n_jobs,
        verbose_eval=1000,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val,
                   n_estimators=600, max_depth=6,
                   subsample=0.8,
                   n_jobs=4):
    print(f"\n=== 訓練 LightGBM (n_jobs={n_jobs}, subsample={subsample}) ===")
    # subsample對應 bagging_fraction
    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        bagging_fraction=subsample,
        learning_rate=0.01,
        random_state=42,
        n_jobs=n_jobs,
        # 需搭配 bagging_freq != 0
        bagging_freq=1,
        verbose_eval=1000,
        early_stopping_rounds=1000
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              )
    return model


# ==========================
# 主流程 (main)
# ==========================
def main():
    try:
        df = pd.read_csv('training_filtered_merged.csv')
        print(f"成功載入資料，資料大小: {df.shape}")
    except FileNotFoundError:
        print("找不到訓練資料檔案，請確認檔案路徑。")
        return

    # 分離特徵與目標變數
    X = df.drop('飆股', axis=1)
    if 'ID' in X.columns:
        X.drop('ID', axis=1, inplace=True)
    y = df['飆股']

    class_counts = y.value_counts()
    print("類別分佈:")
    print(class_counts)
    print(f"類別不平衡比例: 1:{class_counts[0] / class_counts[1]:.2f}")

    # 釋放原始 df
    del df
    gc.collect()
    print("Original DataFrame released.")

    # 切分 train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"訓練集大小: {X_train.shape}, 測試集大小: {X_test.shape}")

    # 再切出 validation (e.g. 20% of train)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # SMOTE (降低補樣比例，以免資料暴增)
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train, ratio=1.0)

    # ============ 訓練 CatBoost ============
    cat_model = train_catboost(
        X_train_balanced, y_train_balanced,
        X_val, y_val,
        n_estimators=10000,
        max_depth=8,
        subsample=0.8,
        thread_count=4,
        use_gpu=torch.cuda.is_available()
    )
    cb_results = evaluate_model(cat_model, X_test, y_test, 'CatBoost')
    cat_model.save_model("models/catboost_model.cbm")

    # ============ 訓練 XGBoost ============
    xgb_model = train_xgboost(
        X_train_balanced, y_train_balanced,
        X_val, y_val,
        n_estimators=10000,
        max_depth=8,
        subsample=0.8,
        n_jobs=4
    )
    xgb_results = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    joblib.dump(xgb_model, "models/xgb_model.pkl")

    # ============ 訓練 LightGBM ============
    lgb_model = train_lightgbm(
        X_train_balanced, y_train_balanced,
        X_val, y_val,
        n_estimators=10000,
        max_depth=8,
        subsample=0.8,
        n_jobs=4
    )
    lgb_results = evaluate_model(lgb_model, X_test, y_test, 'LightGBM')
    joblib.dump(lgb_model, "models/lightgbm_model.pkl")

    # ============ NN (Shallow) ============
    print("\n=== 訓練 Shallow NN ===")
    input_dim = X_train_balanced.shape[1]
    shallow_nn_model = ShallowNN(input_dim).to(device)
    criterion_shallow = nn.BCEWithLogitsLoss()
    optimizer_shallow = optim.AdamW(shallow_nn_model.parameters(), lr=0.001, weight_decay=1e-4)

    # 準備 PyTorch DataLoader
    train_dataset = StockDataset(X_train_balanced, y_train_balanced)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=0)  # num_workers=0 避免過多 CPU Threads
    epochs = 30
    best_val_f1 = -1

    for epoch in range(epochs):
        shallow_nn_model.train()
        epoch_loss = 0.0
        for i, (inputs_batch, labels_batch) in enumerate(train_loader):
            inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
            optimizer_shallow.zero_grad()
            outputs_batch = shallow_nn_model(inputs_batch)
            loss = criterion_shallow(outputs_batch, labels_batch)
            loss.backward()
            optimizer_shallow.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)

        # 簡易 validation，用 X_val
        shallow_nn_model.eval()
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs_val = shallow_nn_model(X_val_tensor)
            val_probs = torch.sigmoid(outputs_val).cpu().numpy().flatten()
        val_preds = (val_probs > 0.5).astype(int)
        val_f1 = f1_score(y_val, val_preds, zero_division=0)

        print(f"[ShallowNN] Epoch {epoch+1}/{epochs}, Loss={avg_epoch_loss:.4f}, Val F1={val_f1:.4f}")
        del X_val_tensor, outputs_val, val_probs, val_preds
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(shallow_nn_model.state_dict(), 'models/shallow_nn_best.pth')

    # 載入最佳權重
    shallow_nn_model.load_state_dict(torch.load('models/shallow_nn_best.pth'))
    shallow_nn_results = evaluate_model(shallow_nn_model, X_test, y_test, 'ShallowNN')

    # 釋放 train_loader
    del train_loader, train_dataset
    gc.collect()

    # ============ NN (Deep) ============
    print("\n=== 訓練 Deep NN ===")
    deep_nn_model = DeepNN(input_dim).to(device)
    criterion_deep = nn.BCEWithLogitsLoss()
    optimizer_deep = optim.AdamW(deep_nn_model.parameters(), lr=0.0005, weight_decay=1e-4)

    train_dataset2 = StockDataset(X_train_balanced, y_train_balanced)
    train_loader2 = DataLoader(train_dataset2, batch_size=128, shuffle=True,
                               num_workers=0)

    epochs = 50
    best_val_f1 = -1

    for epoch in range(epochs):
        deep_nn_model.train()
        epoch_loss = 0.0
        for i, (inputs_batch, labels_batch) in enumerate(train_loader2):
            inputs_batch, labels_batch = inputs_batch.to(device), labels_batch.to(device)
            optimizer_deep.zero_grad()
            outputs_batch = deep_nn_model(inputs_batch)
            loss = criterion_deep(outputs_batch, labels_batch)
            loss.backward()
            optimizer_deep.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader2)

        # validation
        deep_nn_model.eval()
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs_val = deep_nn_model(X_val_tensor)
            val_probs = torch.sigmoid(outputs_val).cpu().numpy().flatten()
        val_preds = (val_probs > 0.5).astype(int)
        val_f1 = f1_score(y_val, val_preds, zero_division=0)

        print(f"[DeepNN] Epoch {epoch+1}/{epochs}, Loss={avg_epoch_loss:.4f}, Val F1={val_f1:.4f}")
        del X_val_tensor, outputs_val, val_probs, val_preds
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(deep_nn_model.state_dict(), 'models/deep_nn_best.pth')

    # 載入最佳權重
    deep_nn_model.load_state_dict(torch.load('models/deep_nn_best.pth'))
    deep_nn_results = evaluate_model(deep_nn_model, X_test, y_test, 'DeepNN')

    del train_loader2, train_dataset2
    gc.collect()

    # === 將結果整合 ===
    results = []
    results.append(cb_results)
    results.append(xgb_results)
    results.append(lgb_results)
    results.append(shallow_nn_results)
    results.append(deep_nn_results)

    results_df = pd.DataFrame(results).set_index('model')
    results_df.sort_values(by='f1_score', ascending=False, inplace=True)
    print("\n=== 模型評估結果比較 ===")
    print(results_df)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/model_comparison.csv')

    # 最佳模型
    best_model_name = results_df['f1_score'].idxmax()
    print(f"\n最佳模型 (以 F1 分數為準): {best_model_name}")
    best_f1 = results_df.loc[best_model_name, 'f1_score']
    print(f"F1 = {best_f1:.4f}")

    with open('results/best_model.txt', 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"F1 Score: {best_f1:.4f}\n")
        f.write(results_df.to_string())

    # 最後釋放
    del X, y
    del X_train, X_val, X_test
    del y_train, y_val, y_test
    gc.collect()
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()

    print("\n=== 訓練流程完成，更新後的 04_training.py 已執行結束 ===")


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    main()
