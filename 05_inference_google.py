import os
import argparse
import pickle

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 命令列參數 ---
parser = argparse.ArgumentParser(description='股票飆股預測推論程式')
parser.add_argument('--voting', type=str, default='hard_majority', 
                    choices=['hard_majority', 'hard_weighted', 'soft_avg', 'soft_weighted', 'ensemble_all'],
                    help='選擇投票機制類型')
parser.add_argument('--threshold', type=float, default=0.5, 
                    help='軟投票的閾值 (對於soft_avg和soft_weighted)')
parser.add_argument('--test_file', type=str, default='filtered_public_x_merged.csv',
                    help='測試數據檔案名稱')
parser.add_argument('--output_file', type=str, default='prediction_results.csv',
                    help='輸出預測結果檔案名稱')
args = parser.parse_args()

# 輸出檔案名稱根據投票方式做後綴
args.output_file = args.output_file.replace('.csv', f'_{args.voting}.csv')

MODEL_DIR = 'models'

# 檢查必要檔案
required_files = [
    'scaler.pkl', 'feature_selector.pkl',
    'gbdt_model.pkl', 'xgb_model.pkl', 'lgbm_model.pkl',
    'shallow_nn_model.pth', 'deep_nn_model.pth'
]
for fn in required_files:
    path = os.path.join(MODEL_DIR, fn)
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少模型檔案：{path}")

# --- 特徵工程（複製自 04_training.py） ---
def preprocess_test_features(X, scaler, feature_engineering=True):
    X_scaled = X.copy()
    # 找出數值型欄位
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    # 標準化
    X_scaled[numerical_features] = scaler.transform(X[numerical_features])
    
    if feature_engineering:
        # 互動項：前兩個價格特徵 × 前兩個交易量特徵
        price_feats = [c for c in numerical_features if 'price' in c.lower() or 'close' in c.lower()]
        vol_feats   = [c for c in numerical_features if 'volume' in c.lower() or 'vol' in c.lower()]
        if price_feats and vol_feats:
            for p in price_feats[:2]:
                for v in vol_feats[:2]:
                    X_scaled[f"{p}_x_{v}"] = X_scaled[p] * X_scaled[v]
        # 群組統計特徵
        groups = []
        for prefix in ['price','volume','ma','rsi']:
            grp = [c for c in numerical_features if prefix in c.lower()]
            if len(grp) >= 3:
                groups.append(grp)
        for i, grp in enumerate(groups):
            X_scaled[f'group_{i}_mean'] = X_scaled[grp].mean(axis=1)
            X_scaled[f'group_{i}_std']  = X_scaled[grp].std(axis=1)
            X_scaled[f'group_{i}_max']  = X_scaled[grp].max(axis=1)
            X_scaled[f'group_{i}_min']  = X_scaled[grp].min(axis=1)
    return X_scaled

# --- 定義神經網路結構（與訓練時完全一致） ---
class ShallowNN(nn.Module):
    def __init__(self, input_dim):
        super(ShallowNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)

class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return self.sigmoid(out)

# --- 載入標準化器、特徵選擇器和模型 ---
print("載入 scaler、selector 和各模型...")
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'feature_selector.pkl'), 'rb') as f:
    selector = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'gbdt_model.pkl'), 'rb') as f:
    gbdt_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'xgb_model.pkl'), 'rb') as f:
    xgb_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'lgbm_model.pkl'), 'rb') as f:
    lgbm_model = pickle.load(f)

# 準備 PyTorch 模型
# 先拿到一個 dummy input_dim（之後在每個 chunk 重新設定）
_dummy_dim = 10
shallow_nn_model = ShallowNN(_dummy_dim)
deep_nn_model    = DeepNN(_dummy_dim)
shallow_nn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'shallow_nn_model.pth')))
deep_nn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'deep_nn_model.pth')))
shallow_nn_model.eval()
deep_nn_model.eval()

# --- 投票機制權重設定 ---
model_weights = {
    'gbdt': 0.15, 'xgboost': 0.25, 'lightgbm': 0.20,
    'shallow_nn': 0.15, 'deep_nn': 0.25
}
# 正規化
total_w = sum(model_weights.values())
for k in model_weights:
    model_weights[k] /= total_w

print(f"使用投票機制: {args.voting}，模型權重: {model_weights}")

# --- 逐批讀取測試資料並預測 ---
all_ids = []
all_preds = []
all_model_probs = []

chunk_size = 10000
for idx, chunk in enumerate(pd.read_csv(args.test_file, chunksize=chunk_size)):
    print(f"處理第 {idx+1} 批資料...")
    ids = chunk['ID'].values
    X_raw = chunk.drop('ID', axis=1)
    
    # 特徵工程 + 標準化
    X_scaled = preprocess_test_features(X_raw, scaler, feature_engineering=True)
    # 套用特徵選擇
    mask = selector.get_support()
    selected_cols = X_scaled.columns[mask]
    X_sel = X_scaled[selected_cols]
    
    # 傳統模型預測
    catboost_prob = cat_model.predict_proba(X_test)[:, 1]
    xgb_prob  = xgb_model.predict_proba(X_sel)[:,1]
    lgbm_prob = lgbm_model.predict_proba(X_sel)[:,1]
    catboost_pred = (catboost_prob > 0.5).astype(int)
    xgb_pred  = (xgb_prob  > 0.5).astype(int)
    lgbm_pred = (lgbm_prob > 0.5).astype(int)
    
    # PyTorch 輸入
    X_tensor = torch.tensor(X_sel.values, dtype=torch.float32)
    # 重新設定模型 input_dim（因為特徵數量可能變動）
    input_dim = X_sel.shape[1]
    shallow_nn_model = ShallowNN(input_dim); shallow_nn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR,'shallow_nn_model.pth'))); shallow_nn_model.eval()
    deep_nn_model    = DeepNN(input_dim);    deep_nn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR,'deep_nn_model.pth')));    deep_nn_model.eval()
    
    with torch.no_grad():
        shallow_prob = shallow_nn_model(X_tensor).cpu().numpy().flatten()
        deep_prob    = deep_nn_model(X_tensor).cpu().numpy().flatten()
    shallow_pred = (shallow_prob > 0.5).astype(int)
    deep_pred    = (deep_prob    > 0.5).astype(int)
    
    # 最終投票
    batch_preds = []
    for i in range(len(X_sel)):
        hard_votes = [gbdt_pred[i], xgb_pred[i], lgbm_pred[i], shallow_pred[i], deep_pred[i]]
        prob_votes = [gbdt_prob[i], xgb_prob[i], lgbm_prob[i], shallow_prob[i], deep_prob[i]]
        sample_probs = {
            'gbdt': gbdt_prob[i], 'xgboost': xgb_prob[i],
            'lightgbm': lgbm_prob[i], 'shallow_nn': shallow_prob[i],
            'deep_nn': deep_prob[i]
        }
        all_model_probs.append(sample_probs)
        
        if args.voting == 'hard_majority':
            final = 1 if sum(hard_votes) >= 3 else 0
        elif args.voting == 'hard_weighted':
            wv = sum(h*model_weights[m] for h,m in zip(hard_votes, model_weights))
            final = 1 if wv >= 0.5 else 0
        elif args.voting == 'soft_avg':
            ap = sum(prob_votes)/len(prob_votes)
            final = 1 if ap >= args.threshold else 0
        elif args.voting == 'soft_weighted':
            wp = sum(p*model_weights[m] for p,m in zip(prob_votes, model_weights))
            final = 1 if wp >= args.threshold else 0
        else:  # ensemble_all
            if all(v==1 for v in hard_votes): final = 1
            elif all(v==0 for v in hard_votes): final = 0
            else:
                wp = sum(p*model_weights[m] for p,m in zip(prob_votes, model_weights))
                final = 1 if wp >= args.threshold else 0
        batch_preds.append(final)
    
    all_ids.extend(ids)
    all_preds.extend(batch_preds)

# --- 儲存結果 ---
res_df = pd.DataFrame({'ID': all_ids, '飆股': all_preds})
res_df.to_csv(args.output_file, index=False)
print(f"預測結果已儲存至 {args.output_file}")

# Optional: 詳細機率存檔
prob_df = pd.DataFrame(all_model_probs)
prob_df['ID'] = all_ids
prob_df['飆股'] = all_preds
prob_df.to_csv(f"detailed_model_predictions_{args.voting}.csv", index=False)

print(f"總樣本數: {len(res_df)}，預測為飆股比例: {res_df['飆股'].mean():.4f}")
