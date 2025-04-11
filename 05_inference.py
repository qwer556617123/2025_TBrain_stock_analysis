import os
import argparse
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# [改動] 新增 CatBoost 載入
from catboost import CatBoostClassifier

# --- Command Line Arguments ---
parser = argparse.ArgumentParser(description='Stock Prediction Inference Script')
parser.add_argument('--voting', type=str, default='hard_majority',
                    choices=['hard_majority', 'hard_weighted', 'soft_avg', 'soft_weighted', 'ensemble_all'],
                    help='Voting mechanism')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for soft voting methods')
parser.add_argument('--test_file', type=str, default='filtered_public_x_merged.csv', help='Test dataset file')
parser.add_argument('--output_file', type=str, default='prediction_results.csv', help='Output prediction file')
args = parser.parse_args()

args.output_file = args.output_file.replace('.csv', f'_{args.voting}.csv')
MODEL_DIR = 'models'

# [改動] 移除了 'gbdt_model.pkl'，改成 'catboost_model.cbm'
required_files = [
    'scaler.pkl',
    'feature_selector.pkl',
    'catboost_model.cbm',  # 這裡對應 cat_model.save_model("catboost_model.cbm")
    'xgb_model.pkl',
    'lgbm_model.pkl',
    'shallow_nn_model.pth',
    'deep_nn_model.pth'
]

for fn in required_files:
    path = os.path.join(MODEL_DIR, fn)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {path}")

# --- Feature Engineering (from 04_training.py) ---
def preprocess_test_features(X, scaler, feature_engineering=True):
    X_scaled = X.copy()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    X_scaled[numerical_features] = scaler.transform(X[numerical_features])

    if feature_engineering:
        price_feats = [c for c in numerical_features if 'price' in c.lower() or 'close' in c.lower()]
        vol_feats = [c for c in numerical_features if 'volume' in c.lower() or 'vol' in c.lower()]
        if price_feats and vol_feats:
            for p in price_feats[:2]:
                for v in vol_feats[:2]:
                    X_scaled[f"{p}_x_{v}"] = X_scaled[p] * X_scaled[v]
        groups = []
        for prefix in ['price', 'volume', 'ma', 'rsi']:
            grp = [c for c in numerical_features if prefix in c.lower()]
            if len(grp) >= 3:
                groups.append(grp)
        for i, grp in enumerate(groups):
            X_scaled[f'group_{i}_mean'] = X_scaled[grp].mean(axis=1)
            X_scaled[f'group_{i}_std']  = X_scaled[grp].std(axis=1)
            X_scaled[f'group_{i}_max']  = X_scaled[grp].max(axis=1)
            X_scaled[f'group_{i}_min']  = X_scaled[grp].min(axis=1)
    return X_scaled

# --- Define Neural Network Models ---
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

# --- Load Scaler, Selector and Models ---
print("Loading scaler, selector, and models...")
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'feature_selector.pkl'), 'rb') as f:
    selector = pickle.load(f)

# [改動] 改成 CatBoost 模型載入
cat_model = CatBoostClassifier()
cat_model.load_model(os.path.join(MODEL_DIR, 'catboost_model.cbm'))

# XGBoost
import joblib
xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_model.pkl'))

# LightGBM
lgbm_model = joblib.load(os.path.join(MODEL_DIR, 'lgbm_model.pkl'))

# Load PyTorch models
_dummy_dim = 10
shallow_nn_model = ShallowNN(_dummy_dim)
deep_nn_model = DeepNN(_dummy_dim)
shallow_nn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'shallow_nn_model.pth')))
deep_nn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'deep_nn_model.pth')))
shallow_nn_model.eval()
deep_nn_model.eval()

# [改動] 改變 weights keys 從 'gbdt' => 'catboost'
model_weights = {
    'catboost': 0.15,
    'xgboost': 0.25,
    'lightgbm': 0.20,
    'shallow_nn': 0.15,
    'deep_nn': 0.25
}
total_w = sum(model_weights.values())
for k in model_weights:
    model_weights[k] /= total_w

print(f"Using voting mechanism: {args.voting} with weights: {model_weights}")

# --- Process Test Data in Batches ---
all_ids = []
all_preds = []
all_model_probs = []

chunk_size = 10000
for idx, chunk in enumerate(pd.read_csv(args.test_file, chunksize=chunk_size)):
    print(f"Processing chunk {idx + 1}...")
    ids = chunk['ID'].values
    X_raw = chunk.drop('ID', axis=1)

    # Feature engineering and scaling
    X_scaled = preprocess_test_features(X_raw, scaler, feature_engineering=True)
    mask = selector.get_support()
    selected_cols = X_scaled.columns[mask]
    X_sel = X_scaled[selected_cols]

    # Predict with catboost
    cat_prob = cat_model.predict_proba(X_sel)[:, 1]  # [改動] 原本是 gbdt_prob
    cat_pred = (cat_prob > 0.5).astype(int)

    # Predict with xgboost
    xgb_prob = xgb_model.predict_proba(X_sel)[:, 1]
    xgb_pred = (xgb_prob > 0.5).astype(int)

    # Predict with lightgbm
    lgbm_prob = lgbm_model.predict_proba(X_sel)[:, 1]
    lgbm_pred = (lgbm_prob > 0.5).astype(int)

    # PyTorch model prediction
    X_tensor = torch.tensor(X_sel.values, dtype=torch.float32)
    input_dim = X_sel.shape[1]
    shallow_nn_model = ShallowNN(input_dim)
    shallow_nn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'shallow_nn_model.pth')))
    shallow_nn_model.eval()

    deep_nn_model = DeepNN(input_dim)
    deep_nn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'deep_nn_model.pth')))
    deep_nn_model.eval()

    with torch.no_grad():
        shallow_prob = shallow_nn_model(X_tensor).cpu().numpy().flatten()
        deep_prob = deep_nn_model(X_tensor).cpu().numpy().flatten()

    shallow_pred = (shallow_prob > 0.5).astype(int)
    deep_pred = (deep_prob > 0.5).astype(int)

    # Final Voting
    batch_preds = []
    for i in range(len(X_sel)):
        hard_votes = [cat_pred[i], xgb_pred[i], lgbm_pred[i], shallow_pred[i], deep_pred[i]]
        prob_votes = [cat_prob[i], xgb_prob[i], lgbm_prob[i], shallow_prob[i], deep_prob[i]]

        # [改動] sample_probs 也改成 'catboost' key
        sample_probs = {
            'catboost':   cat_prob[i],
            'xgboost':    xgb_prob[i],
            'lightgbm':   lgbm_prob[i],
            'shallow_nn': shallow_prob[i],
            'deep_nn':    deep_prob[i]
        }
        all_model_probs.append(sample_probs)

        if args.voting == 'hard_majority':
            # 5 個模型, 3/5 即過
            final = 1 if sum(hard_votes) >= 3 else 0
        elif args.voting == 'hard_weighted':
            # 須確保 zip 順序與 model_weights key 對應
            # 這裡用固定順序 list 來 zip
            model_order = ['catboost','xgboost','lightgbm','shallow_nn','deep_nn']
            wv = sum(h * model_weights[mkey] for h, mkey in zip(hard_votes, model_order))
            final = 1 if wv >= 0.5 else 0
        elif args.voting == 'soft_avg':
            ap = sum(prob_votes) / len(prob_votes)
            final = 1 if ap >= args.threshold else 0
        elif args.voting == 'soft_weighted':
            model_order = ['catboost','xgboost','lightgbm','shallow_nn','deep_nn']
            wp = sum(p * model_weights[mkey] for p, mkey in zip(prob_votes, model_order))
            final = 1 if wp >= args.threshold else 0
        else:  # ensemble_all
            # 如果所有模型都投 1 -> 1, 都投 0 -> 0, 否則以 soft_weighted
            if all(v == 1 for v in hard_votes):
                final = 1
            elif all(v == 0 for v in hard_votes):
                final = 0
            else:
                model_order = ['catboost','xgboost','lightgbm','shallow_nn','deep_nn']
                wp = sum(p * model_weights[mkey] for p, mkey in zip(prob_votes, model_order))
                final = 1 if wp >= args.threshold else 0

        batch_preds.append(final)

    all_ids.extend(ids)
    all_preds.extend(batch_preds)

# --- Save Results ---
res_df = pd.DataFrame({'ID': all_ids, 'Prediction': all_preds})
res_df.to_csv(args.output_file, index=False)
print(f"Predictions saved to {args.output_file}")

# Optional: Save detailed probabilities
prob_df = pd.DataFrame(all_model_probs)
prob_df['ID'] = all_ids
prob_df['Prediction'] = all_preds
prob_df.to_csv(f"detailed_model_predictions_{args.voting}.csv", index=False)

print(f"Total samples: {len(res_df)}, Predicted as 'Stock': {res_df['Prediction'].mean():.4f}")
