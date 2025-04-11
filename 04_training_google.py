import os
ascii_tmp = r"C:\joblib_temp"
os.makedirs(ascii_tmp, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = ascii_tmp
os.environ["TMP"] = ascii_tmp
os.environ["TEMP"] = ascii_tmp

import pandas as pd
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import warnings
import gc # <-- MEMORY RELEASE: Import garbage collector

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 設定隨機種子，確保結果可複現
np.random.seed(42)
torch.manual_seed(42)
if device == torch.device("cuda"):
    torch.cuda.manual_seed(42) # Seed for CUDA as well

# 建立結果目錄
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)



# 資料預處理與特徵工程函數
def preprocess_features(X_train, X_test, feature_engineering=True):
    # 找出數值型特徵
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()

    # 標準化數值型特徵
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

    # 特徵工程（如果啟用）
    if feature_engineering:
        # ... (feature engineering logic remains the same) ...
        # 1. 添加一些基本的特徵交互項（根據金融領域知識選擇）
        price_features = [col for col in numerical_features if 'price' in col.lower() or 'close' in col.lower()]
        volume_features = [col for col in numerical_features if 'volume' in col.lower() or 'vol' in col.lower()]

        if price_features and volume_features:
            for p_feat in price_features[:2]:
                for v_feat in volume_features[:2]:
                    X_train_scaled[f"{p_feat}_x_{v_feat}"] = X_train_scaled[p_feat] * X_train_scaled[v_feat]
                    X_test_scaled[f"{p_feat}_x_{v_feat}"] = X_test_scaled[p_feat] * X_test_scaled[v_feat]

        # 2. 添加一些統計特徵（假設原始特徵中有時間序列特徵）
        groups = []
        for prefix in ['price', 'volume', 'ma', 'rsi']:
            group = [col for col in numerical_features if prefix in col.lower()]
            if len(group) >= 3:
                groups.append(group)

        for i, group in enumerate(groups):
            if len(group) >= 3:
                X_train_scaled[f'group_{i}_mean'] = X_train_scaled[group].mean(axis=1)
                X_train_scaled[f'group_{i}_std'] = X_train_scaled[group].std(axis=1)
                X_train_scaled[f'group_{i}_max'] = X_train_scaled[group].max(axis=1)
                X_train_scaled[f'group_{i}_min'] = X_train_scaled[group].min(axis=1)

                X_test_scaled[f'group_{i}_mean'] = X_test_scaled[group].mean(axis=1)
                X_test_scaled[f'group_{i}_std'] = X_test_scaled[group].std(axis=1)
                X_test_scaled[f'group_{i}_max'] = X_test_scaled[group].max(axis=1)
                X_test_scaled[f'group_{i}_min'] = X_test_scaled[group].min(axis=1)

    return X_train_scaled, X_test_scaled, scaler

# 特徵選擇函數
def select_features(X_train, X_test, y_train, method='model_based', n_features=None):
    if method == 'model_based':
        selector = SelectFromModel(
            XGBClassifier(random_state=42, **({'tree_method': 'gpu_hist'} if device == torch.device("cuda") else {})), # Use GPU if available
            threshold='median'
        )
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()]

    elif method == 'rfe':
        n_features = n_features or max(30, X_train.shape[1] // 3)
        selector = RFE(
            estimator=RandomForestClassifier(random_state=42, n_jobs=4), # Use all cores
            n_features_to_select=n_features,
            step=0.1
        )
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()]

    elif method == 'pca':
        n_components = n_features or min(50, X_train.shape[1] // 2)
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        X_train = pd.DataFrame(X_train_pca, index=X_train.index, columns=[f'PC_{i+1}' for i in range(n_components)]) # Add column names
        X_test = pd.DataFrame(X_test_pca, index=X_test.index, columns=[f'PC_{i+1}' for i in range(n_components)]) # Add column names
        print(f"特徵選擇 (PCA) 後的特徵數量: {n_components}")
        return X_train, X_test, pca

    else: # No selection
        return X_train, X_test, None

    # For model_based and rfe
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    print(f"特徵選擇 ({method}) 後的特徵數量: {len(selected_features)}")
    return X_train_selected, X_test_selected, selector

# 處理類別不平衡
def handle_imbalance(X_train, y_train, method='smote'):
    if method == 'smote':
        print("Applying SMOTE...")
        sm = SMOTE(random_state=42, n_jobs=-1) # Use all cores
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print(f"SMOTE後的訓練資料大小: {X_train_res.shape}")
        print(f"SMOTE後類別分佈:\n{y_train_res.value_counts()}")
        return X_train_res, y_train_res
    elif method == 'class_weight':
        weights = {0: 1, 1: class_counts[0]/class_counts[1]}
        print(f"Using class weights: {weights}")
        return X_train, y_train, weights # Need to return weights
    else:
        return X_train, y_train

# 超參數調整
def tune_hyperparameters(X_train, y_train, model_type):
    class_weight_ratio = class_counts[0] / class_counts[1]
    if model_type == 'catboost':
        weights = [1, class_weight_ratio]
        model = CatBoostClassifier(
            task_type='GPU' if device == torch.device("cuda") else 'CPU',
            devices='0' if device == torch.device("cuda") else None,
            random_state=42,
            # class_weights=weights, # Often better to use scale_pos_weight or SMOTE
            scale_pos_weight=class_weight_ratio, # Alternative for imbalance
            verbose=0,
            eval_metric='F1'
        )
        param_distributions = {
            'iterations': [100, 200, 300, 400], # Expanded range slightly
            'learning_rate': [0.03, 0.05, 0.1, 0.15], # Adjusted range
            'depth': [4, 6, 8, 10], # Expanded range slightly
            'l2_leaf_reg': [1, 3, 5, 7], # Expanded range slightly
            'border_count': [32, 64, 128, 254] # 254 is max for CPU, 128 for GPU often good
        }
        if device == torch.device("cuda"):
             param_distributions['border_count'] = [32, 64, 128] # Limit for GPU
        n_iter_search = 25 # Increase iterations

    elif model_type == 'xgboost':
        model = XGBClassifier(
            tree_method='gpu_hist' if device == torch.device("cuda") else 'hist', # Auto selects hist for CPU
            predictor='gpu_predictor' if device == torch.device("cuda") else None,
            gpu_id=0 if device == torch.device("cuda") else None,
            random_state=42,
            scale_pos_weight=class_weight_ratio, # Use scale_pos_weight for imbalance
            eval_metric='aucpr', # Area under PR curve often good for imbalance
            use_label_encoder=False # Suppress warning
        )
        param_distributions = {
            'n_estimators': [100, 200, 300, 400], # Expanded range slightly
            'learning_rate': [0.03, 0.05, 0.1, 0.15], # Adjusted range
            'max_depth': [3, 5, 7, 9], # Expanded range slightly
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2], # Added gamma for regularization
            'subsample': [0.7, 0.8, 0.9], # Adjusted range
            'colsample_bytree': [0.7, 0.8, 0.9] # Adjusted range
        }
        n_iter_search = 30 # Increase iterations

    elif model_type == 'lightgbm':
        model = LGBMClassifier(
            device='gpu' if device == torch.device("cuda") else 'cpu',
            gpu_platform_id=0 if device == torch.device("cuda") else None,
            gpu_device_id=0 if device == torch.device("cuda") else None,
            random_state=42,
            scale_pos_weight=class_weight_ratio, # Use scale_pos_weight for imbalance
            metric='f1', # Optimize F1 directly if desired
            n_jobs=-1 # Use all cores
        )
        param_distributions = {
            'n_estimators': [100, 200, 300, 400], # Expanded range slightly
            'learning_rate': [0.03, 0.05, 0.1, 0.15], # Adjusted range
            'max_depth': [3, 5, 7, -1], # -1 means no limit
            'num_leaves': [15, 31, 63, 127], # Adjusted range
            'subsample': [0.7, 0.8, 0.9], # Adjusted range
            'colsample_bytree': [0.7, 0.8, 0.9] # Adjusted range
        }
        n_iter_search = 30 # Increase iterations

    else:
        raise ValueError(f"不支援的模型類型: {model_type}")

    # Use StratifiedKFold for CV in RandomizedSearch for better handling of imbalance
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        model, param_distributions,
        n_iter=n_iter_search, # Use adjusted number of iterations
        cv=cv_strategy,       # Use StratifiedKFold
        scoring='f1',         # Keep F1 as the primary optimization metric
        random_state=42,
        n_jobs=2 # Limit jobs for RandomizedSearch if resource contention is an issue, -1 otherwise
    )

    random_search.fit(X_train, y_train)
    print(f"{model_type} 最佳參數: {random_search.best_params_}")
    print(f"{model_type} 最佳F1分數 (CV): {random_search.best_score_:.4f}")

    best_estimator = random_search.best_estimator_

    # <-- MEMORY RELEASE: Clean up RandomizedSearchCV object and intermediate data if possible
    del random_search
    gc.collect()

    return best_estimator

# 評估函數
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n--- Evaluating {model_name} ---")
    if isinstance(model, nn.Module): # PyTorch Model
        model.eval()
        # Ensure X_test is a tensor and on the correct device
        if isinstance(X_test, pd.DataFrame):
             X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
        else: # Assume it's already a tensor (e.g., from DataLoader)
             X_test_tensor = X_test.to(device)

        y_prob_list = []
        y_pred_list = []
        # Evaluate in batches to conserve memory if X_test is large
        temp_dataset = TensorDataset(X_test_tensor)
        temp_loader = DataLoader(temp_dataset, batch_size=1024, shuffle=False)

        with torch.no_grad():
            for [inputs] in temp_loader: # Dataloader returns list
                outputs = model(inputs)
                y_prob_list.append(outputs.cpu())
                y_pred_list.append((outputs > 0.5).float().cpu())

        y_prob = torch.cat(y_prob_list).numpy().flatten()
        y_pred = torch.cat(y_pred_list).numpy().flatten().astype(int)

        # <-- MEMORY RELEASE: Clean up tensors used for evaluation
        del X_test_tensor, temp_dataset, temp_loader, y_prob_list, y_pred_list, inputs, outputs
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    else: # Scikit-learn style model
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0) # Handle zero division
    recall = recall_score(y_test, y_pred, zero_division=0) # Handle zero division
    f1 = f1_score(y_test, y_pred, zero_division=0) # Handle zero division
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError: # Handle cases where only one class is present in y_test or y_pred
        auc = 0.0
        print("Warning: AUC calculation failed (likely due to only one class in test labels or predictions). Setting AUC to 0.")


    results = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

    print(f"{model_name} 模型評估結果:")
    print(f"  準確率: {accuracy:.4f}")
    print(f"  精確率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分數: {f1:.4f}")
    print(f"  AUC值:  {auc:.4f}")

    return results

# 繪製學習曲線
def plot_learning_curve(estimator, X, y, model_name):
    print(f"Plotting learning curve for {model_name}...")
    try:
        # Use StratifiedKFold for CV in learning curve as well
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv_strategy, scoring='f1',
            train_sizes=np.linspace(0.1, 1.0, 8), # Fewer points for speed
            random_state=42, n_jobs=-1, # Use all cores
            exploit_incremental_learning=False # Safer default
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.title(f"Learning Curve ({model_name}) - F1 Score")
        plt.xlabel("Training examples")
        plt.ylabel("F1 Score")
        plt.grid(True) # Enable grid

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        plt.ylim(0.0, 1.05) # Set y-axis limits

        plt.savefig(f"results/learning_curve_{model_name}.png")
        plt.close() # Close the plot to free memory
        print(f"Learning curve saved for {model_name}.")

        # <-- MEMORY RELEASE: Explicitly delete large arrays from learning curve
        del train_sizes, train_scores, test_scores, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std
        gc.collect()

    except Exception as e:
        print(f"Could not plot learning curve for {model_name}: {e}")
        # Clean up plot if it failed midway
        plt.close()


# --- PyTorch Specific Classes ---
# Need TensorDataset for NN evaluation helper
from torch.utils.data import TensorDataset

class StockDataset(Dataset):
    def __init__(self, features, labels):
        # Ensure features and labels are tensors when dataset is created
        self.features = torch.tensor(features.values, dtype=torch.float32) if isinstance(features, pd.DataFrame) else features.clone().detach().float()
        self.labels = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1) if isinstance(labels, pd.Series) else labels.clone().detach().float().unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class ShallowNN(nn.Module):
    def __init__(self, input_dim):
        super(ShallowNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # Add dropout
        self.fc2 = nn.Linear(64, 1)
        # Sigmoid is often combined with BCELoss using BCEWithLogitsLoss for numerical stability
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        # out = self.sigmoid(out) # Removed sigmoid
        return out

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
        self.dropout3 = nn.Dropout(0.2) # Added dropout
        self.fc4 = nn.Linear(32, 1)
        # self.sigmoid = nn.Sigmoid() # Removed sigmoid

    def forward(self, x):
        # Layer 1
        out = self.fc1(x)
        if out.size(0) > 1: out = self.bn1(out) # Apply BN only if batch size > 1
        out = self.relu(out)
        out = self.dropout1(out)
        # Layer 2
        out = self.fc2(out)
        if out.size(0) > 1: out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        # Layer 3
        out = self.fc3(out)
        if out.size(0) > 1: out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout3(out) # Apply dropout
        # Output Layer
        out = self.fc4(out)
        # out = self.sigmoid(out) # Removed sigmoid
        return out

# --- Main Function ---
def main():
    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"訓練集大小: {X_train.shape}, 測試集大小: {X_test.shape}")

    # 資料預處理與特徵工程
    X_train_processed, X_test_processed, scaler = preprocess_features(X_train, X_test, feature_engineering=True)
    print(f"預處理後特徵數量: {X_train_processed.shape[1]}")

    # <-- MEMORY RELEASE: Raw X_train, X_test no longer needed
    del X_train
    del X_test
    gc.collect()
    print("Raw train/test splits released.")

    # 特徵選擇
    # Using 'model_based' as per original script. Change method if needed.
    X_train_selected, X_test_selected, selector = select_features(X_train_processed, X_test_processed, y_train, method='model_based')

    # <-- MEMORY RELEASE: Processed (but not selected) data no longer needed
    del X_train_processed
    del X_test_processed
    gc.collect()
    print("Processed (pre-selection) train/test data released.")

    # 處理類別不平衡
    # Using 'smote' as per original script. Change method if needed.
    # Note: if using 'class_weight', the function returns weights, adjust accordingly
    X_train_balanced, y_train_balanced = handle_imbalance(X_train_selected, y_train, method='smote')

    # It's often better to keep the original y_train for evaluation/CV consistency if not using SMOTE everywhere
    # If using SMOTE, y_train_balanced is the target for subsequent training
    # Let's keep y_train for stacking's meta-model fitting later

    # 儲存預處理器
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    if selector is not None and not isinstance(selector, PCA): # Save selector if it exists and is not PCA (PCA is saved differently if needed)
        with open('models/feature_selector.pkl', 'wb') as f:
            pickle.dump(selector, f)
    elif isinstance(selector, PCA):
         with open('models/pca_transformer.pkl', 'wb') as f:
            pickle.dump(selector, f) # Save PCA object if used
    print("預處理器/選擇器已儲存至models目錄")

    # --- Train Traditional ML Models ---
    results = []
    trained_models = {} # Dictionary to store trained models for ensembling

    # 1. CatBoost GPU
    print("\n=== 訓練 CatBoost 模型 ===")
    cat_model = tune_hyperparameters(X_train_balanced, y_train_balanced, 'catboost')
    trained_models['CatBoost'] = cat_model
    cat_results = evaluate_model(cat_model, X_test_selected, y_test, 'CatBoost')
    results.append(cat_results)
    cat_model.save_model('models/catboost_model.cbm')
    plot_learning_curve(cat_model, X_train_balanced, y_train_balanced, 'CatBoost') # Plot LC on balanced data

    # 2. XGBoost
    print("\n=== 訓練 XGBoost 模型 ===")
    xgb_model = tune_hyperparameters(X_train_balanced, y_train_balanced, 'xgboost')
    trained_models['XGBoost'] = xgb_model
    xgb_results = evaluate_model(xgb_model, X_test_selected, y_test, 'XGBoost')
    results.append(xgb_results)
    with open('models/xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    plot_learning_curve(xgb_model, X_train_balanced, y_train_balanced, 'XGBoost')

    # 3. LightGBM
    print("\n=== 訓練 LightGBM 模型 ===")
    lgbm_model = tune_hyperparameters(X_train_balanced, y_train_balanced, 'lightgbm')
    trained_models['LightGBM'] = lgbm_model
    lgbm_results = evaluate_model(lgbm_model, X_test_selected, y_test, 'LightGBM')
    results.append(lgbm_results)
    with open('models/lgbm_model.pkl', 'wb') as f:
        pickle.dump(lgbm_model, f)
    plot_learning_curve(lgbm_model, X_train_balanced, y_train_balanced, 'LightGBM')

    # --- Prepare Data for PyTorch ---
    # Keep X_test_selected (Pandas DF) for evaluation, create tensors as needed
    input_dim = X_train_balanced.shape[1]

    # Create datasets (can consume memory, manage carefully)
    # Pass pandas objects directly, convert to tensor inside dataset or loader
    train_dataset = StockDataset(X_train_balanced, y_train_balanced)
    # test_dataset = StockDataset(X_test_selected, y_test) # Create test dataset if needed for NN eval loop

    # <-- MEMORY RELEASE: Balanced dataframes potentially released after dataset creation
    # Be cautious if DataFrames are needed elsewhere. Here, they are used by StockDataset.
    # del X_train_balanced # Keep for now, needed by dataset
    # del y_train_balanced # Keep for now, needed by dataset

    batch_size = 128 # Adjusted batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True if device==torch.device('cuda') else False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, num_workers=2, pin_memory=True if device==torch.device('cuda') else False)

    # --- Train Neural Networks ---

    # Common Training Loop Function for NNs
    def train_nn_model(model, train_loader, X_test_eval, y_test_eval, criterion, optimizer, scheduler, model_name, epochs, patience):
        print(f"\n=== 訓練 {model_name} 模型 ===")
        model.to(device)
        best_val_f1 = -1
        epochs_no_improve = 0
        train_losses = []
        val_f1_scores = []
        best_model_path = f'models/{model_name}_best.pth'

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels) # BCEWithLogitsLoss expects raw logits

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)

            # Validation step
            model.eval()
            val_probs_list = []
            # Use X_test_selected directly if small enough, otherwise use a loader
            X_test_tensor_eval = torch.tensor(X_test_eval.values, dtype=torch.float32).to(device)
            with torch.no_grad():
                 # Simple evaluation if test set fits in memory
                 outputs_val = model(X_test_tensor_eval)
                 val_probs = torch.sigmoid(outputs_val).cpu().numpy().flatten() # Apply sigmoid here for evaluation

            val_preds = (val_probs > 0.5).astype(int)
            val_f1 = f1_score(y_test_eval, val_preds, zero_division=0)
            val_f1_scores.append(val_f1)

            scheduler.step(val_f1) # Step scheduler based on validation F1

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Val F1: {val_f1:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

             # <-- MEMORY RELEASE: Intermediate validation tensors
            del X_test_tensor_eval, outputs_val, val_probs, val_preds
            if device == torch.device("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> New best validation F1: {best_val_f1:.4f}. Model saved.")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Finished training {model_name}. Best validation F1: {best_val_f1:.4f}")
        # Load best model
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)

        # Plotting training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title(f'Training Loss ({model_name})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(val_f1_scores)
        plt.title(f'Validation F1 Score ({model_name})')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'results/{model_name}_training_history.png')
        plt.close()

        return model # Return the loaded best model


    # 4. Shallow NN
    shallow_nn_model = ShallowNN(input_dim)
    # Use BCEWithLogitsLoss for stability, combines Sigmoid + BCELoss
    criterion_shallow = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_counts[0]/class_counts[1]).to(device)) # Add pos_weight for imbalance
    optimizer_shallow = optim.AdamW(shallow_nn_model.parameters(), lr=0.001, weight_decay=1e-4) # AdamW often preferred
    scheduler_shallow = optim.lr_scheduler.ReduceLROnPlateau(optimizer_shallow, mode='max', factor=0.2, patience=5, verbose=True)
    shallow_nn_model = train_nn_model(shallow_nn_model, train_loader, X_test_selected, y_test,
                                      criterion_shallow, optimizer_shallow, scheduler_shallow,
                                      'ShallowNN', epochs=50, patience=10)
    trained_models['ShallowNN'] = shallow_nn_model
    # Evaluate using the helper function (which now handles tensors and batches)
    shallow_nn_results = evaluate_model(shallow_nn_model, X_test_selected, y_test, 'ShallowNN')
    results.append(shallow_nn_results)


    # 5. Deep NN
    deep_nn_model = DeepNN(input_dim)
    criterion_deep = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_counts[0]/class_counts[1]).to(device)) # Add pos_weight
    optimizer_deep = optim.AdamW(deep_nn_model.parameters(), lr=0.0005, weight_decay=1e-4) # Lower LR for deeper model
    scheduler_deep = optim.lr_scheduler.ReduceLROnPlateau(optimizer_deep, mode='max', factor=0.2, patience=7, verbose=True) # More patience
    deep_nn_model = train_nn_model(deep_nn_model, train_loader, X_test_selected, y_test,
                                   criterion_deep, optimizer_deep, scheduler_deep,
                                   'DeepNN', epochs=100, patience=15) # Longer training, more patience
    trained_models['DeepNN'] = deep_nn_model
    deep_nn_results = evaluate_model(deep_nn_model, X_test_selected, y_test, 'DeepNN')
    results.append(deep_nn_results)

    # <-- MEMORY RELEASE: NN Training related objects no longer needed
    print("Releasing NN training data loaders and datasets...")
    del train_loader
    # del test_loader # We didn't create a persistent test loader
    del train_dataset
    # del test_dataset # We didn't create a persistent test dataset
    del X_train_balanced # Release balanced data
    del y_train_balanced # Release balanced labels
    gc.collect()
    if device == torch.device("cuda"):
        print("Emptying CUDA cache after NN training...")
        torch.cuda.empty_cache()


    # --- Ensemble Models ---

    # 6. Weighted Ensemble
    print("\n=== 建立加權集成模型 ===")
    ensemble_weights = { # Weights can be optimized, these are examples
        'CatBoost': 0.20,
        'XGBoost':  0.20,
        'LightGBM': 0.20,
        'ShallowNN':0.15,
        'DeepNN':   0.25,
    }
    total_weight = sum(ensemble_weights.values())
    if not np.isclose(total_weight, 1.0):
        print(f"Warning: Ensemble weights do not sum to 1 (sum={total_weight}). Normalizing.")
        ensemble_weights = {k: v / total_weight for k, v in ensemble_weights.items()}

    # Get probabilities (ensure NNs apply sigmoid)
    ensemble_probs_dict = {}
    for name, model in trained_models.items():
        print(f"Getting predictions from {name} for ensemble...")
        if isinstance(model, nn.Module):
            model.eval()
            X_test_tensor_ens = torch.tensor(X_test_selected.values, dtype=torch.float32).to(device)
            with torch.no_grad():
                 outputs_ens = model(X_test_tensor_ens)
                 probs = torch.sigmoid(outputs_ens).cpu().numpy().flatten() # Sigmoid here
            del X_test_tensor_ens, outputs_ens # Release tensor memory
            if device == torch.device("cuda"): torch.cuda.empty_cache()
        else:
            probs = model.predict_proba(X_test_selected)[:, 1]
        ensemble_probs_dict[name] = probs
        gc.collect() # Collect garbage after each model prediction

    # Calculate weighted average
    ensemble_probs_w = np.zeros_like(ensemble_probs_dict['CatBoost'])
    for name, weight in ensemble_weights.items():
        ensemble_probs_w += weight * ensemble_probs_dict[name]

    # Evaluate
    ensemble_preds_w = (ensemble_probs_w > 0.5).astype(int)
    ensemble_accuracy_w = accuracy_score(y_test, ensemble_preds_w)
    ensemble_precision_w = precision_score(y_test, ensemble_preds_w, zero_division=0)
    ensemble_recall_w = recall_score(y_test, ensemble_preds_w, zero_division=0)
    ensemble_f1_w = f1_score(y_test, ensemble_preds_w, zero_division=0)
    ensemble_auc_w = roc_auc_score(y_test, ensemble_probs_w)

    ensemble_results_w = {
        'model': 'WeightedEnsemble',
        'accuracy': ensemble_accuracy_w, 'precision': ensemble_precision_w,
        'recall': ensemble_recall_w, 'f1_score': ensemble_f1_w, 'auc': ensemble_auc_w
    }
    results.append(ensemble_results_w)
    print(f"加權集成模型評估結果:")
    print(f"  準確率: {ensemble_accuracy_w:.4f}, 精確率: {ensemble_precision_w:.4f}, 召回率: {ensemble_recall_w:.4f}, F1分數: {ensemble_f1_w:.4f}, AUC值: {ensemble_auc_w:.4f}")

    # Save config
    ensemble_config = {'weights': ensemble_weights, 'threshold': 0.5}
    with open('models/ensemble_config.pkl', 'wb') as f:
        pickle.dump(ensemble_config, f)

    # <-- MEMORY RELEASE: Individual probabilities for weighted ensemble
    del ensemble_probs_dict
    del ensemble_probs_w
    del ensemble_preds_w
    gc.collect()
    print("Released intermediate weighted ensemble probabilities.")


    # 7. Stacking Ensemble
    print("\n=== 建立堆疊集成模型 (Stacking) ===")

    # Define base models for stacking (excluding NNs due to complexity in CV)
    # If NNs were included, they'd need a separate training loop inside get_stacking_features
    stacking_base_models = {
        'CatBoost': trained_models['CatBoost'], # Use already tuned models
        'XGBoost':  trained_models['XGBoost'],
        'LightGBM': trained_models['LightGBM']
        # NNs excluded here
    }

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=123) # Different seed for CV split

    # Create Out-of-Fold (OOF) predictions for training meta-model
    # Use the original X_train_selected and y_train before SMOTE for OOF generation
    # This avoids data leakage from SMOTE into the validation folds
    print("Generating Out-of-Fold predictions for Stacking meta-model...")
    oof_preds = np.zeros((len(X_train_selected), len(stacking_base_models)))

    # Create predictions on the test set using models trained on full data
    print("Generating Test predictions for Stacking meta-model...")
    test_preds_meta = np.zeros((len(X_test_selected), len(stacking_base_models)))

    for i, (name, model) in enumerate(stacking_base_models.items()):
        print(f"  Processing {name}...")
        # Generate OOF predictions
        oof_preds_col = np.zeros(len(X_train_selected))
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_selected, y_train)):
            X_fold_train, X_fold_val = X_train_selected.iloc[train_idx], X_train_selected.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]

            # Clone the model to avoid retraining the original tuned one
            model_clone = pickle.loads(pickle.dumps(model)) # Simple clone for sklearn models

            # Fit on fold train data (consider applying SMOTE *within* the fold if needed, but simpler not to)
            # model_clone.fit(X_fold_train, y_fold_train) # Fit on original fold data

            # OR fit on SMOTE'd fold data (more complex)
            X_fold_train_bal, y_fold_train_bal = handle_imbalance(X_fold_train, y_fold_train, method='smote')
            model_clone.fit(X_fold_train_bal, y_fold_train_bal)
            del X_fold_train_bal, y_fold_train_bal # Clean up SMOTE'd fold data
            gc.collect()

            oof_preds_col[val_idx] = model_clone.predict_proba(X_fold_val)[:, 1]
            print(f"    Fold {fold+1}/{n_folds} OOF predictions generated.")
            del X_fold_train, X_fold_val, y_fold_train, model_clone # Clean up fold variables
            gc.collect()

        oof_preds[:, i] = oof_preds_col
        print(f"  OOF predictions generated for {name}.")

        # Generate Test predictions (using the main model trained earlier on balanced data)
        test_preds_meta[:, i] = model.predict_proba(X_test_selected)[:, 1]
        print(f"  Test predictions generated for {name}.")
        gc.collect() # Collect after processing each base model


    # Train Meta-Model (Using Logistic Regression or LightGBM is common)
    print("Training Stacking meta-model...")
    # meta_model = LogisticRegression(random_state=42, C=0.1, class_weight='balanced')
    meta_model = LGBMClassifier(random_state=42, n_estimators=150, learning_rate=0.05, num_leaves=15, metric='f1', n_jobs=4)

    # Fit meta-model on OOF predictions and original y_train
    meta_model.fit(oof_preds, y_train)
    print("Meta-model trained.")

    # Evaluate Stacking Model
    stacking_probs = meta_model.predict_proba(test_preds_meta)[:, 1]
    stacking_preds = meta_model.predict(test_preds_meta)

    stacking_accuracy = accuracy_score(y_test, stacking_preds)
    stacking_precision = precision_score(y_test, stacking_preds, zero_division=0)
    stacking_recall = recall_score(y_test, stacking_preds, zero_division=0)
    stacking_f1 = f1_score(y_test, stacking_preds, zero_division=0)
    stacking_auc = roc_auc_score(y_test, stacking_probs)

    stacking_results = {
        'model': 'StackingEnsemble',
        'accuracy': stacking_accuracy, 'precision': stacking_precision,
        'recall': stacking_recall, 'f1_score': stacking_f1, 'auc': stacking_auc
    }
    results.append(stacking_results)
    trained_models['StackingMeta'] = meta_model # Store meta-model if needed

    print(f"堆疊集成模型評估結果:")
    print(f"  準確率: {stacking_accuracy:.4f}, 精確率: {stacking_precision:.4f}, 召回率: {stacking_recall:.4f}, F1分數: {stacking_f1:.4f}, AUC值: {stacking_auc:.4f}")

    # Save meta-model
    with open('models/stacking_meta_model.pkl', 'wb') as f:
        pickle.dump(meta_model, f)

    # <-- MEMORY RELEASE: Stacking intermediate arrays
    del oof_preds
    del test_preds_meta
    del stacking_probs
    del stacking_preds
    gc.collect()
    print("Released intermediate stacking arrays (OOF, test predictions).")

    # --- Final Results Processing ---
    print("\n=== Final Results ===")
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('model')
    results_df.sort_values(by='f1_score', ascending=False, inplace=True) # Sort by F1 score
    print("模型評估結果比較:")
    print(results_df)
    results_df.to_csv('results/model_comparison.csv')

    # Plot comparison
    try:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        num_models = len(results_df)
        plt.figure(figsize=(max(15, num_models * 1.5), 10)) # Adjust figure size based on model count

        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i+1) # Adjust subplot layout if needed
            bars = plt.bar(results_df.index, results_df[metric])
            plt.title(f'{metric.capitalize()} Comparison')
            plt.xticks(rotation=60, ha='right') # Rotate labels more for readability
            plt.ylabel(metric.capitalize())
            plt.ylim(0, 1.05) # Ensure y-axis covers 0 to 1
            # Add value labels on bars
            plt.bar_label(bars, fmt='%.3f', padding=3)
            plt.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid

        plt.tight_layout() # Adjust layout to prevent overlap
        plt.savefig('results/model_metrics_comparison.png')
        plt.close() # Close plot to free memory
        print("Model comparison plot saved.")
    except Exception as e:
        print(f"Could not generate comparison plot: {e}")
        plt.close()


    # Select and report best model based on F1 score
    best_model_name = results_df['f1_score'].idxmax()
    best_f1 = results_df.loc[best_model_name, 'f1_score']
    print(f"\n最佳模型 (基於 F1 分數): {best_model_name} (F1 = {best_f1:.4f})")

    # Save best model details
    with open('results/best_model.txt', 'w') as f:
        f.write(f"Best Model (based on F1 Score): {best_model_name}\n")
        f.write("--- Metrics ---\n")
        for metric in metrics:
             f.write(f"{metric.capitalize()}: {results_df.loc[best_model_name, metric]:.4f}\n")
        f.write("\n--- All Model Results ---\n")
        results_df.to_string(f) # Write full DataFrame to file


    # <-- MEMORY RELEASE: Final cleanup of models before exiting main
    print("\nReleasing trained model objects...")
    del X_train_selected, X_test_selected, y_train, y_test # Release final data splits
    gc.collect()
    if device == torch.device("cuda"):
        print("Emptying CUDA cache at the end.")
        torch.cuda.empty_cache()

    print("\n=== 訓練與評估流程完成 ===")
    print(f"所有模型、預處理器、結果已保存至 models/ 和 results/ 目錄")
    print("記憶體已嘗試釋放.")


if __name__ == "__main__":
    # 載入資料
    try:
        df = pd.read_csv('training_filtered_merged.csv')
        print(f"成功載入資料，資料大小: {df.shape}")
    except FileNotFoundError:
        print("找不到訓練資料檔案，請確認檔案路徑。")
        exit()

    # 分離特徵與目標變數
    X = df.drop('飆股', axis=1)
    y = df['飆股']

    # 檢查類別平衡
    class_counts = y.value_counts()
    print(f"類別分佈: \n{class_counts}")
    print(f"類別不平衡比例: 1:{class_counts[0]/class_counts[1]:.2f}")

    # 移除 'ID' 欄位
    if 'ID' in X.columns:
        X = X.drop('ID', axis=1)

    # <-- MEMORY RELEASE: Original df no longer needed after splitting
    del df
    gc.collect()
    print("Original DataFrame released.")

    # Call main
    main() # main 函數會使用全域的 X 和 y

    # <-- MEMORY RELEASE: Clean up global X, y *after* main completes <--- 在這裡刪除
    print("\nReleasing global X and y after main execution...")
    try:
        del X
        del y
        gc.collect()
        print("Global X and y released.")
    except NameError:
        print("Global X and y were already released or not defined.")