import os
import gc
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb

# 設定暫存目錄，提升處理效率
os.environ['JOBLIB_TEMP_FOLDER'] = os.path.abspath("tmp_joblib")
os.makedirs(os.environ['JOBLIB_TEMP_FOLDER'], exist_ok=True)

# 調整smote比例，避免資料爆炸
SMOTE_RATIO = 0.2

# 使用SMOTE進行輕度過採樣以避免過擬合並降低記憶體需求
from sklearn.neighbors import NearestNeighbors

def apply_smote(X, y, ratio=0.3, random_state=42):
    nn = NearestNeighbors(n_jobs=-1)
    sm = SMOTE(sampling_strategy=ratio, random_state=random_state, k_neighbors=nn)
    return sm.fit_resample(X, y)


# 模型訓練函數
def train_catboost(X_train, y_train, X_val, y_val):
    pos, neg = np.bincount(y_train)
    model = CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.05,
        class_weights=[1.0, neg/pos],
        eval_metric='F1',
        early_stopping_rounds=30,
        task_type='GPU',  # 如果確定有GPU可用，直接設定GPU；若無，設定CPU即可
        verbose=False
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    gc.collect()
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    pos, neg = np.bincount(y_train)
    model = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=neg / pos,
        tree_method='gpu_hist' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'hist',
        n_jobs=-1,
        verbosity=0,
        eval_metric='aucpr'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    gc.collect()
    return model

def train_lightgbm(X_train, y_train, X_val, y_val):
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30)])
    gc.collect()
    return model


def evaluate_model(model, X_val, y_val, threshold=0.5):
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    return classification_report(y_val, y_pred, digits=4)


def main():
    df = pd.read_csv('training_4o_cleaned.csv')
    X = df.drop(columns=['飆股', 'ID'], errors='ignore')
    y = df['飆股']

    # 因資料量大，只取部份作訓練 (若資料超過百萬筆以上)
    if len(df) > 1_000_000:
        df_sample = df.sample(n=500_000, random_state=42)
        X, y = df_sample.drop(columns=['飆股', 'ID']), df_sample['飆股']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # SMOTE 輕度平衡
    X_bal, y_bal = apply_smote(X_train, y_train)

    print("Training models for anomaly detection (imbalance-aware)...")
    models = {
        'catboost': train_catboost(X_bal, y_bal, X_val, y_val),
        'xgboost': train_xgboost(X_bal, y_bal, X_val, y_val),
        'lightgbm': train_lightgbm(X_bal, y_bal, X_val, y_val)
    }

    print("\nEvaluation Results:")
    for name, model in models.items():
        print(f"\n{name} classification report:")
        report = evaluate_model(model, X_val, y_val)
        print(report)

    # 儲存模型
    os.makedirs('models', exist_ok=True)
    for name, model in models.items():
        path = f"models/{name}_anomaly_stock_model.pkl" if name != 'catboost' else f"models/{name}_anomaly_stock_model.cbm"
        if name == 'catboost':
            model.save_model(path)
        else:
            joblib.dump(model, path)

    print("All models saved to ./models")


if __name__ == '__main__':
    main()
