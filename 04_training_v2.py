import os
import gc
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

os.environ['JOBLIB_TEMP_FOLDER'] = os.path.abspath("tmp_joblib")
os.makedirs(os.environ['JOBLIB_TEMP_FOLDER'], exist_ok=True)

SMOTE_RATIO = 0.5

from sklearn.neighbors import NearestNeighbors

def apply_smote(X, y, ratio=SMOTE_RATIO, random_state=42):
    nn = NearestNeighbors(n_jobs=-1)
    sm = SMOTE(sampling_strategy=ratio, random_state=random_state, k_neighbors=nn)
    return sm.fit_resample(X, y)

def train_catboost(X_train, y_train, X_val, y_val):
    pos, neg = np.bincount(y_train)
    model = CatBoostClassifier(
        iterations=600,
        depth=8,
        learning_rate=0.03,
        class_weights=[1.0, 1.5*(neg/pos)],
        eval_metric='F1',
        early_stopping_rounds=30,
        task_type='GPU',
        verbose=False
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    gc.collect()
    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    pos, neg = np.bincount(y_train)
    model = XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.03,
        scale_pos_weight=1.5*(neg / pos),
        tree_method='gpu_hist' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'hist',
        n_jobs=-1,
        verbosity=0,
        eval_metric='aucpr'
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    gc.collect()
    return model

def train_lightgbm(X_train, y_train, X_val, y_val):
    model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        class_weight={0:1, 1:2},
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    gc.collect()
    return model

def find_best_threshold(model, X_val, y_val):
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

def evaluate_model(model, X_val, y_val, threshold):
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    return classification_report(y_val, y_pred, digits=4)

def main():
    df = pd.read_csv('training_4o_cleaned.csv')
    X = df.drop(columns=['飆股', 'ID'], errors='ignore')
    y = df['飆股']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_bal, y_bal = apply_smote(X_train, y_train)
    print(f"SMOTE applied: {len(X_bal)} samples (original: {len(X_train)})")

    print("Training models for anomaly detection (imbalance-aware)...")
    models = {
        'catboost': train_catboost(X_bal, y_bal, X_val, y_val),
        'xgboost': train_xgboost(X_bal, y_bal, X_val, y_val),
        'lightgbm': train_lightgbm(X_bal, y_bal, X_val, y_val)
    }

    print("\nEvaluation Results:")
    for name, model in models.items():
        best_threshold = find_best_threshold(model, X_val, y_val)
        print(f"\n{name} best threshold: {best_threshold:.4f}")
        report = evaluate_model(model, X_val, y_val, best_threshold)
        print(f"\n{name} classification report at best threshold:")
        print(report)

    os.makedirs('models', exist_ok=True)
    for name, model in models.items():
        path = f"models/{name}_anomaly_stock_modelv2.pkl" if name != 'catboost' else f"models/{name}_anomaly_stockv2_model.cbm"
        if name == 'catboost':
            model.save_model(path)
        else:
            joblib.dump(model, path)

    print("All models saved to ./models")

if __name__ == '__main__':
    main()
