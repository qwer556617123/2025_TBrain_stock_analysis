#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import joblib
import argparse
import numpy as np
import pandas as pd
import tempfile
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer

def load_model(name):
    path = f"models/{name}_anomaly_stock_modelv7.pkl" if name != 'catboost' else f"models/{name}_anomaly_stock_modelv7.cbm"
    if name == 'catboost':
        model = CatBoostClassifier()
        model.load_model(path)
    else:
        model = joblib.load(path)
    return model

def voting_predict(models, X, strategy='soft_avg', weights=None):
    probs = [m.predict_proba(X) for m in models]
    if strategy == 'soft_avg':
        avg_prob = np.mean([p[:, 1] for p in probs], axis=0)
        return (avg_prob >= 0.5).astype(int)
    elif strategy == 'soft_weighted':
        assert weights is not None
        weighted_prob = np.average([p[:, 1] for p in probs], axis=0, weights=weights)
        return (weighted_prob >= 0.5).astype(int)
    elif strategy == 'hard_majority':
        votes = [np.argmax(p, axis=1) for p in probs]
        return np.round(np.mean(votes, axis=0)).astype(int)
    elif strategy == 'hard_weighted':
        assert weights is not None
        votes = [np.argmax(p, axis=1) for p in probs]
        weighted = np.zeros((len(votes[0]), 2))
        for i, v in enumerate(votes):
            for j, pred in enumerate(v):
                weighted[j, pred] += weights[i]
        return np.argmax(weighted, axis=1)
    else:
        raise ValueError("Unsupported voting strategy")

def hybrid_predict(stacking_proba, soft_proba, alpha=0.85, threshold=0.75):
    blended = alpha * stacking_proba + (1 - alpha) * soft_proba
    return (blended >= threshold).astype(int)

def main(test_csv, output_dir, strategy):
    df = pd.read_csv(test_csv)
    X = df.drop(columns=['ID'], errors='ignore')
    ids = df['ID'] if 'ID' in df.columns else pd.Series(range(len(df)))

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    model_names = ['catboost', 'xgboost', 'lightgbm', 'easyensemble', 'balancedbagging', 'stacking']
    models = [load_model(name) for name in model_names]

    print("Calculating validation weights for weighted strategies...")
    val_df = pd.read_csv('training_4o_cleaned.csv')
    X_val = val_df.drop(columns=['飆股', 'ID'], errors='ignore')
    y_val = val_df['飆股']
    X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

    with tempfile.TemporaryDirectory(dir="E:\\Tbrain_stock_analysis\\tmp_joblib") as temp_dir:
        os.environ['JOBLIB_TEMP_FOLDER'] = temp_dir
        weights = [f1_score(y_val, m.predict(X_val)) for m in models]

    os.makedirs(output_dir, exist_ok=True)

    print("Running hybrid ensemble prediction...")
    all_test_probas = [m.predict_proba(X)[:, 1] for m in models]
    model_proba_dict = dict(zip(model_names, all_test_probas))
    stacking_proba = model_proba_dict['stacking']
    soft_model_names = [name for name in model_names if name != 'stacking']
    soft_weights = [w for i, w in enumerate(weights) if model_names[i] != 'stacking']
    soft_probs = [model_proba_dict[name] for name in soft_model_names]
    soft_weighted_proba = np.average(soft_probs, axis=0, weights=soft_weights)

    hybrid_preds = hybrid_predict(stacking_proba, soft_weighted_proba, alpha=0.85, threshold=0.75)

    output_df = pd.DataFrame({
        'ID': ids,
        '飆股': hybrid_preds
    })
    out_path = os.path.join(output_dir, f'final_prediction_hybrid_20250422v7.csv')
    output_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    if '飆股' in df.columns:
        f1 = f1_score(df['飆股'], hybrid_preds)
        prec = precision_score(df['飆股'], hybrid_preds)
        recall = recall_score(df['飆股'], hybrid_preds)
        with open(os.path.join(output_dir, 'hybrid_report.txt'), 'w') as f:
            f.write(f"Hybrid Ensemble\nF1 Score: {f1:.4f}\nPrecision: {prec:.4f}\nRecall: {recall:.4f}\n")
        print(f"\nHybrid F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}")

    if strategy != 'ensemble_all':
        preds = voting_predict(models, X, strategy=strategy, weights=weights)
        output_df = pd.DataFrame({
            'ID': ids,
            '飆股': preds
        })
        out_path = os.path.join(output_dir, f'final_prediction_{strategy}_20250422v7.csv')
        output_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        if '飆股' in df.columns:
            f1 = f1_score(df['飆股'], preds)
            prec = precision_score(df['飆股'], preds)
            recall = recall_score(df['飆股'], preds)
            with open(os.path.join(output_dir, f'{strategy}_report.txt'), 'w') as f:
                f.write(f"Strategy: {strategy}\nF1 Score: {f1:.4f}\nPrecision: {prec:.4f}\nRecall: {recall:.4f}\n")
            print(f"\n{strategy} F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, help='Path to the test CSV file', default="E:\\Tbrain_stock_analysis\\test_4o_merged_all.csv")
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save prediction outputs')
    parser.add_argument('--strategy', type=str, default='ensemble_all',
                        choices=['soft_avg', 'soft_weighted', 'hard_majority', 'hard_weighted', 'ensemble_all'],
                        help='Voting strategy to use')
    args = parser.parse_args()

    main(args.test_csv, args.output_dir, args.strategy)
