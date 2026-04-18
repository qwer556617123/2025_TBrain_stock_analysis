import pandas as pd
import numpy as np

# ---------- 設定 ----------
input_file = 'E:\\Tbrain_stock_analysis\\training_4o_cleaned.csv'  # 壓縮後特徵檔案
output_prefix = 'E:\\Tbrain_stock_analysis\\sample_pool\\balanced_set_'  # 輸出檔名前綴
label_column = '飆股'  # 目標欄位
n_sets = 100  # 抽樣次數

# ---------- 載入資料 ----------
df = pd.read_csv(input_file)

df_positive = df[df[label_column] == 1]
df_negative = df[df[label_column] == 0]

n_positive = len(df_positive)

# ---------- 隨機 downsample 三組 ----------
for i in range(1, n_sets + 1):
    df_negative_sampled = df_negative.sample(n=n_positive, replace=False, random_state=42 + i)
    df_balanced = pd.concat([df_positive, df_negative_sampled]).sample(frac=1, random_state=99 + i).reset_index(drop=True)
    df_balanced.to_csv(f'{output_prefix}{i}.csv', index=False)
    print(f"已儲存平衡資料集: {output_prefix}{i}.csv，總筆數: {len(df_balanced)}")
