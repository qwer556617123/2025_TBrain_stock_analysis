import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json

# ---------- 設定區 ----------
data_file = 'E:\\Tbrain_stock_analysis\\training.csv'  # CSV 檔案路徑
column_file = 'column_names.txt'  # 欄位名稱檔案
output_file = 'E:\\Tbrain_stock_analysis\\training_4o_merged.csv'  # 輸出檔案名稱
stats_file = '4o_feature_transform_stats.json'  # 儲存轉換參數
chunk_size = 50000  # 每批處理筆數

# ---------- 讀取欄位分類 ----------
with open(column_file, encoding='utf-8-sig') as f:
    all_columns = [line.strip().replace('\ufeff', '') for line in f if line.strip()]
all_columns = [col for col in all_columns if col != 'ID']

# 類別分類 regex 定義
import re
category_regex = {
    "外資券商": r"^外資券商_",
    "主力券商": r"^主力券商_",
    "官股券商": r"^官股券商_",
    "個股券商分點籌碼分析": r"^個股券商分點籌碼分析_",
    "個股券商分點區域分析": r"^個股券商分點區域分析_",
    "個股主力買賣超統計": r"^個股主力買賣超統計_",
    "日外資": r"^日外資_",
    "日自營": r"^日自營_",
    "日投信": r"^日投信_",
    "技術指標": r"^技術指標_",
    "月營收": r"^月營收_",
    "季IFRS財報": r"^季IFRS財報_",
    "買超分點": r"^買超第\d+名分點",
    "賣超分點": r"^賣超第\d+名分點",
    "飆股": r"^飆股$",
    "類別型": r"(券商代號|等級|信評|異動原因)"
}

category_groups = {key: [] for key in category_regex}
category_groups["其他"] = []

for col in all_columns:
    matched = False
    for cat, regex in category_regex.items():
        if re.search(regex, col):
            category_groups[cat].append(col)
            matched = True
            break
    if not matched:
        category_groups["其他"].append(col)

categorical_groups = ["類別型"]
group_stats = {}  # 統計資訊紀錄用

# ---------- 特徵縮減處理 ----------
def reduce_numeric_features(df, columns, prefix):
    result = {}
    data = df[columns]
    result[f'{prefix}_mean'] = data.mean(axis=1)
    result[f'{prefix}_std'] = data.std(axis=1)
    result[f'{prefix}_max'] = data.max(axis=1)
    result[f'{prefix}_min'] = data.min(axis=1)

    group_stats[prefix] = {
        'mean': data.mean().mean(),
        'std': data.std().mean(),
        'max': data.max().max(),
        'min': data.min().min()
    }

    return result

# ---------- 資料批次處理 ----------
reader = pd.read_csv(data_file, usecols=all_columns, chunksize=chunk_size)
results = []

for chunk in tqdm(reader, desc="Processing chunks"):
    temp = pd.DataFrame(index=chunk.index)

    for group, cols in category_groups.items():
        if not cols or group == '飆股':
            continue
        if group in categorical_groups:
            for col in cols:
                if col in chunk.columns:
                    temp[col + '_encoded'] = chunk[col].astype('category').cat.codes
        else:
            reduced = reduce_numeric_features(chunk, cols, group)
            temp = pd.concat([temp, pd.DataFrame(reduced)], axis=1)

    if '飆股' in chunk.columns:
        temp['飆股'] = chunk['飆股']

    results.append(temp)

# ---------- 合併與輸出 ----------
final_df = pd.concat(results, ignore_index=True)
final_df.to_csv(output_file, index=False)

# 儲存統計資訊
with open(stats_file, 'w', encoding='utf-8') as f:
    json.dump(group_stats, f, ensure_ascii=False, indent=2)

print(f"輸出完成: {output_file}")
print(f"統計參數儲存於: {stats_file}")
