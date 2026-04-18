import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# ---------- 設定 ----------
test_file = 'E:\\Tbrain_stock_analysis\\38_Private_Test_Set_and_Submission_Template_V2\\private_x.csv'  # 測試集資料檔案
column_file = 'E:\\Tbrain_stock_analysis\\try\\column_names_test.txt'  # 欄位名稱
stats_file = 'E:\\Tbrain_stock_analysis\\try\\4o_feature_transform_stats.json'  # 訓練集統計資訊
output_file = 'E:\\Tbrain_stock_analysis\\test_4o_merged_private.csv'  # 測試集輸出檔
chunk_size = 50000

# ---------- 讀取欄位與分類 ----------
with open(column_file, encoding='utf-8-sig') as f:
    all_columns = [line.strip().replace('\ufeff', '') for line in f if line.strip()]
all_columns = [col for col in all_columns if col != 'ID']
all_columns_with_id = ['ID'] + all_columns  # 加入 ID 欄位

with open(stats_file, 'r', encoding='utf-8') as f:
    group_stats = json.load(f)

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
    "類別型": r"(券商代號|等級|信評|異動原因)",
}

category_groups = {key: [] for key in category_regex}
category_groups['其他'] = []

for col in all_columns:
    matched = False
    for cat, regex in category_regex.items():
        if re.search(regex, col):
            category_groups[cat].append(col)
            matched = True
            break
    if not matched:
        category_groups['其他'].append(col)

categorical_groups = ['類別型']

# ---------- 特徵壓縮（使用訓練統計） ----------
def apply_stats_transform(df, columns, prefix):
    result = {}
    data = df[columns]
    result[f'{prefix}_mean'] = data.mean(axis=1)
    result[f'{prefix}_std'] = data.std(axis=1)
    result[f'{prefix}_max'] = data.max(axis=1)
    result[f'{prefix}_min'] = data.min(axis=1)
    return result

reader = pd.read_csv(test_file, usecols=all_columns_with_id, chunksize=chunk_size)
results = []

for chunk in tqdm(reader, desc="Processing test chunks"):
    temp = pd.DataFrame(index=chunk.index)

    # 保留 ID 欄位
    temp['ID'] = chunk['ID']

    for group, cols in category_groups.items():
        if not cols:
            continue
        if group in categorical_groups:
            for col in cols:
                if col in chunk.columns:
                    temp[col + '_encoded'] = chunk[col].astype('category').cat.codes
        elif group in group_stats:
            reduced = apply_stats_transform(chunk, cols, group)
            temp = pd.concat([temp, pd.DataFrame(reduced)], axis=1)

    results.append(temp)

final_df = pd.concat(results, ignore_index=True)
final_df.to_csv(output_file, index=False)
print(f"測試集特徵壓縮完成: {output_file}")
