import pandas as pd
import os

# 讀取要保留的特徵列表
with open('E:\\Tbrain_stock_analysis\\correlation_thr07_3rd\\kept_features.txt', 'r', encoding='utf-8') as f:
    kept_features = [line.strip() for line in f if line.strip()]
    kept_features_set = set(kept_features)

# 先讀取CSV檔案的標頭，以確定哪些列需要保留
df_header = pd.read_csv('E:\\Tbrain_stock_analysis\\38_Public_Test_Set_and_Submmision_Template_V2\\public_x.csv', nrows=0)
all_columns = df_header.columns.tolist()

# 找出kept_features.txt中但不在CSV中的特徵
missing_features = [feature for feature in kept_features if feature not in all_columns]
if missing_features:
    print(f"警告：以下特徵在CSV檔案中不存在：{missing_features}")

# 確定要保留的特徵列表（交集）
valid_features = list(kept_features_set.intersection(set(all_columns)))

# 如果輸出檔案已經存在，先刪除它
output_file = 'filtered_public_x.csv'
if os.path.exists(output_file):
    os.remove(output_file)

# 使用分塊讀取處理大型CSV檔案
# 設定一個合理的分塊大小，根據你的機器記憶體調整
chunksize = 100000  # 每次讀取10萬行

# 初始化計數器
total_rows_processed = 0
header = True  # 第一個分塊需要寫入標頭

# 分塊處理檔案
for chunk in pd.read_csv('E:\\Tbrain_stock_analysis\\38_Public_Test_Set_and_Submmision_Template_V2\\public_x.csv', chunksize=chunksize, usecols=valid_features):
    # 將分塊寫入輸出檔案，只附加不覆蓋
    chunk.to_csv(output_file, mode='a', index=False, header=header)
    
    # 更新已處理的行數
    total_rows_processed += len(chunk)
    print(f"已處理 {total_rows_processed} 行...")
    
    # 後續分塊不再寫入標頭
    if header:
        header = False

print(f"原始檔案所有特徵數量: {len(all_columns)}")
print(f"kept_features.txt中的特徵數量: {len(kept_features)}")
print(f"篩選後保留的特徵數量: {len(valid_features)}")
print(f"總共處理了 {total_rows_processed} 行資料")
print(f"已將篩選後的資料匯出至 '{output_file}'")

# 輸出實際保留的特徵列表，以便參考
with open('actual_kept_features.txt', 'w', encoding='utf-8') as f:
    for feature in valid_features:
        f.write(f"{feature}\n")
print(f"已將實際保留的特徵列表寫入 'actual_kept_features.txt'")