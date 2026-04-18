import pandas as pd

# 讀取 CSV 檔案
input_file = r"E:\\Tbrain_stock_analysis\\training_4o_merged.csv"
output_file = r"E:\\Tbrain_stock_analysis\\training_4o_cleaned.csv"

# 讀取資料
df = pd.read_csv(input_file)
print(f"讀取資料完成，共 {len(df)} 筆資料")

# 移除包含空值或缺失值的資料列
df_cleaned = df.dropna()
print(f"移除空值後，共 {len(df_cleaned)} 筆資料")

# 將清理後的資料儲存為新的 CSV 檔案
df_cleaned.to_csv(output_file, index=False)

print(f"清理完成，已將結果儲存至: {output_file}")