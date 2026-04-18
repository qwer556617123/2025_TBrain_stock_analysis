import pandas as pd

# 讀取 CSV 檔案
csv_file_path = "E:\\Tbrain_stock_analysis\\training_onlySkill.csv"  # 替換為你的 CSV 檔案路徑
output_txt_path = "column_name_onlySkill.txt"  # 輸出檔案名稱

# 使用 pandas 讀取 CSV
df = pd.read_csv(csv_file_path)

# 獲取所有欄位名稱
columns = df.columns.tolist()

# 將欄位名稱寫入 .txt 檔案
with open(output_txt_path, "w", encoding="utf-8") as f:
    for column in columns:
        f.write(column + "\n")

print(f"特徵欄位名稱已輸出至 {output_txt_path}")