import pandas as pd

def filter_large_csv(input_csv_path, output_csv_path, keywords):
    """
    根據字串列表篩選特徵名稱，並匯出篩選後的資料到新的 CSV。
    
    :param input_csv_path: 原始大資料 CSV 檔案路徑
    :param output_csv_path: 篩選後的 CSV 檔案路徑
    :param keywords: 字串列表，用於篩選特徵名稱
    """
    # 使用分塊處理大資料
    chunk_size = 10**5  # 每次讀取 100,000 行
    filtered_columns = None

    # 逐塊讀取 CSV
    for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size):
        # 如果是第一次處理，篩選出符合條件的欄位名稱
        if filtered_columns is None:
            filtered_columns = [col for col in chunk.columns if any(keyword in col for keyword in keywords)]
            if not filtered_columns:
                raise ValueError("沒有符合條件的特徵名稱。")

        # 篩選出符合條件的欄位
        filtered_chunk = chunk[filtered_columns]

        # 將篩選後的資料寫入新的 CSV（追加模式）
        filtered_chunk.to_csv(output_csv_path, mode='a', index=False, header=not pd.io.common.file_exists(output_csv_path))

if __name__ == "__main__":
    # 範例輸入
    input_csv = "E:\\Tbrain_stock_analysis\\38_Public_Test_Set_and_Submmision_Template_V2\\public_x.csv"  # 請替換為實際檔案路徑
    output_csv = "E:\\Tbrain_stock_analysis\\38_Public_Test_Set_and_Submmision_Template_V2\\public_x_onlySkill.csv"  # 請替換為輸出檔案路徑
    keywords_list = ["ID", "技術指標"]  # 請替換為實際字串列表

    filter_large_csv(input_csv, output_csv, keywords_list)