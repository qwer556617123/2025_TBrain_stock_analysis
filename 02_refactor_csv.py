import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc

def filter_correlated_features(
    original_csv_path='training.csv',
    correlation_result_path='correlation_results/combined_correlation_analysis.csv',
    output_csv_path='training_filtered.csv',
    target_column="飆股",  # 目標變數名稱
    correlation_threshold=0.7,
    occurrence_threshold=3,  # 在至少3次抽樣中出現的特徵對會被考慮
    chunk_size=5000  # 每次處理的行數
):
    """
    根據相關性分析結果過濾高相關性特徵，生成新的CSV檔案
    保留目標變數"飆股"不進行過濾
    
    Args:
        original_csv_path: 原始CSV檔案路徑
        correlation_result_path: 相關性分析結果檔案路徑
        output_csv_path: 輸出CSV檔案路徑
        target_column: 目標變數名稱，此列不會被過濾
        correlation_threshold: 相關性閾值，絕對值高於此值的特徵對被視為高相關
        occurrence_threshold: 出現頻率閾值，在至少這麼多次抽樣中出現的特徵對才會被考慮
        chunk_size: 每次處理的行數
    """
    print(f"開始過濾高相關性特徵...")
    
    # 1. 讀取相關性分析結果
    print("讀取相關性分析結果...")
    if not os.path.exists(correlation_result_path):
        raise FileNotFoundError(f"找不到相關性分析結果檔案: {correlation_result_path}")
    
    corr_results = pd.read_csv(correlation_result_path)
    
    # 2. 找出高相關性特徵對
    print("識別高相關性特徵...")
    # 只選擇相關性超過閾值且出現頻率超過指定次數的特徵對
    high_corr_pairs = corr_results[
        (abs(corr_results['AvgCorrelation']) >= correlation_threshold) & 
        (corr_results['Count'] >= occurrence_threshold)
    ]
    
    print(f"發現 {len(high_corr_pairs)} 對高相關性特徵對")
    
    # 3. 為每對高相關特徵，決定要保留哪一個
    # 策略：我們保留在高相關性對中出現較少的特徵（作為更獨立的特徵）
    features_to_remove = set()
    feature_count = {}
    
    # 計算每個特徵在高相關對中出現的次數
    for _, row in high_corr_pairs.iterrows():
        f1, f2 = row['Feature1'], row['Feature2']
        
        if f1 not in feature_count:
            feature_count[f1] = 0
        if f2 not in feature_count:
            feature_count[f2] = 0
            
        feature_count[f1] += 1
        feature_count[f2] += 1
    
    # 為每對高相關特徵，移除出現次數較多的那個
    for _, row in high_corr_pairs.iterrows():
        f1, f2 = row['Feature1'], row['Feature2']
        
        if feature_count[f1] > feature_count[f2]:
            features_to_remove.add(f1)
        else:
            features_to_remove.add(f2)
    
    print(f"將移除 {len(features_to_remove)} 個高相關性特徵")
    
    # 4. 讀取原始CSV的標題，確定要保留的列
    print("分析原始CSV檔案結構...")
    
    # 只讀取標題行以獲取所有列名
    all_columns = pd.read_csv(original_csv_path, nrows=0).columns.tolist()
    print(f"原始CSV有 {len(all_columns)} 個特徵")
    
    # 確認目標變數是否存在
    if target_column not in all_columns:
        print(f"警告：目標變數 '{target_column}' 不在資料集中!")
        target_column = all_columns[-1]  # 使用最後一列作為目標變數
        print(f"使用最後一列 '{target_column}' 作為目標變數")
    
    # 確保目標變數不被移除
    if target_column in features_to_remove:
        features_to_remove.remove(target_column)
        print(f"保留目標變數 '{target_column}'，確保它不被過濾")
    
    # 確定要保留的列
    columns_to_keep = [col for col in all_columns if col not in features_to_remove]
    
    # 確保目標變數在列表中
    if target_column not in columns_to_keep:
        columns_to_keep.append(target_column)
    
    print(f"將保留 {len(columns_to_keep)} 個特徵，包括目標變數 '{target_column}'")
    
    # 將要保留的列清單保存下來（便於後續使用和參考）
    with open('kept_features.txt', 'w', encoding='utf-8') as f:
        for col in columns_to_keep:
            f.write(f"{col}\n")
    
    # 5. 分批讀取原始CSV，只保留需要的列，然後寫入新CSV
    print("開始分批處理原始CSV檔案...")
    
    # 獲取檔案大小（用於進度顯示）
    file_size = os.path.getsize(original_csv_path)
    
    # 創建新檔案的標題
    first_chunk = True
    
    # 初始化進度顯示
    print("處理進度:")
    pbar = tqdm(total=file_size, unit='B', unit_scale=True)
    
    # 分批讀取和寫入
    for chunk in pd.read_csv(original_csv_path, chunksize=chunk_size, usecols=columns_to_keep):
        # 更新進度條
        pbar.update(chunk.memory_usage(deep=True).sum())
        
        # 寫入模式：第一個塊用'w'模式，後續用'a'模式
        mode = 'w' if first_chunk else 'a'
        # 只有第一個塊需要寫入標題
        header = first_chunk
        
        # 保存過濾後的塊
        chunk.to_csv(output_csv_path, mode=mode, index=False, header=header)
        
        # 標記已處理第一個塊
        first_chunk = False
        
        # 清理記憶體
        del chunk
        gc.collect()
    
    pbar.close()
    
    print(f"\n處理完成！過濾後的CSV已保存為: {output_csv_path}")
    print(f"已移除 {len(features_to_remove)} 個高相關性特徵，保留 {len(columns_to_keep)} 個特徵")
    
    # 提供一些統計資訊
    print("\n特徵數量統計：")
    print(f"原始特徵數: {len(all_columns)}")
    print(f"移除特徵數: {len(features_to_remove)}")
    print(f"保留特徵數: {len(columns_to_keep)}")
    print(f"特徵降維比例: {len(features_to_remove)/len(all_columns)*100:.2f}%")
    print(f"目標變數 '{target_column}' 已保留")
    
    # 保存移除的特徵清單（便於參考）
    with open('removed_features.txt', 'w', encoding='utf-8') as f:
        for col in sorted(features_to_remove):
            f.write(f"{col}\n")
    
    return columns_to_keep, features_to_remove

if __name__ == "__main__":
    # 主執行程式
    filter_correlated_features(
        # original_csv_path='training.csv',                  # 原始大型CSV檔案
        original_csv_path="E:\\Tbrain_stock_analysis\\correlation_thr07_2nd\\training_filtered.csv",
        correlation_result_path='correlation_results_3rd/combined_correlation_analysis.csv',  # 相關性分析結果
        output_csv_path='training_filtered.csv',           # 輸出檔案名稱
        target_column="飆股",                              # 目標變數名稱
        correlation_threshold=0.7,                         # 相關性閾值
        occurrence_threshold=3,                            # 出現頻率閾值 (在至少3次抽樣中出現)
        chunk_size=5000                                    # 每次處理的行數
    )