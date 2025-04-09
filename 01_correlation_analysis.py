import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import random
from tqdm import tqdm
import os
import gc

# 檔案設定
# file_path = 'training.csv'
file_path = "E:/Tbrain_stock_analysis/correlation_thr07_2nd/training_filtered.csv"
total_rows = 200864
# total_cols = 10214
total_cols = 2911  # 這是經過篩選後的列數
sample_size = 1000
num_samples = 5
target_column = "飆股"  # 目標變數名稱

def sample_correlation_analysis(file_path, sample_size, sample_num, result_dir="correlation_results"):
    """
    對指定CSV檔案進行抽樣相關性分析，先進行正規化再計算相關係數
    排除目標變數"飆股"不參與相關性分析
    """
    global target_column  # 添加這行來引用全域變數
    
    print(f"開始第 {sample_num} 次抽樣分析...")
    
    # 創建結果目錄
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # 隨機選擇行的索引
    random_indices = random.sample(range(total_rows), sample_size)
    
    # 使用 skiprows 來只讀取選定的行
    # 首先獲取要跳過的行
    skip_rows = [i+1 for i in range(total_rows) if i not in random_indices]  # +1 因為跳過標題行
    
    # 由於列數很多，我們將其分批處理
    chunk_size = 1000  # 每批處理的列數
    all_correlations = {}
    
    # 獲取所有列名
    # 為了節省記憶體，我們只讀取標題行
    all_columns = pd.read_csv(file_path, nrows=0).columns.tolist()
    
    # 確認目標變數是否存在
    if target_column not in all_columns:
        print(f"警告：目標變數 '{target_column}' 不在資料集中!")
        target_index = -1  # 使用最後一列作為目標變數
        target_column = all_columns[-1]
        print(f"使用最後一列 '{target_column}' 作為目標變數")
    else:
        target_index = all_columns.index(target_column)
    
    # 排除目標變數，只對特徵進行相關性分析
    feature_columns = [col for col in all_columns if col != target_column]
    print(f"排除目標變數 '{target_column}'，剩餘 {len(feature_columns)} 個特徵用於相關性分析")
    
    # 根據列數來決定批次數
    num_chunks = (len(feature_columns) + chunk_size - 1) // chunk_size
    
    # 進行主要的相關性分析
    for i in tqdm(range(num_chunks), desc="分析列批次"):
        start_col = i * chunk_size
        end_col = min((i + 1) * chunk_size, len(feature_columns))
        
        # 獲取當前批次的列
        current_cols = feature_columns[start_col:end_col]
        
        # 只讀取我們需要的列
        df_chunk = pd.read_csv(file_path, 
                               skiprows=skip_rows, 
                               usecols=current_cols)
        
        # 正規化數據
        # 1. 移除非數值列，避免錯誤
        numeric_cols = df_chunk.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) == 0:
            continue  # 如果沒有數值列，跳過此批次
        
        # 2. 只處理數值列
        df_numeric = df_chunk[numeric_cols]
        
        # 3. 正規化處理
        scaler = StandardScaler()
        try:
            df_normalized = pd.DataFrame(
                scaler.fit_transform(df_numeric),
                columns=numeric_cols,
                index=df_numeric.index
            )
        except Exception as e:
            print(f"正規化錯誤: {e}")
            # 處理可能的錯誤（如有常數列）
            # 對於常數列，StandardScaler會出錯，我們需要單獨處理
            df_normalized = pd.DataFrame(index=df_numeric.index)
            for col in numeric_cols:
                if df_numeric[col].std() > 0:
                    # 非常數列，正常正規化
                    df_normalized[col] = (df_numeric[col] - df_numeric[col].mean()) / df_numeric[col].std()
                else:
                    # 常數列，直接設為0
                    df_normalized[col] = 0
            
        # 計算相關性矩陣
        corr_matrix = df_normalized.corr(method='pearson', numeric_only=True)
        
        # 儲存到全域結果
        for col1 in corr_matrix.columns:
            if col1 not in all_correlations:
                all_correlations[col1] = {}
            for col2 in corr_matrix.columns:
                all_correlations[col1][col2] = corr_matrix.loc[col1, col2]
        
        # 清除記憶體
        del df_chunk, df_numeric, df_normalized, corr_matrix
        gc.collect()
    
    # 轉換結果為DataFrame
    corr_df = pd.DataFrame(all_correlations)
    
    # 儲存完整相關性矩陣
    corr_df.to_csv(f"{result_dir}/sample_{sample_num}_full_correlation.csv")
    
    # 找出最高相關性的特徵對
    high_corr = []
    
    for col1 in corr_df.columns:
        for col2 in corr_df.index:
            if col1 < col2:  # 避免重複
                corr_value = corr_df.loc[col2, col1]
                if abs(corr_value) > 0.7:  # 高相關性閾值
                    high_corr.append((col1, col2, corr_value))
    
    # 根據相關性絕對值排序
    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # 儲存高相關性結果
    high_corr_df = pd.DataFrame(high_corr, columns=['Feature1', 'Feature2', 'Correlation'])
    high_corr_df.to_csv(f"{result_dir}/sample_{sample_num}_high_correlations.csv", index=False)
    
    # 返回高相關性對，以便後續合併分析
    return high_corr_df

def aggregate_correlation_results(num_samples, result_dir="correlation_results"):
    """
    合併多次抽樣的相關性分析結果
    """
    print("合併多次抽樣結果...")
    
    # 讀取所有高相關性結果
    all_high_corr = []
    
    for i in range(1, num_samples + 1):
        file_path = f"{result_dir}/sample_{i}_high_correlations.csv"
        if os.path.exists(file_path):
            sample_high_corr = pd.read_csv(file_path)
            sample_high_corr['Sample'] = i
            all_high_corr.append(sample_high_corr)
    
    if not all_high_corr:
        print("沒有找到有效的相關性分析結果。")
        return
    
    # 合併所有結果
    combined_high_corr = pd.concat(all_high_corr, ignore_index=True)
    
    # 獲取每個特徵對在不同樣本中的出現次數
    feature_pairs = combined_high_corr.apply(lambda row: (min(row['Feature1'], row['Feature2']), 
                                                         max(row['Feature1'], row['Feature2'])), axis=1)
    combined_high_corr['FeaturePair'] = feature_pairs
    
    # 計算每個特徵對的出現次數和平均相關性
    pair_stats = combined_high_corr.groupby('FeaturePair').agg(
        Count=('Sample', 'count'),
        AvgCorrelation=('Correlation', 'mean'),
        StdCorrelation=('Correlation', 'std'),
        MinCorrelation=('Correlation', 'min'),
        MaxCorrelation=('Correlation', 'max')
    ).reset_index()
    
    # 解包FeaturePair元組
    pair_stats[['Feature1', 'Feature2']] = pd.DataFrame(pair_stats['FeaturePair'].tolist(), index=pair_stats.index)
    pair_stats.drop('FeaturePair', axis=1, inplace=True)
    
    # 根據出現次數和平均相關性排序
    pair_stats.sort_values(by=['Count', 'AvgCorrelation'], ascending=[False, False], inplace=True)
    
    # 儲存合併結果
    pair_stats.to_csv(f"{result_dir}/combined_correlation_analysis.csv", index=False)
    
    # 輸出最穩定的高相關性特徵對（在所有樣本中都出現的）
    stable_pairs = pair_stats[pair_stats['Count'] == num_samples].head(20)
    print("\n在所有樣本中都出現的前20個高相關性特徵對:")
    print(stable_pairs[['Feature1', 'Feature2', 'AvgCorrelation', 'StdCorrelation']])
    
    # 繪製熱圖
    plt.figure(figsize=(12, 10))
    
    # 設定支援中文的字體
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Bitstream Vera Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
    
    # 選出前30個最穩定的特徵對
    top_pairs = pair_stats[pair_stats['Count'] >= num_samples//2].head(30)
    
    # 創建熱圖數據
    unique_features = set()
    for _, row in top_pairs.iterrows():
        unique_features.add(row['Feature1'])
        unique_features.add(row['Feature2'])
    
    unique_features = sorted(list(unique_features))
    heatmap_df = pd.DataFrame(index=unique_features, columns=unique_features)
    
    # 填充熱圖數據
    for _, row in top_pairs.iterrows():
        heatmap_df.loc[row['Feature1'], row['Feature2']] = row['AvgCorrelation']
        heatmap_df.loc[row['Feature2'], row['Feature1']] = row['AvgCorrelation']
    
    # 對角線填充1.0
    for feature in unique_features:
        heatmap_df.loc[feature, feature] = 1.0
    
    # 繪製熱圖
    sns.heatmap(heatmap_df.astype(float), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Top Feature Correlations Across Samples')
    plt.tight_layout()
    plt.savefig(f"{result_dir}/top_correlations_heatmap.png", dpi=300)
    
    return pair_stats

def main():
    """
    主函數：進行多次抽樣相關性分析並合併結果
    """
    print(f"開始對 {file_path} 進行 {num_samples} 次抽樣分析，每次 {sample_size} 筆資料...")
    print(f"排除目標變數 '{target_column}' 不參與相關性分析")
    print("先進行正規化處理再計算相關係數...")
    
    # 進行多次抽樣分析
    for i in range(1, num_samples + 1):
        sample_correlation_analysis(file_path, sample_size, i)
    
    # 合併分析結果
    aggregate_correlation_results(num_samples)
    
    print("\n相關性分析完成！結果已保存在 correlation_results 目錄中。")

if __name__ == "__main__":
    main()