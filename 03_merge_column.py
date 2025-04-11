import pandas as pd
import numpy as np
import time
import gc # Garbage Collector

# --- Configuration ---
input_csv_path = 'filtered_public_x.csv'
output_csv_path = 'filtered_public_x_merged.csv'
features_list_path = 'actual_kept_features.txt' # Path to your features list file

# --- Read Available Features ---
try:
    with open(features_list_path, 'r', encoding='utf-8') as f:
        # Read lines, strip whitespace/newlines, ignore empty lines
        available_columns = set(line.strip() for line in f if line.strip())
    print(f"Read {len(available_columns)} unique column names from {features_list_path}")
    if not available_columns:
        raise ValueError("Features list file is empty or could not be read.")
except FileNotFoundError:
    print(f"Error: Features list file not found at {features_list_path}")
    exit()
except Exception as e:
    print(f"Error reading features list file: {e}")
    exit()

# --- Define Column Groups for Merging (VALIDATED against available_columns) ---

def get_valid_columns(potential_cols, available_set):
    """Filters a list of potential columns, returning only those in the available set."""
    return [col for col in potential_cols if col in available_set]

# Keep these essential columns (ensure they are in the features file too)
essential_cols = get_valid_columns(['ID', '飆股'], available_columns)
if len(essential_cols) != 2:
    print("Warning: 'ID' or '飆股' not found in features list file.")
    # Decide how to handle this - exit or proceed without them?
    # For now, we'll try to proceed but ID/Target might be missing.
    essential_cols = [col for col in ['ID', '飆股'] if col in available_columns]


# Example Group 1: Foreign Broker Activity (Lagged Averages - e.g., 5-day)
foreign_broker_cols = {
    '進出_5d_Avg': get_valid_columns([f'外資券商_前{i}天分點進出' for i in range(1, 6)], available_columns),
    '買賣力_5d_Avg': get_valid_columns([f'外資券商_前{i}天分點買賣力' for i in range(1, 6)], available_columns),
    '吃貨比_5d_Avg': get_valid_columns([f'外資券商_前{i}天分點吃貨比(%)' for i in range(1, 6)], available_columns),
    '出貨比_5d_Avg': get_valid_columns([f'外資券商_前{i}天分點出貨比(%)' for i in range(1, 6)], available_columns),
    '成交力_Avg': get_valid_columns(['外資券商_分點成交力(%)','外資券商_前8天分點成交力(%)', '外資券商_前11天分點成交力(%)', '外資券商_前19天分點成交力(%)'], available_columns)
}
# Add day 0 columns if they exist
if '外資券商_分點進出' in available_columns:
     if '進出_5d_Avg' in foreign_broker_cols: foreign_broker_cols['進出_5d_Avg'].append('外資券商_分點進出')
# ... add similar checks for other day 0 metrics ...
if '外資券商_前1天分點出貨比(%)' in available_columns: # Check one from the error msg
     if '出貨比_5d_Avg' in foreign_broker_cols and '外資券商_前1天分點出貨比(%)' not in foreign_broker_cols['出貨比_5d_Avg']:
           print("Manually adding 外資券商_前1天分點出貨比(%) as it was in error list") # This shouldn't be needed if range(1,6) worked, but as a safeguard.
           foreign_broker_cols['出貨比_5d_Avg'].append('外資券商_前1天分點出貨比(%)')


# Example Group 2: Main Broker Activity (Lagged Averages - e.g., 10-day)
main_broker_potential_cols = [f'主力券商_前{i}天分點成交力(%)' for i in range(1, 11)]
if '主力券商_分點成交力(%)' in available_columns: main_broker_potential_cols.append('主力券商_分點成交力(%)')
main_broker_cols = {
    '成交力_10d_Avg': get_valid_columns(main_broker_potential_cols, available_columns)
}

# Example Group 3: Institutional Daily Totals
institutional_buy_vol_cols = get_valid_columns(['日外資_外資自營商買張', '日自營_自營商買張(自行買賣)', '日自營_自營商買張(避險)', '日投信_投信買張'], available_columns)
institutional_sell_vol_cols = get_valid_columns(['日外資_外資自營商賣張', '日自營_自營商賣張(自行買賣)', '日自營_自營商賣張(避險)', '日投信_投信賣張'], available_columns) # Added 自行買賣 sell
institutional_net_vol_cols = get_valid_columns(['日外資_外資自營商買賣超', '日自營_自營商買賣超(自行買賣)', '日自營_自營商買賣超(避險)', '日投信_投信買賣超'], available_columns)
institutional_holding_ratio_cols = get_valid_columns(['日外資_外資持有比率(%)', '日自營_自營商持股比率(%)', '日投信_投信持股比率(%)'], available_columns)

# Example Group 4: Technical Indicators (Ratios/Differences)
tech_indicators_volatility_cols = get_valid_columns(['技術指標_年化波動度(250D)', '技術指標_年化波動度(21D)'], available_columns)
tech_indicators_beta_cols = get_valid_columns(['技術指標_Beta係數(250D)', '技術指標_Beta係數(21D)'], available_columns)

# Example Group 5: Monthly Revenue Growth
revenue_growth_cols = get_valid_columns(['月營收_單月合併營收年成長(%)', '月營收_單月合併營收月變動(%)', '月營收_累計合併營收成長(%)'], available_columns)

# Example Group 6: IFRS Ratios (Grouped Averages - *Normalization Recommended*)
ifrs_liquidity_cols = get_valid_columns(['季IFRS財報_流動比率(%)', '季IFRS財報_現金與流動資產比率(%)', '季IFRS財報_現金與流動負債比率(%)'], available_columns)
ifrs_profitability_cols = get_valid_columns(['季IFRS財報_毛利率(%)', '季IFRS財報_EBITDA利潤率(%)', '季IFRS財報_歸屬於母公司–稅後權益報酬率(%)', '季IFRS財報_利息保障倍數(倍)'], available_columns)
ifrs_activity_cols = get_valid_columns(['季IFRS財報_應收款項週轉率(次)', '季IFRS財報_存貨週轉率(次)', '季IFRS財報_固定資產週轉率(次)'], available_columns)
ifrs_growth_cols = get_valid_columns(['季IFRS財報_營收成長率(%)', '季IFRS財報_總資產成長率(%)', '季IFRS財報_淨值成長率(%)', '季IFRS財報_營業利益成長率(%)'], available_columns)

# Example Group 7: Top Broker Details (Simplified - Aggregating Top 5 Day 0 - MANUAL & VALIDATED)
# Manually list columns confirmed to be in kept_features.txt for this group
top5_buy_avg_shares_cols = get_valid_columns([f'買超第{i}名分點買均張' for i in range(1, 6)], available_columns) # Uses only ranks 1,2,3,5 based on error msg analysis
top5_sell_avg_shares_cols = get_valid_columns([f'賣超第{i}名分點賣均張' for i in range(1, 6)], available_columns) # Uses only ranks 3,4,5 based on error msg analysis
top5_buy_avg_value_cols = get_valid_columns([f'買超第{i}名分點買均值(千)' for i in range(1, 6)], available_columns) # Uses only ranks 1,2,3,5 based on error msg analysis
top5_sell_avg_value_cols = get_valid_columns([f'賣超第{i}名分點賣均值(千)' for i in range(1, 6)], available_columns) # Uses only ranks 1,3,4,5 based on error msg analysis


# --- Collect all columns needed for loading (Optimization) ---
columns_to_load = set(essential_cols) # Start with essential ID/Target

# Add columns from defined groups, ensuring they are valid
groups_for_usecols = [
    foreign_broker_cols, main_broker_cols # Dictionaries
]
lists_for_usecols = [ # Lists
    institutional_buy_vol_cols, institutional_sell_vol_cols, institutional_net_vol_cols, institutional_holding_ratio_cols,
    tech_indicators_volatility_cols, tech_indicators_beta_cols, revenue_growth_cols,
    ifrs_liquidity_cols, ifrs_profitability_cols, ifrs_activity_cols, ifrs_growth_cols,
    top5_buy_avg_shares_cols, top5_sell_avg_shares_cols, top5_buy_avg_value_cols, top5_sell_avg_value_cols
]

for group_dict in groups_for_usecols:
    for col_list in group_dict.values():
        columns_to_load.update(col_list) # Add only valid columns already filtered

for col_list in lists_for_usecols:
     columns_to_load.update(col_list) # Add only valid columns already filtered


# Convert set back to list for usecols parameter
columns_to_load = sorted(list(columns_to_load)) # Sorting is optional, helps readability
print(f"Attempting to load {len(columns_to_load)} VALIDATED columns.")
# print("Columns to load:", columns_to_load) # Uncomment to verify list if needed

# --- Helper Function for Safe Division ---
def safe_division(numerator, denominator):
    """Performs division, returning 0 if denominator is 0 or NaN."""
    # Ensure inputs are numeric series
    num = pd.to_numeric(numerator, errors='coerce').fillna(0)
    den = pd.to_numeric(denominator, errors='coerce').fillna(0)

    # Create a result series initialized with zeros of the correct float type
    result = pd.Series(0.0, index=num.index, dtype=np.float64) # Use float64 for precision

    # Create a mask for valid denominators (non-zero and not NaN)
    # Because we filled NaN with 0 above, just checking for non-zero is sufficient
    valid_den_mask = (den != 0)

    # Perform division only where the denominator is valid
    result.loc[valid_den_mask] = num.loc[valid_den_mask] / den.loc[valid_den_mask]
    return result

# --- Main Processing ---
start_time = time.time()

try:
    print(f"Loading data from {input_csv_path}...")
    # *** Optimization: Use usecols and potentially dtype ***
    # Consider defining dtypes for further memory saving
    # dtype_spec = {col: 'float32' for col in columns_to_load if col not in essential_cols}
    # if 'ID' in essential_cols: dtype_spec['ID'] = 'int64' # Or appropriate ID type
    # if '飆股' in essential_cols: dtype_spec['飆股'] = 'int8' # Assuming binary target

    df = pd.read_csv(
        input_csv_path,
        usecols=columns_to_load, # Load only needed columns
        # dtype=dtype_spec,      # Uncomment and define dtypes for memory saving
        low_memory=False
        )
    print(f"Data loaded successfully. Shape: {df.shape}")

    # Initialize the new DataFrame using only essential columns that were actually loaded
    merged_df = df[essential_cols].copy()
    print("Initialized merged_df.")

    # --- Feature Merging Calculations ---
    # (The calculation logic below should now work because 'usecols' succeeded
    # and the column lists used here only contain valid, loaded columns)

    # 1. Foreign Broker Activity
    print("Merging Foreign Broker features...")
    for new_col_name, cols_to_avg in foreign_broker_cols.items():
        if cols_to_avg: # Check if list is not empty after validation
             merged_df[f'外資_{new_col_name}'] = df[cols_to_avg].mean(axis=1).fillna(0)
        else:
             print(f"  Skipping {new_col_name}: No valid columns found/loaded.")


    # 2. Main Broker Activity
    print("Merging Main Broker features...")
    for new_col_name, cols_to_avg in main_broker_cols.items():
         if cols_to_avg:
             merged_df[f'主力_{new_col_name}'] = df[cols_to_avg].mean(axis=1).fillna(0)
         else:
             print(f"  Skipping {new_col_name}: No valid columns found/loaded.")

    # 3. Institutional Daily Totals
    print("Merging Institutional Daily features...")
    if institutional_buy_vol_cols: merged_df['法人買張合計'] = df[institutional_buy_vol_cols].sum(axis=1).fillna(0)
    if institutional_sell_vol_cols: merged_df['法人賣張合計'] = df[institutional_sell_vol_cols].sum(axis=1).fillna(0)
    if institutional_net_vol_cols: merged_df['法人買賣超合計'] = df[institutional_net_vol_cols].sum(axis=1).fillna(0)
    if institutional_holding_ratio_cols: merged_df['法人持股比率合計'] = df[institutional_holding_ratio_cols].sum(axis=1).fillna(0)


    # 4. Technical Indicators (Ratios)
    print("Merging Technical Indicator features...")
    # Check if BOTH columns needed for the ratio were successfully loaded
    if len(tech_indicators_volatility_cols) == 2:
        merged_df['技術指標_波動率_短期vs長期'] = safe_division(
            df[tech_indicators_volatility_cols[1]], df[tech_indicators_volatility_cols[0]]
        )
    if len(tech_indicators_beta_cols) == 2:
         merged_df['技術指標_Beta_短期vs長期'] = safe_division(
             df[tech_indicators_beta_cols[1]], df[tech_indicators_beta_cols[0]]
         )

    # 5. Monthly Revenue Growth
    print("Merging Monthly Revenue features...")
    if revenue_growth_cols: merged_df['月營收_綜合成長率_Avg'] = df[revenue_growth_cols].mean(axis=1).fillna(0)

    # 6. IFRS Ratios (Group Averages)
    print("Merging IFRS features...")
    ifrs_groups = {
        'IFRS_流動性_Avg': ifrs_liquidity_cols,
        'IFRS_獲利能力_Avg': ifrs_profitability_cols,
        'IFRS_經營效率_Avg': ifrs_activity_cols,
        'IFRS_成長性_Avg': ifrs_growth_cols
    }
    for new_col_name, cols_to_avg in ifrs_groups.items():
         if cols_to_avg:
             # Consider normalization here before averaging for better results
             merged_df[new_col_name] = df[cols_to_avg].mean(axis=1).fillna(0)
         else:
              print(f"  Skipping {new_col_name}: No valid columns found/loaded.")

    # 7. Top Broker Details (Simplified Day 0 Top 5 Aggregation)
    print("Merging Top Broker features (Simplified)...")
    top_broker_groups = {
        'Top5買超_買均張_Avg': top5_buy_avg_shares_cols,
        'Top5賣超_賣均張_Avg': top5_sell_avg_shares_cols,
        'Top5買超_買均值_Avg': top5_buy_avg_value_cols,
        'Top5賣超_賣均值_Avg': top5_sell_avg_value_cols,
    }
    for new_col_name, cols_to_avg in top_broker_groups.items():
         if cols_to_avg:
             merged_df[new_col_name] = df[cols_to_avg].mean(axis=1).fillna(0)
         else:
              print(f"  Skipping {new_col_name}: No valid columns found/loaded.")


    # --- Cleanup and Save ---
    print("Merging complete. Cleaning up...")
    del df # Free memory
    gc.collect() # Trigger garbage collection

    print(f"Final merged DataFrame shape: {merged_df.shape}")
    print(f"Columns in merged DataFrame: {merged_df.columns.tolist()}")
    print(f"Saving merged data to {output_csv_path}...")
    merged_df.to_csv(output_csv_path, index=False)
    print("Merged data saved successfully.")

except FileNotFoundError:
    print(f"Error: Input file not found at {input_csv_path}")
except MemoryError:
     print(f"Error: MemoryError occurred. The file is too large to process with available RAM.")
     print("Suggestions:")
     print("1. Ensure 'usecols' is loading only necessary columns (check printout).")
     print("2. Specify efficient 'dtype' for columns in pd.read_csv.")
     print("3. Process the file in chunks using the 'chunksize' parameter in pd.read_csv.")
     print("4. Consider using libraries like Dask or Vaex for out-of-core computation.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()


end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")