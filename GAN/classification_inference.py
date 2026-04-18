import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# ========== 模型與特徵欄位定義 ==========
# feature_cols = [
#     '外資券商_mean','外資券商_std','外資券商_max','外資券商_min',
#     '主力券商_mean','主力券商_std','主力券商_max','主力券商_min',
#     '官股券商_mean','官股券商_std','官股券商_max','官股券商_min',
#     '個股券商分點籌碼分析_mean','個股券商分點籌碼分析_std','個股券商分點籌碼分析_max',
#     '個股券商分點籌碼分析_min','個股券商分點區域分析_mean','個股券商分點區域分析_std',
#     '個股券商分點區域分析_max','個股券商分點區域分析_min','個股主力買賣超統計_mean',
#     '個股主力買賣超統計_std','個股主力買賣超統計_max','個股主力買賣超統計_min',
#     '日外資_mean','日外資_std','日外資_max','日外資_min','日自營_mean','日自營_std',
#     '日自營_max','日自營_min','日投信_mean','日投信_std','日投信_max','日投信_min',
#     '技術指標_mean','技術指標_std','技術指標_max','技術指標_min','月營收_mean',
#     '月營收_std','月營收_max','月營收_min','季IFRS財報_mean','季IFRS財報_std',
#     '季IFRS財報_max','季IFRS財報_min','買超分點_mean','買超分點_std','買超分點_max',
#     '買超分點_min','賣超分點_mean','賣超分點_std','賣超分點_max','賣超分點_min',
#     '其他_mean','其他_std','其他_max','其他_min']

feature_cols = [
"技術指標_週RSI(5)", "技術指標_週RSI(10)", "技術指標_週MACD", "技術指標_週K(9)",
"技術指標_週DIF-週MACD", "技術指標_週DIF", "技術指標_週-DI(14)", "技術指標_週D(9)",
"技術指標_週ADX(14)", "技術指標_週+DI(14)", "技術指標_相對強弱比(週)", "技術指標_相對強弱比(日)",
"技術指標_近六月歷史波動率(%)", "技術指標_近三月歷史波動率(%)", "技術指標_近二月歷史波動率(%)",
"技術指標_近九月歷史波動率(%)", "技術指標_近一年歷史波動率(%)", "技術指標_近一月歷史波動率(%)",
"技術指標_季RSI(5)", "技術指標_季RSI(10)", "技術指標_季MACD", "技術指標_季K(9)",
"技術指標_季DIF-季MACD", "技術指標_季DIF", "技術指標_季-DI(14)", "技術指標_季D(9)",
"技術指標_季ADX(14)", "技術指標_季+DI(14)", "技術指標_乖離率(60日)", "技術指標_乖離率(250日)",
"技術指標_乖離率(20日)", "技術指標_年化波動度(250D)", "技術指標_年化波動度(21D)",
"技術指標_月RSI(5)", "技術指標_月RSI(10)", "技術指標_月MACD", "技術指標_月K(9)",
"技術指標_月DIF-月MACD", "技術指標_月DIF", "技術指標_月-DI(14)", "技術指標_月D(9)",
"技術指標_月ADX(14)", "技術指標_月+DI(14)", "技術指標_W%R(5)", "技術指標_W%R(10)",
"技術指標_RSI(5)", "技術指標_RSI(10)", "技術指標_MACD", "技術指標_K(9)",
"技術指標_EWMA波動率(%)", "技術指標_DIF-MACD", "技術指標_DIF", "技術指標_+DI(14)",
"技術指標_-DI(14)", "技術指標_D(9)", "技術指標_Beta係數(65D)", "技術指標_Beta係數(250D)",
"技術指標_Beta係數(21D)", "技術指標_Alpha(250D)", "技術指標_ADX(14)", "技術指標_保力加通道–頂部(20)",
"技術指標_保力加通道–均線(20)", "技術指標_保力加通道–底部(20)", "技術指標_CM-VIX(%)",
"技術指標_SAR", "技術指標_TR(1)", "技術指標_ADXR(14)", "技術指標_+DM(14)",
"技術指標_-DM(14)", "技術指標_週TR(14)", "技術指標_週ADXR(14)", "技術指標_週+DM(14)",
"技術指標_週-DM(14)", "技術指標_月TR(14)", "技術指標_月ADXR(14)", "技術指標_月+DM(14)",
"技術指標_月-DM(14)", "技術指標_季TR(14)", "技術指標_季ADXR(14)", "技術指標_季+DM(14)",
"技術指標_季-DM(14)",
]

# class Classifier(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, 256)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.fc2 = nn.Linear(256, 128)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.fc3 = nn.Linear(128, 64)
#         self.out = nn.Linear(64, 2)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = self.dropout(F.relu(self.bn2(self.fc2(x))))
#         x = F.relu(self.fc3(x))
#         return self.out(x)
    
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # 增加神經元數量
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)  # 降低 Dropout 比例

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.fc4(x))
        return self.out(x)

class InferenceDataset(Dataset):
    def __init__(self, df):
        self.ids = df['ID'].values
        df = df.drop(columns=['ID'])
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(df[feature_cols].values.astype("float32"))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.ids[idx]

# ========== 推論主程式 ==========
def run_inference(model_path, data_path, output_path):
    # 填補缺失值為平均值
    df = pd.read_csv(data_path)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())  # 填補特徵欄位的缺失值

    dataset = InferenceDataset(df)
    loader = DataLoader(dataset, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    with torch.no_grad():
        for x, ids in loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            for pid, p in zip(ids, pred):
                results.append((pid, int(p)))

    # 儲存推論結果
    pd.DataFrame(results, columns=["ID", "飆股"]).to_csv(output_path, index=False)
    print(f"✅ 推論完成，已儲存至 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="E:\\Tbrain_stock_analysis\\classifier_best_fedavg_onlySkill.pth", help='模型檔案路徑')
    parser.add_argument('--data_path', type=str, default="E:\\Tbrain_stock_analysis\\test_onlySkill_all.csv", help='測試資料 CSV 路徑（需含 ID 欄）')
    parser.add_argument('--output_path', type=str, default='E:\\Tbrain_stock_analysis\\outputs\\output_20250418\\inference_result_class_fedavg_onlySkill.csv', help='輸出檔案名稱')
    args = parser.parse_args()

    run_inference(args.model_path, args.data_path, args.output_path)
