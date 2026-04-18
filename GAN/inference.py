import torch
import torch.nn.functional as F
import pandas as pd
from models.generator_deep import Generator
from models.discriminator_deep import Discriminator

# 與 train.py 相同的特徵欄位 (共 60 個)
feature_cols = [
    '外資券商_mean','外資券商_std','外資券商_max','外資券商_min',
    '主力券商_mean','主力券商_std','主力券商_max','主力券商_min',
    '官股券商_mean','官股券商_std','官股券商_max','官股券商_min',
    '個股券商分點籌碼分析_mean','個股券商分點籌碼分析_std','個股券商分點籌碼分析_max',
    '個股券商分點籌碼分析_min','個股券商分點區域分析_mean','個股券商分點區域分析_std',
    '個股券商分點區域分析_max','個股券商分點區域分析_min','個股主力買賣超統計_mean',
    '個股主力買賣超統計_std','個股主力買賣超統計_max','個股主力買賣超統計_min',
    '日外資_mean','日外資_std','日外資_max','日外資_min','日自營_mean','日自營_std',
    '日自營_max','日自營_min','日投信_mean','日投信_std','日投信_max','日投信_min',
    '技術指標_mean','技術指標_std','技術指標_max','技術指標_min','月營收_mean',
    '月營收_std','月營收_max','月營收_min','季IFRS財報_mean','季IFRS財報_std',
    '季IFRS財報_max','季IFRS財報_min','買超分點_mean','買超分點_std','買超分點_max',
    '買超分點_min','賣超分點_mean','賣超分點_std','賣超分點_max','賣超分點_min',
    '其他_mean','其他_std','其他_max','其他_min'
]

# 與 train.py 相同的超參數
noise_dim = 100
feature_dim = len(feature_cols)  # 60
hidden_dim = 64
seq_len = 20  # 雖然訓練時使用序列長度為20，但這裡推論用單步資料即可
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建立模型實例
G = Generator(noise_dim, feature_dim, hidden_dim, seq_len).to(device)
D = Discriminator(feature_dim).to(device)

# 載入訓練後的權重
G.load_state_dict(torch.load("E:\\Tbrain_stock_analysis\\GAN\\fed_gan_adam_csv100\\best_federated_GAN_generator_fedadam.pth", map_location=device))
D.load_state_dict(torch.load("E:\\Tbrain_stock_analysis\\GAN\\fed_gan_adam_csv100\\best_federated_GAN_discriminator_fedadam.pth", map_location=device))

# 設定為推論模式
G.eval()
D.eval()

# 讀取測試資料
test_df = pd.read_csv("E:\\Tbrain_stock_analysis\\test_4o_merged.csv")

# 取出 ID 欄位（假設欄位名稱為 "ID"）
# 如果沒有 ID 欄位，請自行調整
ids = test_df["ID"].values

# 只取特徵欄位 (60 維)
test_features = test_df[feature_cols].values
# 建立張量
test_data = torch.tensor(test_features, dtype=torch.float32).to(device)  # shape: (N, 60)

# 推論
with torch.no_grad():
    # 直接餵入 (N, 60) 到判別器
    adv_out, cls_out = D(test_data)
    prob = F.softmax(cls_out, dim=1)      # shape: (N, 2) 假設是二分類
    predictions = torch.argmax(prob, dim=1)  # 取得類別 0 或 1

# 建立輸出 DataFrame，把預測結果和 ID 一起寫入
# 這邊「飆股」是以 1 表示飆股，0 表示非飆股
res_df = pd.DataFrame({
    "ID": ids,
    "飆股": predictions.cpu().numpy()  # 轉成 CPU numpy
})

# 輸出檔案 (自行修改路徑)
res_df.to_csv("E:\\Tbrain_stock_analysis\\outputs\\output_20250413\\inference_result_gan_fedadam.csv", index=False)
print("推論完成，結果已儲存至 inference_result_gan_fedadam.csv")
