import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
import pickle  # 用於儲存 Scikit-learn, XGBoost, LightGBM 模型和 scaler

# 載入資料
try:
    df = pd.read_csv('E:\\Tbrain_stock_analysis\\training_filtered_merged.csv')
except FileNotFoundError:
    print("找不到 training.csv 檔案，請確認檔案路徑。")
    exit()

# 分離特徵與目標變數
X = df.drop('飆股', axis=1)
y = df['飆股']

# 移除 'ID' 欄位
X = X.drop('ID', axis=1)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 找出數值型特徵 (除了 '飆股' 和 'ID' 以外的所有欄位)
numerical_features = X_train.columns

# 標準化數值型特徵
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# 儲存 scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("StandardScaler 已儲存為 scaler.pkl")

# 將資料轉換為 PyTorch Tensor
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# 創建 Dataset
class StockDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = StockDataset(X_train_tensor, y_train_tensor)
test_dataset = StockDataset(X_test_tensor, y_test_tensor)

# 創建 DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 模型訓練 ---

# 1. Gradient Boosting Decision Tree (GBDT)
gbdt_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbdt_model.fit(X_train, y_train)
gbdt_predictions = gbdt_model.predict(X_test)
gbdt_accuracy = accuracy_score(y_test, gbdt_predictions)
print(f"GBDT 模型準確度: {gbdt_accuracy:.4f}")
# 儲存 GBDT 模型
with open('gbdt_model.pkl', 'wb') as f:
    pickle.dump(gbdt_model, f)
print("GBDT 模型已儲存為 gbdt_model.pkl")

# 2. XGBoost
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
print(f"XGBoost 模型準確度: {xgb_accuracy:.4f}")
# 儲存 XGBoost 模型
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("XGBoost 模型已儲存為 xgb_model.pkl")

# 3. LightGBM
lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train)
lgbm_predictions = lgbm_model.predict(X_test)
lgbm_accuracy = accuracy_score(y_test, lgbm_predictions)
print(f"LightGBM 模型準確度: {lgbm_accuracy:.4f}")
# 儲存 LightGBM 模型
with open('lgbm_model.pkl', 'wb') as f:
    pickle.dump(lgbm_model, f)
print("LightGBM 模型已儲存為 lgbm_model.pkl")

# 4. 深度學習網路 (淺層)
class ShallowNN(nn.Module):
    def __init__(self, input_dim):
        super(ShallowNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

input_dim = X_train_tensor.shape[1]
shallow_nn_model_pt = ShallowNN(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(shallow_nn_model_pt.parameters())
epochs = 10

shallow_nn_model_pt.train()
for epoch in range(epochs):
    for inputs, labels in train_loader:
        # 前向傳播
        outputs = shallow_nn_model_pt(inputs)
        loss = criterion(outputs, labels)

        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 評估淺層模型
shallow_nn_model_pt.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = shallow_nn_model_pt(inputs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

shallow_nn_accuracy_pt = correct / total
print(f"PyTorch 淺層深度學習網路準確度: {shallow_nn_accuracy_pt:.4f}")
# 儲存淺層 PyTorch 模型權重
torch.save(shallow_nn_model_pt.state_dict(), 'shallow_nn_model.pth')
print("淺層深度學習網路模型權重已儲存為 shallow_nn_model.pth")

# 5. 深度學習網路 (較深層)
class DeepNN(nn.Module):
    def __init__(self, input_dim):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

deep_nn_model_pt = DeepNN(input_dim)
criterion_deep = nn.BCELoss()
optimizer_deep = optim.Adam(deep_nn_model_pt.parameters())
epochs_deep = 20

deep_nn_model_pt.train()
for epoch in range(epochs_deep):
    for inputs, labels in train_loader:
        # 前向傳播
        outputs = deep_nn_model_pt(inputs)
        loss = criterion_deep(outputs, labels)

        # 反向傳播和優化
        optimizer_deep.zero_grad()
        loss.backward()
        optimizer_deep.step()

# 評估較深層模型
deep_nn_model_pt.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = deep_nn_model_pt(inputs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

deep_nn_accuracy_pt = correct / total
print(f"PyTorch 較深層深度學習網路準確度: {deep_nn_accuracy_pt:.4f}")
# 儲存較深層 PyTorch 模型權重
torch.save(deep_nn_model_pt.state_dict(), 'deep_nn_model.pth')
print("較深層深度學習網路模型權重已儲存為 deep_nn_model.pth")