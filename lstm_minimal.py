# coding=utf-8
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import os

# 1. 环境检查与配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 正在使用设备: {device} ---")

# 配置参数
SYMBOL = 'RB'  # 螺纹钢
LOOKBACK = 60  # 使用过去60根K线
PREDICT_STEP = 1  # 预测未来1根
HIDDEN_DIM = 64
NUM_LAYERS = 2
EPOCHS = 20
BATCH_SIZE = 32

def load_data():
    db_path = r'D:\期货\回测改造\data\futures_data.db'
    if not os.path.exists(db_path):
        print("错误：未找到数据库文件，请检查路径。")
        return None
    
    conn = sqlite3.connect(db_path)
    # 尝试加载螺纹钢日线数据
    query = f"SELECT time, open, high, low, close, volume FROM bars WHERE symbol = '{SYMBOL}' ORDER BY time ASC"
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
        print(f"错误：数据库中没有 {SYMBOL} 的数据")
        return None
    
    print(f"成功加载 {len(df)} 条数据")
    return df

def prepare_data(df):
    # 只取收盘价做演示
    data = df[['close']].values.astype('float32')
    
    # 归一化 (AI 对原始价格不敏感，必须缩放到 0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 构造序列数据
    x, y = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        x.append(scaled_data[i-LOOKBACK:i, 0])
        y.append(scaled_data[i, 0])
    
    x, y = np.array(x), np.array(y)
    
    # 转为 PyTorch 张量并移动到设备
    x = torch.from_numpy(x).unsqueeze(-1)  # [Samples, TimeSteps, Features]
    y = torch.from_numpy(y).unsqueeze(-1)
    
    # 划分训练集和测试集 (前80%训练)
    split = int(len(x) * 0.8)
    train_x, test_x = x[:split], x[split:]
    train_y, test_y = y[:split], y[split:]
    
    return train_x, train_y, test_x, test_y, scaler

# 2. 定义 LSTM 模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        # 只取序列最后一个时间步的结果
        out = self.fc(out[:, -1, :])
        return out

# 3. 运行主流程
def run():
    df = load_data()
    if df is None: return
    
    train_x, train_y, test_x, test_y, scaler = prepare_data(df)
    
    model = SimpleLSTM(input_dim=1, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=1).to(device)
    criterion = nn.MSELoss() # 均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 数据迭代器
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
    
    print("\n--- 开始训练 ---")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.6f}")

    # 4. 预测与可视化
    print("\n--- 正在生成预测图 ---")
    model.eval()
    with torch.no_grad():
        # 训练集预测 (用于观察拟合)
        train_pred = model(train_x.to(device)).cpu().numpy()
        # 测试集预测 (真正的未来表现)
        test_pred = model(test_x.to(device)).cpu().numpy()
        
    # 反归一化还原价格
    train_pred = scaler.inverse_transform(train_pred)
    train_y_real = scaler.inverse_transform(train_y.numpy())
    test_pred = scaler.inverse_transform(test_pred)
    test_y_real = scaler.inverse_transform(test_y.numpy())
    
    # 绘图
    plt.figure(figsize=(15, 6))
    # 只画测试集部分
    plt.plot(test_y_real, label='Real Price', color='blue', alpha=0.7)
    plt.plot(test_pred, label='AI Prediction', color='red', linestyle='--', alpha=0.9)
    plt.title(f'{SYMBOL} Price Prediction (LSTM Test Set)')
    plt.legend()
    plt.grid(True)
    
    save_path = r'D:\期货\回测改造\lstm_result.png'
    plt.savefig(save_path)
    print(f"--- 预测对比图已保存至: {save_path} ---")
    # plt.show() # 如果在GUI环境下可以取消注释

if __name__ == "__main__":
    run()
