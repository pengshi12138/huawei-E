import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 假设四个位置的数据文件分别为 1km.csv、2km.csv、3km.csv 和 5km.csv
data1 = pd.read_excel('data/coefficient/103.xlsx')
data2 = pd.read_excel('data/coefficient/105.xlsx')
data3 = pd.read_excel('data/coefficient/107.xlsx')
data5 = pd.read_excel('data/coefficient/108.xlsx')

# 合并数据
all_data = pd.concat([data1, data2, data3, data5], ignore_index=True)

# 提取时间、位置作为输入特征，假设时间列名为'time'，位置列名为'location'
time_column = 'start_time'
location_column = 'location'
feature_columns = [time_column, location_column]

# 提取要预测的文件特征列，假设特征列名为'feature_to_predict'
target_column = ['congestion_coefficient']
# 获取 11:30 的时间戳（秒数）
position_start_time = pd.Timestamp('2024-05-01 11:30:00').timestamp()
position_end_time = pd.Timestamp('2024-05-01 16:30:00').timestamp()
# 对时间进行格式化处理，如果需要的话，可以将时间字符串转换为时间戳等格式以便模型处理
all_data[time_column] = ((pd.to_datetime(all_data[time_column]).astype(np.int64) // 10**9 - position_start_time)
                         / (position_end_time - position_start_time))

# 对位置进行编码，可以使用独热编码等方法将位置转换为数值形式以便模型处理
# all_data = pd.get_dummies(all_data, columns=[location_column])

# 数据归一化
all_data[location_column] = all_data[location_column]

# 提取特征和目标
X = all_data[feature_columns].values
y = all_data[target_column].values

# 对目标变量 y 进行归一化
y = 10 * y

# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

# 转换为 PyTorch 的张量
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)


# 定义双向 LSTM 模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 移除 batch_first=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # 调整输入形状
        x = x.unsqueeze(1)  # 添加假序列维数
        h0 = torch.zeros(self.num_layers * 2, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(1), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out.view(out.size(0), -1))
        return out


# 设置模型参数
input_size = X.shape[1]
hidden_size = 64
num_layers = 2

# 确保输出大小与目标变量的大小相匹配
output_size = y.shape[1]
model = BiLSTMModel(input_size, hidden_size, num_layers, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 训练模型并记录损失和测试集误差
train_losses = []
test_losses = []
batch_size = 32
num_epochs = 50
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    # 记录训练集损失
    with torch.no_grad():
        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train)
        train_losses.append(train_loss.item())
    # 记录测试集损失
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses.append(test_loss.item())
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}')

# 绘制训练集和测试集损失曲线
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curve')
plt.legend()
plt.show()
