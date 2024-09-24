#加载保存的模型进行预测
import numpy as np
import pandas as pd
import torch
from torch import nn


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


# 生成时间序列数据和位置数据
position_start_time = pd.Timestamp('11:30:00')
position_end_time = pd.Timestamp('16:30:00')
start_time = pd.Timestamp('12:00:00')
end_time = pd.Timestamp('16:00:00')
time_diff = pd.Timedelta(minutes=5)
times = [start_time - position_start_time + i * time_diff for i in range(int((end_time - start_time) / time_diff))]

times_normalize = [time / (position_end_time - position_start_time) for time in times]

locations = [4] * len(times)

locations_normalize = [4] * len(times)
data = pd.DataFrame({'time': times, 'location': locations,
                     'times_normalize': times_normalize, 'locations_normalize': locations_normalize})

# 将时间列转换为时间戳（以秒为单位）
data['time'] = data['time'].astype(np.int64) // 10 ** 9

# 对位置进行编码，可以使用独热编码等方法将位置转换为数值形式以便模型处理

# 提取特征和目标（这里假设没有实际目标数据，只是演示预测过程）
X = data[['times_normalize', 'locations_normalize']].values

# X = np.array([[1727187600, 4]])
# 设置模型参数
input_size = 2
hidden_size = 64
num_layers = 2
output_size = 1

loaded_model = BiLSTMModel(input_size, hidden_size, num_layers, output_size)
loaded_model.load_state_dict(torch.load('trains/bilstm_model_params.pth'))
loaded_model.eval()
# 转换为 PyTorch 的张量
X_tensor = torch.from_numpy(X).float()

# 使用模型进行预测
with torch.no_grad():
    predictions = loaded_model(X_tensor)
# 输出预测结果

for i, prediction in enumerate(predictions):
    time = data.iloc[i]['time']
    print(f"时间：{time}，位置：4km，预测结果：{prediction[0]}")
# 记录预测结果到 CSV 文件
data['prediction'] = predictions / 10
data.to_csv('prediction_4km_results.csv', index=False)