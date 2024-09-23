import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data_folder = 'data/5-minute_interval_data'
prefixes = ['103', '105', '107', '108']

for prefix in prefixes:
    file_name = [f for f in os.listdir(data_folder) if f.startswith(prefix) and f.endswith('.csv')][0]
    file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(file_path)
    start_time_col = 'start_time'
    density_col = 'density'
    if start_time_col in df.columns and density_col in df.columns:
        # 将 start_time 列转换为 datetime 对象
        df[start_time_col] = pd.to_datetime(df[start_time_col])
        plt.plot(df[start_time_col], df[density_col], label=prefix)
    else:
        print(f'File {file_name} does not have the required columns.')
# 设置 x 轴起始坐标，假设从 11:30 开始显示
start_display_time = pd.Timestamp('2024-05-01 11:30')
end_display_time = pd.Timestamp('2024-05-01 16:20')
# 设置横坐标格式为小时和分钟
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_xlim(left=start_display_time, right=end_display_time)
plt.legend()
plt.show()