import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置 x 轴起始坐标，假设从 11:30 开始显示
start_display_time = pd.Timestamp('2024-05-01 11:30')
end_display_time = pd.Timestamp('2024-05-01 16:20')

# 生成时间序列
start_time = datetime.datetime(2024, 5, 1, 12, 00, 00)
end_time = datetime.datetime(2024, 5, 1, 12, 30, 00)
interval = (end_time - start_time).total_seconds() // 60
times = [start_time + datetime.timedelta(minutes=i) for i in range(int(interval))]

# 读取第一个文件
file1 = 'data/coefficient/107.xlsx'
df1 = pd.read_excel(file1)
label1 = '107'
# 将 start_time 转换为日期时间类型，并提取小时和分钟作为横坐标
df1['start_time'] = pd.to_datetime(df1['start_time'])

# 读取第二个文件
file2 = 'data/coefficient/108.xlsx'
df2 = pd.read_excel(file2)
label2 = '108'
# 同样处理 start_time
df2['start_time'] = pd.to_datetime(df2['start_time'])

file3 = 'prediction_4km_results.csv'
df3 = pd.read_csv(file3)
label3 = 'prediction'
# 同样处理 start_time
df3['start_time'] = [start_display_time + pd.Timedelta(seconds=i) for i in df3['time']]

# 读取第二个文件
file4 = 'data/coefficient/108.xlsx'
df4 = pd.read_excel(file4)
label4 = '108'
# 同样处理 start_time
df4['start_time'] = pd.to_datetime(df4['start_time'])

# 创建图形和坐标轴
fig, ax = plt.subplots()
# 设置区间颜色为蓝色且部分透明
ax.axvspan(start_time, end_time, alpha=0.1, color='red')
# 绘制 y 轴值为 0.5 的直线
ax.axhline(y=0.25, color='red', linestyle='--')

# 添加网格
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# 绘制第一条曲线
plt.plot(df1['start_time'], df1['congestion_coefficient'], label=label1)

# 绘制第二条曲线
plt.plot(df2['start_time'], df2['congestion_coefficient'], label=label2)

plt.plot(df3['start_time'], df3['prediction'], label=label3)
# plt.plot(df4['start_time'], df4['congestion_coefficient'], label=label4)

# 设置横坐标格式为只显示小时和分钟
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# 设置 x 轴为时间格式
ax.set_xlim(left=start_display_time, right=end_display_time)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
# 添加图例
plt.legend()

# 显示图形
plt.show()