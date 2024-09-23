import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# 读取 CSV 文件
df = pd.read_excel('data/108/108.xlsx')

# 计算时间列
start_time = pd.Timestamp('11:35:43')
df['time'] = [start_time + pd.Timedelta(seconds=i * 30.163602) for i in df['frame_time']]

# 设置中文字体为黑体，防止中文显示乱码
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False


# 绘制曲线图
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(df['time'], df['num'], color='black', linewidth=.4)

# 设置 x 轴起始坐标，假设从 11:30:10 开始显示
start_display_time = pd.Timestamp('11:30')
end_display_time = pd.Timestamp('16:20')
ax.set_xlim(left=start_display_time, right=end_display_time)
ax.set_ylim(top=60, bottom=0)
# 设置 x 轴时间格式只显示小时和分钟
date_format = DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(date_format)

# 去掉右边和上面边框
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()