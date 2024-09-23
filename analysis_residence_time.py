import os
from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

record_start_time = "11:30:00"

position_start_times = {
    '103': '12:56:47',
    '105': '11:52:27',
    '107': '11:41:03',
    '108': '11:35:43'
}

miao_zhen_bi = {
    '103': 0.409380952,
    '105': 0.355851648,
    '107': 0.30387037,
    '108': 0.301636029
}

def save_csv(read_name, save_name):
    # 读取CSV文件
    df = pd.read_csv(read_name)

    # 计算帧差值
    df['frame_diff'] = df['last_frame'] - df['first_frame']

    # 定义区间索引
    df['interval_index'] = (df['first_frame'] // 750) * 750

    # 按区间索引分组，并计算每组内frame_diff最大的20个值的平均值
    result = df.groupby('interval_index')['frame_diff'].apply(
        lambda x: x.nlargest(100).mean()
    ).reset_index(name='avg_frame_diff')
    # 计算滞留时间
    result['start_time_second'] = (result['interval_index'] / 750 + 1) * 300

    # 计算滞留时间
    result['dwell_time_second'] = result['avg_frame_diff'] * 0.4

    # 输出结果到CSV文件
    result.to_csv(save_name + '.csv', index=False)

    # 输出结果
    print(result)


def draw():
    data_folder = 'data/residence_time/result'
    files = [f for f in os.listdir(data_folder) if f.endswith('.xlsx')]

    for file_name in files:
        file_path = os.path.join(data_folder, file_name)
        # 去除文件后缀
        label = os.path.splitext(file_name)[0]
        df = pd.read_excel(file_path)
        start_time_col = 'interval_index'
        dwell_time_col = 'dwell_time_second'
        if start_time_col in df.columns and dwell_time_col in df.columns:
            time_str = position_start_times[label]
            time_obj = pd.Timestamp(time_str)
            start_times = [time_obj + pd.Timedelta(seconds=i * miao_zhen_bi[label]) for i in df[start_time_col]]
            dwell_times = df[dwell_time_col]
            plt.plot(start_times, dwell_times, label=label)
        else:
            print(f'File {file_name} does not have the required columns.')

    # 设置 x 轴起始坐标，假设从 11:30 开始显示
    start_display_time = pd.Timestamp('11:30')
    end_display_time = pd.Timestamp('16:20')
    # 设置 x 轴为时间格式
    ax = plt.gca()
    ax.set_xlim(left=start_display_time, right=end_display_time)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # save_csv('data/residence_time/108/20240501_20240501135236_20240501160912_135235_car_id.csv', '108-2')
    draw()
