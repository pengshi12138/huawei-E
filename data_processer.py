import pandas as pd
import numpy as np

data_name = '108'
position_start_times = {
    '103': '2024-05-01 12:56:47',
    '105': '2024-05-01 11:52:27',
    '107': '2024-05-01 11:41:03',
    '108': '2024-05-01 11:35:43'
}
# 读取 xlsx 文件
residence_time_df = pd.read_excel('data/residence_time/result/' + data_name + '.xlsx')
traffic_volume_df = pd.read_excel('data/traffic_volume/result/' + data_name + '.xlsx')


# 提取 second_time 列和 num 列
second_times = traffic_volume_df['second_time']
residence_times_second_times = residence_time_df['second_time']
nums = traffic_volume_df['num']
densities = traffic_volume_df['density']
dwell_time_second = residence_time_df['dwell_time_second']
# 确定 300 的倍数的时间区间边界
boundaries = np.arange(0, np.max(second_times) + 300, 300)

# 进行线性插值
interpolated_nums = np.interp(boundaries, second_times, nums)
interpolated_densities = np.interp(boundaries, second_times, densities)
interpolated_residence_times = np.interp(boundaries, residence_times_second_times, dwell_time_second)

time_obj = pd.Timestamp(position_start_times[data_name])
start_times = [time_obj + pd.Timedelta(seconds=i) for i in boundaries]

# 创建新的 DataFrame 用于保存结果
result_df = pd.DataFrame({'start_time': start_times, 'second_time': boundaries,
                          'flow_sum': interpolated_nums, 'density': interpolated_densities,
                          'residence_time': interpolated_residence_times})

# 将结果保存到 csv 文件
result_df.to_csv(data_name + '.csv', index=False)

