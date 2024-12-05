import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 读取数据
data = pd.read_csv('./data/phase_51.csv', header=None)  # header=None 确保没有使用第一行作为列名
matrix = pd.read_csv('./data/matrix.csv', header=None)
data = data.values
matrix = matrix.values  # 转换为NumPy数组以便处理

# 假设数据集形状为 (61499, 51)
hz_values = np.linspace(9, 11, 51)  # 生成从9到11的51个点

# 设置阈值，定义梯度范围
lower_threshold = -5  # 下限
upper_threshold = 5   # 上限

# 计算每个样本的相邻列的梯度
gradients = np.diff(data, axis=1)  # 计算相邻列的梯度差

# 判断每个样本的梯度是否完全在 (-5, 5) 范围内
keep_samples = np.all((gradients > lower_threshold) & (gradients < upper_threshold), axis=1)

# 使用布尔数组来过滤data和matrix中的样本
cleaned_data = data[keep_samples, :]
cleaned_matrix = matrix[keep_samples, :]  # 同时过滤matrix

# 将清洗后的data保存为CSV
cleaned_data_df = pd.DataFrame(cleaned_data)
cleaned_data_df.to_csv('./data/cleaned_phase_data.csv', index=False, header=False)

# 将清洗后的matrix保存为CSV
cleaned_matrix_df = pd.DataFrame(cleaned_matrix)
cleaned_matrix_df.to_csv('./data/cleaned_matrix_data.csv', index=False, header=False)

# 输出原始和过滤后的数据形状
print(f'Original data shape: {data.shape}')
print(f'Filtered data shape: {cleaned_data.shape}')
print(f'Original matrix shape: {matrix.shape}')
print(f'Filtered matrix shape: {cleaned_matrix.shape}')
