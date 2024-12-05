import numpy as np
import random
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt  # 导入绘图库
from Network import CNN_GRU_Attention_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 目标函数：计算MSE
def calculate_mse(predicted_phase, target_phase):
    """计算预测相位与目标相位的均方误差"""
    return np.mean((predicted_phase - target_phase) ** 2)


# 生成21维的随机二进制解
def generate_random_solution():
    """生成随机的21维二进制解"""
    return np.random.randint(0, 2, size=(1, 21))


# 随机变异1位
def mutate_solution(solution):
    """在21维解中随机变异1位"""
    mutated_solution = solution.copy()
    mutation_index = random.randint(0, 20)  # 随机选择一个位置
    mutated_solution[0, mutation_index] = 1 - mutated_solution[0, mutation_index]  # 翻转该位置
    return mutated_solution


# 随机变异2位
def mutate_two_bits(solution):
    """随机改变解中的两个位置"""
    mutated_solution = solution.copy()
    indices = random.sample(range(21), 2)  # 随机选择两个位置
    mutated_solution[0, indices[0]] = 1 - mutated_solution[0, indices[0]]  # 翻转第一个位置
    mutated_solution[0, indices[1]] = 1 - mutated_solution[0, indices[1]]  # 翻转第二个位置
    return mutated_solution


# 基于敏感度的变异1位
def mutate_solution_based_on_sensitivity(solution, sorted_indices, random_prob=0.8):
    """基于敏感度在21维解中随机变异1位，带有一定概率的随机变异"""
    mutated_solution = solution.copy()

    if random.random() < random_prob:
        # 进行完全随机的变异
        mutation_index = random.randint(0, 20)  # 随机选择一个位置
        mutated_solution[0, mutation_index] = 1 - mutated_solution[0, mutation_index]  # 翻转该位置
    else:
        # 根据敏感度选择变异
        mutation_index = sorted_indices[0]  # 选择敏感度最大的位
        mutated_solution[0, mutation_index] = 1 - mutated_solution[0, mutation_index]  # 翻转该位置

    return mutated_solution





# 评估敏感度的函数
def evaluate_sensitivity(solution, target_phase, cached_mse=None):
    """计算每个位置的敏感度，敏感度高的位应排前"""
    sensitivity = np.zeros(21)  # 初始化敏感度为0
    # 计算原始解的MSE，若已缓存则直接使用
    if cached_mse is None:
        base_mse = calculate_mse(
            forward_neural_network(torch.tensor(solution).float().to(device)).cpu().detach().numpy(),
            target_phase)
    else:
        base_mse = cached_mse

    for i in range(21):
        temp_solution = solution.copy()
        temp_solution[0, i] = 1 - temp_solution[0, i]  # 翻转该位置
        predicted_phase = forward_neural_network(torch.tensor(temp_solution).float().to(device)).cpu().detach().numpy()
        mse = calculate_mse(predicted_phase, target_phase)  # 计算翻转后的MSE

        # 计算MSE的变化量，即增量，注意这里我们期望MSE减小，因此增量越小越敏感
        sensitivity[i] = base_mse - mse  # 敏感度越大，MSE减少越多，敏感度越高

    # 按照敏感度从大到小排序，敏感度大的位排前
    sorted_indices = np.argsort(sensitivity)[::-1]  # 从大到小排序敏感度
    return sorted_indices, sensitivity, base_mse  # 返回排序后的索引和敏感度值


# 转换为对称矩阵
def convert_to_symmetric(particles):
    """将21维的解转换为对称矩阵"""
    particles = torch.reshape(particles, (-1, 21))
    num_particles, particle_length = particles.size()
    result = torch.zeros(num_particles, 6, 6).to(device)
    for i in range(6):
        for j in range(i + 1):
            result[:, i, j] = particles[:, (i * (i + 1) // 2) + j]
            result[:, j, i] = result[:, i, j]
    return result


# 神经网络预测
def forward_neural_network(input_solution):
    """通过判别网络预测相位数据"""
    symmetric_matrix = convert_to_symmetric(input_solution).view(1, -1).to(device)

    # 加载模型
    real_model = CNN_GRU_Attention_Model().to(device)
    imag_model = CNN_GRU_Attention_Model().to(device)
    real_model.load_state_dict(torch.load(r'./model/实部MA-CGFPNetmodel_500.pth', weights_only=True))
    imag_model.load_state_dict(torch.load(r'./model/虚部MA-CGFPNetmodel_500.pth', weights_only=True))

    real_model.eval()
    imag_model.eval()

    with torch.no_grad():
        real_output = real_model(symmetric_matrix)
        imag_output = imag_model(symmetric_matrix)

    predicted_phase = torch.cat((real_output, imag_output), dim=1)
    return predicted_phase


# 读取目标相位数据
Ptarget = pd.read_csv("showdata/sampled_c4_phase_data.csv", header=None).values  # 假设目标相位数据在csv文件中
target_phase = Ptarget[52]  # 取第一行数据作为目标相位数据


# 模拟退火算法
def simulated_annealing(initial_solution, target_phase, initial_temp, cooling_rate, final_temp, iterations, patience,
                        sensitivity_flip_count=3):
    current_solution = initial_solution
    current_temp = initial_temp
    best_solution = current_solution
    # 计算初始解的MSE（预测相位和目标相位的MSE）
    best_mse = calculate_mse(
        forward_neural_network(torch.tensor(best_solution).float().to(device)).cpu().detach().numpy(), target_phase)

    # 输出初始解的MSE
    print("初始解的MSE：", best_mse)

    # 用于记录MSE值变化的列表
    mse_history = [best_mse]

    # 连续无变化的计数器
    no_improvement_counter = 0

    # 1. 在初始阶段根据敏感度翻转最敏感的3个位
    sorted_indices, _, _ = evaluate_sensitivity(current_solution, target_phase)
    for i in range(sensitivity_flip_count):
        current_solution[0, sorted_indices[i]] = 1 - current_solution[0, sorted_indices[i]]  # 翻转最敏感的位

    # 计算翻转后的MSE
    flipped_mse = calculate_mse(
        forward_neural_network(torch.tensor(current_solution).float().to(device)).cpu().detach().numpy(), target_phase)

    # 输出翻转后的解的MSE
    print("翻转后的MSE：", flipped_mse)

    # 2. 开始模拟退火过程
    while current_temp > final_temp:
        for _ in range(iterations):
            sorted_indices, _, base_mse = evaluate_sensitivity(current_solution, target_phase)

            # 基于敏感度的变异
            new_solution = mutate_solution_based_on_sensitivity(current_solution, sorted_indices)

            # 计算新解的预测相位
            predicted_phase = forward_neural_network(
                torch.tensor(new_solution).float().to(device)).cpu().detach().numpy()

            # 计算新解的预测相位与目标相位数据的MSE
            new_mse = calculate_mse(predicted_phase, target_phase)
            delta_mse = new_mse - best_mse

            # 判断是否接受新解
            if delta_mse < 0 or random.random() < math.exp(-delta_mse / current_temp):
                current_solution = new_solution
                if new_mse < best_mse:
                    best_solution = new_solution  # 更新全局最优解
                    best_mse = new_mse
                    print("当前最优解：\n", best_solution)
                    print("当前最优MSE：", best_mse)
                    no_improvement_counter = 0  # 重置无变化计数器

            # 如果全局最优解连续多次没有变化，则基于全局最优解进行敏感度最高的反转
            if no_improvement_counter >= patience:
                print("全局最优解连续多次没有变化，进行基于全局最优解的随机两位变异...")
                sorted_indices, _, _ = evaluate_sensitivity(best_solution, target_phase)
                 # ------------------需要补充逻辑




        current_temp *= cooling_rate  # 降低温度

        mse_history.append(best_mse)  # 记录每次循环的最小MSE

    return best_solution, mse_history


# 主函数入口
if __name__ == "__main__":
    # 设置初始参数
    initial_solution = generate_random_solution()  # 随机生成初始解
    initial_temp = 100  # 初始温度
    cooling_rate = 0.95  # 温度衰减率
    final_temp = 0.01  # 终止温度
    iterations = 100  # 每个温度下的最大迭代次数
    patience = 100  # 连续无变化时进行两位变异
    sensitivity_flip_count = 1  # 初始阶段翻转的敏感位数

    # 使用模拟退火优化
    best_solution, mse_history = simulated_annealing(initial_solution, target_phase, initial_temp, cooling_rate,
                                                     final_temp, iterations, patience, sensitivity_flip_count)

    # 绘制MSE曲线
    plt.plot(mse_history)
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.title('MSE Convergence Curve')
    plt.show()

    print("最优解：\n", best_solution)
    print("最优MSE：", mse_history[-1])