import random
import numpy as np
from Simulator.Approximator import *
import torch
from torch.utils.data import Dataset, DataLoader
from pyomo.environ import *
from Simulator.Plotter import ShapeDrawer_2D
from Simulator import PROJECT_ROOT
import os
import pickle
from Simulator.Plotter import ErrorVisualizer
from DSO_boundary import *
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib

def parse_matrix_from_file(file_path):
    """
    从文件中解析矩阵数据，处理跨行的情况
    """
    rows = []
    current_row = []
    in_row = False

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            # 检查是否是新行的开始
            if line.startswith('['):
                # 如果之前有正在处理的行，先完成它
                if current_row:
                    rows.append(current_row)
                    current_row = []

                # 标记开始新行
                in_row = True
                line = line[1:]  # 移除开头的 [

            # 如果是在行内处理
            if in_row:
                # 检查是否是行结束
                if line.endswith(']'):
                    in_row = False
                    line = line[:-1]  # 移除结尾的 ]

                # 提取当前部分的数字
                pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
                numbers = re.findall(pattern, line)

                # 添加到当前行
                for num in numbers:
                    current_row.append(float(num))

                # 如果行结束了，添加到行列表
                if not in_row and current_row:
                    rows.append(current_row)
                    current_row = []

    # 添加最后一行（如果存在）
    if current_row:
        rows.append(current_row)

    # 转换为numpy数组
    matrix = np.array(rows)
    return matrix


def parse_vector_from_file(file_path):
    """
    从文件中解析向量数据（单列数据）
    """
    data = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过空行
            if not line:
                continue

            try:
                # 尝试将每行转换为浮点数
                value = float(line)
                data.append(value)
            except ValueError:
                print(f"警告: 无法解析行 '{line}'，已跳过")
                continue

    # 转换为numpy数组
    vector = np.array(data)
    return vector


def result_loss_compute(model, A_matrix, b_matrix, num_sample):
    # [up_bound;
    # low_bound;]
    # initialization
    error_feas = []
    error_opt = []
    A_hat = A_matrix
    b_hat = b_matrix
    model.A_hat = A_hat
    model.b_hat = b_hat
    model.update_polytope(model.A_hat, model.b_hat)
    for _ in range(num_sample):
        c = np.random.randn(model.dim)
        # 计算可行性误差
        x_apx = model.optimize_direction(c, in_approx=True)
        x_org = model.project(x_apx) if x_apx is not None else None
        if x_org is not None:
            error_feas.append(np.sum((x_apx - x_org) ** 2))
        # 计算最优性误差
        x_org = model.optimize_direction(c)
        x_apx = model.project(x_org, to_approx=True) if x_org is not None else None
        if x_apx is not None:
            error_opt.append(np.sum((x_apx - x_org) ** 2) / 2 * 10e-3)
    return error_feas, error_opt


# 使用示例
if __name__ == "__main__":
    # 解析文件
    error_path_a = f'{PROJECT_ROOT}\\data_analysis\\data\\error_iterate_varation\\loss_history_0.csv'
    error_path_b = f'{PROJECT_ROOT}\\data_analysis\\data\\error_iterate_varation\\loss_history_a.csv'
    error_path_c = f'{PROJECT_ROOT}\\data_analysis\\data\\error_iterate_varation\\loss_history_b.csv'
    error_path_d = f'{PROJECT_ROOT}\\data_analysis\\data\\error_iterate_varation\\loss_history_c.csv'
    # load data
    error_a = pd.read_csv(error_path_a)
    error_b = pd.read_csv(error_path_b)
    error_c = pd.read_csv(error_path_c)
    error_d = pd.read_csv(error_path_d)

    # 数据平滑参数配置
    smoothing_config = {
        'method': 'gaussian',  # 可选: 'ma', 'ema', 'gaussian', 'savgol'
        'sigma': 2,  # 高斯滤波器的sigma参数
        'window_size': 20,  # 移动平均窗口大小
        'alpha': 0.1,  # 指数移动平均的alpha参数
        'window_length': 51,  # Savitzky-Golay窗口长度
        'polyorder': 3  # Savitzky-Golay多项式阶数
    }

    # 提取原始数据
    feasibility_errors = []
    optimality_errors = []

    # 原始数据
    e_feas_a = error_a.iloc[:, 0].values
    e_feas_b = error_b.iloc[:, 0].values
    e_feas_c = error_c.iloc[:, 0].values
    e_feas_d = error_d.iloc[:, 0].values

    e_opt_a = error_a.iloc[:, 1].values
    e_opt_b = error_b.iloc[:, 1].values
    e_opt_c = error_c.iloc[:, 1].values
    e_opt_d = error_d.iloc[:, 1].values

    feasibility_errors_raw = [e_feas_a, e_feas_b, e_feas_c, e_feas_d]
    optimality_errors_raw = [e_opt_a, e_opt_b, e_opt_c, e_opt_d]

    # 应用平滑
    feasibility_errors_smoothed, optimality_errors_smoothed = apply_smoothing_to_errors(
        feasibility_errors_raw,
        optimality_errors_raw,
        method=smoothing_config['method'],
        sigma=smoothing_config['sigma'],
        window_size=smoothing_config['window_size'],
        alpha=smoothing_config['alpha'],
        window_length=smoothing_config['window_length'],
        polyorder=smoothing_config['polyorder']
    )

    # 确保数据长度一致
    min_length = min(len(feasibility_errors_smoothed[0]), len(optimality_errors_smoothed[0]), 1000)
    iterations = np.arange(1, min_length + 1)

    # 截断数据到相同长度
    feasibility_errors = [feas[:min_length] for feas in feasibility_errors_smoothed]
    optimality_errors = [opt[:min_length] for opt in optimality_errors_smoothed]

    # 更新图例名称
    labels = ['Scenario A', 'Scenario B', 'Scenario C', 'Scenario D']

    # 计算平滑前后对比
    print("数据平滑统计信息:")
    print(f"平滑方法: {smoothing_config['method']}")
    for i in range(4):
        print(f"\n{labels[i]}:")
        print(f"  可行性误差 - 原始标准差: {np.std(feasibility_errors_raw[i][:min_length]):.6f}")
        print(f"  可行性误差 - 平滑后标准差: {np.std(feasibility_errors_smoothed[i][:min_length]):.6f}")
        print(f"  最优性误差 - 原始标准差: {np.std(optimality_errors_raw[i][:min_length]):.6f}")
        print(f"  最优性误差 - 平滑后标准差: {np.std(optimality_errors_smoothed[i][:min_length]):.6f}")

    # ============================================
    # 修改部分：将两个子图分别保存为独立的PDF文件
    # ============================================

    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 28

    # 定义颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # ============================================
    # 子图1：可行性误差 - 独立保存为PDF
    # ============================================
    fig1, ax1 = plt.subplots(figsize=(16, 8))  # 单独一个图形

    # 绘制可行性误差
    for i in range(4):
        ax1.plot(iterations, feasibility_errors_smoothed[i][:min_length],
                 color=colors[i], linewidth=2.5, label=labels[i])

    # 设置坐标轴范围，确保0点位于左下角
    x_min1, x_max1 = 0, max(iterations) if len(iterations) > 0 else 1
    y_min1, y_max1 = 0, max(
        [max(feasibility_errors_smoothed[i][:min_length]) for i in range(4)]) if min_length > 0 else 1

    ax1.set_xlim(x_min1, x_max1)
    ax1.set_ylim(y_min1, y_max1 * 1.05)  # 增加5%的顶部空间

    ax1.set_xlabel('Iteration', fontname='Times New Roman', fontsize=28)
    ax1.set_ylabel("Feasibility Error"+" (MW)", fontname='Times New Roman', fontsize=28)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 添加图例
    ax1.legend(labels=labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=4, fontsize=28, frameon=False)

    plt.tight_layout()

    # 保存可行性误差图为独立的PDF文件
    plt.savefig('error_trends_feasibility.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig('error_trends_feasibility.png', format='png', bbox_inches='tight', dpi=300)
    plt.close(fig1)  # 关闭图形，释放内存
    print("可行性误差图已保存为独立的PDF文件: error_trends_feasibility.pdf")

    # ============================================
    # 子图2：最优性误差 - 独立保存为PDF
    # ============================================
    fig2, ax2 = plt.subplots(figsize=(16, 8))  # 单独一个图形

    # 绘制最优性误差
    for i in range(4):
        ax2.plot(iterations, optimality_errors_smoothed[i][:min_length],
                 color=colors[i], linewidth=2.5, label=labels[i])

    # 设置坐标轴范围，确保0点位于左下角
    x_min2, x_max2 = 0, max(iterations) if len(iterations) > 0 else 1
    y_min2, y_max2 = 0, max(
        [max(optimality_errors_smoothed[i][:min_length]) for i in range(4)]) if min_length > 0 else 1

    ax2.set_xlim(x_min2, x_max2)
    ax2.set_ylim(y_min2, y_max2 * 1.05)  # 增加5%的顶部空间

    ax2.set_xlabel('Iteration', fontname='Times New Roman', fontsize=28)
    ax2.set_ylabel("Optimality Error"+" (MW)", fontname='Times New Roman', fontsize=28)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # 添加图例
    ax2.legend(labels=labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=4, fontsize=28, frameon=False)

    plt.tight_layout()

    # 保存最优性误差图为独立的PDF文件
    plt.savefig('error_trends_optimality.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig('error_trends_optimality.png', format='png', bbox_inches='tight', dpi=300)
    plt.close(fig2)  # 关闭图形，释放内存
    print("最优性误差图已保存为独立的PDF文件: error_trends_optimality.pdf")

    print("\n所有图像文件已保存完成！")
    # 不显示图形，直接结束
    # plt.show()  # 如果需要显示，可以取消注释
