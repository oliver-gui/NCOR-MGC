import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib
from matplotlib import font_manager
import random
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
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Circle
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde

# 设置Times New Roman字体并增大字体尺寸
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 30  # 从26增大到30
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
plt.rcParams['axes.linewidth'] = 2  # 坐标轴线宽


def case_33(total_samples=200, noise_scale=0.15, batch_size=1, model_type='pretrainnet', device='cpu'):
    """固定参数的原始案例实现"""
    # data input
    bus_data = pd.read_csv(r'D:\work space\论文-working\share\case_data\bus-case33.csv')
    branch_data = pd.read_csv(r'D:\work space\论文-working\share\case_data\branch-case33.csv')
    # 固定初始化参数
    node_total = []
    for it in range(bus_data.shape[0]):
        node_total.append(bus_data.iloc[it, 0])
    node_control = [21, 22, 23, 24, 2, 3, 4, 5, 25, 26, 27, 16, 17]
    branch_total = []
    for it in range(branch_data.shape[0]):
        branch_total.append((branch_data.iloc[it, 0], branch_data.iloc[it, 1]))
    # 线路参数 (R, X)
    line_params = input_branch_data_to_line_params(branch_data)
    # 节点负荷 (P, Q)
    load_params = input_bus_data_to_load_params(bus_data)
    p_bound = {
        21: {'lb': -0.55, 'ub': 0.51},
        22: {'lb': -0.61, 'ub': 0.29},
        2: {'lb': -0.74, 'ub': 0.30},
        25: {'lb': -0.48, 'ub': 0.56},
        26: {'lb': -0.65, 'ub': 0.33},
        27: {'lb': -0.33, 'ub': 0.66},
        23: {'lb': -0.31, 'ub': 0.72},
        24: {'lb': -0.37, 'ub': 0.67},
        3: {'lb': -0.34, 'ub': 0.68},
        4: {'lb': -0.32, 'ub': 0.71},
        5: {'lb': -0.61, 'ub': 0.54},
        16: {'lb': -0.48, 'ub': 0.52},
        17: {'lb': -0.47, 'ub': 0.48},
    }
    DSO_base_P = np.zeros(len(p_bound))  # 后续替换为函数求解
    buffer = 0
    for it in p_bound:
        DSO_base_P[buffer] = bus_data.iloc[it, 2]
        buffer += 1
    buffer = 0
    # 创建模型
    model = ConcreteModel()

    # 节点集合 (0: 平衡节点, 1,2,3: PQ节点)
    model.NODES = Set(initialize=node_total)
    model.LINES = Set(initialize=branch_total)
    model.node_control = Set(initialize=node_control)

    model.load_p = Var(model.NODES)
    model.load_q = Var(model.NODES)
    model.P_var = Var(model.node_control, bounds=(-1.5, 1.5))
    # 电压上下限
    V_min, V_max = 0.90, 1.1

    # 定义变量
    # 节点电压平方
    model.V_sq = Var(model.NODES, bounds=(V_min ** 2, V_max ** 2))
    # 线路有功功率
    model.P = Var(model.LINES)
    # 线路无功功率
    model.Q = Var(model.LINES)
    # 平衡节点注入功率
    model.P0 = Var(initialize=0)
    model.Q0 = Var(initialize=0)
    # 求解辅助变量
    model.var_proj = Var(range(len(node_control)))

    def voltage_balance_rule(model, i, j):
        """电压平衡方程: V_j^2 = V_i^2 - 2*(R_ij*P_ij + X_ij*Q_ij)"""
        r_ij = line_params[(i, j)]['r']
        x_ij = line_params[(i, j)]['x']
        return model.V_sq[j] == model.V_sq[i] - 2 * (r_ij * model.P[i, j] + x_ij * model.Q[i, j])

    def loads_p_rule(model, i):
        if i in model.node_control:
            return model.load_p[i] == model.P_var[i]
        else:
            return model.load_p[i] == load_params[i]['p']

    def loads_q_rule(model, i):
        return model.load_q[i] == load_params[i]['q']

    def p_bounds_rule_low_bound(model, i):
        return model.P_var[i] >= p_bound[i]['lb']

    def p_bounds_rule_up_bound(model, i):
        return model.P_var[i] <= p_bound[i]['ub']

    def power_balance_p_rule(model, i):
        """节点有功功率平衡 - 简化版本"""
        if i == 0:  # 平衡节点
            # 平衡节点注入等于流出到第一条线路的功率
            outgoing_lines = [line for line in model.LINES if line[0] == i]
            if outgoing_lines:
                return model.P0 == sum(model.P[line] for line in outgoing_lines)
            else:
                return model.P0 == 0
        else:
            # 找到流入该节点的线路
            incoming_lines = [line for line in model.LINES if line[1] == i]
            # 找到从该节点流出的线路
            outgoing_lines = [line for line in model.LINES if line[0] == i]

            power_in = sum(model.P[line] for line in incoming_lines)
            power_out = sum(model.P[line] for line in outgoing_lines)

            return power_in == power_out + model.load_p[i]

    def power_balance_q_rule(model, i):
        """节点无功功率平衡 - 简化版本"""
        if i == 0:  # 平衡节点
            outgoing_lines = [line for line in model.LINES if line[0] == i]
            if outgoing_lines:
                return model.Q0 == sum(model.Q[line] for line in outgoing_lines)
            else:
                return model.Q0 == 0
        else:
            incoming_lines = [line for line in model.LINES if line[1] == i]
            outgoing_lines = [line for line in model.LINES if line[0] == i]

            power_in = sum(model.Q[line] for line in incoming_lines)
            power_out = sum(model.Q[line] for line in outgoing_lines)

            return power_in == power_out + model.load_q[i]

    def slack_bus_voltage_rule(model):
        """平衡节点电压约束"""
        return model.V_sq[0] == 1.0  # V0 = 1.0 pu

    def model_var_transfer(model, i):
        return model.var_proj[i] == model.P_var[node_control[i]]

    model.voltage_balance = Constraint(model.LINES, rule=voltage_balance_rule)
    model.loads_p = Constraint(model.NODES, rule=loads_p_rule)
    model.loads_q = Constraint(model.NODES, rule=loads_q_rule)
    model.power_balance_p = Constraint(model.NODES, rule=power_balance_p_rule)
    model.power_balance_q = Constraint(model.NODES, rule=power_balance_q_rule)
    model.slack_bus_voltage = Constraint(rule=slack_bus_voltage_rule)
    model.model_var_transfer = Constraint(range(len(model.node_control)), rule=model_var_transfer)
    model.p_var_up_bound = Constraint(model.node_control, rule=p_bounds_rule_up_bound)
    model.p_var_low_bound = Constraint(model.node_control, rule=p_bounds_rule_low_bound)

    # base_point define
    def base_point_dict(point):
        out_dict = {}
        for i in range(len(node_control)):
            out_dict[i] = point[i]
        return out_dict

    base_point = base_point_dict(DSO_base_P)
    model.base_point = Param(range(len(model.node_control)), initialize=base_point, mutable=False)

    original_model = {
        'model': model,
        # 'baseline': baseline,
    }
    dim = len(model.node_control)

    # 数据集配置
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        # def __getitem__(self, idx):
        # return {'theta':torch.normal(0, self.noise_scale, (dim_theta,),device=device)}  # theta维度固定为2

    # DG1 Boundary define
    DG_1 = np.array([[1], [-1]])
    random_vector = np.zeros([2, DG_1.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i, j] ** 2 for j in range(random_vector.shape[1])))
        random_vector[i, :] = random_vector[i, :] / v_norm
    DG_1_randn = np.vstack([DG_1, random_vector])
    # DG2 Boundary define
    DG_2 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 1], [-1, -1, -1]])
    random_vector = np.zeros([8, DG_2.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i, j] ** 2 for j in range(random_vector.shape[1])))
        random_vector[i, :] = random_vector[i, :] / v_norm
    DG_2_randn = np.vstack([DG_2, random_vector])
    # DG3 Boundary define
    DG_3 = np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, -1, 0],
                     [0, 0, 0, 1], [0, 0, 0, -1], [1, 1, 1, 1], [-1, -1, -1, -1]])
    random_vector = np.zeros([20, DG_3.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i, j] ** 2 for j in range(random_vector.shape[1])))
        random_vector[i, :] = random_vector[i, :] / v_norm
    DG_3_randn = np.vstack([DG_3, random_vector])
    # DG3 Boundary define
    DG_4 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [1, 1, 1], [-1, -1, -1]])
    random_vector = np.zeros([8, DG_4.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i, j] ** 2 for j in range(random_vector.shape[1])))
        random_vector[i, :] = random_vector[i, :] / v_norm
    DG_4_randn = np.vstack([DG_4, random_vector])
    # DG5 Boundary define
    DG_5 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]])
    random_vector = np.zeros([6, DG_5.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i, j] ** 2 for j in range(random_vector.shape[1])))
        random_vector[i, :] = random_vector[i, :] / v_norm
    DG_5_randn = np.vstack([DG_5, random_vector])

    A_hat = DG_boundary_to_A_hat([
        DG_1,  # DG1
        DG_2_randn,  # DG2
        DG_3_randn,  # DG3
        DG_4_randn,  # DG4
        DG_5_randn,  # DG4
    ])

    A_hat1 = DG_boundary_to_A_hat([
        DG_1,  # DG1
        DG_2,  # DG2
        DG_3,  # DG3
        DG_4  # DG4
    ])
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='ipopt',
    )
    A_list = [errorcalculator.A_hat]
    b_list = [errorcalculator.b_hat]

    case_name = 'case33_no_bp'
    # case_name = 'case33'
    visualizer = ErrorVisualizer()
    num_sample = 50

    violin_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\violin_plot'
    box_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\box_plot'
    result_folder = f'{PROJECT_ROOT}\\results\\{case_name}'
    kde_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\kde_plot'

    def callback(error_calculator, epoch):
        if not hasattr(callback, "idx_mark"):
            callback.idx_mark = 0  # 初始化计数器
        len_his = len(error_calculator.training_history['feas'])
        print(f"Iter {epoch}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
              f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")
        error_record = visualizer.compute_errors(error_calculator, num_sample=num_sample)
        print((np.mean(visualizer.error_history['error_feas'][-1]), np.mean(visualizer.error_history['error_opt'][-1])))

        # visualization
        # visualizer.plot_dual_violin(save_path=f"{violin_folder}/step{epoch}")
        # visualizer.plot_dual_boxplot(save_path=f"{box_folder}/step{epoch}")
        visualizer.plot_kde_evolution(save_path=f"{kde_folder}/step{epoch}")

        # A,b output
        output_file_A = f"{result_folder}/step{epoch}_A.txt"
        with open(output_file_A, "w") as f:
            for it in range(len(error_calculator.A_hat)):
                f.write(f"{error_calculator.A_hat[it]}\n")

        output_file_b = f"{result_folder}/step{epoch}_b.txt"
        with open(output_file_b, "w") as f:
            for it in range(len(error_calculator.b_hat)):
                f.write(f"{error_calculator.b_hat[it]}\n")

    trainer_configure = {
        "call_interval": 5,
        "training_callback": callback,
        "optimizer": "SGD",
        "lr": 0.25,
        "batch_size": 1,
        "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.95},
        "n_cal": 5,
        "cal_feas": True,
        "cal_opt": True,
        "rate_opt_feas": 1
    }

    params_dict, param_count = pyomo_params_to_numpy(model)
    params = {  # 名字，初值，误差数据集
        'params_dict': params_dict,
        'dataloader': DataLoader(
            CaseData(),
            batch_size=batch_size,
            shuffle=True
        ),
        'count': param_count,
    }
    return {
        'casename': case_name,
        'A_hat': A_hat,
        'b_hat': errorcalculator.b_hat,
        'errorcalculator': errorcalculator,
        'trainer_configure': trainer_configure,
        'params': params,
        'result_path': f'{PROJECT_ROOT}/results/{case_name}/{model_type.lower()}_weights.pth',
        'metadata': {
            'dim': dim
        }
    }


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


def A_hat_diversation(A_hat, divide_mark):
    A_divided = []
    buffer_r = 0
    buffer_c = 0
    for it in range(len(divide_mark)):
        A_divided.append(A_hat[buffer_r:buffer_r + divide_mark[it][0], buffer_c:buffer_c + divide_mark[it][1]])
        buffer_r += divide_mark[it][0]
        buffer_c += divide_mark[it][1]
    return A_divided


def DG_boundary_to_A_hat(DG_boundary):
    A_hat_r = sum(DG_boundary[it].shape[0] for it in range(len(DG_boundary)))
    A_hat_c = sum(DG_boundary[it].shape[1] for it in range(len(DG_boundary)))
    A_hat = np.zeros([A_hat_r, A_hat_c])
    buffer_r = 0
    buffer_c = 0
    for it in range(len(DG_boundary)):
        A_hat[buffer_r:buffer_r + DG_boundary[it].shape[0],
        buffer_c:buffer_c + DG_boundary[it].shape[1]] = DG_boundary[it]
        buffer_r += DG_boundary[it].shape[0]
        buffer_c += DG_boundary[it].shape[1]
    return A_hat


def input_branch_data_to_line_params(input_branch_data):
    line_params = {}
    for it in range(input_branch_data.shape[0]):
        index_0 = input_branch_data.iloc[it, 0]
        index_1 = input_branch_data.iloc[it, 1]
        r = input_branch_data.iloc[it, 2] / 80
        x = input_branch_data.iloc[it, 3] / 80
        line_params[(index_0, index_1)] = {'r': r, 'x': x}
    return line_params


def input_bus_data_to_load_params(input_bus_data):
    load_params = {}
    for it in range(input_bus_data.shape[0]):
        index_0 = input_bus_data.iloc[it, 0]
        p = input_bus_data.iloc[it, 2]
        q = input_bus_data.iloc[it, 3]
        load_params[index_0] = {'p': p, 'q': q}
    return load_params


def compute_density_difference_map(data1, data2, labels=['PIML with RP', 'PIML without RP'],
                                   resolution=100, figsize=(16, 14), dpi=300):
    """
    计算并绘制两组数据的密度差异图
    修改版本：去掉黑色轮廓线，取消坐标轴标签加粗，增大字体
    """
    # 创建图形，增大图形尺寸
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    # 设置Times New Roman字体并增大尺寸
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 32  # 进一步增大
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.linewidth'] = 2.5  # 稍微加粗坐标轴线

    # 计算KDE密度估计
    kde1 = gaussian_kde(data1.T)
    kde2 = gaussian_kde(data2.T)

    # 确定绘图范围（包含所有数据）
    all_data = np.vstack([data1, data2])
    x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
    y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()

    # 扩展范围(10%)
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min, x_max = x_min - 0.1 * x_range, x_max + 0.1 * x_range
    y_min, y_max = y_min - 0.1 * y_range, y_max + 0.1 * y_range

    # 创建网格
    xi, yi = np.mgrid[x_min:x_max:resolution * 1j, y_min:y_max:resolution * 1j]
    grid_coords = np.vstack([xi.flatten(), yi.flatten()])

    # 计算网格点上的密度值
    zi1 = kde1(grid_coords).reshape(xi.shape)
    zi2 = kde2(grid_coords).reshape(xi.shape)

    # 计算密度差异 (Group A - Group B)
    zi_diff = zi1 - zi2

    # 归一化差异用于更好的颜色映射
    max_abs_diff = np.max(np.abs(zi_diff))
    if max_abs_diff > 0:
        zi_diff_normalized = zi_diff / max_abs_diff
    else:
        zi_diff_normalized = zi_diff

    # 创建自定义的Red-Blue颜色映射
    colors = [
        (0, 0, 0.5),  # 深蓝 (PIML without RP密度显著高)
        (0.2, 0.2, 0.8),  # 蓝色
        (0.8, 0.8, 1),  # 浅蓝
        (1, 1, 1),  # 白色 (平衡区域)
        (1, 0.8, 0.8),  # 浅红
        (0.8, 0.2, 0.2),  # 红色
        (0.5, 0, 0)  # 深红 (PIML with RP密度显著高)
    ]

    positions = [0, 0.25, 0.4, 0.5, 0.6, 0.75, 1]
    cmap_diff = matplotlib.colors.LinearSegmentedColormap.from_list(
        'custom_redblue', list(zip(positions, colors)))

    # 绘制密度差异填色图
    contour_levels = np.linspace(-max_abs_diff, max_abs_diff, 21)
    contourf = ax.contourf(xi, yi, zi_diff,
                           levels=contour_levels,
                           cmap=cmap_diff,
                           alpha=0.85,
                           extend='both')

    # 绘制原始数据点（增大点的大小）
    scatter1 = ax.scatter(data1[:, 0], data1[:, 1],
                          color='blue',
                          s=60,  # 从30增大到60
                          alpha=0.35,
                          edgecolors='darkblue',
                          linewidth=1.0,
                          label=labels[0],
                          zorder=3)

    scatter2 = ax.scatter(data2[:, 0], data2[:, 1],
                          color='red',
                          s=60,  # 从30增大到60
                          alpha=0.35,
                          edgecolors='darkred',
                          linewidth=1.0,
                          label=labels[1],
                          zorder=3)

    # 设置坐标轴标签（取消加粗）
    ax.set_xlabel('Feasibility Error (MW)', fontsize=36, fontweight='normal', fontname='Times New Roman')
    ax.set_ylabel('Optimality Error (MW)', fontsize=36, fontweight='normal', fontname='Times New Roman')

    # 设置坐标轴刻度字体大小（增大）
    ax.tick_params(axis='both', which='major', labelsize=32, width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', labelsize=28, width=2, length=8)

    # 添加网格
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1.5)

    # 添加图例（增大字体和点的大小）
    legend = ax.legend(loc='upper right', fontsize=30, frameon=True,
                       fancybox=True, framealpha=0.9, edgecolor='black',
                       markerscale=1.5)  # 增大图例中的标记大小
    legend.get_frame().set_linewidth(2.5)

    # 添加颜色条（增大字体）
    cbar = plt.colorbar(contourf, ax=ax, pad=0.03)
    cbar.ax.tick_params(labelsize=28, width=2.5, length=10)
    cbar.set_ticks([contour_levels.min(), 0, contour_levels.max()])
    cbar.set_ticklabels(['PIML without RP\nHigher', 'Equal', 'PIML with RP\nHigher'])

    # 设置颜色条刻度标签字体
    for label in cbar.ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(26)

    # 设置图形布局
    plt.tight_layout()

    # 返回图形对象和数据
    result = {
        'figure': fig,
        'axis': ax,
        'xi': xi,
        'yi': yi,
        'density_diff': zi_diff,
        'density_groupA': zi1,
        'density_groupB': zi2,
        'contourf': contourf,
    }

    return result


def create_simplified_density_map(data1, data2, labels=['PIML with RP', 'PIML without RP']):
    """
    创建一个简化的密度差异图（适合论文使用）
    修改版本：去掉黑色轮廓线，调整字体和标签
    """
    # 确保数据是二维数组
    if isinstance(data1, tuple) and len(data1) == 2:
        # 如果是元组，将其组合成二维数组
        data1 = np.column_stack(data1)
    if isinstance(data2, tuple) and len(data2) == 2:
        data2 = np.column_stack(data2)

    # 设置Times New Roman字体并增大尺寸
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 32
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.linewidth'] = 2.5

    # 增大图形尺寸
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), dpi=300)

    # 计算KDE
    kde1 = gaussian_kde(data1.T)
    kde2 = gaussian_kde(data2.T)

    # 确定共同范围
    all_data = np.vstack([data1, data2])
    x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
    y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min, x_max = x_min - 0.1 * x_range, x_max + 0.1 * x_range
    y_min, y_max = y_min - 0.1 * y_range, y_max + 0.1 * y_range

    # 创建网格
    xi, yi = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    grid_coords = np.vstack([xi.flatten(), yi.flatten()])

    # 计算密度
    zi1 = kde1(grid_coords).reshape(xi.shape)
    zi2 = kde2(grid_coords).reshape(xi.shape)
    zi_diff = zi1 - zi2

    # 子图1：密度差异填色图
    im1 = ax1.contourf(xi, yi, zi_diff, levels=20, cmap='RdBu_r', alpha=0.85)

    # 绘制数据点（增大点的大小）
    ax1.scatter(data1[:, 0], data1[:, 1], color='blue', s=40, alpha=0.45,
                edgecolors='darkblue', linewidth=1.0)
    ax1.scatter(data2[:, 0], data2[:, 1], color='red', s=40, alpha=0.45,
                edgecolors='darkred', linewidth=1.0)

    # 设置坐标轴标签（取消加粗）
    ax1.set_xlabel('Feasibility Error (MW)', fontsize=36, fontweight='normal', fontname='Times New Roman')
    ax1.set_ylabel('Optimality Error (MW)', fontsize=36, fontweight='normal', fontname='Times New Roman')
    ax1.set_title('Density Difference Map', fontsize=38, fontweight='normal', fontname='Times New Roman')
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=1.5)
    ax1.tick_params(labelsize=32, width=2.5, length=10)

    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.03)
    cbar1.set_label('Density Difference', fontsize=32, fontweight='normal', fontname='Times New Roman')
    cbar1.ax.tick_params(labelsize=28)

    # 子图2：单独组别的密度
    im2 = ax2.contourf(xi, yi, zi1, levels=20, cmap='Blues', alpha=0.65, label=labels[0])
    im3 = ax2.contourf(xi, yi, zi2, levels=20, cmap='Reds', alpha=0.65, label=labels[1])

    # 绘制数据点（增大点的大小）
    ax2.scatter(data1[:, 0], data1[:, 1], color='blue', s=50, alpha=0.5,
                edgecolors='darkblue', linewidth=1.2, label=labels[0])
    ax2.scatter(data2[:, 0], data2[:, 1], color='red', s=50, alpha=0.5,
                edgecolors='darkred', linewidth=1.2, label=labels[1])

    ax2.set_xlabel('Feasibility Error (MW)', fontsize=36, fontweight='normal', fontname='Times New Roman')
    ax2.set_ylabel('Optimality Error (MW)', fontsize=36, fontweight='normal', fontname='Times New Roman')
    ax2.set_title('Individual Group Densities', fontsize=38, fontweight='normal', fontname='Times New Roman')
    ax2.legend(fontsize=30, loc='upper right', markerscale=1.5)
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=1.5)
    ax2.tick_params(labelsize=32, width=2.5, length=10)

    plt.tight_layout()

    return fig


def compact_2d_normal(mean_x, mean_y, compactness=0.1, n_points=500, apply_abs=True):
    """
    生成紧凑的二维正态分布，可选择对所有数据点的x和y取绝对值

    参数:
    mean_x, mean_y: 分布的均值中心
    compactness: 紧凑度（方差值），越小越紧凑
    n_points: 生成的点数
    apply_abs: 是否对x,y取绝对值

    返回:
    x_list, y_list: 两个列表
    """
    # 生成二维正态分布数据
    data = np.random.multivariate_normal(
        mean=[2, 2],
        cov=[[compactness, 0], [0, compactness]],
        size=n_points
    )

    # 如果需要对x和y取绝对值
    if apply_abs:
        data = np.abs(data)

    # 转换为列表
    x_list = (data[:, 0] * mean_x).tolist()
    y_list = (data[:, 1] * mean_y).tolist()

    return x_list, y_list


def compact_2d_normal_nobp(mean_x, mean_y, compactness=0.1, n_points=500, apply_abs=True):
    """
    生成紧凑的二维正态分布，可选择对所有数据点的x和y取绝对值

    参数:
    mean_x, mean_y: 分布的均值中心
    compactness: 紧凑度（方差值），越小越紧凑
    n_points: 生成的点数
    apply_abs: 是否对x,y取绝对值

    返回:
    x_list, y_list: 两个列表
    """
    # 生成二维正态分布数据
    data = np.random.multivariate_normal(
        mean=[1, 1],
        cov=[[compactness, 0], [0, compactness]],
        size=n_points
    )

    # 如果需要对x和y取绝对值
    if apply_abs:
        data = np.abs(data)

    # 转换为列表
    x_list = (data[:, 0] * mean_x).tolist()
    y_list = (data[:, 1] * mean_y).tolist()

    return x_list, y_list


# 主程序
if __name__ == "__main__":
    # declare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parallel = False
    model_type = 'pretrainnet'
    case = case_33(model_type=model_type, device=device)
    n_train = 1000
    divide_mark = [[2, 1], [16, 3], [30, 4], [16, 3], [12, 2]]
    divide_mark1 = [[10, 3]]
    A_hat_model = A_hat_diversation(case['A_hat'], divide_mark)
    model = PreTrainNet(A_hat_model, case['b_hat'], device=device).to(device)
    trainer = Trainer(
        model=model,
        error_calculator=case['errorcalculator'],
        compute_loss=compute_loss,
    )
    trainer.configure(**case['trainer_configure'])
    trainer.initialize()
    # 计算数据
    df1 = pd.read_csv("efeas_bp.csv")
    df2 = pd.read_csv("efeas_nobp.csv")
    df3 = pd.read_csv("eopt_bp.csv")
    df4 = pd.read_csv("eopt_nobp.csv")
    print("Generating sample data...")
    [efeas_bp, eopt_bp] = [df1, df3]
    [efeas_nobp, eopt_nobp] = [df2, df4]
    # 将误差数据转换为二维数组用于密度图
    # 第一维：可行性误差，第二维：最优性误差
    data_bp = np.column_stack([efeas_bp, eopt_bp])
    data_nobp = np.column_stack([efeas_nobp, eopt_nobp])

    # 移除异常值（如果有的话）
    data_bp = data_bp[~np.any(np.isinf(data_bp) | np.isnan(data_bp), axis=1)]
    data_nobp = data_nobp[~np.any(np.isinf(data_nobp) | np.isnan(data_nobp), axis=1)]

    # 创建完整的密度差异图（使用更大尺寸）
    print("Creating density difference map...")
    result = compute_density_difference_map(
        data_bp, data_nobp,
        labels=['PIML with RP', 'PIML without RP'],
        figsize=(14, 10),  # 进一步增大图形尺寸
        dpi=300
    )

    # 创建简化版本（适合论文）
    print("Creating simplified version for publication...")
    fig_simple = create_simplified_density_map(
        data_bp, data_nobp,
        labels=['PIML with RP', 'PIML without RP']
    )

    # 保存图形
    print("Saving figures...")
    result['figure'].savefig('error_density_comparison_detailed.pdf',
                             dpi=300, bbox_inches='tight', facecolor='white')
    result['figure'].savefig('error_density_comparison_detailed.png',
                             dpi=300, bbox_inches='tight', facecolor='white')
    fig_simple.savefig('error_density_comparison_simple.png',
                       dpi=300, bbox_inches='tight', facecolor='white')

    print("\nFigures saved as:")
    print("1. error_density_comparison_detailed.pdf (vector format)")
    print("2. error_density_comparison_detailed.png (raster format)")
    print("3. error_density_comparison_simple.png")

    # 显示图形
    plt.show()

    # 打印统计信息
    print("\nStatistical Summary:")
    print(f"PIML with RP: {len(data_bp)} samples")
    print(f"PIML without RP: {len(data_nobp)} samples")

    print("\nPIML with RP Error Statistics:")
    print(f"  Mean feasibility error: {np.mean(efeas_bp):.6f}")
    print(f"  Mean optimality error: {np.mean(eopt_bp):.6f}")
    print(f"  Max feasibility error: {np.max(efeas_bp):.6f}")
    print(f"  Max optimality error: {np.max(eopt_bp):.6f}")

    print("\nPIML without RP Error Statistics:")
    print(f"  Mean feasibility error: {np.mean(efeas_nobp):.6f}")
    print(f"  Mean optimality error: {np.mean(eopt_nobp):.6f}")
    print(f"  Max feasibility error: {np.max(efeas_nobp):.6f}")
    print(f"  Max optimality error: {np.max(eopt_nobp):.6f}")