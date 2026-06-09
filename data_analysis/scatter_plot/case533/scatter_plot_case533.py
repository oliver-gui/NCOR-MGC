"""
基于已有误差 CSV 数据生成 Case533 密度差异图 / 散点图。

该文件直接读取 scatter_plot_case533.py 生成的误差 CSV 文件作为输入，
不包含模型创建、矩阵解析、误差计算或随机扰动等数据处理逻辑。
"""

import os
import sys

# 将项目根目录加入 sys.path，确保能导入 Simulator 等模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd

from Simulator import PROJECT_ROOT

##===================== 模型创建与数据处理 =====================
def case_533(total_samples=200, noise_scale=0.15, batch_size=1, model_type='pretrainnet', device='cpu'):
    """
    Case 533: 533-bus distribution system from Malmer & Thorin (2023).
    112 controllable nodes (microgrids) grouped into 15 microgrid clusters.
    Clusters are strictly sequentially ordered by node ID.

    Reference: G. Malmer & L. Thorin, "Network reconfiguration for renewable
    generation maximization", Master's thesis, Lund University, 2023.
    """
    # ========================
    # data input
    # ========================
    bus_data = pd.read_csv(r'D:\work space\论文-working\share\case_data\bus-case533.csv', header=None)
    branch_data = pd.read_csv(r'D:\work space\论文-working\share\case_data\branch-case533.csv', header=None)

    # ========================
    # Node and branch initialization
    # ========================
    node_total = []
    for it in range(bus_data.shape[0]):
        node_total.append(bus_data.iloc[it, 0])

    # 112 controllable nodes in 15 clusters (strictly sequential by node ID)
    node_control = [
        # Cluster 1 (9 nodes)
        7,8,13,14,15,16,17,18,20,
        # Cluster 2 (4 nodes)
        23,30,32,33,
        # Cluster 3 (8 nodes)
        42,43,57,60,61,66,67,69,
        # Cluster 4 (5 nodes)
        73,74,78,89,97,
        # Cluster 5 (10 nodes)
        125,145,167,169,170,173,174,175,176,177,
        # Cluster 6 (7 nodes)
        203,205,211,225,226,227,228,
        # Cluster 7 (8 nodes)
        230,237,238,239,243,245,247,259,
        # Cluster 8 (7 nodes)
        269,270,272,274,288,289,292,
        # Cluster 9 (14 nodes)
        295,300,301,306,309,314,315,317,318,319,
        320,321,322,325,
        # Cluster 10 (5 nodes)
        343,369,372,376,377,
        # Cluster 11 (5 nodes)
        391,421,422,423,424,
        # Cluster 12 (5 nodes)
        437,442,443,444,445,
        # Cluster 13 (4 nodes)
        454,461,470,471,
        # Cluster 14 (9 nodes)
        477,478,479,480,485,486,487,489,504,
        # Cluster 15 (12 nodes)
        507,508,509,510,511,512,525,526,527,529,
        530,531,
    ]

    branch_total = []
    for it in range(branch_data.shape[0]):
        branch_total.append((branch_data.iloc[it, 0], branch_data.iloc[it, 1]))

    # ========================
    # Line parameters (R, X) - values are already in p.u. from MATPOWER
    # ========================
    line_params = input_branch_data_to_line_params_533(branch_data)

    # ========================
    # Load parameters (P, Q)
    # ========================
    load_params = input_bus_data_to_load_params(bus_data)

    # ========================
    # P bounds for each controllable node
    # ========================
    p_bound = {}
    for node in node_control:
        pd_abs = abs(load_params[node]['p'])
        scale = max(0.5, min(2.0, pd_abs * 30))
        p_bound[node] = {'lb': -1.0 * scale /4, 'ub': 1.0 * scale /4}

    DSO_base_P = np.zeros(len(node_control))
    for idx, node in enumerate(node_control):
        DSO_base_P[idx] = load_params[node]['p'] * 0.75

    # ========================
    # Create Pyomo model
    # ========================
    model = ConcreteModel()

    model.NODES = Set(initialize=node_total)
    model.LINES = Set(initialize=branch_total)
    model.node_control = Set(initialize=node_control)

    model.load_p = Var(model.NODES)
    model.load_q = Var(model.NODES)
    model.P_var = Var(model.node_control, bounds=(-1.5, 1.5))

    V_min, V_max = 0.9, 1.1

    model.V_sq = Var(model.NODES, bounds=(V_min ** 2, V_max ** 2))
    model.P = Var(model.LINES)
    model.Q = Var(model.LINES)
    model.P0 = Var(initialize=0)
    model.Q0 = Var(initialize=0)
    model.var_proj = Var(range(len(node_control)))

    # ========================
    # Constraints
    # ========================
    def voltage_balance_rule(model, i, j):
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
        if i == 1:  # Slack bus (bus 1 in 533-bus system)
            outgoing_lines = [line for line in model.LINES if line[0] == i]
            if outgoing_lines:
                return model.P0 == sum(model.P[line] for line in outgoing_lines)
            else:
                return model.P0 == 0
        else:
            incoming_lines = [line for line in model.LINES if line[1] == i]
            outgoing_lines = [line for line in model.LINES if line[0] == i]
            power_in = sum(model.P[line] for line in incoming_lines)
            power_out = sum(model.P[line] for line in outgoing_lines)
            return power_in == power_out + model.load_p[i]

    def power_balance_q_rule(model, i):
        if i == 1:  # Slack bus (bus 1 in 533-bus system)
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
        return model.V_sq[1] == 1.0

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

    # Base point
    def base_point_dict(point):
        out_dict = {}
        for i in range(len(node_control)):
            out_dict[i] = point[i]
        return out_dict
    base_point = base_point_dict(DSO_base_P)
    model.base_point = Param(range(len(model.node_control)), initialize=base_point, mutable=False)

    original_model = {'model': model}
    dim = len(model.node_control)

    # Dataset
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale
        def __len__(self):
            return self.size

    # ========================
    # DG Boundary Definitions (15 clusters)
    # Cluster dims: [9, 4, 8, 5, 10, 7, 8, 7, 14, 5, 5, 5, 4, 9, 12]
    # ========================

    def make_dg_boundary(dim):
        """Create a DG boundary matrix for a cluster of given dimension."""
        DG = np.zeros([2 * dim + 2, dim])
        for i in range(dim):
            DG[2 * i, i] = 1
            DG[2 * i + 1, i] = -1
        DG[2 * dim, :] = 1
        DG[2 * dim + 1, :] = -1
        n_rand = 4 * dim
        rand = np.zeros([n_rand, dim])
        for i in range(n_rand):
            rand[i] = np.random.randn(dim)
            v_norm = np.sqrt(np.sum(rand[i] ** 2))
            rand[i, :] = rand[i, :] / v_norm
        return DG, np.vstack([DG, rand])

    DG_list = []
    DG_randn_list = []
    for dim in [9, 4, 8, 5, 10, 7, 8, 7, 14, 5, 5, 5, 4, 9, 12]:
        dg, dg_randn = make_dg_boundary(dim)
        DG_list.append(dg)
        DG_randn_list.append(dg_randn)

    A_hat = DG_boundary_to_A_hat(DG_randn_list)
    A_hat1 = DG_boundary_to_A_hat(DG_list)

    # ========================
    # Error calculator
    # ========================
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='ipopt',
    )

    case_name = 'case533'
    visualizer = ErrorVisualizer()
    num_sample = 80

    violin_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\violin_plot'
    box_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\box_plot'
    result_folder = f'{PROJECT_ROOT}\\results\\{case_name}'

    os.makedirs(violin_folder, exist_ok=True)
    os.makedirs(box_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)

    # Training callback
    def callback(error_calculator, epoch):
        if not hasattr(callback, "idx_mark"):
            callback.idx_mark = 0
        visualizer.plot_dual_violin(save_path=f"{violin_folder}/step{epoch}")
        visualizer.plot_dual_boxplot(save_path=f"{box_folder}/step{epoch}")
        output_file_A = f"{result_folder}/step{epoch}_A.txt"
        with open(output_file_A, "w") as f:
            for it in range(len(error_calculator.A_hat)):
                f.write(f"{error_calculator.A_hat[it]}\n")
        output_file_b = f"{result_folder}/step{epoch}_b.txt"
        with open(output_file_b, "w") as f:
            for it in range(len(error_calculator.b_hat)):
                f.write(f"{error_calculator.b_hat[it]}\n")

    trainer_configure = {
        "call_interval": 100,
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
    params = {
        'params_dict': params_dict,
        'dataloader': DataLoader(CaseData(), batch_size=batch_size, shuffle=True),
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
        'metadata': {'dim': dim}
    }


def A_hat_diversation(A_hat, divide_mark):
    A_divided = []
    buffer_r = 0
    buffer_c = 0
    for it in range(len(divide_mark)):
        A_divided.append(A_hat[buffer_r:buffer_r + divide_mark[it][0],
                               buffer_c:buffer_c + divide_mark[it][1]])
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


def input_branch_data_to_line_params_533(input_branch_data):
    """Branch r, x values are already in p.u. (from MATPOWER)."""
    line_params = {}
    for it in range(input_branch_data.shape[0]):
        index_0 = int(input_branch_data.iloc[it, 0])
        index_1 = int(input_branch_data.iloc[it, 1])
        r = input_branch_data.iloc[it, 2] / 1.0
        x = input_branch_data.iloc[it, 3] / 1.0
        line_params[(index_0, index_1)] = {'r': r, 'x': x}
    return line_params


def input_bus_data_to_load_params(input_bus_data):
    load_params = {}
    for it in range(input_bus_data.shape[0]):
        index_0 = int(input_bus_data.iloc[it, 0])
        p = input_bus_data.iloc[it, 2]
        q = input_bus_data.iloc[it, 3]
        load_params[index_0] = {'p': p, 'q': q}
    return load_params


def base_line_result_loss_compute(model, base_line_result, num_sample):
    error_feas = []
    error_opt = []
    A_hat_baseline = np.zeros([2 * base_line_result.shape[1], base_line_result.shape[1]])
    b_hat_baseline = np.zeros([2 * base_line_result.shape[1], 1])
    for it in range(base_line_result.shape[1]):
        A_hat_baseline[it * 2, it] = 1
        A_hat_baseline[it * 2 + 1, it] = -1
        b_hat_baseline[it * 2, 0] = base_line_result[0, it]
        b_hat_baseline[it * 2 + 1, 0] = -base_line_result[1, it]
    model.A_hat = A_hat_baseline
    model.b_hat = b_hat_baseline
    model.update_polytope(model.A_hat, model.b_hat)
    for _ in range(num_sample):
        c = np.random.randn(model.dim)
        x_apx = model.optimize_direction(c, in_approx=True)
        x_org = model.project(x_apx) if x_apx is not None else None
        if x_org is not None:
            error_feas.append(np.sum((x_apx - x_org) ** 2))
        x_org = model.optimize_direction(c)
        x_apx = model.project(x_org, to_approx=True) if x_org is not None else None
        if x_apx is not None:
            error_opt.append(np.sum((x_apx - x_org) ** 2))
    return error_feas, error_opt



# ===================== 字体与样式设置 =====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 30
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 2


# ===================== 密度差异图 (来自 scatter_plot_case533.py) =====================
def compute_density_difference_map(data1, data2, labels=['PIML with RP', 'PIML without RP'],
                                   resolution=100, figsize=(14, 10), dpi=300):
    """
    计算并绘制两组数据的密度差异图
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 32
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.linewidth'] = 2.5

    # 计算KDE密度估计
    kde1 = gaussian_kde(data1.T)
    kde2 = gaussian_kde(data2.T)

    # 确定绘图范围（x 轴固定 [0, 8e-7]，y 轴固定 [0, 2.4e-2]）
    x_min, x_max = 0.0, 8e-7
    y_min, y_max = 0.0, 2.4e-2

    # 横轴4个刻度(除0外)，纵轴6个刻度(除0外)
    NX_TICKS = 4
    NY_TICKS = 6
    XTICK = x_max / NX_TICKS
    YTICK = y_max / NY_TICKS

    # 创建网格
    xi, yi = np.mgrid[x_min:x_max:resolution * 1j, y_min:y_max:resolution * 1j]
    grid_coords = np.vstack([xi.flatten(), yi.flatten()])

    # 计算网格点上的密度值
    zi1 = kde1(grid_coords).reshape(xi.shape)
    zi2 = kde2(grid_coords).reshape(xi.shape)

    # 计算密度差异 (Group A - Group B)
    zi_diff = zi1 - zi2

    # 计算最大绝对值差异用于颜色映射范围
    max_abs_diff = np.max(np.abs(zi_diff))

    # 创建自定义 Blue-Red 颜色映射
    # 负差异区 (no_RP 密度更高) → 蓝色系；正差异区 (RP 密度更高) → 红色系
    colors = [
        (0, 0, 0.5),        # 深蓝 (no_RP 密度显著高)
        (0.2, 0.2, 0.8),    # 蓝色
        (0.8, 0.8, 1),      # 浅蓝
        (1, 1, 1),          # 白色 (平衡区域)
        (1, 0.8, 0.8),      # 浅红
        (0.8, 0.2, 0.2),    # 红色
        (0.5, 0, 0),        # 深红 (RP 密度显著高)
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

    # 绘制数据点 — RP (data1) 用红色系，no_RP (data2) 用蓝色系
    ax.scatter(data1[:, 0], data1[:, 1],
               color='red',
               s=60, alpha=0.35,
               edgecolors='darkred',
               linewidth=1.0,
               label=labels[0],
               zorder=3)

    ax.scatter(data2[:, 0], data2[:, 1],
               color='blue',
               s=60, alpha=0.35,
               edgecolors='darkblue',
               linewidth=1.0,
               label=labels[1],
               zorder=3)

    ax.set_xlabel('Feasibility Error (p.u.)', fontsize=36, fontweight='normal', fontname='Times New Roman')
    ax.set_ylabel('Optimality Error (p.u.)', fontsize=36, fontweight='normal', fontname='Times New Roman')

    # 显式设置坐标轴范围，防止 MultipleLocator 在自动缩放范围内生成过多刻度
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 每个刻度显示科学计数法 (整数尾数不显示小数点和0)
    def fmt_sci(v, p):
        if v == 0:
            return '0'
        exp = int(np.floor(np.log10(abs(v))))
        mantissa = v / 10**exp
        if abs(mantissa - round(mantissa)) < 1e-9:
            return f'{mantissa:.0f}e{exp}'
        else:
            return f'{mantissa:.1f}e{exp}'
    sci_formatter = FuncFormatter(fmt_sci)
    ax.xaxis.set_major_formatter(sci_formatter)
    ax.yaxis.set_major_formatter(sci_formatter)
    ax.xaxis.set_major_locator(MultipleLocator(XTICK))
    ax.yaxis.set_major_locator(MultipleLocator(YTICK))

    ax.tick_params(axis='both', which='major', labelsize=32, width=2.5, length=10)
    ax.tick_params(axis='both', which='minor', labelsize=28, width=2, length=8)

    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1.5)

    legend = ax.legend(loc='upper right', fontsize=30, frameon=True,
                       fancybox=True, framealpha=0.9, edgecolor='black',
                       markerscale=1.5)
    legend.get_frame().set_linewidth(2.5)

    plt.tight_layout()

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
    创建简化的密度差异图（适合论文使用）
    """
    if isinstance(data1, tuple) and len(data1) == 2:
        data1 = np.column_stack(data1)
    if isinstance(data2, tuple) and len(data2) == 2:
        data2 = np.column_stack(data2)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 32
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.linewidth'] = 2.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), dpi=300)

    # 计算KDE
    kde1 = gaussian_kde(data1.T)
    kde2 = gaussian_kde(data2.T)

    # 确定共同范围（x 轴固定 [0, 8e-7]，y 轴固定 [0, 2.4e-2]）
    x_min, x_max = 0.0, 8e-7
    y_min, y_max = 0.0, 2.4e-2

    # 横轴4个刻度(除0外)，纵轴6个刻度(除0外)
    NX_TICKS = 4
    NY_TICKS = 6
    XTICK = x_max / NX_TICKS
    YTICK = y_max / NY_TICKS

    # 创建网格
    xi, yi = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    grid_coords = np.vstack([xi.flatten(), yi.flatten()])

    zi1 = kde1(grid_coords).reshape(xi.shape)
    zi2 = kde2(grid_coords).reshape(xi.shape)
    zi_diff = zi1 - zi2

    # 子图1：密度差异填色图
    im1 = ax1.contourf(xi, yi, zi_diff, levels=20, cmap='RdBu', alpha=0.85)

    # 数据点 — RP (data1) 用红色系，no_RP (data2) 用蓝色系
    ax1.scatter(data1[:, 0], data1[:, 1], color='red', s=40, alpha=0.45,
                edgecolors='darkred', linewidth=1.0)
    ax1.scatter(data2[:, 0], data2[:, 1], color='blue', s=40, alpha=0.45,
                edgecolors='darkblue', linewidth=1.0)

    ax1.set_xlabel('Feasibility Error (p.u.)', fontsize=36, fontweight='normal', fontname='Times New Roman')
    ax1.set_ylabel('Optimality Error (p.u.)', fontsize=36, fontweight='normal', fontname='Times New Roman')
    ax1.set_title('Density Difference Map', fontsize=38, fontweight='normal', fontname='Times New Roman')
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=1.5)
    ax1.tick_params(labelsize=32, width=2.5, length=10)

    # 显式设置坐标轴范围，防止 MultipleLocator 在自动缩放范围内生成过多刻度
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    # 每个刻度显示科学计数法 (整数尾数不显示小数点和0)
    def fmt_sci_simple(v, p):
        if v == 0:
            return '0'
        exp = int(np.floor(np.log10(abs(v))))
        mantissa = v / 10**exp
        if abs(mantissa - round(mantissa)) < 1e-9:
            return f'{mantissa:.0f}e{exp}'
        else:
            return f'{mantissa:.1f}e{exp}'
    sci_formatter = FuncFormatter(fmt_sci_simple)
    ax1.xaxis.set_major_formatter(sci_formatter)
    ax1.yaxis.set_major_formatter(sci_formatter)
    ax1.xaxis.set_major_locator(MultipleLocator(XTICK))
    ax1.yaxis.set_major_locator(MultipleLocator(YTICK))

    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.03)
    cbar1.set_label('Density Difference', fontsize=32, fontweight='normal', fontname='Times New Roman')
    cbar1.ax.tick_params(labelsize=28)

    # 子图2：单独组别的密度 — RP (data1) 用 Reds，no_RP (data2) 用 Blues
    ax2.contourf(xi, yi, zi1, levels=20, cmap='Reds', alpha=0.65, label=labels[0])
    ax2.contourf(xi, yi, zi2, levels=20, cmap='Blues', alpha=0.65, label=labels[1])

    ax2.scatter(data1[:, 0], data1[:, 1], color='red', s=50, alpha=0.5,
                edgecolors='darkred', linewidth=1.2, label=labels[0])
    ax2.scatter(data2[:, 0], data2[:, 1], color='blue', s=50, alpha=0.5,
                edgecolors='darkblue', linewidth=1.2, label=labels[1])

    ax2.set_xlabel('Feasibility Error (p.u.)', fontsize=36, fontweight='normal', fontname='Times New Roman')
    ax2.set_ylabel('Optimality Error (p.u.)', fontsize=36, fontweight='normal', fontname='Times New Roman')
    ax2.set_title('Individual Group Densities', fontsize=38, fontweight='normal', fontname='Times New Roman')
    ax2.legend(fontsize=30, loc='upper right', markerscale=1.5)
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=1.5)
    ax2.tick_params(labelsize=32, width=2.5, length=10)

    # 显式设置坐标轴范围
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    ax2.xaxis.set_major_formatter(sci_formatter)
    ax2.yaxis.set_major_formatter(sci_formatter)
    ax2.xaxis.set_major_locator(MultipleLocator(XTICK))
    ax2.yaxis.set_major_locator(MultipleLocator(YTICK))

    plt.tight_layout()

    return fig


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("Case533 密度差异图生成 (从 CSV 读取误差数据)")
    print("=" * 60)

    # ---- 1. 配置路径 ----
    # 输入: scatter_plot_case533.py 生成的误差 CSV
    csv_dir = os.path.join(PROJECT_ROOT, 'data_analysis', 'results')
    csv_path = os.path.join(csv_dir, 'case533_errors_300samples.csv')

    # 输出目录
    output_dir = csv_dir
    os.makedirs(output_dir, exist_ok=True)

    # ---- 2. 读取 CSV 数据 ----
    print(f"\n[1/3] 从 CSV 读取误差数据: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"  错误: CSV 文件不存在，请先运行 scatter_plot_case533.py 生成数据。")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"  读取到 {len(df)} 行数据")
    print(f"  列名: {list(df.columns)}")

    # 判断是两组数据 (RP + No RP) 还是单组数据 (仅 RP)
    has_no_rp = 'feasibility_error_no_rp' in df.columns and 'optimality_error_no_rp' in df.columns

    # ---- 3. 整理数据 ----
    print(f"\n[2/3] 整理数据...")

    feas_rp = df['feasibility_error_rp'].values
    opt_rp = df['optimality_error_rp'].values
    data_rp = np.column_stack([feas_rp, opt_rp])
    data_rp = data_rp[~np.any(np.isinf(data_rp) | np.isnan(data_rp), axis=1)]
    print(f"  RP 有效数据点: {len(data_rp)}")

    if has_no_rp:
        feas_no_rp = df['feasibility_error_no_rp'].values
        opt_no_rp = df['optimality_error_no_rp'].values
        data_no_rp = np.column_stack([feas_no_rp, opt_no_rp])
        data_no_rp = data_no_rp[~np.any(np.isinf(data_no_rp) | np.isnan(data_no_rp), axis=1)]
        print(f"  No RP 有效数据点: {len(data_no_rp)}")

    # ---- 4. 绘制并保存图形 ----
    print(f"\n[3/3] 绘制图形...")

    if has_no_rp:
        labels = ['PIML with RP', 'PIML without RP']

        print("  生成密度差异图...")
        result = compute_density_difference_map(
            data_rp, data_no_rp,
            labels=labels,
            figsize=(14, 10),
            dpi=300
        )

        print("  生成简化版密度图...")
        fig_simple = create_simplified_density_map(
            data_rp, data_no_rp,
            labels=labels
        )

        # 保存
        print("  保存图形...")
        result['figure'].savefig(os.path.join(output_dir, 'case533_density_comparison_detailed.pdf'),
                                 dpi=300, bbox_inches='tight', facecolor='white')
        result['figure'].savefig(os.path.join(output_dir, 'case533_density_comparison_detailed.png'),
                                 dpi=300, bbox_inches='tight', facecolor='white')
        fig_simple.savefig(os.path.join(output_dir, 'case533_density_comparison_simple.pdf'),
                           dpi=300, bbox_inches='tight', facecolor='white')
        fig_simple.savefig(os.path.join(output_dir, 'case533_density_comparison_simple.png'),
                           dpi=300, bbox_inches='tight', facecolor='white')

        print(f"\n图形已保存至: {output_dir}")
        print("  - case533_density_comparison_detailed.pdf / .png")
        print("  - case533_density_comparison_simple.pdf / .png")

        # 统计信息
        print("\n" + "=" * 60)
        print("统计摘要")
        print("=" * 60)
        print(f"\nPIML with RP ({len(data_rp)} valid samples):")
        print(f"  可行性误差 - Mean: {np.mean(feas_rp):.6e}, "
              f"Max: {np.max(feas_rp):.6e}, Min: {np.min(feas_rp):.6e}")
        print(f"  最优性误差 - Mean: {np.mean(opt_rp):.6e}, "
              f"Max: {np.max(opt_rp):.6e}, Min: {np.min(opt_rp):.6e}")

        print(f"\nPIML without RP ({len(data_no_rp)} valid samples):")
        print(f"  可行性误差 - Mean: {np.mean(feas_no_rp):.6e}, "
              f"Max: {np.max(feas_no_rp):.6e}, Min: {np.min(feas_no_rp):.6e}")
        print(f"  最优性误差 - Mean: {np.mean(opt_no_rp):.6e}, "
              f"Max: {np.max(opt_no_rp):.6e}, Min: {np.min(opt_no_rp):.6e}")

    else:
        # 仅 RP 数据 → 绘制单组散点图 + KDE
        print("  生成 RP 散点图...")
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)

        feas = data_rp[:, 0]
        opt = data_rp[:, 1]

        # KDE
        if len(feas) > 10:
            kde = gaussian_kde(data_rp.T)
            x_min, x_max = feas.min(), feas.max()
            y_min, y_max = opt.min(), opt.max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min, x_max = x_min - 0.1 * x_range, x_max + 0.1 * x_range
            y_min, y_max = y_min - 0.1 * y_range, y_max + 0.1 * y_range

            xi, yi = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            grid_coords = np.vstack([xi.flatten(), yi.flatten()])
            zi = kde(grid_coords).reshape(xi.shape)
            ax.contourf(xi, yi, zi, levels=15, cmap='Blues', alpha=0.55)

        ax.scatter(feas, opt,
                   color='steelblue', s=50, alpha=0.45,
                   edgecolors='darkblue', linewidth=0.8,
                   label=f'Case533 PIML with RP (n={len(feas)})', zorder=3)

        ax.set_xlabel('Feasibility Error (MW)', fontsize=36, fontweight='normal', fontname='Times New Roman')
        ax.set_ylabel('Optimality Error (MW)', fontsize=36, fontweight='normal', fontname='Times New Roman')
        ax.set_title('Case533: Error Distribution (RP only)\n(300 Sample Points)',
                     fontsize=38, fontweight='normal', fontname='Times New Roman')
        ax.tick_params(axis='both', which='major', labelsize=28, width=2.5, length=10)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=1.5)
        legend = ax.legend(loc='upper right', fontsize=26, frameon=True,
                           fancybox=True, framealpha=0.9, edgecolor='black', markerscale=1.5)
        legend.get_frame().set_linewidth(2.5)
        plt.tight_layout()

        print("  保存图形...")
        fig.savefig(os.path.join(output_dir, 'case533_scatter_rp_only.pdf'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(output_dir, 'case533_scatter_rp_only.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')

        print(f"\n图形已保存至: {output_dir}")
        print("  - case533_scatter_rp_only.pdf / .png")

        print(f"\nPIML with RP ({len(data_rp)} valid samples):")
        print(f"  可行性误差 - Mean: {np.mean(feas_rp):.6e}, "
              f"Max: {np.max(feas_rp):.6e}, Min: {np.min(feas_rp):.6e}")
        print(f"  最优性误差 - Mean: {np.mean(opt_rp):.6e}, "
              f"Max: {np.max(opt_rp):.6e}, Min: {np.min(opt_rp):.6e}")

    plt.show()
    print("\n程序结束。")
