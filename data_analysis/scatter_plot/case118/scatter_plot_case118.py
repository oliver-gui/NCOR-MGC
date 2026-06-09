"""
基于已有误差 CSV 数据生成 Case118 密度差异图 / 散点图。

该文件直接读取 scatter_plot_case118.py 生成的误差 CSV 文件作为输入，
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
def case_118_modified(total_samples = 200, noise_scale=0.15, batch_size=1, model_type = 'pretrainnet',device = 'cpu'):
    """固定参数的原始案例实现"""
    # data input
    bus_data = pd.read_csv(r'D:\work space\论文-working\share\case_data\bus-case118.csv')
    branch_data = pd.read_csv(r'D:\work space\论文-working\share\case_data\branch-case118.csv')
    # 固定初始化参数
    node_total = []
    for it in range(bus_data.shape[0]):
        node_total.append(bus_data.iloc[it, 0])
    node_control = [1,2,9,10,11,12,
                    17,18,19,20,
                    21,22,23,24,25,26,
                    37,38,39,40,41,42,43,44,45,
                    54,55,56,57,58,
                    65,66,67,68,69,70,71,72,
                    89,90,91,92,93,94,
                    104,105,106,107,108,
                    113,114,115,116]
    branch_total = []
    for it in range(branch_data.shape[0]):
        branch_total.append((branch_data.iloc[it,0], branch_data.iloc[it,1]))
    # 线路参数 (R, X)
    line_params = input_branch_data_to_line_params(branch_data)
    # 节点负荷 (P, Q)
    load_params = input_bus_data_to_load_params(bus_data)
    p_bound = {
        1: {'lb': -1.35, 'ub': 2.4},
        2: {'lb': -0.92, 'ub': 1.66},
        9: {'lb': -1.61, 'ub': 0.73},
        10: {'lb': -1.73, 'ub': 2.22},
        11: {'lb': -1.08, 'ub': 1.89},
        12: {'lb': -1.78, 'ub': 2.44},
        17: {'lb': -0.8, 'ub': 0.85},
        18: {'lb': -1.58, 'ub': 0.79},
        19: {'lb': -1.43, 'ub': 1.5},
        20: {'lb': -1.28, 'ub': 1.01},
        21: {'lb': -1.07, 'ub': 0.69},
        22: {'lb': -1.45, 'ub': 1.17},
        23: {'lb': -1.25, 'ub': 2.05},
        24: {'lb': -1.56, 'ub': 1.48},
        25: {'lb': -1.09, 'ub': 0.5},
        26: {'lb': -1.07, 'ub': 0.76},
        37: {'lb': -1.72, 'ub': 2.39},
        38: {'lb': -0.64, 'ub': 2.1},
        39: {'lb': -1.43, 'ub': 0.61},
        40: {'lb': -0.98, 'ub': 1.32},
        41: {'lb': -1.65, 'ub': 1.44},
        42: {'lb': -1.76, 'ub': 2.31},
        43: {'lb': -1.49, 'ub': 1.79},
        44: {'lb': -1.43, 'ub': 1.49},
        45: {'lb': -1.14, 'ub': 0.79},
        54: {'lb': -0.64, 'ub': 2.03},
        55: {'lb': -0.67, 'ub': 2.28},
        56: {'lb': -1.08, 'ub': 2.34},
        57: {'lb': -1.69, 'ub': 0.81},
        58: {'lb': -1.75, 'ub': 1.08},
        65: {'lb': -1.33, 'ub': 0.97},
        66: {'lb': -0.81, 'ub': 1.15},
        67: {'lb': -1.46, 'ub': 1.54},
        68: {'lb': -1.63, 'ub': 2.08},
        69: {'lb': -1.71, 'ub': 2.47},
        70: {'lb': -0.87, 'ub': 0.82},
        71: {'lb': -1.79, 'ub': 2.11},
        72: {'lb': -0.95, 'ub': 1.93},
        89: {'lb': -0.87, 'ub': 0.56},
        90: {'lb': -1.37, 'ub': 0.64},
        91: {'lb': -0.76, 'ub': 1.71},
        92: {'lb': -1.4, 'ub': 0.53},
        93: {'lb': -1.43, 'ub': 1.08},
        94: {'lb': -0.92, 'ub': 1.74},
        104: {'lb': -0.74, 'ub': 1.39},
        105: {'lb': -1.66, 'ub': 1.9},
        106: {'lb': -0.89, 'ub': 1.58},
        107: {'lb': -0.87, 'ub': 1.44},
        108: {'lb': -1.17, 'ub': 1.3},
        113: {'lb': -1.77, 'ub': 0.63},
        114: {'lb': -1.76, 'ub': 1.74},
        115: {'lb': -1.42, 'ub': 1.47},
        116: {'lb': -0.71, 'ub': 0.92},
    }
    DSO_base_P = np.zeros(len(node_control))
    buffer_it = 0
    for it in range(len(load_params)):
        if it == node_control[buffer_it]:
            DSO_base_P[buffer_it] = load_params[it]['p'] * 0.75
            buffer_it += 1
            if buffer_it == len(node_control):
                break

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
    V_min, V_max = 0.95, 1.05

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
    model.base_point = Param(range(len(model.node_control)), initialize=base_point,mutable=False)

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

        #def __getitem__(self, idx):
            #return {'theta':torch.normal(0, self.noise_scale, (dim_theta,),device=device)}  # theta维度固定为2
    # DG1 Boundary define
    DG_1 = np.array([[1,0,0,0,0,0], [-1,0,0,0,0,0], [0,1,0,0,0,0], [0,-1,0,0,0,0], [0,0,1,0,0,0], [0,0,-1,0,0,0],
                     [0,0,0,1,0,0], [0,0,0,-1,0,0], [0,0,0,0,1,0], [0,0,0,0,-1,0], [0,0,0,0,0,1], [0,0,0,0,0,-1],
                     [1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1]])
    random_vector = np.zeros([28,DG_1.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i,j]**2 for j in range(random_vector.shape[1])))
        random_vector[i,:] = random_vector[i,:]/v_norm
    DG_1_randn = np.vstack([DG_1, random_vector])
    # DG2 Boundary define
    DG_2 = np.array([[1,0,0,0], [-1,0,0,0],[0,1,0,0], [0,-1,0,0], [0,0,1,0], [0,0,-1,0],
                     [0,0,0,1], [0,0,0,-1],[1,1,1,1],[-1,-1,-1,-1]])
    random_vector = np.zeros([20, DG_2.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i, j] ** 2 for j in range(random_vector.shape[1])))
        random_vector[i, :] = random_vector[i, :] / v_norm
    DG_2_randn = np.vstack([DG_2, random_vector])
    # DG3 Boundary define
    DG_3 = np.array([[1,0,0,0,0,0], [-1,0,0,0,0,0], [0,1,0,0,0,0], [0,-1,0,0,0,0], [0,0,1,0,0,0], [0,0,-1,0,0,0],
                     [0,0,0,1,0,0], [0,0,0,-1,0,0], [0,0,0,0,1,0], [0,0,0,0,-1,0], [0,0,0,0,0,1], [0,0,0,0,0,-1],
                     [1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1]])
    random_vector = np.zeros([28, DG_3.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i, j] ** 2 for j in range(random_vector.shape[1])))
        random_vector[i, :] = random_vector[i, :] / v_norm
    DG_3_randn = np.vstack([DG_3, random_vector])
    # DG4 Boundary define
    DG_4 = np.array([[1,0,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,-1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,-1,0,0,0,0,0,0],
                     [0,0,0,1,0,0,0,0,0], [0,0,0,-1,0,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,-1,0,0,0,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,-1,0,0,0],
                     [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,-1,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,-1,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,-1],
                     [1,1,1,1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1,-1,-1,-1]])
    random_vector = np.zeros([40, DG_4.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i, j] ** 2 for j in range(random_vector.shape[1])))
        random_vector[i, :] = random_vector[i, :] / v_norm
    DG_4_randn = np.vstack([DG_4, random_vector])
    # DG5 Boundary define
    DG_5 = np.array([[1,0,0,0,0], [-1,0,0,0,0], [0,1,0,0,0], [0,-1,0,0,0], [0,0,1,0,0], [0,0,-1,0,0],
                     [0,0,0,1,0], [0,0,0,-1,0], [0,0,0,0,1], [0,0,0,0,-1], [1,1,1,1,1], [-1,-1,-1,-1,-1]])
    random_vector = np.zeros([24, DG_5.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i, j] ** 2 for j in range(random_vector.shape[1])))
        random_vector[i, :] = random_vector[i, :] / v_norm
    DG_5_randn = np.vstack([DG_5, random_vector])
    # DG6 Boundary define
    DG_6 = np.array([[1,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,-1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,-1,0,0,0,0,0],
                     [0,0,0,1,0,0,0,0], [0,0,0,-1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,-1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,-1,0,0],
                     [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,-1,0], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,-1],
                     [1,1,1,1,1,1,1,1], [-1,-1,-1,-1,-1,-1,-1,-1]])
    random_vector = np.zeros([36, DG_6.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i, j] ** 2 for j in range(random_vector.shape[1])))
        random_vector[i, :] = random_vector[i, :] / v_norm
    DG_6_randn = np.vstack([DG_6, random_vector])
    # DG7 Boundary define
    DG_7 = np.array([[1,0,0,0,0,0], [-1,0,0,0,0,0], [0,1,0,0,0,0], [0,-1,0,0,0,0], [0,0,1,0,0,0], [0,0,-1,0,0,0],
                     [0,0,0,1,0,0], [0,0,0,-1,0,0], [0,0,0,0,1,0], [0,0,0,0,-1,0], [0,0,0,0,0,1], [0,0,0,0,0,-1],
                     [1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1]])
    random_vector = np.zeros([28, DG_7.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i, j] ** 2 for j in range(random_vector.shape[1])))
        random_vector[i, :] = random_vector[i, :] / v_norm
    DG_7_randn = np.vstack([DG_7, random_vector])
    # DG8 Boundary define
    DG_8 = np.array([[1,0,0,0,0], [-1,0,0,0,0], [0,1,0,0,0], [0,-1,0,0,0], [0,0,1,0,0], [0,0,-1,0,0],
                     [0,0,0,1,0], [0,0,0,-1,0], [0,0,0,0,1], [0,0,0,0,-1], [1,1,1,1,1], [-1,-1,-1,-1,-1]])
    random_vector = np.zeros([24, DG_8.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i, j] ** 2 for j in range(random_vector.shape[1])))
        random_vector[i, :] = random_vector[i, :] / v_norm
    DG_8_randn = np.vstack([DG_8, random_vector])
    # DG9 Boundary define
    DG_9 = np.array([[1,0,0,0], [-1,0,0,0],[0,1,0,0], [0,-1,0,0], [0,0,1,0], [0,0,-1,0],
                     [0,0,0,1], [0,0,0,-1],[1,1,1,1],[-1,-1,-1,-1]])
    random_vector = np.zeros([20, DG_9.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i, j] ** 2 for j in range(random_vector.shape[1])))
        random_vector[i, :] = random_vector[i, :] / v_norm
    DG_9_randn = np.vstack([DG_9, random_vector])

    A_hat = DG_boundary_to_A_hat([
        DG_1_randn,  # DG1
        DG_2_randn,  # DG2
        DG_3_randn,  # DG3
        DG_4_randn,  # DG4
        DG_5_randn,  # DG5
        DG_6_randn,  # DG6
        DG_7_randn,  # DG7
        DG_8_randn,  # DG8
        DG_9_randn   # DG9
    ])

    A_hat1 = DG_boundary_to_A_hat([
        DG_1,  # DG1
        DG_2,  # DG2
        DG_3,  # DG3
        DG_4,  # DG4
        DG_5,  # DG5
        DG_6,  # DG6
        DG_7,  # DG7
        DG_8,  # DG8
        DG_9   # DG9
    ])
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='ipopt',
    )
    A_list = [errorcalculator.A_hat]
    b_list = [errorcalculator.b_hat]

    #case_name = 'case118_no_bp'
    case_name = 'case118_modified'
    visualizer = ErrorVisualizer()
    num_sample = 80

    violin_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\violin_plot'
    box_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\box_plot'
    result_folder = f'{PROJECT_ROOT}\\results\\{case_name}'

    def callback(error_calculator, epoch):
        if not hasattr(callback, "idx_mark"):
            callback.idx_mark = 0  # 初始化计数器
        # visualization
        visualizer.plot_dual_violin(save_path=f"{violin_folder}/step{epoch}")
        visualizer.plot_dual_boxplot(save_path=f"{box_folder}/step{epoch}")

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
        r = input_branch_data.iloc[it, 2] / 160
        x = input_branch_data.iloc[it, 3] / 160
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

def base_line_result_loss_compute(model, base_line_result, num_sample):
    #[up_bound;
    # low_bound;]
    # initialization
    error_feas = []
    error_opt = []
    A_hat_baseline = np.zeros([2 * base_line_result.shape[1], base_line_result.shape[1]])
    b_hat_baseline = np.zeros([2 * base_line_result.shape[1], 1])
    for it in range(base_line_result.shape[1]):
        A_hat_baseline[it * 2, it] = 1
        A_hat_baseline[it * 2 + 1, it] = -1
        b_hat_baseline[it * 2, 0] = base_line_result[0,it]
        b_hat_baseline[it * 2 + 1, 0] = -base_line_result[1,it]
    model.A_hat = A_hat_baseline
    model.b_hat = b_hat_baseline
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
            error_opt.append(np.sum((x_apx - x_org) ** 2))
    return error_feas, error_opt

# ===================== 字体与样式设置 =====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 30
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 2


# ===================== 密度差异图 (来自 scatter_plot_case118.py) =====================
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

    # 确定绘图范围（x/y 轴均从 0 开始）
    all_data = np.vstack([data1, data2])
    x_min, x_max = 0.0, 2e-7
    y_min, y_max = 0.0, 3e-6

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

    # 确定共同范围（x/y 轴均从 0 开始）
    all_data = np.vstack([data1, data2])
    x_min, x_max = 0.0, 2e-7
    y_min, y_max = 0.0, 3e-6

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

    ax2.xaxis.set_major_formatter(sci_formatter)
    ax2.yaxis.set_major_formatter(sci_formatter)
    ax2.xaxis.set_major_locator(MultipleLocator(XTICK))
    ax2.yaxis.set_major_locator(MultipleLocator(YTICK))

    plt.tight_layout()

    return fig


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("=" * 60)
    print("Case118 密度差异图生成 (从 CSV 读取误差数据)")
    print("=" * 60)

    # ---- 1. 配置路径 ----
    # 输入: scatter_plot_case118.py 生成的误差 CSV
    csv_dir = os.path.join(PROJECT_ROOT, 'data_analysis', 'results')
    csv_path = os.path.join(csv_dir, 'case118_errors_300samples.csv')

    # 输出目录
    output_dir = csv_dir
    os.makedirs(output_dir, exist_ok=True)

    # ---- 2. 读取 CSV 数据 ----
    print(f"\n[1/3] 从 CSV 读取误差数据: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"  错误: CSV 文件不存在，请先运行 scatter_plot_case118.py 生成数据。")
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
        result['figure'].savefig(os.path.join(output_dir, 'case118_density_comparison_detailed.pdf'),
                                 dpi=300, bbox_inches='tight', facecolor='white')
        result['figure'].savefig(os.path.join(output_dir, 'case118_density_comparison_detailed.png'),
                                 dpi=300, bbox_inches='tight', facecolor='white')
        fig_simple.savefig(os.path.join(output_dir, 'case118_density_comparison_simple.pdf'),
                           dpi=300, bbox_inches='tight', facecolor='white')
        fig_simple.savefig(os.path.join(output_dir, 'case118_density_comparison_simple.png'),
                           dpi=300, bbox_inches='tight', facecolor='white')

        print(f"\n图形已保存至: {output_dir}")
        print("  - case118_density_comparison_detailed.pdf / .png")
        print("  - case118_density_comparison_simple.pdf / .png")

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
                   label=f'Case118 PIML with RP (n={len(feas)})', zorder=3)

        ax.set_xlabel('Feasibility Error (MW)', fontsize=36, fontweight='normal', fontname='Times New Roman')
        ax.set_ylabel('Optimality Error (MW)', fontsize=36, fontweight='normal', fontname='Times New Roman')
        ax.set_title('Case118: Error Distribution (RP only)\n(300 Sample Points)',
                     fontsize=38, fontweight='normal', fontname='Times New Roman')
        ax.tick_params(axis='both', which='major', labelsize=28, width=2.5, length=10)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=1.5)
        legend = ax.legend(loc='upper right', fontsize=26, frameon=True,
                           fancybox=True, framealpha=0.9, edgecolor='black', markerscale=1.5)
        legend.get_frame().set_linewidth(2.5)
        plt.tight_layout()

        print("  保存图形...")
        fig.savefig(os.path.join(output_dir, 'case118_scatter_rp_only.pdf'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(os.path.join(output_dir, 'case118_scatter_rp_only.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')

        print(f"\n图形已保存至: {output_dir}")
        print("  - case118_scatter_rp_only.pdf / .png")

        print(f"\nPIML with RP ({len(data_rp)} valid samples):")
        print(f"  可行性误差 - Mean: {np.mean(feas_rp):.6e}, "
              f"Max: {np.max(feas_rp):.6e}, Min: {np.min(feas_rp):.6e}")
        print(f"  最优性误差 - Mean: {np.mean(opt_rp):.6e}, "
              f"Max: {np.max(opt_rp):.6e}, Min: {np.min(opt_rp):.6e}")

    plt.show()
    print("\n程序结束。")
