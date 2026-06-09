import random

import numpy as np
from Simulator.Approximator import *
import torch
from torch.utils.data import Dataset, DataLoader
from pyomo.environ import *
from Simulator.Plotter import ShapeDrawer_2D, ShapeDrawer_3D
import matplotlib.pyplot as plt
from Simulator import PROJECT_ROOT
import os
import pickle
from Simulator.Plotter import ErrorVisualizer
from DSO_boundary import *
import pandas as pd

def case_modified(total_samples = 200, noise_scale=0.15, batch_size=1, model_type = 'pretrainnet',device = 'cpu'):
    """固定参数的原始案例实现"""
    # 固定初始化参数
    node_total = [0, 1, 2, 3]
    node_load = [0]
    node_control = [1, 2, 3]
    branch_total = [(0, 1), (1, 2), (0, 3)]
    # 线路参数 (R, X)
    line_params = {
        (0, 1): {'r': 0.55 / 3 , 'x': 1.33 / 3 },
        (1, 2): {'r': 0.55 / 3 , 'x': 1.33 / 3 },
        (0, 3): {'r': 0.55 / 3 , 'x': 1.33 / 3 }
    }
    # 节点负荷 (P, Q)
    load_params = {
        0: {'p': 0.0, 'q': 0.0},  # 平衡节点
        1: {'p': 0.05, 'q': 0.03},
        2: {'p': 0.08, 'q': 0.05},
        3: {'p': 0.06, 'q': 0.04}
    }
    p_bound = {
        1: {'lb': -0.75, 'ub': 0.2},
        2: {'lb': -0.32, 'ub': 0.8},
        3: {'lb': -0.50, 'ub': 0.4},
    }
    DSO_base_P = [0.05 * 0.75, 0.08 * 0.75, 0.06 * 0.75]  # 后续替换为函数求解
    DSO_base_P_2p = [0.05 * 0.75, 0.08 * 0.75]

    # 创建模型
    model = ConcreteModel()

    # 节点集合 (0: 平衡节点, 1,2,3: PQ节点)
    model.NODES = Set(initialize=node_total)
    model.LINES = Set(initialize=branch_total)
    model.node_load = Set(initialize=node_load)
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
    DG_1 = np.array([[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,-1]])
    random_vector = np.zeros([10,DG_1.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i,j]**2 for j in range(random_vector.shape[1])))
        random_vector[i,:] = random_vector[i,:]/v_norm
    DG_1_randn = np.vstack([DG_1, random_vector])
    #DG_1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    # DG2 Boundary define
    DG_2 = np.array([[1], [-1]])
    random_vector = np.zeros([5, DG_2.shape[1]])
    for i in range(random_vector.shape[0]):
        random_vector[i] = np.random.randn(random_vector.shape[1])
        v_norm = sqrt(sum(random_vector[i, j] ** 2 for j in range(random_vector.shape[1])))
        random_vector[i, :] = random_vector[i, :] / v_norm
    DG_2_randn = np.vstack([DG_2, random_vector])
    # DG3 Boundary define
    DG_3 = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,0],[-1,-1,0],[1,1,1],[-1,-1,-1]])

    A_hat = DG_boundary_to_A_hat([
        DG_1_randn,  # DG1
        DG_2,  # DG2
    ])

    A_hat1 = DG_boundary_to_A_hat([
        DG_1,  # DG2
        DG_2,  # DG2
    ])
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat1,
        solver='ipopt',
    )
    A_list = [errorcalculator.A_hat]
    b_list = [errorcalculator.b_hat]

    case_name = 'test'
    visualizer = ErrorVisualizer()
    num_sample = 50

    plt.figure(figsize=(8, 6))
    plotter = ShapeDrawer_2D()
    xlim = [-2, 2]
    ylim = [-2, 2]

    figure_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\figures'
    violin_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\violin_plot'
    box_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\box_plot'
    kde_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\kde_plot'
    result_folder = f'{PROJECT_ROOT}\\results\\{case_name}'

    def callback(error_calculator, epoch):
        if not hasattr(callback, "idx_mark"):
            callback.idx_mark = 0  # 初始化计数器
        # visualization
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

        # plot polygon
        if epoch == 0:
            plotter.plot_polygon(error_calculator.A_hat[0:16,0:2], error_calculator.b_hat[0:16], xlim=xlim, ylim=ylim,
                             facecolor='green', label='Approximation'
                             , title=f'Training step {epoch}')
        else:
            plotter.remove_shape(plotter.shapes[-1]['id'])
            plotter.plot_polygon(error_calculator.A_hat[0:16,0:2], error_calculator.b_hat[0:16], xlim=xlim, ylim=ylim,
                                 facecolor='green', label='Approximation'
                                 , title=f'Training step {epoch}')
        plotter.save(f"{figure_folder}/step{epoch}")

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
        'A_hat': A_hat1,
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
        r = input_branch_data.iloc[it, 2]
        x = input_branch_data.iloc[it, 3]
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

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parallel = False
    model_type = 'pretrainnet'
    case = case_modified(model_type=model_type, device=device)
    n_train = 1000
    divide_mark = [[6,2],[2,1]]
    divide_mark1 = [[16,2]]
    A_hat_model = A_hat_diversation(case['A_hat'], divide_mark)
    model = PreTrainNet(A_hat_model, case['b_hat'], device=device).to(device)
    trainer = Trainer(
        model=model,
        error_calculator=case['errorcalculator'],
        compute_loss=compute_loss,
    )
    trainer.configure(**case['trainer_configure'])
    trainer.initialize()
    trainer.train(n_train=n_train, params_data=case['params'], parallel=parallel)
    print("Program Ended")
