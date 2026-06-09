import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import random

import numpy as np
from Simulator.Approximator import *
import torch
from torch.utils.data import Dataset, DataLoader
from pyomo.environ import *
from Simulator.Plotter import ShapeDrawer_2D
import matplotlib.pyplot as plt
from Simulator import PROJECT_ROOT
import os
import pickle
from Simulator.Plotter import ErrorVisualizer
from DSO_boundary import *
import pandas as pd


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
        "rate_opt_feas": 1,
        "rate_feas": 10e-3,
        "rate_opt": 10e-1,
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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parallel = False
    model_type = 'pretrainnet'
    case = case_533(model_type=model_type, device=device)
    n_train = 1000

    # Cluster dims: [9, 4, 8, 5, 10, 7, 8, 7, 14, 5, 5, 5, 4, 9, 12]
    divide_mark = [
        [56, 9],    # DG1:  6*9+2=56 rows,  9 cols
        [26, 4],    # DG2:  6*4+2=26 rows,  4 cols
        [50, 8],    # DG3:  6*8+2=50 rows,  8 cols
        [32, 5],    # DG4:  6*5+2=32 rows,  5 cols
        [62, 10],   # DG5:  6*10+2=62 rows, 10 cols
        [44, 7],    # DG6:  6*7+2=44 rows,  7 cols
        [50, 8],    # DG7:  6*8+2=50 rows,  8 cols
        [44, 7],    # DG8:  6*7+2=44 rows,  7 cols
        [86, 14],   # DG9:  6*14+2=86 rows, 14 cols
        [32, 5],    # DG10: 6*5+2=32 rows,  5 cols
        [32, 5],    # DG11: 6*5+2=32 rows,  5 cols
        [32, 5],    # DG12: 6*5+2=32 rows,  5 cols
        [26, 4],    # DG13: 6*4+2=26 rows,  4 cols
        [56, 9],    # DG14: 6*9+2=56 rows,  9 cols
        [74, 12],   # DG15: 6*12+2=74 rows, 12 cols
    ]

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
