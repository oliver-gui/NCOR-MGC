"""
组合箱型散点图：在一张图中以方法为分组（Clustered, Non-Clustered, Ref.[21]），
每组内展示 case33、case118、case533 三组算例的最优性误差分布。

数据来源：
  - case33: data_analysis_share/box_plot/*.csv
  - case118: data/box_plot/case118_*.csv 
  - case533: data/box_plot/case533_*.csv 

输出：
  - data/box_plot/combined_boxplot.pdf
  - data/box_plot/combined_boxplot.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# ===================== 数据加载 =====================

# 当前脚本所在目录 (data/box_plot/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# case33 数据目录
CASE33_DIR = os.path.join(
    os.path.dirname(os.path.dirname(BASE_DIR)),
    'data_analysis_share', 'box_plot'
)

# ---- case33: 无表头，直接读取 ----
case33_clustered = pd.read_csv(
    os.path.join(CASE33_DIR, 'data_proposed_clustering.csv'), header=None
).values.ravel()
case33_noclustered = pd.read_csv(
    os.path.join(CASE33_DIR, 'data_proposed_no_clustering.csv'), header=None
).values.ravel()
case33_ref = pd.read_csv(
    os.path.join(CASE33_DIR, 'data_ref_21.csv'), header=None
).values.ravel()

# ---- case118: 有表头 'optimality_error' ----
case118_clustered = pd.read_csv(
    os.path.join(BASE_DIR, 'case118_proposed_clustering.csv')
)['optimality_error'].values
case118_noclustered = pd.read_csv(
    os.path.join(BASE_DIR, 'case118_proposed_no_clustering.csv')
)['optimality_error'].values
case118_ref = pd.read_csv(
    os.path.join(BASE_DIR, 'case118_ref_21.csv')
)['optimality_error'].values

# ---- case533: 有表头 'optimality_error' ----
case533_clustered = pd.read_csv(
    os.path.join(BASE_DIR, 'case533_proposed_clustering.csv')
)['optimality_error'].values
case533_noclustered = pd.read_csv(
    os.path.join(BASE_DIR, 'case533_proposed_no_clustering.csv')
)['optimality_error'].values
case533_ref = pd.read_csv(
    os.path.join(BASE_DIR, 'case533_ref_21.csv')
)['optimality_error'].values

# ===================== 数据汇总 =====================

print("=" * 60)
print("三组算例最优性误差统计")
print("=" * 60)

for case_name, clustered, noclustered, ref in [
    ('Case 33', case33_clustered, case33_noclustered, case33_ref),
    ('Case 118', case118_clustered, case118_noclustered, case118_ref),
    ('Case 533', case533_clustered, case533_noclustered, case533_ref),
]:
    print(f"\n{case_name}:")
    print(f"  {'方法':<28} {'Mean':<14} {'Std':<14} {'Median':<14}")
    print(f"  {'-'*66}")
    for method_name, data in [
        ('Clustered', clustered),
        ('Non-Clustered', noclustered),
        ('Ref.[21]', ref),
    ]:
        print(f"  {method_name:<28} {np.mean(data):<14.6e} "
              f"{np.std(data):<14.6e} {np.median(data):<14.6e}")

# ===================== 绘图 =====================

# 字体设置
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 21

# ============================================================
# 数据排列：按方法分组，每组内按 case33, case118, case533 排列
#   Group 1 (positions 1,2,3):   Clustered    — case33, case118, case533
#   Group 2 (positions 5,6,7):   Non-Clustered — case33, case118, case533
#   Group 3 (positions 9,10,11): Ref.[21]     — case33, case118, case533
# ============================================================
data = [
    case33_clustered, case118_clustered, case533_clustered,   # Clustered 组
    case33_noclustered, case118_noclustered, case533_noclustered,  # Non-Clustered 组
    case33_ref, case118_ref, case533_ref,                     # Ref.[21] 组
]

# 箱体位置：组内间距0.9，组间间距1.3，整体更紧凑
positions = [1, 1.9, 2.8,           # Clustered
             4.1, 5.0, 5.9,         # Non-Clustered
             7.2, 8.1, 9.0]         # Ref.[21]

# ============================================================
# 颜色定义：以算例区分颜色
#   case33  → 蓝色系
#   case118 → 绿色系
#   case533 → 橙/红色系
# 每组方法内三种颜色循环：case33, case118, case533
# ============================================================
case_colors = {
    'case33':  {'box': '#5DADE2', 'jitter': '#2E86C1'},   # 蓝色
    'case118': {'box': '#58D68D', 'jitter': '#27AE60'},   # 绿色
    'case533': {'box': '#F1948A', 'jitter': '#E74C3C'},   # 红色
}

# 按数据顺序分配颜色（每组方法内 case33, case118, case533）
box_colors = [
    case_colors['case33']['box'], case_colors['case118']['box'], case_colors['case533']['box'],  # Clustered
    case_colors['case33']['box'], case_colors['case118']['box'], case_colors['case533']['box'],  # Non-Clustered
    case_colors['case33']['box'], case_colors['case118']['box'], case_colors['case533']['box'],  # Ref.[21]
]
jitter_colors = [
    case_colors['case33']['jitter'], case_colors['case118']['jitter'], case_colors['case533']['jitter'],
    case_colors['case33']['jitter'], case_colors['case118']['jitter'], case_colors['case533']['jitter'],
    case_colors['case33']['jitter'], case_colors['case118']['jitter'], case_colors['case533']['jitter'],
]

# 创建图形（宽度调小以适应更紧凑的布局）
fig, ax = plt.subplots(figsize=(13, 9))

# 绘制箱型图
bp = ax.boxplot(
    data,
    patch_artist=True,
    widths=0.5,
    positions=positions,
    showfliers=True,
    flierprops=dict(marker='o', markerfacecolor='gray', markersize=5,
                    alpha=0.4, linestyle='none'),
)

# 设置箱体颜色
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# 设置线条属性
for element in ['whiskers', 'caps']:
    for line in bp[element]:
        line.set_color('#333333')
        line.set_linewidth(1.2)

# 中位线
for median in bp['medians']:
    median.set_color('darkred')
    median.set_linewidth(2.0)

# 添加抖动散点
jitter_amount = 0.13
for i, (d, pos, jitter_color) in enumerate(zip(data, positions, jitter_colors)):
    x_jitter = np.random.normal(0, jitter_amount, len(d))
    x_positions = pos + x_jitter
    ax.scatter(
        x_positions, d,
        alpha=0.35, color=jitter_color,
        s=28, edgecolor='white', linewidth=0.25, zorder=3,
    )

# ---- X 轴设置 ----
# 三组方法的标签位置（组中心）
group_centers = [1.9, 5.0, 8.1]
ax.set_xticks(group_centers)
ax.set_xticklabels(
    ['Proposed Method\nClustered',
     'Proposed Method\nNon-Clustered',
     'Ref.[21] Method\nNon-Clustered'],
    fontsize=22, fontname='Times New Roman',
)


# ---- Y 轴设置（线性尺度，从 0 到 0.5） ----
ax.set_ylim(bottom=0, top=0.5)
ax.set_ylabel(r'Optimality Error (p.u.)', fontsize=27, fontname='Times New Roman')

# ---- Y 轴刻度：主刻度每 0.1，次刻度每 0.02 ----
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))

# ---- 图例（以算例区分） ----
legend_elements = [
    Patch(facecolor=case_colors['case33']['box'], alpha=0.75,
          label='Case 33 (13-bus)'),
    Patch(facecolor=case_colors['case118']['box'], alpha=0.75,
          label='Case 118 (118-bus)'),
    Patch(facecolor=case_colors['case533']['box'], alpha=0.75,
          label='Case 533 (533-bus)'),
]
ax.legend(
    handles=legend_elements,
    loc='upper left',
    fontsize=24,
    framealpha=0.9,
    edgecolor='gray',
    ncol=1,
    borderaxespad=0.3,
)

# ---- 网格 ----
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, which='major')
ax.yaxis.grid(True, which='major', alpha=0.35)

# ---- 坐标轴字体 ----
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_fontsize(21)
ax.yaxis.label.set_fontname('Times New Roman')
ax.yaxis.label.set_fontsize(27)

# ---- 调整布局 ----
plt.tight_layout(rect=[0, 0.02, 1, 0.96])

# ---- 保存 ----
output_pdf = os.path.join(BASE_DIR, 'combined_boxplot.pdf')
output_png = os.path.join(BASE_DIR, 'combined_boxplot.png')
plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"\n图形已保存:")
print(f"  {output_pdf}")
print(f"  {output_png}")

plt.show()
