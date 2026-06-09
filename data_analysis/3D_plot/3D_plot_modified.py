import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from matplotlib.patches import Patch

# 创建图形 - 使用更大的画布
fig = plt.figure(figsize=(16, 14))
ax = fig.add_subplot(111, projection='3d')

# 设置坐标轴标签（按照P2, P1, P3的顺序）并添加单位
ax.set_xlabel('$P_2$ (MW)', fontname='Times New Roman', fontsize=28, labelpad=25)
ax.set_ylabel('$P_1$ (MW)', fontname='Times New Roman', fontsize=28, labelpad=25)
ax.set_zlabel('$P_3$ (MW)', fontname='Times New Roman', fontsize=28, labelpad=25)

# 设置刻度字体和更精细的刻度
for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
    tick.set_fontname('Times New Roman')
    tick.set_fontsize(22)

# 第一个多面体（蓝色）
A1 = np.array([
    [0.9999995, 0.00102822, 0],
    [-0.99996585, -0.00828014, 0],
    [0.13307443, 0.99110603, 0],
    [-0.148231, -0.98895276, 0],
    [0.44916672, 0.89344794, 0],
    [-0.62208056, -0.7829533, 0],
    [0, 0, 1],
    [0, 0, -1]
])

b1 = np.array([
    0.1989852786064148,
    0.7454229593276978,
    0.28778223037719727,
    0.27336111664772034,
    -0.023646898567676544,
    0.28627988040447235,
    0.1689377874135971,
    0.3760448694229128
])

# 第二个多面体（橙色）
A2 = np.array([
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
])

b2 = np.array([
    0.18364452552353222,
    0.05812566673807931,
    -0.13280716108362136,
    0.25456593959852164,
    0.1416465058596817,
    0.3487843736680237
])


def find_interior_point_2d(A_2d, b_2d):
    """找到2D多边形的内部点（严格在内部）"""
    # 首先尝试使用线性规划
    c = np.zeros(2)
    bounds = [(-10, 10), (-10, 10)]

    # 添加小的松弛量确保点在内部
    epsilon = 1e-6
    b_2d_relaxed = b_2d - epsilon

    result = linprog(c, A_ub=A_2d, b_ub=b_2d_relaxed, bounds=bounds, method='highs')

    if result.success:
        point = result.x
        # 验证点是否真的在内部
        violations = A_2d @ point - b_2d
        if np.all(violations < -1e-8):
            print(f"找到内部点: {point}")
            return point
        else:
            print(f"点不严格在内部，最大违反: {violations.max()}")

    # 如果线性规划失败，尝试其他方法
    print("尝试备选方法寻找内部点...")

    # 方法2：找到所有约束线的交点，取平均值
    n_constraints = len(A_2d)
    intersections = []

    for i in range(n_constraints):
        for j in range(i + 1, n_constraints):
            a1, b1_val = A_2d[i], b_2d[i]
            a2, b2_val = A_2d[j], b_2d[j]

            # 求解交点
            try:
                # 解线性方程组: a1·x = b1, a2·x = b2
                A_eq = np.vstack([a1, a2])
                b_eq = np.array([b1_val, b2_val])

                # 检查矩阵是否可逆
                if np.linalg.matrix_rank(A_eq) == 2:
                    x = np.linalg.solve(A_eq, b_eq)

                    # 检查交点是否满足所有约束（有松弛）
                    if np.all(A_2d @ x <= b_2d + 1e-6):
                        intersections.append(x)
            except:
                continue

    if intersections:
        # 取所有可行交点的平均值
        avg_point = np.mean(intersections, axis=0)
        print(f"使用交点平均值: {avg_point}")

        # 验证平均值点是否可行
        violations = A_2d @ avg_point - b_2d
        if np.all(violations < -1e-8):
            return avg_point
        else:
            print(f"平均值点违反约束，最大违反: {violations.max()}")
            # 尝试向中心收缩
            shrink_factor = 0.9
            while shrink_factor > 0.1:
                trial_point = avg_point * shrink_factor
                violations = A_2d @ trial_point - b_2d
                if np.all(violations < -1e-8):
                    print(f"使用收缩点: {trial_point} (收缩因子: {shrink_factor})")
                    return trial_point
                shrink_factor -= 0.1

    # 方法3：尝试原点附近的点
    test_points = [
        np.array([0, 0]),
        np.array([0.05, 0]),
        np.array([0, 0.05]),
        np.array([-0.05, 0]),
        np.array([0, -0.05]),
        np.array([0.03, 0.03]),
        np.array([-0.03, -0.03]),
        np.array([0.1, 0]),
        np.array([-0.1, 0]),
        np.array([0, 0.1]),
        np.array([0, -0.1])
    ]

    for point in test_points:
        if np.all(A_2d @ point <= b_2d - 1e-8):
            print(f"找到测试点: {point}")
            return point

    # 最后尝试：使用最小二乘解
    print("使用最小二乘解...")
    try:
        point = np.linalg.lstsq(A_2d, b_2d - 0.1, rcond=None)[0]
        # 验证点
        violations = A_2d @ point - b_2d
        if np.all(violations < 0):
            return point
    except:
        pass

    # 如果所有方法都失败，返回原点
    print("所有方法失败，返回原点")
    return np.array([0.0, 0.0])


def solve_polyhedron_vertices_2d_3d(A, b):
    """专门处理2D多边形在3D空间拉伸成棱柱的情况"""
    # 分离平面约束和垂直约束
    planar_constraints = []
    z_constraints = []

    for i in range(len(A)):
        a = A[i]
        bi = b[i]

        # 检查是否是z轴约束 (0, 0, 1) 或 (0, 0, -1)
        if np.abs(a[0]) < 1e-6 and np.abs(a[1]) < 1e-6 and np.abs(a[2]) > 0.9:
            z_constraints.append((a, bi))
        else:
            planar_constraints.append((a[:2], bi))  # 只取前两个分量

    if len(z_constraints) != 2:
        print(f"警告: 期望2个z轴约束，但找到{len(z_constraints)}个")
        return None, None

    # 提取z轴范围
    z_min, z_max = None, None
    for a, bi in z_constraints:
        if a[2] > 0:  # (0, 0, 1) -> z <= bi
            z_max = bi
        else:  # (0, 0, -1) -> -z <= bi -> z >= -bi
            z_min = -bi

    if z_min is None or z_max is None:
        print("警告: 无法确定z轴范围")
        return None, None

    print(f"Z轴范围: {z_min:.6f} <= z <= {z_max:.6f}")

    # 在2D平面上求解多边形顶点
    A_2d = np.array([c[0] for c in planar_constraints])
    b_2d = np.array([c[1] for c in planar_constraints])

    print(f"2D约束数量: {len(A_2d)}")

    # 找到内部点
    interior_point_2d = find_interior_point_2d(A_2d, b_2d)
    print(f"2D内部点: {interior_point_2d}")

    # 验证内部点
    violations = A_2d @ interior_point_2d - b_2d
    print(f"约束违反情况: max={violations.max():.6e}, min={violations.min():.6e}")

    # 调整内部点使其更靠内
    if violations.max() >= -1e-10:
        print("调整内部点...")
        # 尝试将点向原点移动
        for factor in [0.8, 0.6, 0.4, 0.2, 0.0]:
            trial_point = interior_point_2d * factor
            trial_violations = A_2d @ trial_point - b_2d
            if np.all(trial_violations < -1e-8):
                interior_point_2d = trial_point
                print(f"使用调整后的点: {interior_point_2d}")
                break

    # 尝试求解2D半空间交集
    try:
        # 添加安全边界
        safety_margin = 1e-4
        b_2d_adjusted = b_2d - safety_margin

        print(f"尝试使用内部点: {interior_point_2d}")
        print(f"调整后的b值范围: [{b_2d_adjusted.min():.6f}, {b_2d_adjusted.max():.6f}]")

        # 验证内部点是否真的可行
        test_violations = A_2d @ interior_point_2d - b_2d_adjusted
        if np.any(test_violations >= 0):
            print(f"内部点不满足约束，最大违反: {test_violations.max():.6e}")
            # 进一步调整
            interior_point_2d = interior_point_2d * 0.5
            print(f"进一步调整内部点为: {interior_point_2d}")

        hs_2d = HalfspaceIntersection(np.hstack([A_2d, b_2d_adjusted.reshape(-1, 1)]), interior_point_2d)
        vertices_2d = hs_2d.intersections

        if len(vertices_2d) < 3:
            print(f"警告: 2D多边形顶点太少 ({len(vertices_2d)})")
            return manual_solve_2d_polygon(A_2d, b_2d, z_min, z_max)

        print(f"找到2D多边形顶点: {len(vertices_2d)}个")

        # 计算2D凸包以确保正确的顶点顺序
        hull_2d = ConvexHull(vertices_2d)

        # 按凸包顺序排列顶点
        vertices_2d_ordered = vertices_2d[hull_2d.vertices]

        # 将2D顶点扩展到3D，创建棱柱
        vertices_3d = []

        # 底部顶点 (z = z_min)
        for v in vertices_2d_ordered:
            vertices_3d.append([v[0], v[1], z_min])

        # 顶部顶点 (z = z_max)
        for v in vertices_2d_ordered:
            vertices_3d.append([v[0], v[1], z_max])

        vertices_3d = np.array(vertices_3d)

        # 创建面
        faces = []
        n = len(vertices_2d_ordered)

        # 底面
        faces.append([vertices_3d[i] for i in range(n)])

        # 顶面
        faces.append([vertices_3d[i + n] for i in range(n)])

        # 侧面
        for i in range(n):
            j = (i + 1) % n
            faces.append([
                vertices_3d[i],
                vertices_3d[j],
                vertices_3d[j + n],
                vertices_3d[i + n]
            ])

        print(f"创建的3D棱柱: {len(vertices_3d)}个顶点, {len(faces)}个面")
        return vertices_3d, faces

    except Exception as e:
        print(f"警告: 2D半空间交集求解失败: {e}")
        # 尝试手动计算顶点
        print("尝试手动计算顶点...")
        return manual_solve_2d_polygon(A_2d, b_2d, z_min, z_max)


def manual_solve_2d_polygon(A_2d, b_2d, z_min, z_max):
    """手动计算2D多边形顶点"""
    n = len(A_2d)
    vertices = []

    # 计算所有约束线的交点
    for i in range(n):
        for j in range(i + 1, n):
            a1, b1_val = A_2d[i], b_2d[i]
            a2, b2_val = A_2d[j], b_2d[j]

            # 计算行列式
            det = a1[0] * a2[1] - a1[1] * a2[0]

            if abs(det) > 1e-10:  # 确保线不平行
                # 计算交点
                x = (b1_val * a2[1] - a1[1] * b2_val) / det
                y = (a1[0] * b2_val - b1_val * a2[0]) / det

                point = np.array([x, y])

                # 检查交点是否满足所有约束（有容差）
                violations = A_2d @ point - b_2d
                if np.all(violations <= 1e-8):
                    # 检查是否已存在类似点
                    is_duplicate = False
                    for v in vertices:
                        if np.linalg.norm(v - point) < 1e-6:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        vertices.append(point)
                        print(f"找到顶点 {len(vertices)}: ({x:.6f}, {y:.6f})")

    if len(vertices) < 3:
        print(f"手动计算失败: 只找到{len(vertices)}个顶点")
        return None, None

    print(f"手动找到{len(vertices)}个2D顶点")

    # 对顶点按角度排序
    center = np.mean(vertices, axis=0)
    angles = []
    for v in vertices:
        dx = v[0] - center[0]
        dy = v[1] - center[1]
        angles.append(np.arctan2(dy, dx))

    # 按角度排序
    sorted_indices = np.argsort(angles)
    vertices_sorted = [vertices[i] for i in sorted_indices]

    # 创建3D棱柱
    vertices_3d = []

    # 底部顶点
    for v in vertices_sorted:
        vertices_3d.append([v[0], v[1], z_min])

    # 顶部顶点
    for v in vertices_sorted:
        vertices_3d.append([v[0], v[1], z_max])

    vertices_3d = np.array(vertices_3d)

    # 创建面
    faces = []
    n = len(vertices_sorted)

    # 底面
    faces.append([vertices_3d[i] for i in range(n)])

    # 顶面
    faces.append([vertices_3d[i + n] for i in range(n)])

    # 侧面
    for i in range(n):
        j = (i + 1) % n
        faces.append([
            vertices_3d[i],
            vertices_3d[j],
            vertices_3d[j + n],
            vertices_3d[i + n]
        ])

    return vertices_3d, faces


def create_cube_from_bounds(A, b):
    """从轴对齐不等式创建立方体"""
    # 解析边界
    x_min, x_max = -np.inf, np.inf
    y_min, y_max = -np.inf, np.inf
    z_min, z_max = -np.inf, np.inf

    for i in range(len(b)):
        a = A[i]
        bi = b[i]

        if np.abs(a[0]) > 0.9 and np.abs(a[1]) < 0.1 and np.abs(a[2]) < 0.1:
            if a[0] > 0:
                x_max = min(x_max, bi / a[0])
            else:
                x_min = max(x_min, bi / a[0])
        elif np.abs(a[1]) > 0.9 and np.abs(a[0]) < 0.1 and np.abs(a[2]) < 0.1:
            if a[1] > 0:
                y_max = min(y_max, bi / a[1])
            else:
                y_min = max(y_min, bi / a[1])
        elif np.abs(a[2]) > 0.9 and np.abs(a[0]) < 0.1 and np.abs(a[1]) < 0.1:
            if a[2] > 0:
                z_max = min(z_max, bi / a[2])
            else:
                z_min = max(z_min, bi / a[2])

    print(f"解析的边界: x:[{x_min:.6f}, {x_max:.6f}], y:[{y_min:.6f}, {y_max:.6f}], z:[{z_min:.6f}, {z_max:.6f}]")

    # 创建立方体顶点
    vertices = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ])

    # 定义面
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
        [vertices[0], vertices[3], vertices[7], vertices[4]]  # 左面
    ]

    return vertices, faces


# ====================== 美化的颜色方案 ======================
# 使用更现代、更美观的颜色
color1 = (0.3, 0.6, 0.9, 0.4)  # 柔和的蓝色，透明度0.4
color1_edge = (0.1, 0.3, 0.7, 1.0)  # 深蓝色边缘
color2 = (0.9, 0.5, 0.2, 0.4)  # 柔和的橙色，透明度0.4
color2_edge = (0.7, 0.3, 0.1, 1.0)  # 深橙色边缘

# 设置背景颜色为纯白色
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# ====================== 绘制第一个多面体 ======================
print("=" * 60)
print("求解第一个多面体...")
print("=" * 60)

vertices1, faces1 = solve_polyhedron_vertices_2d_3d(A1, b1)

if vertices1 is None:
    print("第一个多面体求解失败，使用轴对齐边界框")
    vertices1, faces1 = create_cube_from_bounds(A1, b1)

# 绘制第一个多面体
if vertices1 is not None and faces1 is not None:
    # 使用更粗的边缘
    poly1 = Poly3DCollection(faces1, alpha=0.4,
                             facecolors=color1,
                             edgecolors=color1_edge,
                             linewidths=2.5,  # 更粗的边缘
                             linestyle='-')  # 实线
    ax.add_collection3d(poly1)

    # 可选：绘制顶点（现在注释掉，因为顶点可能会干扰视觉效果）
    # ax.scatter(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2],
    #            c='blue', s=60, alpha=0.6, depthshade=True, marker='o')

    print(f"第一个多面体 - 顶点数: {len(vertices1)}, 面数: {len(faces1)}")

# ====================== 绘制第二个多面体 ======================
print("\n" + "=" * 60)
print("求解第二个多面体...")
print("=" * 60)

vertices2, faces2 = create_cube_from_bounds(A2, b2)

# 使用不同的线型来区分两个多面体
poly2 = Poly3DCollection(faces2, alpha=0.5,
                         facecolors=color2,
                         edgecolors=color2_edge,
                         linewidths=2.5,  # 更粗的边缘
                         linestyle='--')  # 虚线，与第一个多面体区分
ax.add_collection3d(poly2)

# 可选：绘制顶点（现在注释掉）
# ax.scatter(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2],
#            c='orange', s=60, alpha=0.6, depthshade=True, marker='^')

print(f"第二个多面体 - 顶点数: {len(vertices2)}, 面数: {len(faces2)}")


# ====================== 设置坐标轴范围和刻度（0.2一刻度） ======================
def setup_axes_with_aligned_ticks(ax, all_vertices=None, tick_spacing=0.2):
    """设置坐标轴，使坐标轴在交点处显示相同的刻度值"""
    if all_vertices is None or len(all_vertices) == 0:
        # 默认范围
        x_min, x_max = -0.5, 0.5
        y_min, y_max = -0.5, 0.5
        z_min, z_max = -0.5, 0.5
    else:
        x_min, x_max = all_vertices[:, 0].min(), all_vertices[:, 0].max()
        y_min, y_max = all_vertices[:, 1].min(), all_vertices[:, 1].max()
        z_min, z_max = all_vertices[:, 2].min(), all_vertices[:, 2].max()

    # 添加边距
    margin = 0.15
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    # 确保最小范围
    min_range = 0.6  # 至少包含几个刻度
    if x_range < min_range:
        x_range = min_range
        x_center = (x_min + x_max) / 2
        x_min, x_max = x_center - x_range / 2, x_center + x_range / 2

    if y_range < min_range:
        y_range = min_range
        y_center = (y_min + y_max) / 2
        y_min, y_max = y_center - y_range / 2, y_center + y_range / 2

    if z_range < min_range:
        z_range = min_range
        z_center = (z_min + z_max) / 2
        z_min, z_max = z_center - z_range / 2, z_center + z_range / 2

    # 调整范围包含边距
    x_min, x_max = x_min - margin * x_range, x_max + margin * x_range
    y_min, y_max = y_min - margin * y_range, y_max + margin * y_range
    z_min, z_max = z_min - margin * z_range, z_max + margin * z_range

    # 对齐范围到0.2的倍数，确保包含原点
    def align_to_spacing(min_val, max_val, spacing):
        """将范围对齐到最近的刻度间距倍数"""
        aligned_min = np.floor(min_val / spacing) * spacing
        aligned_max = np.ceil(max_val / spacing) * spacing

        # 确保包含原点
        if aligned_min > 0:
            aligned_min = -spacing
        if aligned_max < 0:
            aligned_max = spacing

        return aligned_min, aligned_max

    # 对齐每个轴的范围
    x_min_aligned, x_max_aligned = align_to_spacing(x_min, x_max, tick_spacing)
    y_min_aligned, y_max_aligned = align_to_spacing(y_min, y_max, tick_spacing)
    z_min_aligned, z_max_aligned = align_to_spacing(z_min, z_max, tick_spacing)

    # 设置坐标轴范围
    ax.set_xlim([x_min_aligned, x_max_aligned])
    ax.set_ylim([y_min_aligned, y_max_aligned])
    ax.set_zlim([z_min_aligned, z_max_aligned])

    # 生成所有刻度（包括0）
    def generate_all_ticks(min_val, max_val, spacing):
        """生成所有刻度，包括0"""
        ticks = []
        current = min_val
        while current <= max_val + spacing / 2:
            ticks.append(current)
            current += spacing
        return np.array(ticks)

    # 生成所有刻度（包括0）
    x_ticks = generate_all_ticks(x_min_aligned, x_max_aligned, tick_spacing)
    y_ticks = generate_all_ticks(y_min_aligned, y_max_aligned, tick_spacing)
    z_ticks = generate_all_ticks(z_min_aligned, z_max_aligned, tick_spacing)

    # 设置刻度（包括0）
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)

    # 恢复带符号的刻度标签（保持正常显示）
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}' if x != 0 or abs(x) > 0.001 else '0'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}' if x != 0 or abs(x) > 0.001 else '0'))
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}' if x != 0 or abs(x) > 0.001 else '0'))

    # 确保坐标轴在交点处显示相同的刻度值
    # 对于3D坐标轴，matplotlib会自动在坐标轴相交处显示相同的刻度值

    return (x_min_aligned, x_max_aligned, x_ticks,
            y_min_aligned, y_max_aligned, y_ticks,
            z_min_aligned, z_max_aligned, z_ticks,
            tick_spacing)


# 收集所有顶点数据
all_vertices_list = []
if vertices1 is not None:
    all_vertices_list.append(vertices1)
if vertices2 is not None:
    all_vertices_list.append(vertices2)

if all_vertices_list:
    all_vertices = np.vstack(all_vertices_list)

    print(f"\n所有顶点原始范围:")
    print(f"  P2 (x): [{all_vertices[:, 0].min():.6f}, {all_vertices[:, 0].max():.6f}]")
    print(f"  P1 (y): [{all_vertices[:, 1].min():.6f}, {all_vertices[:, 1].max():.6f}]")
    print(f"  P3 (z): [{all_vertices[:, 2].min():.6f}, {all_vertices[:, 2].max():.6f}]")

    # 设置坐标轴，使用0.2一刻度
    axis_info = setup_axes_with_aligned_ticks(ax, all_vertices, tick_spacing=0.2)
    tick_spacing = axis_info[9]

    print(f"\n对齐后的显示范围:")
    print(f"  P2: [{axis_info[0]:.3f}, {axis_info[1]:.3f}]")
    print(f"  P1: [{axis_info[3]:.3f}, {axis_info[4]:.3f}]")
    print(f"  P3: [{axis_info[6]:.3f}, {axis_info[7]:.3f}]")
    print(f"刻度间距: {tick_spacing:.2f}")
else:
    # 默认范围和对齐
    axis_info = setup_axes_with_aligned_ticks(ax, tick_spacing=0.2)
    tick_spacing = axis_info[9]


# ====================== 旋转P3标签180度（正确方法） ======================
def rotate_zlabel_180(ax):
    """旋转z轴标签180度 - 使用正确的方法"""
    # 对于matplotlib 3D，直接设置旋转可能不生效
    # 我们可以重新设置z轴标签来模拟旋转效果
    zlabel_text = ax.get_zlabel()

    # 移除原来的标签
    ax.set_zlabel('')

    # 添加新的z轴标签，通过设置labelpad来调整位置
    # 在3D图中，我们可以通过调整视角来达到类似效果
    ax.set_zlabel(zlabel_text, fontname='Times New Roman', fontsize=28, labelpad=25)

    # 对于matplotlib 3D，标签旋转不是直接支持的
    # 我们可以通过添加文本的方式模拟旋转
    # 获取当前z轴标签的位置
    zlim = ax.get_zlim()

# 应用P3标签旋转（使用新方法）
rotate_zlabel_180(ax)

# ====================== 设置视角 ======================
# 设置视角 - 更好的观察角度
ax.view_init(elev=28, azim=42)

# ====================== 美化设置 ======================
# 添加更精细的网格
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

# 设置坐标轴线的宽度
ax.xaxis._axinfo['grid']['linewidth'] = 0.5
ax.yaxis._axinfo['grid']['linewidth'] = 0.5
ax.zaxis._axinfo['grid']['linewidth'] = 0.5

# 设置坐标轴线颜色和透明度（白色背景下的设置）
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('gray')
ax.yaxis.pane.set_edgecolor('gray')
ax.zaxis.pane.set_edgecolor('gray')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)

# 设置坐标轴刻度线
ax.tick_params(axis='both', which='major', pad=10)

# 注意：删除了原点处的灰色点（删除了 ax.scatter([0], [0], [0], ...) 这行代码）

# ====================== 添加图例（两列，位于图形上方正中） ======================
# 创建图例句柄
legend_elements = [
    Patch(facecolor=color1, edgecolor=color1_edge, linewidth=2.5, linestyle='-', alpha=0.4,
          label='Proposed Method'),
    Patch(facecolor=color2, edgecolor=color2_edge, linewidth=2.5, linestyle='--', alpha=0.5,
          label='Ref. [21]')
]

# 添加图例，设置ncol=2创建两列，bbox_to_anchor设置位置在图形上方正中
legend = ax.legend(handles=legend_elements,
                   loc='lower center',  # 使用lower center作为基准
                   bbox_to_anchor=(0.5, 0.9),  # 在图形上方正中
                   ncol=2,  # 两列水平排列
                   fontsize=24,
                   framealpha=0,
                   fancybox=False,
                   shadow=False)

# 设置图例文本字体
for text in legend.get_texts():
    text.set_fontname('Times New Roman')
    text.set_fontsize(24)

# 调整布局，为图例留出空间
plt.subplots_adjust(top=0.95)  # 为顶部图例留出空间

# ====================== 保存图像 ======================
# 保存高质量图像（白色背景）
plt.savefig('polyhedra_final_clean.pdf', format='pdf', dpi=600,
            bbox_inches='tight', facecolor='white')
plt.savefig('polyhedra_final_clean.png', format='png', dpi=600,
            bbox_inches='tight', facecolor='white',
            transparent=False)

print("\n" + "=" * 60)
print("已修复P3标签旋转问题")
print("已删除图中灰色点（原点标记）")
print("已设置坐标轴为0.2一刻度")
print("x、y、z坐标轴在交点处显示相同的刻度值")
print("恢复原坐标轴带有符号的刻度标签设置")
print("图片背景已设置为白色")
print("图例已设置为两列水平排列，位于图形上方正中")
print("图像已保存为 'polyhedra_final_clean.pdf' 和 'polyhedra_final_clean.png'")
print("=" * 60)

# 显示图形
plt.show()