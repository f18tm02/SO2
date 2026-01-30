import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# ==========================================
# 1. 参数设置 (Parameters Setup)
# ==========================================
TARGET_MASS = 100 * 10**6      # 1亿吨
START_YEAR = 2050
DISCOUNT_RATE = 0.03           # 贴现率 3%

# --- 约束条件 (Constraints) ---
MIN_ANNUAL_SUPPLY = 1.5 * 10**6  # 每年最少运 150万吨
MAX_THROUGHPUT = 5.0 * 10**6     # 港口每年最大吞吐 500万吨

# --- 初始成本 (Initial Costs) ---
COST_DIRECT_INIT = 1.9 * 10**6   # 火箭初始成本 ($/ton)
COST_HYBRID_INIT = 0.3 * 10**6   # 电梯初始成本 ($/ton)

# --- 技术进化参数 (Tech Evolution) ---
# 1. 太空电梯 (指数增长)
SE_INITIAL_CAP = 0.54 * 10**6
SE_GROWTH_RATE = 0.05            # 运力年增长 5%
SE_COST_DECAY = 0.01             # 成本年下降 1%

# 2. 火箭 (稳步改进)
ROCKET_INITIAL_CAP = 2.0 * 10**6
ROCKET_CAP_GROWTH = 0.03         # 运力年增长 3%
ROCKET_COST_DECAY = 0.02         # 成本年下降 2%

# ==========================================
# 2. 迭代优化生成帕累托前沿 (Iterative Optimization)
# ==========================================
T_min = 10
T_max = 100
feasible_years = []
min_costs = []

print('正在模拟帕累托前沿 (Simulating Pareto Frontier)...')

# 抑制 linprog 的输出信息
options = {'disp': False}

for T in range(T_min, T_max + 1):
    num_vars = 2 * T
    years_idx = np.arange(T)
    
    # --- 构建目标函数系数 c (Costs) ---
    # Python的linprog形式是 min c^T x
    # 计算每年的单位成本
    c_rocket = COST_DIRECT_INIT * ((1 - ROCKET_COST_DECAY) ** years_idx)
    c_elev = COST_HYBRID_INIT * ((1 - SE_COST_DECAY) ** years_idx)
    
    # 计算贴现因子
    disc = (1 + DISCOUNT_RATE) ** (-years_idx)
    
    # 最终的目标函数系数向量 [c_rocket_0...c_rocket_T, c_elev_0...c_elev_T]
    # 注意：linprog默认求解变量顺序，我们需要将火箭和电梯的变量拼接
    # 为了方便，我们将变量设为 [R_0, R_1..., E_0, E_1...] (前半部分火箭，后半部分电梯)
    f = np.concatenate([c_rocket * disc, c_elev * disc])
    
    # --- 构建约束条件 A_eq, b_eq (等式约束) ---
    # 总运量 = TARGET_MASS
    # sum(R_t) + sum(E_t) = TARGET_MASS
    A_eq = np.ones((1, num_vars))
    b_eq = np.array([TARGET_MASS])
    
    # --- 构建约束条件 A_ub, b_ub (不等式约束) ---
    # linprog 默认是 A_ub * x <= b_ub
    
    # 1. 最小年供应量: R_t + E_t >= MIN_ANNUAL_SUPPLY
    # 变换为: -R_t - E_t <= -MIN_ANNUAL_SUPPLY
    A_min = np.zeros((T, num_vars))
    b_min = -MIN_ANNUAL_SUPPLY * np.ones(T)
    
    # 2. 最大吞吐量: R_t + E_t <= MAX_THROUGHPUT
    A_max = np.zeros((T, num_vars))
    b_max = MAX_THROUGHPUT * np.ones(T)
    
    for t in range(T):
        # 填充矩阵：第t年火箭(索引t) 和 第t年电梯(索引T+t)
        # 最小约束
        A_min[t, t] = -1
        A_min[t, T + t] = -1
        # 最大约束
        A_max[t, t] = 1
        A_max[t, T + t] = 1
        
    A_ub = np.vstack([A_min, A_max])
    b_ub = np.concatenate([b_min, b_max])
    
    # --- 构建变量边界 bounds ---
    # 0 <= R_t <= Rocket_Cap_t
    # 0 <= E_t <= Elev_Cap_t
    
    caps_rock = ROCKET_INITIAL_CAP * ((1 + ROCKET_CAP_GROWTH) ** years_idx)
    caps_elev = SE_INITIAL_CAP * ((1 + SE_GROWTH_RATE) ** years_idx)
    
    # 生成bounds列表 [(0, cap_r_0), ..., (0, cap_e_0), ...]
    bounds = []
    # 火箭的边界
    for cap in caps_rock:
        bounds.append((0, cap))
    # 电梯的边界
    for cap in caps_elev:
        bounds.append((0, cap))
        
    # --- 求解 ---
    res = linprog(c=f, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res.success:
        feasible_years.append(START_YEAR + T)
        min_costs.append(res.fun / 10**12) # 转为万亿

# 转换为numpy数组方便处理
feasible_years = np.array(feasible_years)
min_costs = np.array(min_costs)

# ==========================================
# 3. 拐点分析 (Knee Point Analysis)
# ==========================================
if len(feasible_years) > 0:
    # 归一化
    y_norm = (feasible_years - feasible_years.min()) / (feasible_years.max() - feasible_years.min())
    c_norm = (min_costs - min_costs.min()) / (min_costs.max() - min_costs.min())
    
    # 起点和终点
    P_start = np.array([y_norm[0], c_norm[0]])
    P_end = np.array([y_norm[-1], c_norm[-1]])
    Vec_line = P_end - P_start
    
    # 计算距离
    dists = []
    for i in range(len(y_norm)):
        P_curr = np.array([y_norm[i], c_norm[i]])
        # 点到直线的距离公式 (二维向量叉乘模长 / 底边长)
        # np.cross在二维时返回标量
        d = np.abs(np.cross(Vec_line, P_curr - P_start)) / np.linalg.norm(Vec_line)
        dists.append(d)
        
    best_idx = np.argmax(dists)
    best_year = feasible_years[best_idx]
    best_cost = min_costs[best_idx]
    
    print(f"推荐最佳完成年份: {best_year}, 预估成本: ${best_cost:.2f} Trillion")

    # ==========================================
    # 4. 可视化 1: 帕累托前沿 (Pareto Frontier)
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(feasible_years, min_costs, linewidth=3, color='#9C27B0', label='Pareto Frontier')
    plt.scatter(best_year, best_cost, s=150, c='red', marker='p', label='Optimal Solution', zorder=5)
    
    plt.xlabel('Completion Year', fontsize=12)
    plt.ylabel('Cost (Trillion USD)', fontsize=12)
    plt.title('Fig 1: Pareto Frontier (Fair Competition)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    # ==========================================
    # 5. 提取并绘制最佳年份的详细计划
    # ==========================================
    print(f'正在提取 {best_year} 年的详细计划...')
    
    T_opt = best_year - START_YEAR
    num_vars = 2 * T_opt
    years_idx = np.arange(T_opt)
    
    # --- 重新构建 T_opt 的参数 (同上) ---
    c_rocket = COST_DIRECT_INIT * ((1 - ROCKET_COST_DECAY) ** years_idx)
    c_elev = COST_HYBRID_INIT * ((1 - SE_COST_DECAY) ** years_idx)
    disc = (1 + DISCOUNT_RATE) ** (-years_idx)
    f = np.concatenate([c_rocket * disc, c_elev * disc])
    
    A_eq = np.ones((1, num_vars))
    b_eq = np.array([TARGET_MASS])
    
    A_min = np.zeros((T_opt, num_vars))
    b_min = -MIN_ANNUAL_SUPPLY * np.ones(T_opt)
    A_max = np.zeros((T_opt, num_vars))
    b_max = MAX_THROUGHPUT * np.ones(T_opt)
    
    for t in range(T_opt):
        A_min[t, t] = -1
        A_min[t, T_opt + t] = -1
        A_max[t, t] = 1
        A_max[t, T_opt + t] = 1
        
    A_ub = np.vstack([A_min, A_max])
    b_ub = np.concatenate([b_min, b_max])
    
    caps_rock = ROCKET_INITIAL_CAP * ((1 + ROCKET_CAP_GROWTH) ** years_idx)
    caps_elev = SE_INITIAL_CAP * ((1 + SE_GROWTH_RATE) ** years_idx)
    
    bounds = []
    for cap in caps_rock: bounds.append((0, cap))
    for cap in caps_elev: bounds.append((0, cap))
        
    # --- 求解具体调度 ---
    res_opt = linprog(c=f, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if res_opt.success:
        # 分离变量
        sched_rocket = res_opt.x[:T_opt] / 10**6       # 百万吨
        sched_elevator = res_opt.x[T_opt:] / 10**6     # 百万吨
        years_axis = np.arange(START_YEAR + 1, best_year + 1)
        
        # ==========================================
        # 6. 可视化 2: 堆叠柱状图 (Detailed Schedule)
        # ==========================================
        plt.figure(figsize=(12, 7))
        
        # 绘制堆叠柱状图
        # 注意：matplotlib的bar stacked需要使用bottom参数
        plt.bar(years_axis, sched_elevator, color='#42A5F5', label='Elevator Flow', alpha=0.9)
        plt.bar(years_axis, sched_rocket, bottom=sched_elevator, color='#EF5350', label='Rocket Flow', alpha=0.9)
        
        # 绘制运力上限线
        plt.plot(years_axis, caps_elev / 10**6, 'b--', linewidth=2, label='Elevator Limit')
        plt.plot(years_axis, caps_rock / 10**6, 'r--', linewidth=2, label='Rocket Limit')
        
        # 绘制港口限制
        plt.axhline(y=MAX_THROUGHPUT / 10**6, color='k', linestyle=':', linewidth=2, label='Port Limit')
        
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Transport Mass (Million Tons)', fontsize=12)
        plt.title(f'Fig 2: Optimal Logistics Schedule (Target Year: {best_year})', fontsize=14)
        plt.suptitle('Hybrid Strategy under Fair Competition Model', fontsize=12, y=0.92)
        
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(START_YEAR, best_year + 1)
        plt.show()
        
    else:
        print("Error: 无法求解最佳年份的详细调度。")

else:
    print("未找到可行解 (No feasible solution found). 请检查约束条件。")
