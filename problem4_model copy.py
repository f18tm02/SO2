# -*- coding: utf-8 -*-
"""
=============================================================================
MCM 2026 Problem B - Question 4: NSGA-III Optimization
基于NSGA-III算法的多目标环境影响优化模型
Multi-Objective Evolutionary Optimization using NSGA-III
=============================================================================

作者: MCM Team MI2601448 (Powered by Gemini 3 Pro Thinking)
日期: 2026年2月

核心算法:
    NSGA-III (Non-dominated Sorting Genetic Algorithm III)
    用于解决高维多目标优化问题 (Many-Objective Optimization)

决策变量 (Decision Variables):
    x1: 电梯运输比例 (Elevator Allocation Ratio) [0, 1]
    x2: 火箭绿色技术投入等级 (Rocket Green Tech Level) [0, 1] - 增加成本，减少排放
    x3: 电梯清洁能源投入等级 (Elevator Clean Energy Level) [0, 1] - 增加成本，减少排放

目标函数 (Objectives) - Minimize all:
    f1: 总成本 (Total Cost, Trillion USD)
    f2: 建设工期 (Construction Time, Years)
    f3: 总碳排放 (Total Emissions, Million Tons CO2)

依赖库: numpy, matplotlib, pymoo, json
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
from datetime import datetime

# 引入pymoo核心组件
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from pymoo.visualization.pcp import PCP


# =============================================================================
# 全局设置 & 参数
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'pareto': '#4C72B0',
    'best': '#C44E52',
    'base': '#8C8C8C'
}

os.makedirs('figures_nsga3', exist_ok=True)
os.makedirs('results', exist_ok=True)

class ModelParams:
    # 基础物理参数
    TOTAL_MATERIAL = 100_000_000      # 1亿吨
    
    # 运力参数 (吨/年)
    CAPACITY_ELEVATOR = 526_260
    CAPACITY_ROCKET = 406_250
    
    # 基础成本 (美元/公斤)
    COST_BASE_ELEVATOR = 100
    COST_BASE_ROCKET = 1500
    
    # 基础排放 (吨CO2/吨载荷)
    EMISSION_BASE_ELEVATOR = 0.1
    EMISSION_BASE_ROCKET = 4.0
    
    # 技术升级参数
    # 火箭绿色技术: 每投入10%成本增加，减少15%排放 (非线性收益递减)
    ROCKET_TECH_COST_FACTOR = 0.5     # 最大增加50%成本
    ROCKET_TECH_EMISSION_REDUCT = 0.4 # 最大减少40%排放
    
    # 电梯清洁能源: 每投入5%成本增加，减少60%排放
    ELEVATOR_TECH_COST_FACTOR = 0.2   # 最大增加20%成本
    ELEVATOR_TECH_EMISSION_REDUCT = 0.8 # 最大减少80%排放

# =============================================================================
# 定义优化问题 (pymoo Problem Class)
# =============================================================================
class MoonColonyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=3,             # 3个决策变量
            n_obj=3,             # 3个目标函数 (Cost, Time, CO2)
            n_ieq_constr=0,      # 0个不等式约束
            xl=np.array([0.0, 0.0, 0.0]), # 下界
            xu=np.array([1.0, 1.0, 1.0])  # 上界
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        核心评估函数
        x[0]: alpha (电梯分配比例)
        x[1]: beta (火箭技术投入)
        x[2]: gamma (电梯技术投入)
        """
        alpha = x[0]
        beta = x[1]
        gamma = x[2]
        
        P = ModelParams
        
        # 1. 物资分配
        mat_elev = P.TOTAL_MATERIAL * alpha
        mat_rocket = P.TOTAL_MATERIAL * (1 - alpha)
        
        # 2. 计算动态单价和排放因子
        # 火箭: 成本随技术投入增加，排放随技术投入减少
        unit_cost_rocket = P.COST_BASE_ROCKET * (1 + beta * P.ROCKET_TECH_COST_FACTOR)
        unit_emission_rocket = P.EMISSION_BASE_ROCKET * (1 - beta * P.ROCKET_TECH_EMISSION_REDUCT)
        
        # 电梯: 成本随技术投入增加，排放随技术投入减少
        unit_cost_elev = P.COST_BASE_ELEVATOR * (1 + gamma * P.ELEVATOR_TECH_COST_FACTOR)
        unit_emission_elev = P.EMISSION_BASE_ELEVATOR * (1 - gamma * P.ELEVATOR_TECH_EMISSION_REDUCT)
        
        # 3. 计算目标函数 f1: 总成本 (Trillion USD)
        # Cost = (Material_kg * Unit_Cost) / 1e12
        cost = (mat_elev * 1000 * unit_cost_elev + 
                mat_rocket * 1000 * unit_cost_rocket) / 1e12
        
        # 4. 计算目标函数 f2: 工期 (Years)
        # 并行工作，取决于最慢的一方。如果某方分配为0，时间为0。
        time_elev = mat_elev / P.CAPACITY_ELEVATOR if mat_elev > 0 else 0
        time_rocket = mat_rocket / P.CAPACITY_ROCKET if mat_rocket > 0 else 0
        # 实际上系统是并行的，总工期由瓶颈决定
        if alpha == 0:
            time = time_rocket
        elif alpha == 1:
            time = time_elev
        else:
            time = max(time_elev, time_rocket)
            
        # 5. 计算目标函数 f3: 总排放 (Million Tons CO2)
        emission = (mat_elev * unit_emission_elev + 
                    mat_rocket * unit_emission_rocket) / 1e6
        
        # 输出结果 (pymoo要求minimize，这里三个都是越小越好，无需变号)
        out["F"] = [cost, time, emission]

# =============================================================================
# 可视化与分析函数
# =============================================================================
def run_nsga3_optimization():
    """执行NSGA-III优化"""
    print("[Optimization] Setting up NSGA-III Algorithm...")
    
    # 1. 创建参考方向 (Reference Directions)
    # NSGA-III 需要参考点来维持高维空间的多样性
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
    
    # 2. 定义算法
    algorithm = NSGA3(
        pop_size=100,            # 种群大小
        ref_dirs=ref_dirs,       # 参考方向
        n_offsprings=50,         # 每次繁衍后代数
        eliminate_duplicates=True
    )
    
    # 3. 定义终止条件
    termination = get_termination("n_gen", 150) # 迭代150代
    
    # 4. 执行优化
    print("[Optimization] Running evolution (this may take a moment)...")
    problem = MoonColonyProblem()
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=2026,               # 固定随机种子以便复现
        save_history=True,
        verbose=True
    )
    
    print(f"[Optimization] Done. Found {len(res.X)} non-dominated solutions.")
    return res

def analyze_solutions(res):
    """分析最优解集"""
    X = res.X # 决策变量 [alpha, beta, gamma]
    F = res.F # 目标函数 [Cost, Time, Emission]
    
    # 寻找特殊解
    # 1. 最低成本 (Index of min Cost)
    idx_min_cost = np.argmin(F[:, 0])
    # 2. 最短工期 (Index of min Time)
    idx_min_time = np.argmin(F[:, 1])
    # 3. 最低排放 (Index of min Emission)
    idx_min_emission = np.argmin(F[:, 2])
    # 4. 最佳折衷解 (使用归一化距离最近原点的点 - Utopia Point)
    # Normalize objectives
    F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-6)
    dist_to_ideal = np.linalg.norm(F_norm, axis=1)
    idx_balance = np.argmin(dist_to_ideal)
    
    special_solutions = {
        'Min Cost': {'idx': int(idx_min_cost), 'F': F[idx_min_cost], 'X': X[idx_min_cost]},
        'Min Time': {'idx': int(idx_min_time), 'F': F[idx_min_time], 'X': X[idx_min_time]},
        'Min Emission': {'idx': int(idx_min_emission), 'F': F[idx_min_emission], 'X': X[idx_min_emission]},
        'Best Compromise': {'idx': int(idx_balance), 'F': F[idx_balance], 'X': X[idx_balance]}
    }
    
    return special_solutions

def plot_3d_pareto(res, special_sols):
    """绘制3D帕累托前沿散点图"""
    F = res.F
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制所有非支配解
    sc = ax.scatter(F[:, 0], F[:, 1], F[:, 2], 
                    c=F[:, 2], cmap='viridis', s=40, alpha=0.8, edgecolors='grey')
    
    # 标记特殊解
    markers = {'Min Cost': 'v', 'Min Time': '^', 'Min Emission': 's', 'Best Compromise': '*'}
    colors = {'Min Cost': 'red', 'Min Time': 'orange', 'Min Emission': 'green', 'Best Compromise': 'magenta'}
    
    for name, data in special_sols.items():
        f_val = data['F']
        ax.scatter(f_val[0], f_val[1], f_val[2], c=colors[name], s=200, marker=markers[name], label=name, edgecolors='black', zorder=10)
    
    ax.set_xlabel('Total Cost ($ Trillion)', fontweight='bold')
    ax.set_ylabel('Construction Time (Years)', fontweight='bold')
    ax.set_zlabel('CO2 Emissions (Million Tons)', fontweight='bold')
    ax.set_title('3D Pareto Frontier (NSGA-III Optimization)', fontsize=14)
    
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Emissions (Color Scale)')
    ax.legend(loc='upper left')
    
    # 调整视角
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('figures_nsga3/nsga3_3d_pareto.png', dpi=300)
    print("  [Plot] 3D Pareto saved.")

def plot_parallel_coordinates(res):
    """绘制平行坐标图 (展示多维权衡)"""
    F = res.F
    
    # 归一化数据用于可视化
    F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 每一行代表一个解，颜色映射到排放量 (objective 2 in 0-indexed)
    # 使用matplotlib手动绘制，比pymoo原生更美观
    for i in range(len(F)):
        color = plt.cm.viridis(F_norm[i, 2]) # Color by Emission
        ax.plot(['Cost', 'Time', 'Emission'], F_norm[i], color=color, alpha=0.5)
        
    ax.set_title('Parallel Coordinates Plot of Optimal Solutions', fontsize=14)
    ax.set_ylabel('Normalized Objective Value (0=Best, 1=Worst)')
    ax.grid(True, axis='x')
    
    # 添加解释性文字
    ax.text(0, -0.1, f"Range: [{F[:,0].min():.1f}, {F[:,0].max():.1f}] T$", ha='center', transform=ax.get_xaxis_transform())
    ax.text(1, -0.1, f"Range: [{F[:,1].min():.0f}, {F[:,1].max():.0f}] Yrs", ha='center', transform=ax.get_xaxis_transform())
    ax.text(2, -0.1, f"Range: [{F[:,2].min():.1f}, {F[:,2].max():.1f}] Mt", ha='center', transform=ax.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig('figures_nsga3/nsga3_parallel_coords.png', dpi=300)
    print("  [Plot] Parallel Coordinates saved.")

def plot_decision_space(res):
    """绘制决策变量分布"""
    X = res.X
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Alpha (Elevator Ratio) vs Beta (Rocket Tech)
    axes[0].scatter(X[:, 0], X[:, 1], c=res.F[:, 2], cmap='viridis', alpha=0.7)
    axes[0].set_xlabel('Elevator Ratio (x1)')
    axes[0].set_ylabel('Rocket Green Tech (x2)')
    axes[0].set_title('Elevator Allocation vs. Rocket Tech')
    
    # 2. Alpha vs Gamma (Elevator Tech)
    axes[1].scatter(X[:, 0], X[:, 2], c=res.F[:, 2], cmap='viridis', alpha=0.7)
    axes[1].set_xlabel('Elevator Ratio (x1)')
    axes[1].set_ylabel('Elevator Clean Tech (x3)')
    axes[1].set_title('Elevator Allocation vs. Elevator Tech')
    
    # 3. Objectives Trade-off (Cost vs Time)
    sc = axes[2].scatter(res.F[:, 0], res.F[:, 1], c=res.F[:, 2], cmap='viridis', s=40)
    axes[2].set_xlabel('Cost ($ Trillion)')
    axes[2].set_ylabel('Time (Years)')
    axes[2].set_title('Cost vs. Time (Color=Emission)')
    plt.colorbar(sc, ax=axes[2], label='Emission (Mt)')
    
    plt.tight_layout()
    plt.savefig('figures_nsga3/nsga3_decision_space.png', dpi=300)
    print("  [Plot] Decision Space saved.")

def save_results(special_sols):
    """保存结果到JSON"""
    # Convert numpy types to native python types
    output = {}
    for name, data in special_sols.items():
        output[name] = {
            'Objectives': {
                'Cost_Trillion': float(data['F'][0]),
                'Time_Years': float(data['F'][1]),
                'Emission_Mt': float(data['F'][2])
            },
            'Decisions': {
                'Elevator_Ratio_x1': float(data['X'][0]),
                'Rocket_Tech_Level_x2': float(data['X'][1]),
                'Elevator_Tech_Level_x3': float(data['X'][2])
            }
        }
    
    with open('results/nsga3_optimization_results.json', 'w') as f:
        json.dump(output, f, indent=4)
    print("  [Data] JSON results saved.")
    
    # 打印简报
    print(" " + "="*60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*60)
    print(f"{'Scenario':<20} | {'Cost ($T)':<10} | {'Time (Yr)':<10} | {'CO2 (Mt)':<10}")
    print("-" * 60)
    for name, data in special_sols.items():
        f = data['F']
        print(f"{name:<20} | {f[0]:<10.2f} | {f[1]:<10.1f} | {f[2]:<10.1f}")
    print("="*60)
    
    # 建议文本
    comp = special_sols['Best Compromise']
    print("[AI Recommendation]")
    print(f"Based on NSGA-III analysis, the Best Compromise Strategy is:")
    print(f" > Allocate {comp['X'][0]*100:.1f}% material to Space Elevators.")
    print(f" > Invest {comp['X'][1]*100:.1f}% capability in Green Rocket Propulsion.")
    print(f" > Invest {comp['X'][2]*100:.1f}% capability in Clean Energy for Elevators.")
    print(f" > Expected Outcome: Cost ${comp['F'][0]:.2f}T, Time {comp['F'][1]:.1f} Years, CO2 {comp['F'][2]:.1f} Mt.")

def visualize_pareto_frontier_advanced(res, special_sols):
    """
    高级帕累托前沿可视化
    包含：
    1. 3D 交互式散点图 (带有投影，增强空间感)
    2. 2D 两两目标投影矩阵图 (更易于在论文中分析)
    """
    F = res.F  # 目标函数值矩阵 [Cost, Time, Emission]
    
    # =========================================================================
    # 图表 1: 高级 3D 帕累托前沿散点图 (3D Scatter Plot with Projections)
    # =========================================================================
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. 绘制主散点 (帕累托前沿面)
    # 使用颜色映射 (c=F[:, 2]) 代表碳排放强度
    img = ax.scatter(F[:, 0], F[:, 1], F[:, 2], 
                     c=F[:, 2], cmap='viridis', s=60, alpha=0.9, edgecolors='w', linewidth=0.5, label='Pareto Optimal Solutions')
    
    # 2. 标记特殊解 (Best Compromise, etc.)
    # 最佳折衷解 (星号)
    comp = special_sols['Best Compromise']['F']
    ax.scatter(comp[0], comp[1], comp[2], c='red', s=200, marker='*', 
               label='Best Compromise', edgecolors='black', zorder=10)
    
    # 最低成本 (圆点)
    cost_min = special_sols['Min Cost']['F']
    ax.scatter(cost_min[0], cost_min[1], cost_min[2], c='blue', s=100, marker='v', 
               label='Min Cost', edgecolors='white', zorder=10)

    # 3. 添加"投影" (Shadows) 到墙壁上，增强 3D 感
    # 这是一个让图表看起来更专业的技巧
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    
    # 投影到底面 (Cost vs Time) - 灰色影子
    ax.scatter(F[:, 0], F[:, 1], np.full_like(F[:, 2], zlim[0]), c='gray', s=20, alpha=0.2)
    # 投影到侧面 (Time vs Emission)
    ax.scatter(np.full_like(F[:, 0], xlim[0]), F[:, 1], F[:, 2], c='gray', s=20, alpha=0.2)
    # 投影到背面 (Cost vs Emission)
    ax.scatter(F[:, 0], np.full_like(F[:, 1], ylim[1]), F[:, 2], c='gray', s=20, alpha=0.2)

    # 4. 标签与美化
    ax.set_xlabel('Total Cost ($ Trillion)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Construction Time (Years)', fontsize=11, fontweight='bold')
    ax.set_zlabel('CO2 Emissions (Million Tons)', fontsize=11, fontweight='bold')
    ax.set_title('NSGA-III Pareto Frontier Surface(Trade-off between Cost, Time, and Environment)', fontsize=14)
    
    # 调整视角 (最佳观看角度)
    ax.view_init(elev=25, azim=135)
    
    # 添加颜色条
    cbar = plt.colorbar(img, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Emission Level (Color Scale)')
    
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('figures_nsga3/fig_pareto_3d_advanced.png', dpi=300)
    print("  [Plot] Generated Advanced 3D Pareto Frontier: figures_nsga3/fig_pareto_3d_advanced.png")

    # =========================================================================
    # 图表 2: 2D 投影矩阵 (2D Projection Matrix) - 论文常用
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 子图 1: Cost vs Time
    sc1 = axes[0].scatter(F[:, 0], F[:, 1], c=F[:, 2], cmap='viridis', s=40, edgecolors='grey')
    axes[0].scatter(comp[0], comp[1], c='red', marker='*', s=150, edgecolors='black', label='Compromise')
    axes[0].set_xlabel('Total Cost ($ Trillion)')
    axes[0].set_ylabel('Time (Years)')
    axes[0].set_title('View 1: Cost vs. Time')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 子图 2: Cost vs Emission
    sc2 = axes[1].scatter(F[:, 0], F[:, 2], c=F[:, 1], cmap='magma', s=40, edgecolors='grey') # Color by Time
    axes[1].scatter(comp[0], comp[2], c='red', marker='*', s=150, edgecolors='black')
    axes[1].set_xlabel('Total Cost ($ Trillion)')
    axes[1].set_ylabel('Emission (Million Tons)')
    axes[1].set_title('View 2: Cost vs. Emission')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    plt.colorbar(sc2, ax=axes[1], label='Time (Years)') # Colorbar specific to this plot
    
    # 子图 3: Time vs Emission
    sc3 = axes[2].scatter(F[:, 1], F[:, 2], c=F[:, 0], cmap='coolwarm', s=40, edgecolors='grey') # Color by Cost
    axes[2].scatter(comp[1], comp[2], c='red', marker='*', s=150, edgecolors='black')
    axes[2].set_xlabel('Time (Years)')
    axes[2].set_ylabel('Emission (Million Tons)')
    axes[2].set_title('View 3: Time vs. Emission')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    plt.colorbar(sc3, ax=axes[2], label='Cost ($ Trillion)')
    
    plt.tight_layout()
    plt.savefig('figures_nsga3/fig_pareto_2d_projections.png', dpi=300)
    print("  [Plot] Generated 2D Pareto Projections: figures_nsga3/fig_pareto_2d_projections.png")

# =============================================================================
# 整合到主程序
# =============================================================================
# 请在主流程的 plot_decision_space(res) 之后调用此函数：
# visualize_pareto_frontier_advanced(res, special_sols)


# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    print("MCM 2026 Problem B - Question 4 (NSGA-III Mode)")
    print("Identity: Gemini 3 Pro Thinking")
    
    try:
        # 1. 运行优化
        res = run_nsga3_optimization()
        
        # 2. 分析结果
        special_sols = analyze_solutions(res)
        
        # 3. 绘图
        plot_3d_pareto(res, special_sols)
        plot_parallel_coordinates(res)
        plot_decision_space(res)
        visualize_pareto_frontier_advanced(res, special_sols)

        # 4. 保存
        save_results(special_sols)
        
    except ImportError as e:
        print("[ERROR] Missing required library 'pymoo'.")
        print("Please install it using: pip install pymoo")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

