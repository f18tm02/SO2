# -*- coding: utf-8 -*-
"""
MCM 2026 Problem B - Question 4 Solution
Algorithm: NSGA-III (Many-Objective Optimization)
Objectives: Minimize [Emissions, Cost, Time]
"""

import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

# =============================================================================
# 1. 定义问题环境 (Problem Definition)
# =============================================================================

class SpaceTransportProblem(ElementwiseProblem):
    def __init__(self):
        # 决策变量：30年的火箭使用率 (0.0 ~ 1.0)
        # 目标数量：3 (排放, 成本, 时间)
        super().__init__(n_var=30, n_obj=3, n_ieq_constr=0, xl=0.0, xu=1.0)
        
        # --- 仿真参数 ---
        self.total_demand = 100.0  # 1亿吨 (单位：百万吨)
        self.years = 30
        
        # 预计算随时间变化的参数
        t = np.linspace(0, 1, self.years)
        
        # 电梯运力：S型增长 (从0.5M到5M)
        self.cap_elev = 0.5 + 4.5 * (1 / (1 + np.exp(-10 * (t - 0.5))))
        
        # 火箭运力：假设每年最大运力恒定 3M
        self.cap_rocket_max = np.full(self.years, 3.0)
        
        # 排放因子 (单位：吨CO2/吨)
        # 火箭：技术微弱进步 (4.0 -> 3.5)
        self.emit_factor_rocket = 4.0 - 0.5 * t
        # 电梯：电网清洁化 (0.1 -> 0.0)
        self.emit_factor_elev = 0.1 - 0.1 * t
        
        # 成本因子 (单位：相对单位)
        self.cost_rocket = 15.0  # 火箭非常贵
        self.cost_elev = 1.0     # 电梯便宜

    def _evaluate(self, x, out, *args, **kwargs):
        """
        x: 一个个体的基因 (长度30的数组，代表每年的火箭使用率)
        """
        # 1. 计算每年的实际运输量
        mass_elev = self.cap_elev  # 电梯总是拉满
        mass_rocket = x * self.cap_rocket_max # 火箭按决策比例
        
        total_capacity = mass_elev + mass_rocket
        cumulative_mass = np.cumsum(total_capacity)
        
        # 2. 计算完工时间 (Objective 3)
        # 找到第一个累计运输量 >= total_demand 的年份
        idx = np.searchsorted(cumulative_mass, self.total_demand)
        
        if idx < self.years:
            completion_time = idx + 1 # 年份索引从0开始
            
            # 3. 截断计算 (任务完成后不再产生排放和成本)
            # 对于完成的那一年，只计算需要的比例
            needed_last_year = self.total_demand - cumulative_mass[idx-1] if idx > 0 else self.total_demand
            capacity_last_year = total_capacity[idx]
            ratio_last_year = needed_last_year / capacity_last_year
            
            # 构建一个掩码，完成年之前全算，完成年算部分，之后不算
            mask = np.zeros(self.years)
            mask[:idx] = 1.0
            mask[idx] = ratio_last_year
            
        else:
            # 惩罚：如果30年没运完
            completion_time = self.years + (self.total_demand - cumulative_mass[-1]) * 10
            mask = np.ones(self.years) # 全部都要算，并且还要受罚

        # 4. 计算总排放 (Objective 1)
        annual_emissions = (mass_elev * self.emit_factor_elev) + \
                           (mass_rocket * self.emit_factor_rocket)
        total_emission = np.sum(annual_emissions * mask)
        
        # 5. 计算总成本 (Objective 2)
        annual_costs = (mass_elev * self.cost_elev) + \
                       (mass_rocket * self.cost_rocket)
        total_cost = np.sum(annual_costs * mask)
        
        # 输出目标向量 [排放, 成本, 时间]
        out["F"] = [total_emission, total_cost, completion_time]

# =============================================================================
# 2. 配置 NSGA-III 算法
# =============================================================================

# 创建参考方向 (Reference Directions)
# n_obj=3, n_partitions=12 是常用设置，产生约91个参考点
ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

# 实例化算法
algorithm = NSGA3(
    pop_size=100,
    ref_dirs=ref_dirs,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(prob=0.03, eta=20),
    eliminate_duplicates=True
)

# 定义终止条件
termination = get_termination("n_gen", 300)

# =============================================================================
# 3. 运行优化
# =============================================================================
print("Running NSGA-III Optimization...")
problem = SpaceTransportProblem()

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               verbose=True)

print(f"Optimization Finished. Found {len(res.F)} non-dominated solutions.")

# =============================================================================
# 4. 结果可视化 (3D Pareto Front)
# =============================================================================
F = res.F # 目标函数值矩阵 [N, 3]

# 归一化以便绘图
F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0))

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点
img = ax.scatter(F[:, 0], F[:, 1], F[:, 2], c=F[:, 2], cmap='viridis', s=50, alpha=0.8)

ax.set_xlabel('Total Emissions (Million Tons CO2)')
ax.set_ylabel('Total Cost (Relative Unit)')
ax.set_zlabel('Completion Time (Years)')
ax.set_title('Pareto Front: Trade-off between Environment, Cost, and Time')

# 添加颜色条
cbar = fig.colorbar(img, ax=ax, pad=0.1, shrink=0.6)
cbar.set_label('Completion Time (Years)')

# 视角调整
ax.view_init(elev=30, azim=45)

plt.savefig('nsga3_pareto_front.png', dpi=300)
plt.show()

# =============================================================================
# 5. 策略分析 (Parallel Coordinate Plot)
# =============================================================================
# 展示其中一个折中解（例如：时间适中，排放较低）
# 找到时间最接近 20 年的解
idx_mid = np.abs(F[:, 2] - 20).argmin()
best_sol_X = res.X[idx_mid]
best_sol_F = res.F[idx_mid]

print("-" * 50)
print(f"Selected Balanced Strategy (Time ~ 20 Years):")
print(f"  Time: {best_sol_F[2]:.1f} Years")
print(f"  Emissions: {best_sol_F[0]:.2f}")
print(f"  Cost: {best_sol_F[1]:.2f}")
print("-" * 50)

# 绘制该策略的时间线
plt.figure(figsize=(10, 5))
years = np.arange(1, 31)
rocket_usage = best_sol_X * problem.cap_rocket_max
elev_usage = problem.cap_elev

plt.stackplot(years, rocket_usage, elev_usage, labels=['Rocket Usage', 'Elevator Usage'],
              colors=['#DD8452', '#4C72B0'], alpha=0.8)
plt.xlabel("Year")
plt.ylabel("Transport Mass (Million Tons)")
plt.title(f"Optimal Transport Schedule (Balanced Strategy)")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.savefig('optimal_schedule.png')
plt.show()
