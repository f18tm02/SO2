import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# 定义问题参数
# 总运输量：100,000,000吨

# 甲车：一年能运537000吨货物，成本是1.56亿美元
annual_capacity_car1 = 537000  # 年运输量（吨）
cost_car1_per_year = 1560000000  # 年成本（美元）

# 乙车：一次能运125吨货物，一天能发三次，一次的成本是1亿美元
# 现在有10个乙车同时运输
num_cars_car2 = 10  # 乙车数量
capacity_car2_per_trip_per_car = 125  # 每辆车每次运输量（吨）
trips_per_day = 3  # 每天次数
cost_car2_per_trip_per_car = 100000000  # 每辆车每次成本（美元）

# 计算甲车的时间和经济成本
total_cargo = 100000000  # 总运输量（吨）

time_cost_car1 = total_cargo / annual_capacity_car1  # 时间成本（年）
economic_cost_car1 = (total_cargo / annual_capacity_car1) * cost_car1_per_year  # 经济成本（美元）

# 计算乙车的时间和经济成本
capacity_car2_per_trip = capacity_car2_per_trip_per_car * num_cars_car2  # 10辆车每次总运输量（吨）
cost_car2_per_trip = cost_car2_per_trip_per_car * num_cars_car2  # 10辆车每次总成本（美元）
annual_capacity_car2 = capacity_car2_per_trip * trips_per_day * 365  # 年运输量（吨）
time_cost_car2 = total_cargo / annual_capacity_car2  # 时间成本（年）
economic_cost_car2 = (total_cargo / capacity_car2_per_trip) * cost_car2_per_trip  # 经济成本（美元）

# 方案一：甲车（经济成本低，时间成本高）
cost1_time = time_cost_car1  # 时间成本（年）
cost1_economic = economic_cost_car1  # 经济成本（美元）

# 方案二：乙车（时间成本低，经济成本高）
cost2_time = time_cost_car2  # 时间成本（年）
cost2_economic = economic_cost_car2  # 经济成本（美元）

# 权重范围（方案一的比例，方案二的比例为1-权重）
weight_range = np.linspace(0, 1, 1000)

# 实现帕累托最优筛选函数
def pareto_frontier(costs):
    """
    筛选帕累托最优解
    costs: 二维数组，每行包含时间成本和经济成本
    返回帕累托最优解的索引
    """
    # 初始化帕累托前沿
    pareto_indices = []
    
    # 对每个解进行检查
    for i, cost in enumerate(costs):
        # 检查是否存在其他解在所有目标上都不劣于当前解
        is_pareto = True
        for j, other_cost in enumerate(costs):
            if i != j:
                # 如果其他解的时间成本和经济成本都不大于当前解，且至少有一个严格小于
                if other_cost[0] <= cost[0] and other_cost[1] <= cost[1]:
                    if other_cost[0] < cost[0] or other_cost[1] < cost[1]:
                        is_pareto = False
                        break
        
        if is_pareto:
            pareto_indices.append(i)
    
    return pareto_indices

# 生成权重组合并计算对应的总成本
def calculate_costs(weight):
    """
    计算给定权重下的时间成本和经济成本
    weight: 方案一的权重（0-1之间）
    返回：(时间成本, 经济成本)
    """
    time_cost = weight * cost1_time + (1 - weight) * cost2_time
    economic_cost = weight * cost1_economic + (1 - weight) * cost2_economic
    return time_cost, economic_cost

# 计算所有权重下的成本
all_costs = np.array([calculate_costs(w) for w in weight_range])
all_time_costs = all_costs[:, 0]
all_economic_costs = all_costs[:, 1]

# 筛选帕累托最优解
pareto_indices = pareto_frontier(all_costs)
pareto_weights = weight_range[pareto_indices]
pareto_time_costs = all_time_costs[pareto_indices]
pareto_economic_costs = all_economic_costs[pareto_indices]

# 定义目标函数来找到最佳权重
def find_optimal_weight():
    """
    找到最佳权重，考虑时间成本和经济成本的平衡
    使用帕累托前沿的拐点来确定权重
    返回：(最佳权重, 对应的时间成本, 对应的经济成本)
    """
    # 使用帕累托前沿的拐点来确定最佳权重
    if len(pareto_indices) <= 1:
        # 如果只有一个或没有帕累托最优解，返回第一个
        optimal_index = pareto_indices[0] if len(pareto_indices) > 0 else 0
    else:
        # 计算帕累托前沿上的斜率变化，找到拐点
        # 首先对帕累托前沿按时间成本排序
        sorted_indices = np.argsort(pareto_time_costs)
        sorted_time = pareto_time_costs[sorted_indices]
        sorted_economic = pareto_economic_costs[sorted_indices]
        sorted_weights = pareto_weights[sorted_indices]
        
        # 计算相邻点之间的斜率
        slopes = []
        for i in range(1, len(sorted_time)):
            delta_time = sorted_time[i] - sorted_time[i-1]
            delta_economic = sorted_economic[i] - sorted_economic[i-1]
            if delta_time != 0:
                slope = delta_economic / delta_time
                slopes.append(slope)
        
        # 找到斜率变化最大的点（拐点）
        if slopes:
            # 计算斜率的变化率
            slope_changes = []
            for i in range(1, len(slopes)):
                slope_change = abs(slopes[i] - slopes[i-1])
                slope_changes.append(slope_change)
            
            if slope_changes:
                # 找到斜率变化最大的索引
                max_change_index = np.argmax(slope_changes) + 1  # +1 因为斜率变化是相对于前一个点
                optimal_weight = sorted_weights[max_change_index]
            else:
                # 如果只有一个斜率，选择中间点
                optimal_weight = sorted_weights[len(sorted_weights)//2]
        else:
            # 如果没有斜率（所有点时间成本相同），选择经济成本最低的
            min_economic_index = np.argmin(sorted_economic)
            optimal_weight = sorted_weights[min_economic_index]
        
        # 找到最优权重对应的索引
        optimal_index = np.argmin(np.abs(weight_range - optimal_weight))
    
    optimal_weight = weight_range[optimal_index]
    optimal_time = all_time_costs[optimal_index]
    optimal_economic = all_economic_costs[optimal_index]
    
    return optimal_weight, optimal_time, optimal_economic

# 计算最佳权重
optimal_weight, optimal_time, optimal_economic = find_optimal_weight()

# 可视化帕累托前沿和最优解
def visualize_results():
    """
    可视化帕累托前沿和最优解
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制所有可能的解
    plt.scatter(all_time_costs, all_economic_costs, c='lightgray', alpha=0.3, label='所有可能的解')
    
    # 绘制帕累托前沿
    plt.plot(pareto_time_costs, pareto_economic_costs, 'r-', linewidth=2, label='帕累托前沿')
    plt.scatter(pareto_time_costs, pareto_economic_costs, c='red', s=50, alpha=0.7)
    
    # 标记最优解
    plt.scatter(optimal_time, optimal_economic, c='blue', s=100, marker='*', label='最佳解')
    
    # 标记两个极端方案
    plt.scatter(cost1_time, cost1_economic, c='green', s=80, marker='s', label='方案一')
    plt.scatter(cost2_time, cost2_economic, c='purple', s=80, marker='s', label='方案二')
    
    # 添加图表元素
    plt.title('时间成本与经济成本的帕累托最优分析', fontsize=14)
    plt.xlabel('时间成本（小时）', fontsize=12)
    plt.ylabel('经济成本（元）', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 调整坐标轴范围
    plt.xlim(min(all_time_costs) - 0.5, max(all_time_costs) + 0.5)
    plt.ylim(min(all_economic_costs) - 50, max(all_economic_costs) + 50)
    
    # 保存图表
    plt.savefig('pareto_frontier.png', dpi=300, bbox_inches='tight')
    plt.show()

# 可视化结果
visualize_results()

# 分析结果并输出最佳方案建议
def analyze_results():
    """
    分析结果并输出最佳方案建议
    """
    print("=== 帕累托最优分析结果 ===")
    print(f"方案一：时间成本 = {cost1_time} 小时，经济成本 = {cost1_economic} 元")
    print(f"方案二：时间成本 = {cost2_time} 小时，经济成本 = {cost2_economic} 元")
    print()
    print(f"最佳权重：方案一占比 {optimal_weight:.2%}，方案二占比 {1-optimal_weight:.2%}")
    print(f"对应的时间成本：{optimal_time:.2f} 小时")
    print(f"对应的经济成本：{optimal_economic:.2f} 元")
    print()
    
    # 分析与两个极端方案的对比
    time_saving = cost1_time - optimal_time
    economic_saving = cost2_economic - optimal_economic
    
    print("=== 成本分析 ===")
    print(f"相比纯方案一：节省时间 {time_saving:.2f} 小时，增加经济成本 {optimal_economic - cost1_economic:.2f} 元")
    print(f"相比纯方案二：增加时间 {optimal_time - cost2_time:.2f} 小时，节省经济成本 {economic_saving:.2f} 元")
    print()
    
    # 提供决策建议
    print("=== 决策建议 ===")
    if optimal_weight > 0.7:
        print("建议：主要使用方案一，辅以少量方案二")
        print("适用场景：对经济成本敏感，时间要求不紧迫")
    elif optimal_weight < 0.3:
        print("建议：主要使用方案二，辅以少量方案一")
        print("适用场景：对时间要求紧迫，经济成本不是主要考虑因素")
    else:
        print("建议：平衡使用两个方案")
        print("适用场景：时间和经济成本都需要考虑，寻求最佳平衡")
    print()
    
    # 分析帕累托前沿
    print("=== 帕累托前沿分析 ===")
    print(f"找到 {len(pareto_indices)} 个帕累托最优解")
    print("帕累托最优解代表了时间成本和经济成本的最佳平衡")
    print("在这些解中，无法同时降低时间成本和经济成本")

# 分析结果并输出建议
analyze_results()
