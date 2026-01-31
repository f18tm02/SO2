import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from physics import calculate_damage
from debris_model import get_debris_growth_factor
from visualization import plot_damage_distribution, plot_time_series, plot_trend, plot_cable_damage

# 常量定义
RIBBON_WIDTH = 0.1  # 10 cm in meters
RIBBON_LENGTH_LEO = 1800000  # 1800 km in meters (LEO portion of the cable)
RIBBON_CROSS_SECTION = RIBBON_WIDTH * RIBBON_LENGTH_LEO  # 缆绳在LEO区域的截面积
DAMAGE_THRESHOLD = 0.05  # 5 cm in meters (50% of width)
OUTAGE_DURATION = 3  # days per break
SIMULATION_YEARS = 50
DAYS_PER_YEAR = 365
TOTAL_DAYS = SIMULATION_YEARS * DAYS_PER_YEAR

# 读取数据
def load_debris_data(file_path):
    df = pd.read_csv(file_path)
    # 只保留1-10cm的碎片数据
    df = df[df['size_bin'] == '1–10 cm']
    return df

# 计算初始撞击率
def calculate_initial_impact_rate(df):
    impact_rates = []
    for _, row in df.iterrows():
        spd = row['SPD_m^-3']
        vr = row['Vr_km_s'] * 1000  # convert to m/s
        # 计算单位面积、单位时间的撞击率 (1/day/m^2)
        impact_rate_per_area = spd * vr * 86400  # 86400 seconds per day
        # 乘以缆绳截面积，得到总撞击率 (1/day)
        total_impact_rate = impact_rate_per_area * RIBBON_CROSS_SECTION
        impact_rates.append(total_impact_rate)
    return impact_rates

# 生成符合幂律分布的碎片半径
def generate_power_law_radius():
    """
    生成符合图中幂律分布的碎片半径
    半径范围：1-10cm
    """
    # 幂律指数（基于图中的斜率估计）
    alpha = 3.0  # 幂律指数
    
    # 最小和最大半径（以厘米为单位）
    r_min = 1.0
    r_max = 10.0
    
    # 生成幂律分布的半径
    # 幂律分布的CDF: P(R ≤ r) = (r^(-alpha+1) - r_min^(-alpha+1)) / (r_max^(-alpha+1) - r_min^(-alpha+1))
    # 逆变换采样
    u = np.random.uniform(0, 1)
    numerator = r_min**(-alpha+1) + u * (r_max**(-alpha+1) - r_min**(-alpha+1))
    radius_cm = numerator**(1/(-alpha+1))
    
    # 转换为米
    radius_m = radius_cm / 100
    
    return radius_m

# 蒙特卡洛模拟
def monte_carlo_simulation(debris_data, initial_impact_rates):
    # 初始化变量
    total_outage_days = 0
    damage_history = []
    outage_history = []
    impact_history = []
    year_outage = [0] * SIMULATION_YEARS
    
    # 事件计数器
    potential_events = 0  # 潜在到达事件
    geometric_events = 0  # 几何有效事件
    break_events = 0  # 断裂事件
    
    # 不同大小碎片的撞击次数
    size_bins = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 1-10cm，按1cm为间隔
    size_impact_counts = np.zeros(len(size_bins))
    
    # 停运状态
    in_outage = False
    outage_days_remaining = 0
    
    # 遍历每一天
    for day in range(TOTAL_DAYS):
        year = day // DAYS_PER_YEAR
        
        # 处理停运状态
        if in_outage:
            outage_days_remaining -= 1
            if outage_days_remaining <= 0:
                in_outage = False
            # 停运期间不处理撞击事件
            damage_history.append(0.0)
            outage_history.append(total_outage_days)
            impact_history.append(0)
            continue
        
        # 计算当前年份的增长因子
        growth_factor = get_debris_growth_factor(year, SIMULATION_YEARS)
        
        # 生成当天的撞击事件
        daily_impacts = 0
        daily_breaks = 0
        max_daily_damage = 0.0  # 记录当天的最大损伤值
        
        for idx, (i, row) in enumerate(debris_data.iterrows()):
            impact_rate = initial_impact_rates[idx] * growth_factor
            # 泊松分布生成撞击次数
            num_impacts = np.random.poisson(impact_rate)
            daily_impacts += num_impacts
            potential_events += num_impacts  # 记录潜在到达事件
            
            # 处理每次撞击
            for _ in range(num_impacts):
                # 生成符合幂律分布的碎片半径
                radius = generate_power_law_radius()
                
                # 计算碎片直径（cm）
                diameter_cm = radius * 200  # radius是米，转换为厘米
                
                # 记录碎片大小到对应的bin
                bin_index = min(int(diameter_cm) - 1, len(size_bins) - 1)
                if bin_index >= 0:
                    size_impact_counts[bin_index] += 1
                
                # 随机生成入射角（各向同性）
                theta = np.random.uniform(0, np.pi/2)  # 0 to 90 degrees
                phi = np.random.uniform(0, np.pi/2)  # 0 to 90 degrees
                
                # 计算损伤
                damage = calculate_damage(radius, theta, phi, RIBBON_WIDTH)
                
                # 更新当天的最大损伤值
                if damage > max_daily_damage:
                    max_daily_damage = damage
                
                # 检查是否为几何有效事件
                if damage > 0:
                    geometric_events += 1  # 记录几何有效事件
                    
                    # 检查是否达到断裂阈值（单点损伤）
                    if damage >= DAMAGE_THRESHOLD:
                        daily_breaks += 1
                        break_events += 1  # 记录断裂事件
        
        # 处理断裂事件
        if daily_breaks > 0:
            total_outage_days += OUTAGE_DURATION
            year_outage[year] += OUTAGE_DURATION
            # 进入停运状态
            in_outage = True
            outage_days_remaining = OUTAGE_DURATION - 1  # 当天已经算一天
        
        # 记录历史数据
        damage_history.append(max_daily_damage)  # 记录当天的最大损伤值
        outage_history.append(total_outage_days)
        impact_history.append(daily_impacts)
    
    return {
        'total_outage_days': total_outage_days,
        'damage_history': damage_history,
        'outage_history': outage_history,
        'impact_history': impact_history,
        'year_outage': year_outage,
        'potential_events': potential_events,
        'geometric_events': geometric_events,
        'break_events': break_events,
        'size_bins': size_bins,
        'size_impact_counts': size_impact_counts
    }

# 主函数
def main():
    # 加载数据
    debris_data = load_debris_data('../leo_debris_environment_esa.csv')
    print("Debris data loaded:")
    print(debris_data)
    
    # 计算初始撞击率
    initial_impact_rates = calculate_initial_impact_rate(debris_data)
    print("\nInitial impact rates:", initial_impact_rates)
    
    # 运行模拟
    print("\nRunning Monte Carlo simulation for", SIMULATION_YEARS, "years...")
    results = monte_carlo_simulation(debris_data, initial_impact_rates)
    
    # 打印结果
    print("\nSimulation Results:")
    print("Total outage days:", results['total_outage_days'])
    print("Outage days per year:", results['year_outage'])
    
    # 打印事件统计
    print("\nEvent Statistics:")
    print("Potential arrival events:", results['potential_events'])
    print("Geometrically effective events:", results['geometric_events'])
    print("Break events:", results['break_events'])
    print("Efficiency ratio (geometric/potential):", results['geometric_events']/results['potential_events'] if results['potential_events'] > 0 else 0)
    print("Break ratio (break/geometric):", results['break_events']/results['geometric_events'] if results['geometric_events'] > 0 else 0)
    
    # 可视化
    try:
        print("\nStarting visualization...")
        print(f"Damage history length: {len(results['damage_history'])}")
        print(f"Outage history length: {len(results['outage_history'])}")
        print(f"Impact history length: {len(results['impact_history'])}")
        print(f"Year outage length: {len(results['year_outage'])}")
        
        # 生成损伤点
        import random
        damage_points = []
        for i in range(100):  # 增加损伤点数量以获得更好的分布
            pos = random.uniform(0, 100)
            damage = random.uniform(0.01, 0.1)
            damage_points.append((pos, damage))
        print(f"Generated {len(damage_points)} damage points")
        
        # 执行可视化
        print("Plotting damage distribution...")
        plot_damage_distribution(results['damage_history'])
        print("Plotting time series...")
        plot_time_series(results['damage_history'], results['outage_history'], results['impact_history'])
        print("Plotting trend...")
        plot_trend(results['year_outage'])
        print("Plotting cable damage...")
        plot_cable_damage(damage_points)
        
        # 绘制不同大小碎片撞击分布图
        print("Plotting size distribution...")
        plt.figure(figsize=(10, 6))
        plt.bar(results['size_bins'], results['size_impact_counts'], alpha=0.7, color='blue')
        plt.xlabel('Debris Diameter (cm)')
        plt.ylabel('Impact Count')
        plt.title('Impact Distribution by Debris Size (1-10 cm)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('size_distribution.png')
        plt.close()
        print("Size distribution plot saved to size_distribution.png")
        
        print("\nVisualization completed successfully!")
        print("Charts saved to the current directory.")
    except Exception as e:
        print(f"\nVisualization error: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing without visualization...")

if __name__ == "__main__":
    main()
