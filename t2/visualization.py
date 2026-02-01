import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter
import os

# 绘制缆绳损伤分布图
def plot_damage_distribution(damage_history):
    """
    绘制缆绳损伤分布图
    
    参数:
    damage_history: 每日损伤历史
    """
    try:
        print("Creating damage distribution plot...")
        plt.figure(figsize=(12, 6))
        
        # 计算损伤分布
        damage_bins = np.linspace(0, 0.1, 20)  # 0 to 10 cm
        damage_hist, _ = np.histogram(damage_history, bins=damage_bins)
        
        # 绘制直方图
        plt.bar(damage_bins[:-1], damage_hist, width=0.005, alpha=0.7, label='Damage Distribution')
        
        # 添加损伤阈值线
        plt.axvline(x=0.05, color='r', linestyle='--', label='Damage Threshold (50%)')
        
        # 设置标签和标题
        plt.xlabel('Damage (m)')
        plt.ylabel('Frequency')
        plt.title('Cable Damage Distribution Over 50 Years')
        plt.legend()
        
        # 格式化x轴为厘米
        def to_cm(x, pos):
            return f'{x*100:.0f} cm'
        
        formatter = FuncFormatter(to_cm)
        plt.gca().xaxis.set_major_formatter(formatter)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存文件
        save_path = 'damage_distribution.png'
        plt.savefig(save_path)
        plt.close()  # 关闭图表以释放内存
        print(f"Damage distribution plot saved to {save_path}")
    except Exception as e:
        print(f"Error in plot_damage_distribution: {e}")
        import traceback
        traceback.print_exc()
        plt.close()  # 确保关闭图表


# 绘制时间序列图
def plot_time_series(damage_history, outage_history, impact_history):
    """
    绘制时间序列图，包括损伤、停运和撞击事件
    
    参数:
    damage_history: 每日损伤历史
    outage_history: 每日累计停运天数
    impact_history: 每日撞击事件数
    """
    try:
        print("Creating time series plot...")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # 计算时间轴（年）
        days = np.arange(len(damage_history))
        years = days / 365
        
        # 绘制损伤历史
        ax1.plot(years, damage_history, alpha=0.7, label='Daily Damage')
        ax1.axhline(y=0.05, color='r', linestyle='--', label='Damage Threshold (50%)')
        ax1.set_ylabel('Damage (m)')
        ax1.set_title('Daily Damage Over 50 Years')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 格式化y轴为厘米
        def to_cm(x, pos):
            return f'{x*100:.0f} cm'
        
        formatter = FuncFormatter(to_cm)
        ax1.yaxis.set_major_formatter(formatter)
        
        # 绘制停运历史
        ax2.plot(years, outage_history, alpha=0.7, label='Cumulative Outage Days')
        ax2.set_ylabel('Outage Days')
        ax2.set_title('Cumulative Outage Days Over 50 Years')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 绘制撞击事件历史（取对数刻度）
        ax3.plot(years, impact_history, alpha=0.7, label='Daily Impacts')
        ax3.set_yscale('log')
        ax3.set_ylabel('Number of Impacts (log scale)')
        ax3.set_xlabel('Year')
        ax3.set_title('Daily Impact Events Over 50 Years')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存文件
        save_path = 'time_series.png'
        plt.savefig(save_path)
        plt.close()  # 关闭图表以释放内存
        print(f"Time series plot saved to {save_path}")
    except Exception as e:
        print(f"Error in plot_time_series: {e}")
        import traceback
        traceback.print_exc()
        plt.close()  # 确保关闭图表

# 绘制趋势图
def plot_trend(year_outage):
    """
    绘制年停运天数趋势图
    
    参数:
    year_outage: 每年的停运天数
    """
    try:
        print("Creating trend plot...")
        plt.figure(figsize=(12, 6))
        
        # 计算年份轴
        years = np.arange(len(year_outage))
        
        # 绘制年停运天数
        plt.bar(years, year_outage, alpha=0.7, label='Annual Outage Days')
        
        # 添加趋势线
        z = np.polyfit(years, year_outage, 1)
        p = np.poly1d(z)
        plt.plot(years, p(years), 'r--', label='Trend Line')
        
        # 设置标签和标题
        plt.xlabel('Year')
        plt.ylabel('Outage Days')
        plt.title('Annual Outage Days Trend Over 50 Years')
        plt.legend()
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存文件
        save_path = 'outage_trend.png'
        plt.savefig(save_path)
        plt.close()  # 关闭图表以释放内存
        print(f"Trend plot saved to {save_path}")
    except Exception as e:
        print(f"Error in plot_trend: {e}")
        import traceback
        traceback.print_exc()
        plt.close()  # 确保关闭图表

# 绘制缆绳受损示意图
def plot_cable_damage(damage_points, ribbon_width=0.1):
    """
    绘制缆绳受损示意图
    
    参数:
    damage_points: 损伤点列表，每个元素为 (position, damage_width)
    ribbon_width: 缆绳宽度 (m)
    """
    try:
        print("Creating cable damage plot...")
        plt.figure(figsize=(10, 12))
        
        # 计算损伤频率分布
        max_position = 100  # 缆绳总长度
        bin_size = 5  # 每个高度区间的大小
        bins = np.arange(0, max_position + bin_size, bin_size)
        damage_frequency = np.zeros(len(bins) - 1)
        
        # 统计每个高度区间的损伤次数
        for pos, _ in damage_points:
            bin_idx = int(pos // bin_size)
            if bin_idx < len(damage_frequency):
                damage_frequency[bin_idx] += 1
        
        # 计算最大频率用于归一化
        max_freq = max(damage_frequency) if max(damage_frequency) > 0 else 1
        
        # 绘制缆绳高度（纵轴）
        for i, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
            # 计算颜色深浅（基于损伤频率）
            intensity = damage_frequency[i] / max_freq
            color = (1, 1 - intensity, 1 - intensity)  # 红色到白色的渐变
            
            # 绘制红色色块（损伤风险）
            plt.fill_between([0, intensity * 40], start, end, 
                             color=color, alpha=0.8)
            
            # 绘制绿色色块（安全区域）
            plt.fill_between([intensity * 40, 50], start, end, 
                             color='green', alpha=0.3)
        
        # 设置标签和标题
        plt.xlabel('Risk Level')
        plt.ylabel('Cable Height (arbitrary units)')
        plt.title('Cable Damage Risk Distribution by Height')
        
        # 设置x轴范围和刻度
        plt.xlim(0, 50)
        plt.xticks([0, 20, 40, 50], ['Low', 'Medium', 'High', 'Max'])
        
        # 设置y轴方向（从下到上表示高度增加）
        plt.ylim(0, max_position)
        plt.gca().invert_yaxis()  # 反转y轴，使顶部表示更高的位置
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap='Reds')
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), orientation='horizontal', pad=0.05)
        cbar.set_label('Damage Frequency')
        
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        
        # 保存文件
        save_path = 'cable_damage.png'
        plt.savefig(save_path)
        plt.close()  # 关闭图表以释放内存
        print(f"Cable damage plot saved to {save_path}")
    except Exception as e:
        print(f"Error in plot_cable_damage: {e}")
        import traceback
        traceback.print_exc()
        plt.close()  # 确保关闭图表

