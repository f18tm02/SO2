import numpy as np

# 获取太空垃圾增长因子
def get_debris_growth_factor(year, total_years, growth_type='logarithmic'):
    """
    计算指定年份的太空垃圾增长因子
    
    参数:
    year: 当前年份 (0-based)
    total_years: 总模拟年份
    growth_type: 增长类型 ('linear', 'exponential' 或 'logarithmic')
    
    返回:
    growth_factor: 增长因子
    """
    if growth_type == 'linear':
        # 线性增长，50年后增长到初始值的3倍
        max_growth = 3.0
        growth_factor = 1.0 + (max_growth - 1.0) * (year / total_years)
    elif growth_type == 'exponential':
        # 指数增长，50年后增长到初始值的5倍
        max_growth = 5.0
        growth_factor = np.exp(np.log(max_growth) * (year / total_years))
    elif growth_type == 'logarithmic':
        # 对数增长，50年后增长到初始值的3倍
        max_growth = 3.0
        # 使用自然对数，确保增长速度逐渐放缓
        growth_factor = 1.0 + (max_growth - 1.0) * (np.log(year + 1) / np.log(total_years + 1))
    else:
        # 默认无增长
        growth_factor = 1.0
    
    return growth_factor

# 计算特定年份的碎片密度
def get_debris_density(base_density, year, total_years, growth_type='exponential'):
    """
    计算指定年份的碎片密度
    
    参数:
    base_density: 初始碎片密度
    year: 当前年份 (0-based)
    total_years: 总模拟年份
    growth_type: 增长类型 ('linear' 或 'exponential')
    
    返回:
    density: 当年碎片密度
    """
    growth_factor = get_debris_growth_factor(year, total_years, growth_type)
    return base_density * growth_factor

# 预测未来碎片数量
def predict_debris_count(base_count, years, growth_type='exponential'):
    """
    预测未来指定年份的碎片数量
    
    参数:
    base_count: 初始碎片数量
    years: 预测年数
    growth_type: 增长类型 ('linear' 或 'exponential')
    
    返回:
    predicted_count: 预测的碎片数量
    """
    growth_factor = get_debris_growth_factor(years, years, growth_type)
    return base_count * growth_factor
