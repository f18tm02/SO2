import numpy as np
import pandas as pd

# 工具函数模块

# 单位转换
def meters_to_centimeters(meters):
    """
    将米转换为厘米
    
    参数:
    meters: 长度（米）
    
    返回:
    centimeters: 长度（厘米）
    """
    return meters * 100

def centimeters_to_meters(centimeters):
    """
    将厘米转换为米
    
    参数:
    centimeters: 长度（厘米）
    
    返回:
    meters: 长度（米）
    """
    return centimeters / 100

# 角度转换
def degrees_to_radians(degrees):
    """
    将角度转换为弧度
    
    参数:
    degrees: 角度
    
    返回:
    radians: 弧度
    """
    return np.radians(degrees)

def radians_to_degrees(radians):
    """
    将弧度转换为角度
    
    参数:
    radians: 弧度
    
    返回:
    degrees: 角度
    """
    return np.degrees(radians)

# 数据处理
def calculate_statistics(data):
    """
    计算数据的基本统计信息
    
    参数:
    data: 数据列表或数组
    
    返回:
    stats: 包含统计信息的字典
    """
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }
    return stats

# 生成随机角度（各向同性）
def generate_isotropic_angles():
    """
    生成各向同性的随机角度
    
    返回:
    theta, phi: 两个随机角度（弧度）
    """
    # 生成0-90度的随机角度
    theta = np.random.uniform(0, np.pi/2)
    phi = np.random.uniform(0, np.pi/2)
    return theta, phi
