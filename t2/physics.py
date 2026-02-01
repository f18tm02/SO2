import numpy as np

# 计算单次撞击的损伤宽度
def calculate_damage(radius, theta, phi, ribbon_width):
    """
    计算单次撞击对缆绳的损伤宽度
    
    参数:
    radius: 碎片半径 (m)
    theta: 入射角 (rad)
    phi: 另一个角度参数 (rad)
    ribbon_width: 缆绳宽度 (m)
    
    返回:
    damage: 损伤宽度 (m)
    """
    # 使用用户提供的公式计算损伤
    # r + r*sin(Φ)/tan(θ) ≥ w*f/2
    
    # 碎片直径
    diameter = 2 * radius
    
    # 只有当碎片直径大于1 cm时才会产生显著损伤
    if diameter < 0.01:  # 小于1 cm
        return 0.0
    
    # 基础损伤为碎片直径
    base_damage = 2 * radius
    
    # 计算额外损伤
    if theta > 0:  # 避免除以零
        additional_damage = radius * np.sin(phi) / np.tan(theta)
    else:
        additional_damage = 0
    
    # 总损伤
    total_damage = base_damage + additional_damage
    
    # 损伤不能超过缆绳宽度
    total_damage = min(total_damage, ribbon_width)
    
    # 确保损伤为非负值
    total_damage = max(total_damage, 0.0)
    
    return total_damage

# 计算碎片的动能
def calculate_kinetic_energy(mass, velocity):
    """
    计算碎片的动能
    
    参数:
    mass: 碎片质量 (kg)
    velocity: 碎片速度 (m/s)
    
    返回:
    ke: 动能 (J)
    """
    return 0.5 * mass * velocity**2

# 估算碎片质量（基于直径和密度假设）
def estimate_debris_mass(diameter):
    """
    估算碎片质量
    
    参数:
    diameter: 碎片直径 (m)
    
    返回:
    mass: 估算质量 (kg)
    """
    # 假设平均密度为2700 kg/m^3（铝的密度）
    density = 2700
    volume = (4/3) * np.pi * (diameter/2)**3
    return density * volume
