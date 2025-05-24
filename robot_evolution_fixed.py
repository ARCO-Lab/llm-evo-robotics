import numpy as np
import pybullet as p
import pybullet_data
import tempfile
import os
import time
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover as SBX
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling as LHS
from pymoo.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

import logging




def decode_gene(x, config=None, adaptation_state=None):
    """
    使用可配置参数和自适应机制解码机器人设计基因
    
    Args:
        x (numpy.ndarray): 基因数组
        config (dict, optional): 解码配置参数，包含参数范围、默认值等
        adaptation_state (dict, optional): 包含学习到的最优参数分布的状态
        
    Returns:
        dict: 解码后的机器人配置字典
    """
    import numpy as np
    import pybullet as p
    
    # 默认配置参数
    default_config = {
        'min_links': 4,           # 最小连杆数
        'max_links': 500,          # 最大连杆数
        'base_params': 6,         # 底座参数数量
        'link_params': 13,        # 每个额外连杆的参数数量
        'joint_type_ranges': [0.25, 0.5, 0.75],  # 关节类型概率阈值
        'wheel_radius_range': [0.03, 0.1],       # 轮子半径范围
        'wheel_width_range': [0.02, 0.08],       # 轮子宽度范围
        'link_size_range': [0.05, 0.25],         # 普通连杆尺寸范围
        'joint_limit_range': [-3.14, 3.14],      # 关节限制范围
        'damping_range': [0.1, 1.0],             # 阻尼系数范围
        'min_wheel_prob': 0.0,                   # 设为轮子的最小概率
        'default_joint_axis': [0, 0, 1],         # 默认关节轴向
        'wheel_probability': 0.4,                # 连杆成为轮子的基础概率
    }
    
    # 合并用户配置与默认配置
    if config is not None:
        for key, value in config.items():
            default_config[key] = value
    
    config = default_config
    
    # 应用自适应状态（如果提供）
    if adaptation_state is not None:
        # 从自适应状态更新配置
        for key, value in adaptation_state.items():
            if key in config:
                config[key] = value
    
    # 1. 确定连杆数量 - 使用更灵活的编码
    min_links = config['min_links']
    max_links = config['max_links']
    
    # 使用基因值确定连杆数 - 更均匀的分布
    num_links = min_links + int(x[0] * (max_links - min_links + 1))
    
    # 初始化数组
    joint_types, has_motor, shapes = [], [], []
    is_wheel, wheel_types, joint_axes = [], [], []
    link_sizes, link_materials = [], []
    joint_limits, joint_damping = [], []
    
    # 计算可用的基因长度，确保不超出范围
    gene_length = len(x)
    required_length = 1 + config['base_params'] + (num_links - 1) * config['link_params']
    
    # 如果基因不够长，调整连杆数
    if required_length > gene_length:
        available_links = 1 + (gene_length - 1 - config['base_params']) // config['link_params']
        num_links = max(min_links, available_links)
        if num_links < min_links:
            # 如果连杆数少于最小要求，使用默认值填充
            num_links = min_links
    
    # 2. 解码底座参数
    # 车身/底板 (总是第一个连杆)
    joint_types.append(p.JOINT_FIXED)  # 第一个关节总是固定的
    has_motor.append(False)
    
    base_idx = 1  # 基因中的起始索引
    
    # 确保索引不越界
    if base_idx < gene_length:
        # 车身形状: 0-0.33: 盒子, 0.33-0.66: 圆柱, 0.66-1.0: 球
        shape_val = x[base_idx]
        shapes.append(int(shape_val * 3))
        base_idx += 1
    else:
        shapes.append(0)  # 默认盒子形状
    
    is_wheel.append(False)  # 车身不是轮子
    wheel_types.append(0)
    
    # 车身尺寸
    if base_idx + 2 < gene_length:
        # 支持更灵活的底座尺寸
        size_range = config.get('base_size_range', config['link_size_range'])
        min_size, max_size = size_range
        size_diff = max_size - min_size
        
        size_x = min_size + x[base_idx] * size_diff
        size_y = min_size + x[base_idx+1] * size_diff
        # 车身高度稍小
        size_z = min_size + x[base_idx+2] * size_diff * 0.5  
        link_sizes.append([size_x, size_y, size_z])
        base_idx += 3
    else:
        # 默认尺寸
        link_sizes.append([0.2, 0.2, 0.1])
    
    # 车身材质
    if base_idx < gene_length:
        link_materials.append(int(x[base_idx] * 3))  # 0: 金属, 1: 塑料, 2: 橡胶
        base_idx += 1
    else:
        link_materials.append(0)  # 默认金属
    
    joint_axes.append(config['default_joint_axis'])  # 车身的固定关节方向
    joint_limits.append([0, 0])    # 固定关节没有限制
    joint_damping.append(config['damping_range'][0])  # 最小阻尼
    
    # 3. 解码其余连杆
    for i in range(1, num_links):
        # 为每个连杆计算基因起始位置
        gene_start = 1 + config['base_params'] + (i-1) * config['link_params']
        
        # 如果基因不够长，使用默认值
        if gene_start + config['link_params'] > gene_length:
            # 添加默认连杆 - 使用自适应机制决定默认类型
            wheel_prob = config['wheel_probability']
            is_default_wheel = np.random.random() < wheel_prob
            
            if is_default_wheel:
                # 默认轮子
                joint_types.append(p.JOINT_REVOLUTE)
                has_motor.append(True)
                shapes.append(1)  # 圆柱形
                is_wheel.append(True)
                wheel_types.append(0)  # 普通轮
                joint_axes.append([0, 0, 1])  # Z轴旋转轮子
                link_sizes.append([0.06, 0.06, 0.04])  # 默认轮子尺寸
                link_materials.append(2)  # 橡胶
                joint_limits.append(config['joint_limit_range'])
                joint_damping.append(config['damping_range'][0])
            else:
                # 默认连杆（非轮子）
                joint_types.append(p.JOINT_REVOLUTE)
                has_motor.append(True)
                shapes.append(0)  # 盒子形状
                is_wheel.append(False)
                wheel_types.append(0)
                joint_axes.append(config['default_joint_axis'])
                link_sizes.append([0.1, 0.1, 0.1])  # 默认连杆尺寸
                link_materials.append(0)  # 金属
                joint_limits.append(config['joint_limit_range'])
                joint_damping.append(config['damping_range'][0])
            continue
        
        # 关节类型
        if gene_start < gene_length:
            joint_type_val = x[gene_start]
            ranges = config['joint_type_ranges']
            
            if joint_type_val < ranges[0]:
                joint_types.append(p.JOINT_FIXED)
            elif joint_type_val < ranges[1]:
                joint_types.append(p.JOINT_REVOLUTE)
            elif joint_type_val < ranges[2]:
                joint_types.append(p.JOINT_PRISMATIC)
            else:
                joint_types.append(p.JOINT_SPHERICAL)
            gene_start += 1
        else:
            joint_types.append(p.JOINT_REVOLUTE)  # 默认旋转关节
        
        # 是否有电机
        if gene_start < gene_length:
            has_motor.append(x[gene_start] > 0.5)
            gene_start += 1
        else:
            has_motor.append(True)  # 默认有电机
        
        # 连杆形状
        if gene_start < gene_length:
            shape_val = x[gene_start]
            shapes.append(int(shape_val * 3))
            gene_start += 1
        else:
            shapes.append(0)  # 默认盒子形状
        
        # 是否是轮子 - 考虑自适应概率
        min_wheel_prob = config['min_wheel_prob']
        if gene_start < gene_length:
            # 轮子概率受到最小概率约束
            wheel_threshold = 1.0 - min_wheel_prob - config['wheel_probability']
            is_wheel_val = x[gene_start] > wheel_threshold
            is_wheel.append(is_wheel_val)
            gene_start += 1
        else:
            is_wheel.append(False)  # 默认非轮子
        
        # 轮子类型
        if gene_start < gene_length:
            wheel_types.append(int(x[gene_start] > 0.5))
            gene_start += 1
        else:
            wheel_types.append(0)  # 默认普通轮
        
        # 连杆尺寸
        if gene_start + 2 < gene_length:
            if is_wheel[-1]:
                # 轮子使用特殊尺寸: 半径和宽度
                min_r, max_r = config['wheel_radius_range']
                min_w, max_w = config['wheel_width_range']
                
                wheel_radius = min_r + x[gene_start] * (max_r - min_r)
                wheel_width = min_w + x[gene_start+1] * (max_w - min_w)
                link_sizes.append([wheel_radius, wheel_radius, wheel_width])
            else:
                # 普通连杆尺寸
                min_size, max_size = config['link_size_range']
                size_diff = max_size - min_size
                
                size_x = min_size + x[gene_start] * size_diff
                size_y = min_size + x[gene_start+1] * size_diff
                size_z = min_size + x[gene_start+2] * size_diff
                link_sizes.append([size_x, size_y, size_z])
            gene_start += 3
        else:
            # 默认尺寸
            if is_wheel[-1]:
                link_sizes.append([0.06, 0.06, 0.04])  # 默认轮子尺寸
            else:
                link_sizes.append([0.1, 0.1, 0.1])  # 默认连杆尺寸
        
        # 连杆材质
        if gene_start < gene_length:
            link_materials.append(int(x[gene_start] * 3))
            gene_start += 1
        else:
            link_materials.append(0)  # 默认金属
        
        # 关节轴向
        if gene_start + 2 < gene_length:
            # 归一化向量以确保单位长度
            axis_x = x[gene_start] * 2 - 1     # -1到1
            axis_y = x[gene_start+1] * 2 - 1   # -1到1
            axis_z = x[gene_start+2] * 2 - 1   # -1到1
            
            # 确保轮子的关节轴适合其运动
            if is_wheel[-1]:
                # 轮子强制使用Z轴旋转
                joint_axes.append([0, 0, 1])
            else:
                # 防止零向量
                norm = np.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
                if norm < 0.001:
                    joint_axes.append(config['default_joint_axis'])
                else:
                    joint_axes.append([axis_x/norm, axis_y/norm, axis_z/norm])
            gene_start += 3
        else:
            # 默认轴向
            if is_wheel[-1]:
                joint_axes.append([0, 0, 1])  # 轮子z轴旋转
            else:
                joint_axes.append(config['default_joint_axis'])
        
        # 关节限制
        if gene_start + 1 < gene_length:
            min_limit, max_limit = config['joint_limit_range']
            limit_range = max_limit - min_limit
            
            if not is_wheel[-1] and joint_types[-1] != p.JOINT_FIXED:
                # 非轮子的关节限制
                limit_lower = min_limit + x[gene_start] * limit_range
                limit_upper = min_limit + x[gene_start+1] * limit_range
                
                # 确保下限小于上限
                if limit_lower > limit_upper:
                    limit_lower, limit_upper = limit_upper, limit_lower
                    
                # 棱柱关节的特殊处理
                if joint_types[-1] == p.JOINT_PRISMATIC:
                    # 确保棱柱关节的限制范围不为零
                    if abs(limit_upper - limit_lower) < 0.05:
                        limit_upper = limit_lower + 0.05
                
                joint_limits.append([limit_lower, limit_upper])
            else:
                # 轮子使用完全旋转范围
                joint_limits.append(config['joint_limit_range'])
            gene_start += 2
        else:
            # 默认限制
            joint_limits.append(config['joint_limit_range'])
        
        # 关节阻尼
        if gene_start < gene_length:
            min_damp, max_damp = config['damping_range']
            damping = min_damp + x[gene_start] * (max_damp - min_damp)
            joint_damping.append(damping)
            gene_start += 1
        else:
            # 默认阻尼
            joint_damping.append(config['damping_range'][0])
    
    # 创建并返回机器人配置字典
    return {
        'num_links': len(joint_types),
        'joint_types': joint_types,
        'has_motor': has_motor,
        'shapes': shapes,
        'is_wheel': is_wheel,
        'wheel_types': wheel_types,
        'joint_axes': joint_axes,
        'link_sizes': link_sizes,
        'link_materials': link_materials,
        'joint_limits': joint_limits,
        'joint_damping': joint_damping
    }

def update_adaptation_state(adaptation_state, evaluated_designs, current_generation):
    """
    基于历史评估更新适应状态，用于指导后续解码
    
    Args:
        adaptation_state (dict): 当前适应状态，如果为None则创建新的
        evaluated_designs (list): 已评估的设计列表，包含性能信息
        current_generation (int): 当前进化代数
        
    Returns:
        dict: 更新后的适应状态
    """
    import numpy as np
    
    # 如果没有适应状态，创建新的
    if adaptation_state is None:
        adaptation_state = {
            'max_links': 8,                 # 默认最大连杆数
            'joint_type_ranges': [0.25, 0.5, 0.75],  # 默认关节类型阈值
            'wheel_probability': 0.4,       # 默认轮子概率
            'min_wheel_prob': 0.0,          # 默认最小轮子概率
            'generation': 0,                # 当前代数
            'success_rates': {}             # 各设计类型的成功率
        }
    
    # 如果没有评估数据，直接返回
    if not evaluated_designs or len(evaluated_designs) < 5:
        return adaptation_state
    
    # 更新代数
    adaptation_state['generation'] = current_generation
    
    # 找出表现最好的前20%设计
    top_designs = sorted(evaluated_designs, key=lambda d: d.get('performance', 0.0), reverse=True)
    top_n = max(1, len(evaluated_designs) // 5)
    top_designs = top_designs[:top_n]
    
    # 分析顶级设计的特性
    num_links_values = []
    wheel_counts = []
    joint_type_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # 不同关节类型的计数
    
    for design in top_designs:
        config = design.get('config', {})
        
        # 收集连杆数量
        if 'num_links' in config:
            num_links_values.append(config['num_links'])
        
        # 收集轮子数量
        if 'is_wheel' in config:
            wheel_count = sum(1 for w in config['is_wheel'] if w)
            wheel_counts.append(wheel_count)
        
        # 收集关节类型
        if 'joint_types' in config:
            for jt in config['joint_types']:
                if jt in joint_type_counts:
                    joint_type_counts[jt] += 1
    
    # 更新最大连杆数 - 基于顶级设计的平均值
    if num_links_values:
        avg_links = sum(num_links_values) / len(num_links_values)
        # 动态调整最大连杆数 - 逐渐增加上限
        current_max = adaptation_state['max_links']
        if avg_links > current_max * 0.8:
            # 如果平均连杆数接近当前上限，提高上限
            adaptation_state['max_links'] = min(20, current_max + 1)
        elif avg_links < current_max * 0.5 and current_max > 8:
            # 如果平均连杆数远低于上限，适当降低上限
            adaptation_state['max_links'] = max(8, current_max - 1)
    
    # 更新轮子概率 - 基于顶级设计的平均轮子比例
    if wheel_counts and num_links_values:
        avg_wheel_ratio = sum(wheel_counts) / sum(num_links_values)
        # 限制最小轮子概率，确保基本移动能力
        min_wheel_prob = max(0.1, avg_wheel_ratio - 0.2)
        adaptation_state['min_wheel_prob'] = min_wheel_prob
        adaptation_state['wheel_probability'] = max(min_wheel_prob, avg_wheel_ratio)
    
    # 更新关节类型分布 - 基于顶级设计中的关节类型比例
    total_joints = sum(joint_type_counts.values())
    if total_joints > 0:
        # 累计概率分布
        cumulative_probs = []
        cum_prob = 0.0
        for jt in range(4):  # 4种关节类型
            cum_prob += joint_type_counts[jt] / total_joints
            cumulative_probs.append(cum_prob)
        
        # 确保最后一个概率为1.0
        if cumulative_probs:
            cumulative_probs[-1] = 1.0
            
        # 限制在合理范围内
        joint_type_ranges = []
        for prob in cumulative_probs[:-1]:  # 不包括最后一个1.0
            joint_type_ranges.append(max(0.1, min(0.9, prob)))
        
        # 确保递增
        for i in range(1, len(joint_type_ranges)):
            if joint_type_ranges[i] <= joint_type_ranges[i-1]:
                joint_type_ranges[i] = joint_type_ranges[i-1] + 0.1
        
        # 如果有效，更新关节类型范围
        if len(joint_type_ranges) == 3 and joint_type_ranges[-1] <= 0.9:
            adaptation_state['joint_type_ranges'] = joint_type_ranges
    
    return adaptation_state


# --- 改进的URDF生成函数 ---
def generate_urdf(gene):
    urdf = '<?xml version="1.0"?>\n<robot name="evolved_robot">\n'
    
    # 材质定义
    urdf += '''  <material name="metal">
    <color rgba="0.7 0.7 0.7 1.0"/>
  </material>
  <material name="plastic">
    <color rgba="0.3 0.3 0.9 1.0"/>
  </material>
  <material name="rubber">
    <color rgba="0.1 0.1 0.1 1.0"/>
  </material>
  <material name="red">
    <color rgba="1.0 0.0 0.0 1.0"/>
  </material>
  <material name="wheel_material">
    <color rgba="0.2 0.2 0.2 1.0"/>
  </material>\n'''
    
    # 获取基因参数
    num_links = gene['num_links']
    joint_types = gene['joint_types']
    shapes = gene['shapes']
    is_wheel = gene['is_wheel']
    link_sizes = gene['link_sizes']
    link_materials = gene['link_materials']
    joint_axes = gene['joint_axes']
    joint_limits = gene['joint_limits']
    joint_damping = gene['joint_damping']
    
    # 材质映射
    material_names = ["metal", "plastic", "rubber"]
    
    # 计算轮子数量
    wheel_count = sum(is_wheel[1:])
    
    # 添加基础连杆 (索引0)
    base_size = link_sizes[0]
    base_material = material_names[link_materials[0]]
    
    # 添加基础质量
    base_mass = 5.0
    inertia_factor = 10.0  # 增加惯性以增强稳定性
    
    if shapes[0] == 0:  # 盒子
        urdf += f'''  <link name="base_link">
    <visual>
      <geometry><box size="{base_size[0]} {base_size[1]} {base_size[2]}"/></geometry>
      <material name="{base_material}"/>
    </visual>
    <collision>
      <geometry><box size="{base_size[0]} {base_size[1]} {base_size[2]}"/></geometry>
    </collision>
    <inertial>
      <mass value="{base_mass}"/>
      <inertia ixx="{inertia_factor * (base_size[1]**2 + base_size[2]**2)/12}" ixy="0" ixz="0" 
               iyy="{inertia_factor * (base_size[0]**2 + base_size[2]**2)/12}" iyz="0" 
               izz="{inertia_factor * (base_size[0]**2 + base_size[1]**2)/12}"/>
    </inertial>
  </link>\n'''
    elif shapes[0] == 1:  # 圆柱
        radius = (base_size[0] + base_size[1]) / 2 / 2  # 平均宽度的一半作为半径
        length = base_size[2]  # z轴作为高度
        urdf += f'''  <link name="base_link">
    <visual>
      <geometry><cylinder radius="{radius}" length="{length}"/></geometry>
      <material name="{base_material}"/>
    </visual>
    <collision>
      <geometry><cylinder radius="{radius}" length="{length}"/></geometry>
    </collision>
    <inertial>
      <mass value="{base_mass}"/>
      <inertia ixx="{inertia_factor * (3*radius**2 + length**2)/12}" ixy="0" ixz="0" 
               iyy="{inertia_factor * (3*radius**2 + length**2)/12}" iyz="0" 
               izz="{inertia_factor * radius**2/2}"/>
    </inertial>
  </link>\n'''
    else:  # 球
        radius = (base_size[0] + base_size[1] + base_size[2]) / 3 / 2  # 平均大小的一半
        urdf += f'''  <link name="base_link">
    <visual>
      <geometry><sphere radius="{radius}"/></geometry>
      <material name="{base_material}"/>
    </visual>
    <collision>
      <geometry><sphere radius="{radius}"/></geometry>
    </collision>
    <inertial>
      <mass value="{base_mass}"/>
      <inertia ixx="{inertia_factor * 2*radius**2/5}" ixy="0" ixz="0" 
               iyy="{inertia_factor * 2*radius**2/5}" iyz="0" 
               izz="{inertia_factor * 2*radius**2/5}"/>
    </inertial>
  </link>\n'''
    
    # 计算其他连杆的相对位置
    # 使用更智能的方法放置连杆
    if num_links > 1:
        base_size = link_sizes[0]
        
        # 计算轮子连接的基本位置
        wheel_indices = [i for i in range(1, num_links) if is_wheel[i]]
        non_wheel_indices = [i for i in range(1, num_links) if not is_wheel[i]]
        
        # 轮子位置策略 - 尝试形成稳定的配置
        wheel_positions = []
        if len(wheel_indices) >= 2:
            chassis_x, chassis_y, chassis_z = base_size
            
            # 标准位置
            standard_positions = [
                [chassis_x/2 - 0.02, chassis_y/2, -chassis_z/2],  # 左前
                [chassis_x/2 - 0.02, -chassis_y/2, -chassis_z/2],  # 右前
                [-chassis_x/2 + 0.02, chassis_y/2, -chassis_z/2],  # 左后
                [-chassis_x/2 + 0.02, -chassis_y/2, -chassis_z/2]  # 右后
            ]
            
            # 根据实际轮子数量选择位置
            num_wheels = len(wheel_indices)
            if num_wheels == 2:
                # 两轮 - 左右放置
                wheel_positions = [standard_positions[0], standard_positions[2]]  # 左前和左后
            elif num_wheels == 3:
                # 三轮 - 三角形
                wheel_positions = [standard_positions[0], standard_positions[2], standard_positions[3]]  # 左前、左后和右后
            elif num_wheels >= 4:
                # 四轮或更多 - 在角落和中间分布
                wheel_positions = standard_positions[:min(4, num_wheels)]
                
                # 如果有额外的轮子，在侧面添加
                for i in range(4, num_wheels):
                    extra_pos = [0, (i % 2) * 2 - 1 * chassis_y/1.5, -chassis_z/2]  # 在侧面交错放置
                    wheel_positions.append(extra_pos)
        
        # 非轮子连杆位置 (如机械臂段等)
        non_wheel_positions = []
        if len(non_wheel_indices) > 0:
            for i in range(len(non_wheel_indices)):
                non_wheel_positions.append([0, 0, base_size[2]/2 + i * 0.05])
        
        # 分配位置
        all_positions = [None] * num_links  # 所有连杆的位置数组
        all_positions[0] = None  # 底盘位置为None
        
        # 分配轮子位置
        for i, wheel_idx in enumerate(wheel_indices):
            if i < len(wheel_positions):
                all_positions[wheel_idx] = wheel_positions[i]
        
        # 分配非轮子位置
        for i, non_wheel_idx in enumerate(non_wheel_indices):
            if i < len(non_wheel_positions):
                all_positions[non_wheel_idx] = non_wheel_positions[i]
        
        # 添加其余连杆
        for i in range(1, num_links):
            link_name = f"link{i}"
            joint_name = f"joint{i}"
            parent_name = "base_link"  # 所有连杆连接到底盘，确保稳定性
            
            # 连杆形状和尺寸
            link_shape = shapes[i]
            link_size = link_sizes[i]
            link_material = material_names[link_materials[i]]
            
            # 特殊处理轮子
            if is_wheel[i]:
                # 轮子通常是圆柱形
                wheel_radius = link_size[0]  # 使用x尺寸作为半径
                wheel_width = link_size[2]  # 使用z尺寸作为宽度
                wheel_mass = 1.0  # 增加轮子质量以增强稳定性
                
                urdf += f'''  <link name="{link_name}">
    <visual>
      <geometry>
        <cylinder radius="{wheel_radius}" length="{wheel_width}"/>
      </geometry>
      <material name="wheel_material"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="{wheel_radius}" length="{wheel_width}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{wheel_mass}"/>
      <inertia ixx="{(wheel_radius**2/4 + wheel_width**2/12)}" ixy="0" ixz="0" 
               iyy="{(wheel_radius**2/4 + wheel_width**2/12)}" iyz="0" 
               izz="{wheel_radius**2/2}"/>
    </inertial>
  </link>\n'''
            else:
                # 基于形状创建非轮子连杆
                if link_shape == 0:  # 盒子
                    urdf += f'''  <link name="{link_name}">
    <visual>
      <geometry><box size="{link_size[0]} {link_size[1]} {link_size[2]}"/></geometry>
      <material name="{link_material}"/>
    </visual>
    <collision>
      <geometry><box size="{link_size[0]} {link_size[1]} {link_size[2]}"/></geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="{(link_size[1]**2 + link_size[2]**2)/12}" ixy="0" ixz="0" 
               iyy="{(link_size[0]**2 + link_size[2]**2)/12}" iyz="0" 
               izz="{(link_size[0]**2 + link_size[1]**2)/12}"/>
    </inertial>
  </link>\n'''
                elif link_shape == 1:  # 圆柱
                    radius = (link_size[0] + link_size[1]) / 4  # 平均宽度的一半作为半径
                    length = link_size[2]  # z轴作为高度
                    urdf += f'''  <link name="{link_name}">
    <visual>
      <geometry><cylinder radius="{radius}" length="{length}"/></geometry>
      <material name="{link_material}"/>
    </visual>
    <collision>
      <geometry><cylinder radius="{radius}" length="{length}"/></geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="{(3*radius**2 + length**2)/12}" ixy="0" ixz="0" 
               iyy="{(3*radius**2 + length**2)/12}" iyz="0" 
               izz="{radius**2/2}"/>
    </inertial>
  </link>\n'''
                else:  # 球
                    radius = min(link_size) / 2  # 使用最小尺寸以确保不会过大
                    urdf += f'''  <link name="{link_name}">
    <visual>
      <geometry><sphere radius="{radius}"/></geometry>
      <material name="{link_material}"/>
    </visual>
    <collision>
      <geometry><sphere radius="{radius}"/></geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="{2*radius**2/5}" ixy="0" ixz="0" 
               iyy="{2*radius**2/5}" iyz="0" 
               izz="{2*radius**2/5}"/>
    </inertial>
  </link>\n'''
            
            # 添加关节
            joint_type = joint_types[i]
            joint_type_str = "fixed"
            if joint_type == p.JOINT_REVOLUTE:
                joint_type_str = "revolute" if not is_wheel[i] else "continuous"
            elif joint_type == p.JOINT_PRISMATIC:
                joint_type_str = "prismatic"
            elif joint_type == p.JOINT_SPHERICAL:
                joint_type_str = "floating"  # PyBullet中没有直接对应球形关节
            
            # 关节位置和方向
            pos = all_positions[i] if all_positions[i] is not None else [0, 0, 0]
            joint_axis = joint_axes[i]
            
            # 对于轮子，设置合适的方向
            rpy = "0 0 0"
            if is_wheel[i]:
                # 让轮子侧向放置，以便它们可以滚动
                rpy = "1.5708 0 0"  # 90度旋转，使轮子垂直于地面
            
            # 关节限制和阻尼
            limits_str = ""
            if joint_type_str in ["revolute", "prismatic"] and not is_wheel[i]:
                limits = joint_limits[i]
                # 确保棱柱关节必须有限制，否则pybullet会报错
                if joint_type_str == "prismatic" and limits[0] >= limits[1]:
                    limits = [-0.5, 0.5]  # 设置默认的棱柱关节运动范围
                limits_str = f"\n    <limit lower=\"{limits[0]}\" upper=\"{limits[1]}\" effort=\"100\" velocity=\"100\"/>"
            
            damping = joint_damping[i]
            
            urdf += f'''  <joint name="{joint_name}" type="{joint_type_str}">
    <parent link="{parent_name}"/>
    <child link="{link_name}"/>
    <origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="{rpy}"/>
    <axis xyz="{joint_axis[0]} {joint_axis[1]} {joint_axis[2]}"/>
    <dynamics damping="{damping}" friction="0.1"/>{limits_str}
  </joint>\n'''
    
    urdf += '</robot>'
    return urdf 

def simulate_robot_multi(gene, config=None):
    """
    模拟机器人在指定地形上的运动表现，并计算多个性能指标。
    
    Args:
        gene: 机器人的基因参数，用于生成URDF
        config: 配置字典，包含模拟参数
            - gui: 是否使用图形界面 (默认False)
            - sim_time: 模拟时间 (默认5.0)
            - use_self_collision: 是否使用自碰撞检测 (默认True)
            - terrain_type: 地形类型 (默认"flat")
            - goal_pos: 目标位置 (默认[5.0, 0, 0.1])
            - terrain_params: 地形生成参数
    
    Returns:
        (dist_score, neg_path_linearity, stability, energy_efficiency): 性能指标元组
    """
    # 默认配置
    default_config = {
        'gui': False,
        'sim_time': 5.0,
        'use_self_collision': True,
        'terrain_type': 'flat',
        'goal_pos': [5.0, 0, 0.1],
        'terrain_params': {
            'size': 256,
            'scale': [0.05, 0.05, 0.2],
            'height_range': [0, 0.4]
        },
        'wheel_control': {
            'velocity': 10.0,
            'force': 100.0
        },
        'joint_control': {
            'force': 10.0
        },
        'stability_threshold': 1.5,  # 大约85度
        'sampling_rate': 100         # 每多少步记录一次数据
    }
    
    # 合并用户配置与默认配置
    if config is None:
        config = {}
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    # 启动物理引擎
    cid = p.connect(p.GUI if config['gui'] else p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    try:
        # 加载地形
        terrain_id = create_terrain(config['terrain_type'], config['terrain_params'])
        
        # 设置目标点
        if config['gui']:
            visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 0.7])
            p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=config['goal_pos'])
        
        # 生成并加载机器人URDF
        robot_id, urdf_path = load_robot_from_gene(gene, config['use_self_collision'])
        
        # 设置相机位置
        if config['gui']:
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0, 
                cameraYaw=0, 
                cameraPitch=-20, 
                cameraTargetPosition=[1.0, 0, 0]
            )
        
        # 识别并控制机器人关节
        joint_data = setup_robot_joints(robot_id, gene, config)
        
        # 执行仿真并收集数据
        simulation_data = run_simulation(
            robot_id, 
            config['goal_pos'], 
            joint_data, 
            config
        )
        
        # 计算性能指标
        performance_metrics = calculate_performance_metrics(simulation_data, config)
        
        return performance_metrics
        
    except Exception as e:
        logging.error(f"Simulation error: {str(e)}", exc_info=True)
        # 返回较差的性能指标
        return 999.0, -0.0, 3.14, 1000.0
        
    finally:
        if 'urdf_path' in locals() and os.path.exists(urdf_path):
            os.unlink(urdf_path)
        p.disconnect()

def create_terrain(terrain_type, params):
    """创建指定类型的地形"""
    size = params['size']
    scale = params['scale']
    
    if terrain_type == "flat":
        return p.loadURDF("plane.urdf")
    
    # 创建高度场地形
    heightfield_data = [0] * (size * size)
    
    if terrain_type == "stairs":
        # 创建台阶地形
        step_size = size // 4
        for i in range(size):
            for j in range(size):
                if i < step_size:
                    heightfield_data[i + j * size] = 0
                elif i < 2 * step_size:
                    heightfield_data[i + j * size] = 1
                elif i < 3 * step_size:
                    heightfield_data[i + j * size] = 2
                else:
                    heightfield_data[i + j * size] = 3
    
    elif terrain_type == "rough":
        # 创建随机不平地形
        height_range = params['height_range']
        for i in range(size):
            for j in range(size):
                heightfield_data[i + j * size] = np.random.uniform(
                    height_range[0], height_range[1]
                )
    
    terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=scale,
        heightfieldData=heightfield_data,
        numHeightfieldRows=size,
        numHeightfieldColumns=size
    )
    terrain = p.createMultiBody(0, terrain_shape)
    p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])
    
    return terrain

def load_robot_from_gene(gene, use_self_collision):
    """从基因生成并加载机器人URDF"""
    urdf_string = generate_urdf(gene)
    with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as f:
        f.write(urdf_string.encode('utf-8'))
        urdf_path = f.name
    
    flags = p.URDF_USE_SELF_COLLISION if use_self_collision else 0
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 1], flags=flags)
    
    return robot_id, urdf_path

def setup_robot_joints(robot_id, gene, config):
    """设置机器人关节控制并返回关节数据"""
    joint_indices = []
    x_axis_wheels = []
    y_axis_wheels = []
    z_axis_wheels = []
    
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        
        # 跳过固定关节
        if joint_type == p.JOINT_FIXED:
            continue
            
        joint_indices.append(i)
        joint_axis = joint_info[13]
        
        # 判断是否为轮子
        is_wheel = is_joint_wheel(joint_name, joint_axis, i, gene)
        
        if is_wheel:
            # 根据主轴分类轮子
            max_axis_idx = np.argmax(np.abs(joint_axis))
            
            if max_axis_idx == 0:  # X轴
                x_axis_wheels.append(i)
                p.setJointMotorControl2(
                    robot_id, i, p.VELOCITY_CONTROL, 
                    targetVelocity=-config['wheel_control']['velocity'], 
                    force=config['wheel_control']['force']
                )
            elif max_axis_idx == 1:  # Y轴
                y_axis_wheels.append(i)
                p.setJointMotorControl2(
                    robot_id, i, p.VELOCITY_CONTROL, 
                    targetVelocity=-config['wheel_control']['velocity'], 
                    force=config['wheel_control']['force']
                )
            else:  # Z轴
                z_axis_wheels.append(i)
                p.setJointMotorControl2(
                    robot_id, i, p.VELOCITY_CONTROL, 
                    targetVelocity=config['wheel_control']['velocity'], 
                    force=config['wheel_control']['force']
                )
        else:
            # 非轮子关节 - 锁定位置
            p.setJointMotorControl2(
                robot_id, i, p.POSITION_CONTROL, 
                targetPosition=0, 
                force=config['joint_control']['force']
            )
    
    if config['gui']:
        logging.info(f"找到轮子关节: X轴={len(x_axis_wheels)}个, Y轴={len(y_axis_wheels)}个, Z轴={len(z_axis_wheels)}个")
    
    return {
        'joint_indices': joint_indices,
        'x_axis_wheels': x_axis_wheels,
        'y_axis_wheels': y_axis_wheels,
        'z_axis_wheels': z_axis_wheels
    }

def is_joint_wheel(joint_name, joint_axis, joint_idx, gene):
    """判断关节是否为轮子"""
    # 1. 通过名称判断
    if "wheel" in joint_name.lower():
        return True
    
    # 2. 通过基因参数判断
    if 'is_wheel' in gene and joint_idx < len(gene['is_wheel']) and gene['is_wheel'][joint_idx]:
        return True
    
    # 3. 通过轴向判断 (主要在X或Y方向的关节可能是轮子)
    max_axis_value = max(abs(joint_axis[0]), abs(joint_axis[1]), abs(joint_axis[2]))
    if abs(joint_axis[0]) == max_axis_value or abs(joint_axis[1]) == max_axis_value:
        return True
    
    return False

def run_simulation(robot_id, goal_pos, joint_data, config):
    """运行仿真并收集数据"""
    # 获取初始位置
    start_pos, _ = p.getBasePositionAndOrientation(robot_id)
    
    # 初始化数据收集
    data = {
        'trajectory': [start_pos],
        'energy': 0.0,
        'max_roll_pitch': 0.0,
        'tipped_over': False
    }
    
    # 模拟步长
    dt = 1.0/240.0
    steps = int(config['sim_time'] / dt)
    sampling_rate = config['sampling_rate']
    
    # 仿真循环
    for step in range(steps):
        p.stepSimulation()
        
        # 按采样率记录数据
        if step % sampling_rate == 0:
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            data['trajectory'].append(pos)
            
            # 计算姿态稳定性
            euler = p.getEulerFromQuaternion(orn)
            roll_pitch = max(abs(euler[0]), abs(euler[1]))
            data['max_roll_pitch'] = max(data['max_roll_pitch'], roll_pitch)
            
            # 计算能量消耗
            for joint_idx in joint_data['joint_indices']:
                joint_state = p.getJointState(robot_id, joint_idx)
                velocity = joint_state[1]  # 关节速度
                force = joint_state[3]     # 关节力/扭矩
                power = abs(velocity * force)  # 功率 = 速度 * 力
                data['energy'] += power * dt  # 能量 = 功率 * 时间
            
            # 检查机器人是否翻倒
            if abs(euler[0]) > config['stability_threshold'] or abs(euler[1]) > config['stability_threshold']:
                data['tipped_over'] = True
                if config['gui']:
                    logging.info("机器人翻倒，提前结束仿真。")
                break
        
        if config['gui']:
            time.sleep(dt)
    
    # 记录最终位置
    end_pos, end_orn = p.getBasePositionAndOrientation(robot_id)
    data['end_pos'] = end_pos
    data['end_orn'] = end_orn
    data['goal_pos'] = goal_pos
    data['start_pos'] = start_pos
    
    return data

def calculate_performance_metrics(data, config):
    """计算性能指标"""
    # 1. 距离目标的距离
    dist_to_goal = np.linalg.norm(np.array(data['end_pos']) - np.array(data['goal_pos']))
    
    # 2. 路径直线性
    path_linearity = 1.0
    if len(data['trajectory']) > 2:
        total_dist = sum(
            np.linalg.norm(np.array(data['trajectory'][i]) - np.array(data['trajectory'][i-1])) 
            for i in range(1, len(data['trajectory']))
        )
        direct_dist = np.linalg.norm(np.array(data['trajectory'][-1]) - np.array(data['trajectory'][0]))
        if total_dist > 0:
            path_linearity = direct_dist / total_dist
    
    # 3. 稳定性
    stability = data['max_roll_pitch']
    
    # 4. 能量效率
    distance_traveled = np.linalg.norm(np.array(data['end_pos']) - np.array(data['start_pos']))
    energy_efficiency = data['energy'] / max(0.1, distance_traveled)
    
    # 考虑前进距离
    forward_distance = data['end_pos'][0] - data['start_pos'][0]
    
    # 根据前进/后退计算最终得分
    if forward_distance > 0:
        dist_score = max(0, dist_to_goal - forward_distance)
    else:
        dist_score = dist_to_goal + abs(forward_distance)
    
    # 如果机器人翻倒，额外惩罚
    if data['tipped_over']:
        dist_score *= 1.5
        stability *= 1.5
        energy_efficiency *= 1.5
    
    return dist_score, -path_linearity, stability, energy_efficiency



# --- 定义多目标优化问题 ---
class RobotMultiObjectiveProblem(Problem):
    def __init__(self, max_links=8, use_self_collision=True, terrain_type="flat"):
        # 计算变量数量：
        # 1个基础变量确定连杆数量 + 6个变量用于底座 + 每个额外连杆13个变量
        # base params: shape, size_x, size_y, size_z, material, unused
        # link params: joint_type, has_motor, shape, is_wheel, wheel_type, 
        #             size_x, size_y, size_z, material, axis_x, axis_y, axis_z, 
        #             limit_lower, limit_upper, damping
        n_vars = 1 + 6 + (max_links-1) * 13
        super().__init__(n_var=n_vars, n_obj=4, n_constr=0, xl=0.0, xu=1.0)
        self.max_links = max_links
        self.use_self_collision = use_self_collision
        self.terrain_type = terrain_type
        
    def _evaluate(self, X, out, *args, **kwargs):
        f1, f2, f3, f4 = [], [], [], []  # 距离、路径直线性、稳定性、能量效率
        
        for x in X:
            # 修改基因，强制底盘更大，更稳定
            modified_x = x.copy()
            # 底盘参数 - 确保更宽、更平稳
            modified_x[2] = 0.7 + 0.3 * x[2]  # X尺寸 (0.7-1.0)
            modified_x[3] = 0.7 + 0.3 * x[3]  # Y尺寸 (0.7-1.0)
            modified_x[4] = 0.2 * x[4]       # Z尺寸 (0.0-0.2) 较扁平
            
            # 确保至少有2个轮子
            wheel_positions = [7, 20, 33, 46]  # 典型轮子位置的基因索引
            wheel_count = 0
            for wheel_idx in wheel_positions:
                if wheel_idx + 3 < len(modified_x):  # 确保有足够的基因位置
                    # 调整连杆参数为轮子
                    modified_x[wheel_idx] = 0.3  # 关节类型 (旋转)
                    modified_x[wheel_idx+1] = 1.0  # 有电机
                    modified_x[wheel_idx+2] = 0.5  # 形状 (圆柱)
                    modified_x[wheel_idx+3] = 1.0  # 是轮子
                    wheel_count += 1
                    if wheel_count >= 2:  # 确保至少有2个轮子
                        break
            
            gene = decode_gene(modified_x, self.max_links)
            d, lin, roll, energy = simulate_robot_multi(
                gene, 
                gui=False, 
                use_self_collision=self.use_self_collision,
                terrain_type=self.terrain_type
            )
            # 额外惩罚不稳定的机器人
            stability_penalty = 5.0 if roll > 1.5 else 0.0  # 如果过度倾斜，增加额外的距离惩罚
            f1.append(d + stability_penalty)  # 加入稳定性惩罚
            f2.append(lin)  # 注意：路径直线性从仿真中已经取负，所以这里直接用
            f3.append(roll)
            f4.append(energy)
            
        out["F"] = np.column_stack([f1, f2, f3, f4])

# --- 可视化最佳机器人设计 ---
def visualize_robot(urdf_file, sim_time=10.0, terrain_type="flat"):
    """可视化机器人设计并进行简单仿真"""
    print(f"\nVisualizing {urdf_file} on {terrain_type} terrain...")
    cid = p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # 创建地形
    if terrain_type == "flat":
        p.loadURDF("plane.urdf")
    elif terrain_type == "stairs":
        # 创建台阶地形
        heightfield_data = [0] * 256 * 256
        for i in range(256):
            for j in range(256):
                if i < 50:
                    heightfield_data[i + j * 256] = 0
                elif i < 100:
                    heightfield_data[i + j * 256] = 1
                elif i < 150:
                    heightfield_data[i + j * 256] = 2
                else:
                    heightfield_data[i + j * 256] = 3
        
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[0.05, 0.05, 0.2],
            heightfieldData=heightfield_data,
            numHeightfieldRows=256,
            numHeightfieldColumns=256
        )
        terrain = p.createMultiBody(0, terrain_shape)
        p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])
    elif terrain_type == "rough":
        # 创建随机不平地形
        heightfield_data = [0] * 256 * 256
        for i in range(256):
            for j in range(256):
                heightfield_data[i + j * 256] = np.random.uniform(0, 0.4)
        
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[0.05, 0.05, 0.2],
            heightfieldData=heightfield_data,
            numHeightfieldRows=256,
            numHeightfieldColumns=256
        )
        terrain = p.createMultiBody(0, terrain_shape)
        p.resetBasePositionAndOrientation(terrain, [-6.0, -6.0, 0], [0, 0, 0, 1])
    
    # 加载机器人
    robot_id = p.loadURDF(urdf_file, basePosition=[0, 0, 0.1])
    
    # 打印机器人信息
    num_joints = p.getNumJoints(robot_id)
    print(f"Robot has {num_joints} joints")
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        print(f"Joint {i}: {joint_info[1].decode('utf-8')}, Type: {joint_info[2]}")
    
    # 设置相机
    p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[1.0, 0, 0])
    
    # 设置目标点
    goal_pos = [5.0, 0, 0.1]
    visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 0.7])
    p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=goal_pos)
    
    # 为所有轮子设置控制
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        
        if joint_type != p.JOINT_FIXED and ("wheel" in joint_name.lower() or "joint" in joint_name.lower()):
            # 轮子向前转动
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=-10.0, force=100.0)
    
    # 添加调试线条，显示轨迹
    line_id = None
    prev_pos = None
    
    # 运行仿真
    print("Running simulation... Press Ctrl+C to stop.")
    try:
        for step in range(int(sim_time / (1./240.))):
            p.stepSimulation()
            time.sleep(1./240.)
            
            # 获取机器人位置和姿态
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            euler = p.getEulerFromQuaternion(orn)
            
            # 绘制轨迹
            if prev_pos is not None:
                if line_id is not None:
                    p.removeUserDebugItem(line_id)
                line_id = p.addUserDebugLine(prev_pos, pos, [0, 1, 0], 2.0)
            prev_pos = pos
            
            # 每秒打印一次位置
            if step % 240 == 0:
                dist_to_goal = np.linalg.norm(np.array(pos) - np.array(goal_pos))
                print(f"Time: {step/240:.1f}s, Pos: {pos}, Roll/Pitch: {euler[0]:.2f}/{euler[1]:.2f}, Dist to goal: {dist_to_goal:.2f}")
                
            # 检查机器人是否翻倒
            if abs(euler[0]) > 1.5 or abs(euler[1]) > 1.5:  # 大约85度
                print("Robot tipped over, ending simulation.")
                break
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        p.disconnect()

# --- 主函数: 设置和运行优化 ---
def main():
    print("开始机器人进化优化...")
    
    try:
        # 设置问题
        problem = RobotMultiObjectiveProblem(max_links=5, use_self_collision=True, terrain_type="flat")
        
        # 配置NSGA-II算法
        algorithm = NSGA2(
            pop_size=10,  # 种群大小
            crossover=SBX(prob=0.9, eta=20),  # 模拟二进制交叉
            mutation=PM(prob=0.2, eta=20),    # 多项式变异
            eliminate_duplicates=True,
            sampling=LHS()  # 拉丁超立方采样进行初始化
        )
        
        # 运行优化
        print("开始优化过程...这可能需要较长时间")
        res = minimize(problem,
                       algorithm,
                       termination=('n_gen', 3),  # 进行3代进化
                       seed=1,
                       verbose=True)

        # 绘制3D帕累托前沿
        F = res.F
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(F[:, 0], F[:, 1], F[:, 2], c=F[:, 3], cmap='viridis', marker='o')
        ax.set_xlabel('Distance to Goal (minimize)')
        ax.set_ylabel('Path Linearity (maximize)')
        ax.set_zlabel('Max Roll/Pitch (minimize)')
        ax.set_title('Pareto Front of Robot Morphology Optimization')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Energy Efficiency (minimize)')
        plt.savefig('pareto_front_fixed.png')
        print("\nOptimization completed. Pareto front saved to 'pareto_front_fixed.png'")
        plt.close()

        # 找到帕累托前沿上的最佳解
        X = res.X
        
        # 根据各个目标找到最佳解
        best_distance_idx = np.argmin(F[:, 0])  # 最小距离
        best_linearity_idx = np.argmax(F[:, 1])  # 最大直线性（注意：直线性是负的，所以使用argmax）
        best_stability_idx = np.argmin(F[:, 2])  # 最小翻滚/俯仰
        best_energy_idx = np.argmin(F[:, 3])     # 最小能量消耗

        # 计算综合评分 - 归一化每个指标并加权求和
        normalized_F = np.zeros_like(F)
        for j in range(F.shape[1]):
            min_val = np.min(F[:, j])
            max_val = np.max(F[:, j])
            if max_val > min_val:
                if j == 1:  # 路径直线性需要特殊处理，因为我们要最大化它
                    normalized_F[:, j] = 1 - (F[:, j] - min_val) / (max_val - min_val)
                else:  # 其他指标都是最小化
                    normalized_F[:, j] = (F[:, j] - min_val) / (max_val - min_val)
            else:
                normalized_F[:, j] = 0

        # 权重可以根据需要调整
        weights = np.array([0.5, 0.2, 0.2, 0.1])  # 距离权重最高
        scores = np.sum(normalized_F * weights, axis=1)
        best_overall_idx = np.argmin(scores)

        # 记录最佳设计
        best_designs = {
            'best_overall': decode_gene(X[best_overall_idx]),
            'best_distance': decode_gene(X[best_distance_idx]),
            'best_linearity': decode_gene(X[best_linearity_idx]),
            'best_stability': decode_gene(X[best_stability_idx]),
            'best_energy': decode_gene(X[best_energy_idx])
        }

        # 保存最佳设计的URDF并打印性能
        for name, design in best_designs.items():
            urdf = generate_urdf(design)
            with open(f"{name}.urdf", "w") as f:
                f.write(urdf)
            
            print(f"\n{name.replace('_', ' ').title()} robot design:")
            print(f"- Number of links: {design['num_links']}")
            print(f"- Wheels: {sum(design['is_wheel'][1:])} of {design['num_links']-1} links")
            
            # 根据索引显示相应的性能
            idx = best_overall_idx
            if name == 'best_distance': idx = best_distance_idx
            elif name == 'best_linearity': idx = best_linearity_idx
            elif name == 'best_stability': idx = best_stability_idx
            elif name == 'best_energy': idx = best_energy_idx
            
            print(f"- Performance: Distance={F[idx, 0]:.3f}, Linearity={F[idx, 1]:.3f}, Stability={F[idx, 2]:.3f}, Energy={F[idx, 3]:.3f}")
        
        # 自动可视化综合最佳设计
        print("\n自动可视化最佳设计中...")
        visualize_robot("best_overall.urdf", sim_time=15.0)
        
        return best_designs
        
    except KeyboardInterrupt:
        print("\n用户中断优化过程")
        return None
    except Exception as e:
        print(f"\n仿真过程中发生错误: {e}")
        return None

# 如果直接运行此脚本，则执行主函数
if __name__ == "__main__":
    main() 