import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import tempfile
import datetime
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover as SBX
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling as LHS
from pymoo.optimize import minimize

# Import functions from robot_evolution_fixed
from robot_evolution_fixed import decode_gene, generate_urdf, simulate_robot_multi

def create_diverse_gene():
    """创建多样化的机器人基因，支持更多种结构类型，包括多足和立体结构"""
    gene = np.zeros(100)
    
    # 1. 连杆数量 - 支持更多连杆以便创建复杂结构
    gene[0] = 0.3 + np.random.random() * 0.7  # 增加连杆基数 (3-8个连杆)
    
    # 2. 车身参数 - 更多样化
    # 形状选择概率
    shape_prob = np.random.random()
    # 更丰富的形状选择，增加复杂形状的概率
    gene[1] = shape_prob
    
    # 尺寸 - 更广泛的尺寸范围，但保持在物理合理范围内
    gene[2] = 0.2 + np.random.random() * 0.5  # 尺寸X：0.2-0.7
    gene[3] = 0.2 + np.random.random() * 0.5  # 尺寸Y：0.2-0.7
    gene[4] = 0.1 + np.random.random() * 0.3  # 尺寸Z：0.1-0.4，支持更高的结构
    
    # 材质 - 更丰富的材质选择
    gene[5] = np.random.random()
    
    # 3. 连杆配置 - 极大提高多样性
    # 随机确定结构类型
    structure_type = np.random.random()
    
    # 四种主要结构类型：轮式(0-0.4)、腿式(0.4-0.6)、混合(0.6-0.8)、自由形态(0.8-1.0)
    if structure_type < 0.4:  # 轮式结构
        # 轮子数量可变
        num_wheels = np.random.randint(2, 7)  # 2-6个轮子
        # 非轮子连杆数量
        num_arms = np.random.randint(0, 3)  # 0-2个机械臂
        
        # 选择轮子和机械臂的索引
        available_indices = list(range(1, 8))
        wheel_indices = np.random.choice(available_indices, size=min(num_wheels, len(available_indices)), replace=False)
        available_indices = [i for i in available_indices if i not in wheel_indices]
        arm_indices = np.random.choice(available_indices, size=min(num_arms, len(available_indices)), replace=False)
        
    elif structure_type < 0.6:  # 腿式结构
        # 为多足结构准备，主要是关节连杆
        num_legs = np.random.randint(2, 7)  # 2-6条腿
        # 每条腿可以有1-2个关节
        leg_indices = []
        available_indices = list(range(1, 8))
        
        # 分配腿部连杆
        for _ in range(min(num_legs, len(available_indices))):
            if available_indices:
                leg_idx = np.random.choice(available_indices)
                leg_indices.append(leg_idx)
                available_indices.remove(leg_idx)
                
        wheel_indices = []  # 无轮子
        arm_indices = leg_indices  # 腿作为机械臂处理
        
    elif structure_type < 0.8:  # 混合结构
        # 轮腿混合结构
        num_wheels = np.random.randint(1, 4)  # 1-3个轮子
        num_legs = np.random.randint(1, 4)  # 1-3条腿
        
        available_indices = list(range(1, 8))
        wheel_indices = np.random.choice(available_indices, size=min(num_wheels, len(available_indices)), replace=False)
        available_indices = [i for i in available_indices if i not in wheel_indices]
        arm_indices = np.random.choice(available_indices, size=min(num_legs, len(available_indices)), replace=False)
        
    else:  # 自由形态结构
        # 完全随机分配
        num_components = np.random.randint(3, 8)
        wheel_prob = 0.3  # 30%的概率是轮子
        
        wheel_indices = []
        arm_indices = []
        
        for i in range(1, min(num_components+1, 8)):
            if np.random.random() < wheel_prob:
                wheel_indices.append(i)
            else:
                arm_indices.append(i)
    
    # 初始化所有潜在连杆
    for i in range(1, 8):
        # 每个连杆的基因起始位置
        idx = 7 + (i-1) * 13
        
        # 关节类型：对腿足结构增加棱柱关节概率
        if i in arm_indices:
            # 关节类型: 0-0.2固定, 0.2-0.6旋转, 0.6-0.9棱柱, 0.9-1.0球形
            joint_type_val = np.random.random()
            if joint_type_val < 0.2:
                gene[idx] = 0.1  # 固定关节
            elif joint_type_val < 0.6:
                gene[idx] = 0.35  # 旋转关节
            elif joint_type_val < 0.9:
                gene[idx] = 0.65  # 棱柱关节(腿部常用)
            else:
                gene[idx] = 0.9  # 球形关节
        else:
            # 轮子或其他连杆，偏向旋转关节
            gene[idx] = 0.3 + np.random.random() * 0.2
            
        # 有电机的概率
        if i in wheel_indices or i in arm_indices:
            gene[idx+1] = 0.6 + np.random.random() * 0.4  # 高概率有电机
        else:
            gene[idx+1] = np.random.random()  # 随机
        
        # 连杆形状：为不同结构类型选择合适形状
        if i in wheel_indices:
            gene[idx+2] = 0.4  # 偏向圆柱形状
            gene[idx+3] = 0.7 + np.random.random() * 0.3  # 是轮子标志
            gene[idx+4] = np.random.random()  # 轮子类型
            
            # 轮子尺寸 - 限制在合理范围内
            gene[idx+5] = 0.3 + np.random.random() * 0.4  # 轮半径
            gene[idx+6] = 0.3 + np.random.random() * 0.4  # 轮宽度
            gene[idx+7] = 0.0  # 不使用
            
            # 轮子材质 - 增加材质多样性
            gene[idx+8] = np.random.random()  # 允许任何材质
            
            # 关节轴 - 轮子允许更多轴向变化
            # 增加X轴、Y轴和Z轴旋转轮的比例
            axis_choice = np.random.random()
            if axis_choice < 0.3:  # 30%概率主要使用X轴
                # X轴为主旋转轴
                gene[idx+9] = 0.7 + np.random.random() * 0.3  # X轴分量大
                gene[idx+10] = 0.1 + np.random.random() * 0.2  # Y轴分量小
                gene[idx+11] = 0.1 + np.random.random() * 0.2  # Z轴分量小
            elif axis_choice < 0.6:  # 30%概率主要使用Y轴
                # Y轴为主旋转轴
                gene[idx+9] = 0.1 + np.random.random() * 0.2  # X轴分量小
                gene[idx+10] = 0.7 + np.random.random() * 0.3  # Y轴分量大
                gene[idx+11] = 0.1 + np.random.random() * 0.2  # Z轴分量小
            elif axis_choice < 0.9:  # 30%概率主要使用Z轴
                # Z轴为主旋转轴 - 全向轮效果
                gene[idx+9] = 0.1 + np.random.random() * 0.2  # X轴分量小
                gene[idx+10] = 0.1 + np.random.random() * 0.2  # Y轴分量小
                gene[idx+11] = 0.7 + np.random.random() * 0.3  # Z轴分量大
            else:  # 10%概率使用更复杂的旋转轴组合
                # 混合轴
                axis_x = np.random.random()
                axis_y = np.random.random()
                axis_z = np.random.random()
                # 标准化
                total = axis_x + axis_y + axis_z
                if total > 0:
                    gene[idx+9] = axis_x / total
                    gene[idx+10] = axis_y / total
                    gene[idx+11] = axis_z / total
                else:
                    gene[idx+9] = 0.33
                    gene[idx+10] = 0.33
                    gene[idx+11] = 0.34
        elif i in arm_indices:
            # 其他代码保持不变
            gene[idx+2] = np.random.random()  # 形状随机
            gene[idx+3] = np.random.random() * 0.3  # 不是轮子
            
            # 连杆尺寸 - 腿/臂通常细长
            gene[idx+4] = 0.2 + np.random.random() * 0.5  # 主长度
            gene[idx+5] = 0.1 + np.random.random() * 0.3  # 次长度
            gene[idx+6] = 0.1 + np.random.random() * 0.3  # 次长度
            
            # 材质 - 随机
            gene[idx+7:idx+9] = np.random.random(2)
            
            # 关节位置 - 允许3D空间分布
            # 为腿足结构设置更多立体空间的位置
            gene[idx+9] = np.random.random()    # X轴位置 - 完全随机
            gene[idx+10] = np.random.random()   # Y轴位置 - 完全随机
            # 增加Z轴变化，支持立体结构
            gene[idx+11] = 0.2 + np.random.random() * 0.6  # Z轴有明显变化
        else:
            # 其他连杆 - 完全随机
            gene[idx+2:idx+12] = np.random.random(10)
            
        # 阻尼与摩擦 - 增加物理特性多样性
        # 不同材质对应不同阻尼和摩擦系数
        if np.random.random() > 0.7:  # 30%概率有特殊的动力学特性
            gene[idx+12] = 0.1 + np.random.random() * 0.9  # 宽范围阻尼
        else:
            gene[idx+12] = 0.3 + np.random.random() * 0.4  # 中等阻尼
    
    return gene

def fix_prismatic_joints(robot_config):
    """修复棱柱关节的限制问题"""
    for i in range(robot_config['num_links']):
        # 检查是否为棱柱关节(PRISMATIC)
        if robot_config['joint_types'][i] == p.JOINT_PRISMATIC:
            # 确保关节限制是有效的
            limits = robot_config['joint_limits'][i]
            if (limits[0] >= limits[1]) or (limits[0] == 0 and limits[1] == 0):
                # 设置默认的限制范围
                robot_config['joint_limits'][i] = [-0.5, 0.5]
            
            # 确保关节不是轮子
            if i > 0 and robot_config['is_wheel'][i]:
                # 将棱柱关节的轮子改为旋转关节
                robot_config['joint_types'][i] = p.JOINT_REVOLUTE
    
    return robot_config

def fix_connection_structure(robot_config, verbose=False):
    """修复零件连接结构问题，防止零件远离主体而没有连接，同时支持更多样的3D结构
    
    Args:
        robot_config: 机器人配置字典
        verbose: 是否打印详细日志，默认为False
    
    Returns:
        修复后的机器人配置
    """
    num_links = robot_config['num_links']
    if num_links <= 1:
        return robot_config  # 只有一个连杆，无需修复
    
    # 检查是否存在parent_indices，如果不存在则初始化
    if 'parent_indices' not in robot_config:
        # 创建默认的父连杆索引 - 所有连杆都连接到主体(索引0)
        robot_config['parent_indices'] = [0] * num_links
        # 第一个连杆(主体)没有父连杆
        robot_config['parent_indices'][0] = -1
        
    # 检查是否存在joint_positions，如果不存在则初始化
    if 'joint_positions' not in robot_config:
        # 创建默认的关节位置
        robot_config['joint_positions'] = []
        # 主体关节位置为原点
        robot_config['joint_positions'].append([0, 0, 0])
        
        # 为其余连杆创建简单的环形排列位置
        for i in range(1, num_links):
            angle = 2 * np.pi * (i / float(num_links))
            # 根据连杆尺寸确定距离，保持在0.05米到0.5米范围内
            radius = 0.2  # 默认值
            if 'link_sizes' in robot_config and i < len(robot_config['link_sizes']):
                # 根据主体和连杆尺寸设置合理距离
                body_size = max(0.05, np.mean(robot_config['link_sizes'][0]))
                link_size = max(0.05, np.mean(robot_config['link_sizes'][i]))
                # 计算距离，确保在合理范围内
                radius = min(0.5, max(0.05 + body_size + link_size, body_size + 2 * link_size))
                
            # 创建环形排列
            pos = [radius * np.cos(angle), radius * np.sin(angle), 0.0]
            robot_config['joint_positions'].append(pos)
    
    # 获取当前的连接结构
    parent_indices = robot_config['parent_indices']
    joint_positions = robot_config['joint_positions']
    
    # 确保joint_positions列表足够长
    while len(joint_positions) < num_links:
        # 添加默认位置
        i = len(joint_positions)
        angle = 2 * np.pi * (i / float(num_links))
        radius = 0.15  # 采用安全的默认距离
        pos = [radius * np.cos(angle), radius * np.sin(angle), 0.0]
        joint_positions.append(pos)
    
    link_sizes = robot_config['link_sizes']
    
    # 标记已修复的连接
    fixed_connections = [False] * num_links
    fixed_connections[0] = True  # 主体默认已修复
    
    # 计算连杆之间的距离
    def calc_distance(pos1, pos2):
        return np.sqrt(np.sum((np.array(pos1) - np.array(pos2))**2))
    
    # 计算连杆的大致尺寸
    def get_link_size(idx):
        if idx == 0:  # 主体
            return max(0.05, max(link_sizes[0]))
        else:
            # 对于轮子或其他零件，使用平均尺寸
            return max(0.05, np.mean(link_sizes[idx]))
    
    # 计算连杆的大致位置（基于关节位置）
    link_positions = [None] * num_links
    link_positions[0] = [0, 0, 0]  # 主体位置
    
    # 第一轮：尝试找到每个连杆的位置
    for i in range(1, num_links):
        if parent_indices[i] >= 0:
            # 关节位置是相对于父连杆的，我们需要计算绝对位置
            parent_pos = link_positions[parent_indices[i]]
            if parent_pos is not None:
                link_positions[i] = [
                    parent_pos[0] + joint_positions[i][0],
                    parent_pos[1] + joint_positions[i][1],
                    parent_pos[2] + joint_positions[i][2]
                ]
    
    # 第二轮：检查并修复连接问题
    max_iterations = 3  # 最大修复迭代次数
    for iteration in range(max_iterations):
        # 检查每个未修复的连接
        all_fixed = True
        for i in range(1, num_links):
            if fixed_connections[i]:
                continue
                
            parent_idx = parent_indices[i]
            
            # 问题1：检查父连杆是否有效
            if parent_idx < 0 or parent_idx >= num_links:
                # 修复：连接到主体
                robot_config['parent_indices'][i] = 0
                parent_idx = 0
                
            # 问题2：检查父连杆是否已修复
            if not fixed_connections[parent_idx]:
                all_fixed = False
                continue  # 等待父连杆被修复
            
            # 问题3：检查与父连杆的距离是否合适
            if link_positions[i] is not None and link_positions[parent_idx] is not None:
                distance = calc_distance(link_positions[i], link_positions[parent_idx])
                parent_size = get_link_size(parent_idx)
                current_size = get_link_size(i)
                
                # 计算最小和最大允许距离
                min_allowed_distance = 0.05 + parent_size/2 + current_size/2  # 至少留出5cm空间
                max_allowed_distance = 0.5  # 最大距离限制为50cm
                
                # 检查是否需要修复
                needs_fix = False
                if distance < min_allowed_distance:
                    needs_fix = True
                    if verbose:
                        print(f"连杆 {i} 距父连杆 {parent_idx} 太近: {distance:.3f}m, 调整中...")
                elif distance > max_allowed_distance:
                    needs_fix = True
                    if verbose:
                        print(f"连杆 {i} 距父连杆 {parent_idx} 太远: {distance:.3f}m, 调整中...")
                
                if needs_fix:
                    # 修复：调整关节位置，使其保持在合理范围内
                    # 计算从父连杆到当前连杆的单位向量
                    direction = np.array(link_positions[i]) - np.array(link_positions[parent_idx])
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        # 如果位置相同，生成真正随机的3D方向
                        random_dir = np.random.random(3) - 0.5
                        direction = random_dir / np.linalg.norm(random_dir)
                    
                    # 设置新的关节位置，确保距离在允许范围内
                    # 使用随机因子确定目标距离，在最小和最大允许距离之间
                    distance_factor = np.random.random() * 0.7 + 0.3  # 0.3-1.0
                    target_distance = min_allowed_distance + distance_factor * (max_allowed_distance - min_allowed_distance)
                    target_distance = min(max_allowed_distance, max(min_allowed_distance, target_distance))
                    
                    new_joint_pos = target_distance * direction
                    
                    # 更新关节位置
                    robot_config['joint_positions'][i] = new_joint_pos.tolist()
                    
                    # 更新连杆位置
                    link_positions[i] = [
                        link_positions[parent_idx][0] + new_joint_pos[0],
                        link_positions[parent_idx][1] + new_joint_pos[1],
                        link_positions[parent_idx][2] + new_joint_pos[2]
                    ]
                    
                    if verbose:
                        print(f"已调整连杆 {i} 与父连杆 {parent_idx} 的距离为 {target_distance:.3f}m")
            
            # 标记为已修复
            fixed_connections[i] = True
        
        # 如果所有连接都已修复，提前结束
        if all_fixed:
            break
    
    # 最后一轮：检查是否还有未修复的连接
    for i in range(1, num_links):
        if not fixed_connections[i]:
            # 将任何未修复的连接直接连到主体，并给定合理位置
            robot_config['parent_indices'][i] = 0
            
            # 给定相对于主体的合理位置（支持3D空间分布）
            # 使用球坐标系生成更多样的位置
            theta = 2 * np.pi * (i / float(num_links))  # 水平角度
            phi = np.random.random() * np.pi - np.pi/2  # 垂直角度 (-π/2 到 π/2)
            
            # 确保距离在0.05米到0.5米之间
            body_size = get_link_size(0)
            current_size = get_link_size(i)
            min_distance = 0.05 + body_size/2 + current_size/2
            radius = min(0.3, max(min_distance, body_size + current_size))
            
            # 根据是否是腿部/机械臂，决定是否使用3D位置
            use_3d = np.random.random() > 0.5  # 50%概率使用3D位置
            
            if use_3d:
                # 3D位置 - 使用球坐标
                x = radius * np.cos(phi) * np.cos(theta)
                y = radius * np.cos(phi) * np.sin(theta)
                z = radius * np.sin(phi) * 0.5  # 缩小Z方向的范围
                new_pos = [x, y, z]
            else:
                # 平面位置 - 传统方法
                new_pos = [
                    radius * np.cos(theta),
                    radius * np.sin(theta),
                    0.0  # 保持在同一平面
                ]
            
            robot_config['joint_positions'][i] = new_pos
            if verbose:
                print(f"将未连接的连杆 {i} 直接连接到主体，距离: {radius:.3f}m")
    
    return robot_config 

def check_connection_quality(robot_config, verbose=False):
    """检查机器人连接质量，判断是否存在零件距离过远或过近的问题
    
    Args:
        robot_config: 机器人配置字典
        verbose: 是否打印详细日志，默认为False
    
    Returns:
        (bool, str): (是否合格, 问题描述)
    """
    num_links = robot_config['num_links']
    if num_links <= 1:
        return True, ""  # 只有一个连杆，无需检查
    
    # 检查是否存在必要的键
    if 'parent_indices' not in robot_config or 'joint_positions' not in robot_config:
        return False, "缺少parent_indices或joint_positions参数"
    
    parent_indices = robot_config['parent_indices']
    joint_positions = robot_config['joint_positions']
    link_sizes = robot_config['link_sizes']
    
    # 计算连杆之间的距离
    def calc_distance(pos1, pos2):
        return np.sqrt(np.sum((np.array(pos1) - np.array(pos2))**2))
    
    # 计算连杆的大致尺寸
    def get_link_size(idx):
        if idx == 0:  # 主体
            return max(0.05, max(link_sizes[0]))
        else:
            # 对于轮子或其他零件，使用平均尺寸
            return max(0.05, np.mean(link_sizes[idx]))
    
    # 计算连杆的大致位置（基于关节位置）
    link_positions = [None] * num_links
    link_positions[0] = [0, 0, 0]  # 主体位置
    
    # 计算每个连杆的绝对位置
    for i in range(1, num_links):
        if parent_indices[i] >= 0:
            # 关节位置是相对于父连杆的，我们需要计算绝对位置
            parent_pos = link_positions[parent_indices[i]]
            if parent_pos is not None:
                link_positions[i] = [
                    parent_pos[0] + joint_positions[i][0],
                    parent_pos[1] + joint_positions[i][1],
                    parent_pos[2] + joint_positions[i][2]
                ]
    
    # 检查连接问题
    connection_issues = []
    
    for i in range(1, num_links):
        parent_idx = parent_indices[i]
        
        # 检查父连杆是否有效
        if parent_idx < 0 or parent_idx >= num_links:
            connection_issues.append(f"连杆 {i} 的父连杆索引 {parent_idx} 无效")
            continue
            
        # 检查与父连杆的距离是否合适
        if link_positions[i] is not None and link_positions[parent_idx] is not None:
            distance = calc_distance(link_positions[i], link_positions[parent_idx])
            parent_size = get_link_size(parent_idx)
            current_size = get_link_size(i)
            
            # 计算最小和最大允许距离
            min_allowed_distance = 0.05 + parent_size/2 + current_size/2  # 至少留出5cm空间
            max_allowed_distance = 0.5  # 最大距离限制为50cm
            
            if distance < min_allowed_distance * 0.8:  # 给予20%的容错空间
                connection_issues.append(f"连杆 {i} 距父连杆 {parent_idx} 太近: {distance:.3f}m < {min_allowed_distance:.3f}m")
            elif distance > max_allowed_distance * 1.2:  # 给予20%的容错空间
                connection_issues.append(f"连杆 {i} 距父连杆 {parent_idx} 太远: {distance:.3f}m > {max_allowed_distance:.3f}m")
    
    if connection_issues:
        if verbose:
            for issue in connection_issues:
                print(f"连接问题: {issue}")
        return False, "; ".join(connection_issues)
    
    return True, ""

class RobotDesignProblem(Problem):
    """机器人设计多目标优化问题"""
    
    def __init__(self, n_var=100, use_gui=False, verbose=False, pause_after_eval=False, add_diversity=True):
        # 定义约束条件数量
        n_constraints = 4  # 设置4个约束条件
        
        super().__init__(
            n_var=n_var,         # 基因变量数量
            n_obj=5 if add_diversity else 4,  # 增加一个多样性目标
            n_constr=n_constraints,  # 约束条件数量 - 从0修改为4
            xl=np.zeros(n_var),  # 基因下限
            xu=np.ones(n_var)    # 基因上限
        )
        
        # 其他参数保持不变
        self.use_gui = use_gui
        self.verbose = verbose
        self.pause_after_eval = pause_after_eval
        self.add_diversity = add_diversity
        self.evaluated_designs = []  # 记录已评估的设计，用于计算多样性
        
        # 添加约束参数
        self.min_stability = 0.75  # 降低稳定性要求，允许更多样化结构
        self.max_energy = 2000  # 提高能耗上限，允许腿式结构的更高能耗
        self.connection_quality_threshold = 0.8  # 连接质量阈值

        self.structure_types = {
            'wheeled': 0,  # 轮式结构数量
            'legged': 0,   # 腿式结构数量
            'hybrid': 0,   # 混合结构数量
            'other': 0     # 其他结构数量
        }
        
        # 添加进化过程跟踪变量
        self.current_generation = 0
        self.total_generations = 0
        self.total_populations = 0

    def _evaluate(self, X, out, *args, **kwargs):
        """评估机器人设计的适应度和约束条件"""
        n_individuals = X.shape[0]
        F = np.zeros((n_individuals, self.n_obj))
        G = np.zeros((n_individuals, self.n_constr))  # 创建约束违反数组
        
        # 统计当前种群的结构类型
        current_population_types = {
            'wheeled': 0,
            'legged': 0,
            'hybrid': 0,
            'other': 0
        }
        
        # 打印当前进化过程信息
        print(f"\n当前进化状态:")
        print(f"- 当前代数: {self.current_generation + 1}/{self.total_generations}")
        print(f"- 当前种群: {n_individuals} 个个体")
        print(f"- 剩余代数: {self.total_generations - self.current_generation - 1}")
        print(f"- 总种群数: {self.total_populations}")
        print(f"- 已评估个体数: {len(self.evaluated_designs)}")
        
        for i in range(n_individuals):
            gene = X[i, :]
            print(f"\n评估个体 {i+1}/{n_individuals}")
            
            # 解码基因为机器人配置
            robot_config = decode_gene(gene)
            
            # 检查并修复棱柱关节的限制 (保留此修复功能是合理的)
            robot_config = fix_prismatic_joints(robot_config)
            
            # 确保robot_config中有必要的键 (保留数据完整性检查)
            if 'parent_indices' not in robot_config:
                num_links = robot_config['num_links']
                robot_config['parent_indices'] = [0] * num_links
                robot_config['parent_indices'][0] = -1
                
            if 'joint_positions' not in robot_config:
                num_links = robot_config['num_links']
                robot_config['joint_positions'] = []
                robot_config['joint_positions'].append([0, 0, 0])
                
                # 为其余连杆创建环形排列位置
                for j in range(1, num_links):
                    angle = 2 * np.pi * (j / float(num_links))
                    radius = 0.2
                    if j < len(robot_config['link_sizes']):
                        radius = max(0.2, np.mean(robot_config['link_sizes'][j]) * 2)
                    pos = [radius * np.cos(angle), radius * np.sin(angle), 0.0]
                    robot_config['joint_positions'].append(pos)
            
            # 修复零件连接问题 (仍然需要此修复以确保物理模拟可行)
            robot_config = fix_connection_structure(robot_config, verbose=self.verbose)
            
            # ===== 评估约束条件 =====
            # 约束1: 最小轮子数量
            connection_ok, connection_issues = check_connection_quality(robot_config, verbose=False)
            G[i, 0] = 0 if connection_ok else 1  # 如果连接有问题，约束违反值为1
            
            # 约束2: 稳定性约束
            stability_score = self.estimate_stability(robot_config)
            G[i, 1] = self.min_stability - stability_score  # 如果稳定性低于阈值，值为正
            
            # 识别结构类型 - 新增
            structure_type = self.classify_structure_type(robot_config)
            # 更新当前种群的结构类型统计
            current_population_types[structure_type] += 1
            
            # 如果连接质量检查不合格，我们仍然可以继续计算目标函数，
            # 但NSGA-II会优先考虑满足约束的解
            
            try:
                # 模拟机器人并获取性能指标
                metrics = simulate_robot_multi(
                    robot_config, 
                    gui=self.use_gui,
                    sim_time=10.0,
                    terrain_type="rough"
                )
                
                # 记录性能指标
                F[i, 0] = -metrics[0]  # 距离 (最大化，所以取负)
                F[i, 1] = -metrics[1]  # 路径直线性 (最大化，所以取负)
                F[i, 2] = metrics[2]   # 稳定性 (最小化)
                F[i, 3] = metrics[3]   # 能量消耗 (最小化)
                
                # 约束4: 能耗约束 - 使用实际能耗
                G[i, 2] = metrics[3] - self.max_energy  # 如果能耗超过限制，值为正
                
                # 为腿式结构提供奖励 - 移动距离目标
                if structure_type == 'legged':
                    # 轻微降低腿式结构的距离标准，使其更容易被保留
                    distance_bonus = 0.1 * metrics[0]
                    F[i, 0] -= distance_bonus  # 注意这里是减负数，实际上是增加正值
                
                # 记录结构信息 - 添加结构类型
                self.evaluated_designs.append({
                    'gene': gene.copy(),
                    'config': {
                        'num_links': robot_config['num_links'],
                        'num_wheels': sum(robot_config['is_wheel']),
                        'shape_type': robot_config['shapes'][0],
                        'joint_types': robot_config['joint_types'].copy(),
                        'structure_type': structure_type  # 新增
                    }
                })
                
                # 更新结构类型统计
                self.structure_types[structure_type] += 1
                
                # 如果启用多样性目标，计算多样性得分
                if self.add_diversity and len(self.evaluated_designs) > 1:
                    # 增强版多样性计算 - 考虑结构类型稀有度
                    diversity_score = self.calculate_enhanced_diversity(robot_config, structure_type, i)
                    F[i, 4] = -diversity_score  # 最大化多样性，所以取负
                elif self.add_diversity:
                    F[i, 4] = 0.0  # 第一个个体的多样性得分设为0
                
            except Exception as e:
                print(f"模拟过程出错: {str(e)}")
                # 如果模拟失败，给个体评分为最差
                F[i, 0] = 0.0    # 距离为0
                F[i, 1] = 0.0    # 直线性为0
                F[i, 2] = 3.14   # 最大稳定性问题
                F[i, 3] = 1000   # 最大能耗
                
                # 同时设置能耗约束为严重违反
                G[i, 3] = 500.0  # 严重违反能耗约束
                
                if self.add_diversity:
                    F[i, 4] = 0.0  # 多样性得分为0
            
            # 打印该个体的评估结果
            print(f"个体评估结果:")
            print(f"- 移动距离: {-F[i, 0]:.2f}")
            print(f"- 路径直线性: {-F[i, 1]:.2f}")
            print(f"- 稳定性指标: {F[i, 2]:.2f}")
            print(f"- 能量消耗: {F[i, 3]:.2f}")
            if self.add_diversity:
                print(f"- 结构多样性: {-F[i, 4]:.2f}")
            print(f"约束违反情况:")
            print(f"- 连接质量约束: {G[i, 0]:.2f}")
            print(f"- 稳定性约束: {G[i, 1]:.2f}")
            print(f"- 能耗约束: {G[i, 2]:.2f}")
            
            # 如果启用了暂停，让用户决定何时继续
            if self.use_gui and self.pause_after_eval:
                input("按Enter键继续评估下一个个体...")

        # 打印当前种群的结构类型分布
        print("\n当前种群结构类型分布:")
        total = sum(current_population_types.values())
        for type_name, count in current_population_types.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"- {type_name}: {count} ({percentage:.1f}%)")
        
        # 输出目标函数值和约束违反值
        out["F"] = F
        out["G"] = G  # 这是关键一步！

    def classify_structure_type(self, robot_config):
        """识别机器人的结构类型"""
        num_wheels = sum(robot_config['is_wheel'])
        num_links = robot_config['num_links']
        num_non_wheels = num_links - 1 - num_wheels  # 减去1是因为主体不算
        
        # 检查关节类型
        joint_types = robot_config['joint_types']
        prismatic_joints = sum(1 for jt in joint_types if jt == p.JOINT_PRISMATIC)
        
        # 计算轮子的平均高度（Z坐标）
        wheel_heights = []
        if 'joint_positions' in robot_config:
            for i in range(1, num_links):
                if robot_config['is_wheel'][i]:
                    wheel_heights.append(robot_config['joint_positions'][i][2])
        
        # 检查轮子是否在不同高度 - 可能表示有腿部结构
        height_variation = 0
        if wheel_heights:
            height_variation = max(wheel_heights) - min(wheel_heights)
        
        # 分类标准
        if num_wheels == 0:
            # 无轮结构
            if prismatic_joints > 0:
                return 'legged'  # 假设棱柱关节用于腿部伸缩
            else:
                return 'other'   # 其他非轮式结构
        elif num_wheels <= 2 and num_non_wheels >= 2:
            # 少量轮子，但有多个非轮连杆
            return 'legged'      # 可能是腿式结构
        elif num_wheels >= 3 and height_variation > 0.1:
            # 多轮结构但高度不同
            return 'hybrid'      # 可能是轮腿混合结构
        elif num_wheels >= 1 and num_non_wheels >= 3:
            # 有轮子但非轮连杆较多
            return 'hybrid'      # 混合结构
        else:
            # 默认为轮式结构
            return 'wheeled'

    def calculate_enhanced_diversity(self, robot_config, structure_type, current_idx):
        """增强版多样性计算 - 考虑结构类型稀有度"""
        # 提取当前设计的特征
        current_features = np.array([
            robot_config['num_links'],
            sum(robot_config['is_wheel']),
            robot_config['shapes'][0],
            np.mean(robot_config['joint_types']),
            np.std(robot_config['joint_types']),
            np.mean(robot_config['link_sizes'])
        ])
        
        # 计算结构类型稀有度奖励
        total_designs = sum(self.structure_types.values())
        if total_designs > 0:
            type_ratio = self.structure_types[structure_type] / total_designs
            # 稀有结构获得更高奖励 (最大奖励系数为2倍)
            rarity_bonus = 1.0 + (1.0 - min(type_ratio, 0.5) * 2.0)
        else:
            rarity_bonus = 1.0
        
        # 与之前所有设计计算差异度
        diversity_scores = []
        for idx, design in enumerate(self.evaluated_designs):
            if idx == current_idx:
                continue  # 跳过当前设计
                
            # 提取历史设计特征
            prev_config = design['config']
            prev_features = np.array([
                prev_config['num_links'],
                prev_config['num_wheels'],
                prev_config['shape_type'],
                np.mean(prev_config['joint_types']),
                np.std(prev_config['joint_types']),
                0.1  # 替代值
            ])
            
            # 计算欧氏距离
            distance = np.linalg.norm(current_features - prev_features)
            
            # 如果结构类型不同，增加差异度
            if 'structure_type' in prev_config and prev_config['structure_type'] != structure_type:
                distance *= 1.2  # 不同结构类型获得20%额外差异度
                
            diversity_scores.append(distance)
        
        # 如果没有历史设计，返回基础分数
        if not diversity_scores:
            return 1.0 * rarity_bonus
            
        # 返回平均差异度乘以稀有度奖励
        return np.mean(diversity_scores) * rarity_bonus
    
    # def estimate_stability(self, robot_config):
    #     """估计机器人的稳定性分数，基于车身形状和轮子配置"""
    #     # 获取车身形状和尺寸
    #     body_shape = robot_config['shapes'][0]
    #     body_size = robot_config['link_sizes'][0]
        
    #     # 计算轮子的数量和分布
    #     num_wheels = sum(robot_config['is_wheel'])
    #     wheel_positions = []
        
    #     # 如果有joint_positions信息，计算轮子位置
    #     if 'joint_positions' in robot_config and 'parent_indices' in robot_config:
    #         # 对于每个被标记为轮子的连杆
    #         for i in range(1, robot_config['num_links']):
    #             if robot_config['is_wheel'][i]:
    #                 # 获取关节位置
    #                 wheel_positions.append(robot_config['joint_positions'][i])
        
    #     # 基础稳定性分数
    #     stability_score = 0.0
        
    #     # 1. 基于车身形状的稳定性 - 盒子最稳定，球体最不稳定
    #     shape_stability = {
    #         0: 1.0,  # 盒子
    #         1: 0.7,  # 圆柱
    #         2: 0.5   # 球体
    #     }
    #     # 将连续值映射到离散类别
    #     shape_idx = min(int(body_shape * 3), 2)
    #     stability_score += shape_stability[shape_idx] * 0.3  # 形状贡献30%
        
    #     # 2. 基于车身尺寸的稳定性 - 较宽的底座更稳定
    #     # 计算底面面积
    #     if shape_idx == 0:  # 盒子
    #         base_area = body_size[0] * body_size[1]  # 底面积
    #         height = body_size[2]  # 高度
    #     elif shape_idx == 1:  # 圆柱
    #         radius = max(body_size[0], body_size[1]) / 2  # 估计半径
    #         base_area = 3.14159 * radius * radius  # 底面积
    #         height = body_size[2]  # 高度
    #     else:  # 球体
    #         radius = max(body_size) / 2  # 估计半径
    #         base_area = 3.14159 * radius * radius  # 虚拟底面积
    #         height = 2 * radius  # 高度
        
    #     # 计算高度与底面积的比例 - 较低的值表示更稳定
    #     if base_area > 0:
    #         aspect_ratio = height / np.sqrt(base_area)
    #         # 转换为0-1分数，数值越大越稳定
    #         size_stability = 1.0 / (1.0 + aspect_ratio)
    #     else:
    #         size_stability = 0.1  # 防止除以零
        
    #     stability_score += size_stability * 0.3  # 尺寸贡献30%
        
    #     # 3. 基于轮子配置的稳定性
    #     if num_wheels >= 3:
    #         # 3个或更多轮子通常更稳定
    #         wheel_stability = min(1.0, num_wheels / 6.0)  # 最多6个轮子算满分
            
    #         # 如果有足够的轮子位置信息，评估轮子分布
    #         if len(wheel_positions) >= 3:
    #             # 计算轮子的凸包面积或分布
    #             # 这里简化为检查轮子是否分布在不同象限
    #             quadrants = set()
    #             for pos in wheel_positions:
    #                 if pos[0] >= 0 and pos[1] >= 0:
    #                     quadrants.add(1)
    #                 elif pos[0] < 0 and pos[1] >= 0:
    #                     quadrants.add(2)
    #                 elif pos[0] < 0 and pos[1] < 0:
    #                     quadrants.add(3)
    #                 else:
    #                     quadrants.add(4)
                
    #             # 分布在更多象限通常更稳定
    #             distribution_factor = len(quadrants) / 4.0
    #             wheel_stability *= (0.5 + 0.5 * distribution_factor)  # 轮子分布影响稳定性
    #     else:
    #         # 少于3个轮子通常不稳定
    #         wheel_stability = 0.2 * num_wheels  # 最多0.4分
        
    #     stability_score += wheel_stability * 0.4  # 轮子配置贡献40%
        
    #     return stability_score  # 范围0-1，值越大越稳定


    import numpy as np
    from scipy.spatial import ConvexHull

    def estimate_stability(self, robot_config):
        """估计机器人结构的稳定性
        
        纯粹基于物理原理：评估质心位置相对于支撑多边形的关系。
        稳定性取决于质心的水平投影是否位于支撑多边形内部，以及质心高度。
        
        Args:
            robot_config (dict): 机器人配置字典
            
        Returns:
            float: 稳定性得分 (0-1，越高越稳定)
        """
        # 1. 确定支撑点 - 所有轮子和接触地面的固定关节
        support_points = []
        
        for i in range(robot_config['num_links']):
            # 检查是否是轮子或接触地面的固定点
            is_wheel = robot_config['is_wheel'][i] if i < len(robot_config['is_wheel']) else False
            
            if is_wheel:
                # 轮子作为支撑点
                if i < len(robot_config['joint_positions']):
                    support_points.append(robot_config['joint_positions'][i])
                    
            # 检查其他可能接触地面的点 (如固定关节且z坐标接近0)
            elif i > 0 and i < len(robot_config['joint_positions']):
                position = robot_config['joint_positions'][i]
                # 假设z坐标小于特定阈值的点可能接触地面
                if position[2] < 0.05:  # 5厘米阈值
                    support_points.append(position)
        
        # 如果支撑点少于3个，无法形成稳定的支撑多边形
        if len(support_points) < 3:
            # 支撑点数量越少，稳定性越差
            return len(support_points) / 3.0
        
        # 2. 计算质心位置
        # 假设每个连杆质量相等，或根据连杆尺寸估计质量
        masses = []
        positions = []
        
        for i in range(robot_config['num_links']):
            if i < len(robot_config['joint_positions']):
                position = robot_config['joint_positions'][i]
                
                # 估计质量 - 可以基于连杆尺寸或使用均匀质量
                if 'link_sizes' in robot_config and i < len(robot_config['link_sizes']):
                    # 基于连杆体积估计质量
                    link_size = robot_config['link_sizes'][i]
                    volume = link_size[0] * link_size[1] * link_size[2]
                    mass = max(0.1, volume)  # 最小质量为0.1kg
                else:
                    mass = 1.0  # 默认质量为1kg
                    
                masses.append(mass)
                positions.append(position)
        
        # 计算质心
        total_mass = sum(masses)
        if total_mass <= 0:
            return 0.0  # 如果没有质量，返回最低稳定性
            
        center_of_mass = np.zeros(3)
        for i in range(len(masses)):
            center_of_mass += np.array(positions[i]) * masses[i]
        
        center_of_mass /= total_mass
        
        # 3. 计算支撑多边形 (仅考虑xy平面)
        support_points_2d = [[p[0], p[1]] for p in support_points]
        
        try:
            # 计算支撑点形成的凸包
            hull = ConvexHull(support_points_2d)
            vertices = hull.vertices
            hull_points = np.array(support_points_2d)[vertices]
            
            # 计算支撑多边形的面积
            area = hull.volume  # 在2D中，volume实际上是面积
            
            # 4. 判断质心投影是否在支撑多边形内
            com_2d = [center_of_mass[0], center_of_mass[1]]
            
            # 使用射线法检查点是否在多边形内
            def is_point_in_polygon(point, polygon):
                """检查点是否在多边形内部"""
                x, y = point
                n = len(polygon)
                inside = False
                
                p1x, p1y = polygon[0]
                for i in range(1, n + 1):
                    p2x, p2y = polygon[i % n]
                    if y > min(p1y, p2y):
                        if y <= max(p1y, p2y):
                            if x <= max(p1x, p2x):
                                if p1y != p2y:
                                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or x <= xinters:
                                    inside = not inside
                    p1x, p1y = p2x, p2y
                
                return inside
            
            point_in_polygon = is_point_in_polygon(com_2d, hull_points)
            
            # 5. 计算稳定性得分
            if not point_in_polygon:
                # 质心不在支撑多边形内 - 计算到最近边的距离
                # 这种情况下机器人不稳定，但我们可以评估不稳定程度
                
                # 计算质心到多边形每条边的最短距离
                min_distance = float('inf')
                for i in range(len(hull_points)):
                    p1 = hull_points[i]
                    p2 = hull_points[(i + 1) % len(hull_points)]
                    
                    # 计算点到线段的距离
                    def dist_point_to_segment(p, s1, s2):
                        """计算点到线段的距离"""
                        line_vec = np.array(s2) - np.array(s1)
                        point_vec = np.array(p) - np.array(s1)
                        line_len = np.linalg.norm(line_vec)
                        
                        # 避免除以零
                        if line_len == 0:
                            return np.linalg.norm(point_vec)
                            
                        line_unitvec = line_vec / line_len
                        point_vec_scaled = point_vec / line_len
                        
                        t = np.dot(line_unitvec, point_vec_scaled)
                        
                        if t < 0.0:
                            t = 0.0
                        elif t > 1.0:
                            t = 1.0
                            
                        nearest = np.array(s1) + t * line_vec
                        return np.linalg.norm(nearest - np.array(p))
                    
                    distance = dist_point_to_segment(com_2d, p1, p2)
                    min_distance = min(min_distance, distance)
                
                # 根据距离计算不稳定性
                # 归一化距离，考虑支撑多边形的尺度
                polygon_radius = np.sqrt(area / np.pi)  # 估计多边形的"半径"
                normalized_distance = min_distance / max(polygon_radius, 0.1)
                
                # 距离越大，稳定性越差
                stability = max(0.0, 0.5 - normalized_distance)
            else:
                # 质心在支撑多边形内 - 完全稳定的基础分数
                base_stability = 0.6
                
                # 计算质心到支撑多边形中心的距离
                polygon_center = np.mean(hull_points, axis=0)
                center_distance = np.linalg.norm(np.array(com_2d) - polygon_center)
                
                # 归一化距离
                polygon_radius = np.sqrt(area / np.pi)
                normalized_center_distance = center_distance / max(polygon_radius, 0.1)
                
                # 质心越接近支撑多边形中心，稳定性越高
                center_stability = max(0.0, 0.3 * (1.0 - normalized_center_distance))
                
                # 考虑质心高度 - 高度越低越稳定
                height = center_of_mass[2]
                height_factor = max(0.0, 0.1 * (1.0 - height))
                
                stability = base_stability + center_stability + height_factor
        
        except Exception as e:
            # 如果凸包计算失败（例如，所有点共线）
            # 退化到基于支撑点数量的简单估计
            stability = min(0.3, len(support_points) / 10.0)
        
        # 确保稳定性在0-1范围内
        return min(1.0, max(0.0, stability))


    def calculate_diversity(self, robot_config, current_idx):
        """计算当前设计与之前设计的结构差异度"""
        # 提取当前设计的特征
        current_features = np.array([
            robot_config['num_links'],
            sum(robot_config['is_wheel']),
            robot_config['shapes'][0],
            np.mean(robot_config['joint_types']),
            np.std(robot_config['joint_types']),
            np.mean(robot_config['link_sizes'])
        ])
        
        # 与之前所有设计计算差异度
        diversity_scores = []
        for idx, design in enumerate(self.evaluated_designs):
            if idx == current_idx:
                continue  # 跳过当前设计
                
            # 提取历史设计特征
            prev_config = design['config']
            prev_features = np.array([
                prev_config['num_links'],
                prev_config['num_wheels'],
                prev_config['shape_type'],
                np.mean(prev_config['joint_types']),
                np.std(prev_config['joint_types']),
                0.1  # 替代值
            ])
            
            # 计算欧氏距离
            distance = np.linalg.norm(current_features - prev_features)
            diversity_scores.append(distance)
        
        # 如果没有历史设计，返回0
        if not diversity_scores:
            return 0.0
            
        # 返回平均差异度
        return np.mean(diversity_scores)

def create_legged_gene():
        """创建基于腿部结构的机器人基因"""
        gene = np.zeros(100)
        
        # 1. 连杆数量 - 腿式结构需要更多连杆
        gene[0] = 0.4 + np.random.random() * 0.6  # 5-8个连杆
        
        # 2. 车身参数 - 扁平化设计
        # 形状 - 偏向于盒子形状(更稳定)
        gene[1] = np.random.random() * 0.3  # 90%概率是盒子
        
        # 尺寸 - 确保合理的车身比例
        gene[2] = 0.4 + np.random.random() * 0.3  # 尺寸X - 中等 (0.4-0.7)
        gene[3] = 0.4 + np.random.random() * 0.3  # 尺寸Y - 中等 (0.4-0.7)
        gene[4] = 0.1 + np.random.random() * 0.15  # 尺寸Z - 更扁平 (0.1-0.25)
        
        # 材质 - 随机
        gene[5] = np.random.random()
        
        # 3. 腿部设计 - 2-4条腿
        num_legs = np.random.randint(2, 5)
        leg_indices = np.random.choice(range(1, 8), size=min(num_legs, 7), replace=False)
        
        # 每条腿的基因起始位置
        for i in range(1, 8):
            idx = 7 + (i-1) * 13
            
            # 检查是否为腿部连杆
            if i in leg_indices:
                # 是腿部 - 设置腿部特性
                joint_type_val = np.random.random()
                if joint_type_val < 0.4:
                    gene[idx] = 0.3  # 旋转关节 (40%概率)
                else:
                    gene[idx] = 0.65  # 棱柱关节 (60%概率) - 用于伸缩腿
                    
                gene[idx+1] = 0.6 + np.random.random() * 0.4  # 有电机
                gene[idx+2] = 0.2 + np.random.random() * 0.6  # 形状多样化
                gene[idx+3] = 0.0 + np.random.random() * 0.2  # 不是轮子
                
                # 腿部尺寸 - 细长
                gene[idx+4] = 0.1 + np.random.random() * 0.3  # 宽度较小
                gene[idx+5] = 0.1 + np.random.random() * 0.3  # 宽度较小
                gene[idx+6] = 0.5 + np.random.random() * 0.5  # 长度较大
                
                # 腿部材质 - 随机
                gene[idx+7] = np.random.random()
                gene[idx+8] = np.random.random()
                
                # 腿部位置 - 分布在车身四周
                # 使用极坐标，确保均匀分布
                angle = 2.0 * np.pi * (np.where(leg_indices == i)[0][0] / float(num_legs))
                
                # 转换为0-1范围的值
                x_pos = 0.5 + 0.4 * np.cos(angle)  # 0.1-0.9
                y_pos = 0.5 + 0.4 * np.sin(angle)  # 0.1-0.9
                
                gene[idx+9] = x_pos
                gene[idx+10] = y_pos
                
                # Z轴位置 - 腿部通常比车身低
                gene[idx+11] = 0.0 + np.random.random() * 0.4  # 0.0-0.4，表示向下
                
                # 关节动力学属性
                gene[idx+12] = 0.3 + np.random.random() * 0.5  # 阻尼适中偏高
            else:
                # 非腿部连杆 - 可以是支架或传感器等
                gene[idx] = 0.1  # 倾向于固定关节
                gene[idx+1] = 0.1 + np.random.random() * 0.3  # 电机概率低
                
                # 随机形状和尺寸
                gene[idx+2:idx+7] = np.random.random(5)
                gene[idx+3] = np.random.random() * 0.2  # 确保不是轮子
                
                # 其他参数随机
                gene[idx+7:idx+13] = np.random.random(6)
        
        return gene

def control_legged_robot(robot_id, robot_config):
        """为腿式机器人生成协调的运动控制"""
        # 识别所有关节
        all_joints = []
        for i in range(p.getNumJoints(robot_id)):
            joint_info = p.getJointInfo(robot_id, i)
            joint_type = joint_info[2]
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                all_joints.append(i)
        
        if not all_joints:
            return []  # 没有可用关节
        
        # 获取腿部关节 (非轮子关节)
        leg_joints = []
        for i in range(1, robot_config['num_links']):
            if i < len(robot_config['joint_types']) and not robot_config['is_wheel'][i]:
                joint_type = robot_config['joint_types'][i]
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    # 找到物理引擎中对应的关节索引
                    for j in all_joints:
                        joint_info = p.getJointInfo(robot_id, j)
                        if joint_info[1].decode('utf-8') == f"joint{i}":
                            leg_joints.append(j)
                            break
        
        # 如果没有找到腿部关节，尝试使用所有非轮子关节
        if not leg_joints:
            # 查找轮子关节
            wheel_joints = []
            for i in all_joints:
                joint_info = p.getJointInfo(robot_id, i)
                joint_name = joint_info[1].decode('utf-8')
                if "wheel" in joint_name.lower():
                    wheel_joints.append(i)
            
            # 使用非轮子关节作为腿部
            leg_joints = [j for j in all_joints if j not in wheel_joints]
        
        # 如果仍然没有腿部关节，返回所有关节
        if not leg_joints:
            return all_joints
        
        # 按腿分组关节（简化：假设每条腿只有一个关节）
        num_legs = len(leg_joints)
        
        # 设置腿的相位差，使其形成交替模式
        for t in range(100):  # 100个时间步
            for i, joint_idx in enumerate(leg_joints):
                # 获取关节类型
                joint_info = p.getJointInfo(robot_id, joint_idx)
                joint_type = joint_info[2]
                
                # 计算该腿的相位
                phase = 2.0 * np.pi * (i / float(num_legs))
                phase_offset = 2.0 * np.pi * (t / 20.0)  # 20步一个完整周期
                
                if joint_type == p.JOINT_PRISMATIC:
                    # 棱柱关节 - 伸缩运动
                    extension = 0.1 * np.sin(phase + phase_offset)
                    p.setJointMotorControl2(
                        robot_id, joint_idx, 
                        p.POSITION_CONTROL, 
                        targetPosition=extension,
                        force=10.0
                    )
                else:  # 旋转关节
                    # 旋转关节 - 摆动运动
                    angle = 0.5 * np.sin(phase + phase_offset)
                    p.setJointMotorControl2(
                        robot_id, joint_idx, 
                        p.POSITION_CONTROL, 
                        targetPosition=angle,
                        force=5.0
                    )
        
        return leg_joints

def test_robot_with_gene(gene):
        """测试使用基因参数生成的机器人"""
    # 解码基因为机器人配置
        robot_config = decode_gene(gene)
        
        # 确保必要的键
        if 'parent_indices' not in robot_config:
            # ... (保持原有代码)
            pass
            
        if 'joint_positions' not in robot_config:
            # ... (保持原有代码)
            pass
        
        # 应用修复
        robot_config = fix_prismatic_joints(robot_config)
        robot_config = fix_connection_structure(robot_config, verbose=True)
        
        # 打印机器人配置信息
        print_robot_structure(robot_config)
        
        # 识别结构类型
        num_wheels = sum(robot_config['is_wheel'])
        joint_types = robot_config['joint_types']
        prismatic_joints = sum(1 for jt in joint_types if jt == p.JOINT_PRISMATIC)
        
        # 简单结构分类
        if num_wheels == 0:
            structure_type = 'legged' if prismatic_joints > 0 else 'other'
        elif num_wheels <= 2:
            structure_type = 'legged'  # 少量轮子，可能是腿式或混合结构
        else:
            structure_type = 'wheeled'  # 多轮结构
        
        print(f"\n检测到的结构类型: {structure_type}")
        
        # 生成URDF
        urdf = generate_urdf(robot_config)
        with open("gene_robot.urdf", "w") as f:
            f.write(urdf)
        print("\n已生成基于基因的机器人URDF")  
        
        # 初始化PyBullet
        p.connect(p.GUI)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # 加载地面
        p.loadURDF("plane.urdf")
        
        # 加载机器人
        robot_id = p.loadURDF("gene_robot.urdf", basePosition=[0, 0, 0.1])
        
        # 设置相机
        p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0, 0, 0])
        
        # 目标点
        goal_pos = [2.0, 0, 0.1]
        visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 0.7])
        p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=goal_pos)
        
        # 根据结构类型选择控制方式
        if structure_type == 'wheeled':
            # 轮式结构 - 使用传统控制方法
            wheel_joints = []
            for i in range(p.getNumJoints(robot_id)):
                joint_info = p.getJointInfo(robot_id, i)
                joint_name = joint_info[1].decode('utf-8')
                joint_type = joint_info[2]
                
                is_wheel = False
                if "wheel" in joint_name.lower() or (i < len(robot_config['is_wheel']) and robot_config['is_wheel'][i]):
                    is_wheel = True
                elif joint_type == p.JOINT_REVOLUTE:
                    joint_axis = p.getJointInfo(robot_id, i)[13]
                    max_axis = max(abs(joint_axis[0]), abs(joint_axis[1]), abs(joint_axis[2]))
                    if abs(joint_axis[0]) == max_axis or abs(joint_axis[1]) == max_axis:
                        is_wheel = True
                
                if is_wheel:
                    wheel_joints.append(i)
            
            print(f"\n找到 {len(wheel_joints)} 个轮子关节，使用轮式控制")
            
            # 为轮子设置控制
            for i in wheel_joints:
                p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=10.0, force=10.0)
                
        elif structure_type == 'legged':
            # 腿式结构 - 使用步态生成控制
            print("\n检测到腿式结构，使用腿部控制策略")
            leg_joints = control_legged_robot(robot_id, robot_config)
            print(f"已设置 {len(leg_joints)} 个腿部关节的协调控制")
            
        else:
            # 混合或其他结构 - 尝试通用控制
            print("\n检测到混合或特殊结构，尝试通用控制")
            for i in range(p.getNumJoints(robot_id)):
                joint_type = p.getJointInfo(robot_id, i)[2]
                if joint_type != p.JOINT_FIXED:
                    p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=5.0, force=5.0)
        
        # 模拟循环
        print("\n开始模拟 - 按Ctrl+C停止")
        try:
            for _ in range(10000):  # 运行约40秒
                # 如果是腿式结构，每20步更新一次控制
                if structure_type == 'legged' and _ % 20 == 0:
                    control_legged_robot(robot_id, robot_config)
                    
                p.stepSimulation()
                time.sleep(1/240.0)
        except KeyboardInterrupt:
            print("\n模拟被用户中断")
        finally:
            p.disconnect()
        
        print("\n模拟完成")
            

# def run_diverse_genetic_optimization(pop_size=10, n_gen=5, use_gui=True, 
#                                     verbose=False, pause_after_eval=False,
#                                     save_designs=True, checkpoint_interval=5,
#                                     resume_from=None):
#     """运行多样化结构的遗传算法优化机器人设计，使用约束优化"""
#     print("\n开始遗传算法优化机器人设计...")
#     print(f"种群大小: {pop_size}, 进化代数: {n_gen}")
#     print(f"使用结构约束: 是")
#     print(f"增加结构多样性: 是")
#     print(f"使用约束优化: 是")
#     print(f"结构均衡: 是 (确保初始种群包含多种结构)")
#     print(f"显示模拟可视化: {'是' if use_gui else '否'}")
#     print(f"打印详细结构信息: {'是' if verbose else '否'}")
#     print(f"每次评估后暂停: {'是' if pause_after_eval else '否'}")
#     print(f"保存机器人设计: {'是' if save_designs else '否'}")
    
#     try:
#         # 定义问题
#         problem = RobotDesignProblem(n_var=100, 
#                                     use_gui=use_gui,
#                                     verbose=verbose, 
#                                     pause_after_eval=pause_after_eval, 
#                                     add_diversity=True)
        
#         # 设置进化过程跟踪变量
#         problem.total_generations = n_gen
#         problem.total_populations = pop_size * n_gen
        
#         # 创建带有约束的初始种群，确保包含多种结构类型
#         initial_pop = np.zeros((pop_size, 100))
        
#         # 分配不同结构类型的比例
#         wheeled_ratio = 0.4  # 40% 轮式
#         legged_ratio = 0.4   # 40% 腿式
#         hybrid_ratio = 0.2   # 20% 混合/多样
        
#         wheeled_count = int(pop_size * wheeled_ratio)
#         legged_count = int(pop_size * legged_ratio)
#         hybrid_count = pop_size - wheeled_count - legged_count
        
#         print(f"\n初始种群结构分布:")
#         print(f"- 轮式结构: {wheeled_count} ({wheeled_ratio*100:.0f}%)")
#         print(f"- 腿式结构: {legged_count} ({legged_ratio*100:.0f}%)")
#         print(f"- 混合/多样结构: {hybrid_count} ({hybrid_ratio*100:.0f}%)")
        
#         # 生成不同类型的个体
#         for i in range(pop_size):
#             if i < wheeled_count:
#                 # 轮式结构
#                 gene = create_constrained_gene()  # 使用原有的轮式结构生成函数
#                 print(f"生成个体 {i+1}: 轮式结构")
#             elif i < wheeled_count + legged_count:
#                 # 腿式结构
#                 gene = create_legged_gene()  # 使用新的腿式结构生成函数
#                 print(f"生成个体 {i+1}: 腿式结构")
#             else:
#                 # 混合/多样结构
#                 gene = create_diverse_gene()  # 使用多样化结构生成函数
#                 print(f"生成个体 {i+1}: 混合/多样结构")
            
#             # 尝试10次生成有效设计
#             for attempt in range(10):
#                 # 检查基本可行性
#                 robot_config = decode_gene(gene)
#                 robot_config = fix_prismatic_joints(robot_config)
                
#                 # 确保基本配置
#                 if 'parent_indices' not in robot_config:
#                     num_links = robot_config['num_links']
#                     robot_config['parent_indices'] = [0] * num_links
#                     robot_config['parent_indices'][0] = -1
                    
#                 if 'joint_positions' not in robot_config:
#                     # ... (原有代码)
#                     pass
                
#                 # 修复连接
#                 robot_config = fix_connection_structure(robot_config, verbose=False)
                
#                 # 检查连接质量
#                 connection_ok, issues = check_connection_quality(robot_config, verbose=False)
                
#                 # 检查稳定性
#                 stability_score = problem.estimate_stability(robot_config)
#                 stability_ok = (stability_score >= problem.min_stability)
                
#                 # 基本检查合格
#                 if connection_ok and stability_ok:
#                     initial_pop[i] = gene
#                     break
                    
#                 # 如果检查不通过，重新生成该类型的基因
#                 if i < wheeled_count:
#                     gene = create_constrained_gene()
#                 elif i < wheeled_count + legged_count:
#                     gene = create_legged_gene()
#                 else:
#                     gene = create_diverse_gene()
            
#             # 如果10次尝试都失败，使用备选方案
#             if attempt == 9 and (not connection_ok or not stability_ok):
#                 print(f"警告: 个体 {i+1} 10次尝试后仍不符合基本要求，使用备选设计")
#                 # 根据所需类型选择备选设计方法
#                 if i < wheeled_count:
#                     # 轮式备选设计 - 使用最安全的轮式设计
#                     gene = np.zeros(100)
#                     # 设置基本参数 (保证稳定的四轮设计)
#                     gene[0] = 0.4  # 连杆数量 (~5)
#                     gene[1] = 0.1  # 盒子形状
#                     gene[2:5] = [0.6, 0.6, 0.2]  # 扁平宽大的车身
#                     # ... 轮子参数设置
#                 elif i < wheeled_count + legged_count:
#                     # 腿式备选设计 - 较简单的双腿结构
#                     gene = np.zeros(100)
#                     gene[0] = 0.3  # 连杆数量 (~4)
#                     gene[1] = 0.1  # 盒子形状
#                     gene[2:5] = [0.5, 0.3, 0.1]  # 扁平长方形车身
#                     # ... 腿部参数设置
#                 else:
#                     # 多样化备选设计
#                     gene = create_diverse_gene()
                
#                 initial_pop[i] = gene
        
#         # 使用自定义初始种群
#         from pymoo.core.sampling import Sampling
        
#         class CustomSampling(Sampling):
#             def __init__(self, initial_pop):
#                 super().__init__()
#                 self.initial_pop = initial_pop
                
#             def _do(self, problem, n_samples, **kwargs):
#                 return self.initial_pop
                
#         sampling = CustomSampling(initial_pop)
        
#         # 修改交叉算子和变异算子，增加多样性
#         crossover = SBX(prob=0.9, eta=10)  # 降低eta增加多样性
#         mutation = PM(prob=0.2, eta=15)    # 增加变异概率
        
#         # 设置NSGA-II算法，显式指定约束处理方法
#         algorithm = NSGA2(
#             pop_size=pop_size,
#             sampling=sampling,
#             crossover=crossover,
#             mutation=mutation,
#             eliminate_duplicates=True,
#             constraint_handling="feasibility_first"  # 首先考虑可行性
#         )
        
#         # 记录开始时间
#         start_time = time.time()
        
#         # 定义回调函数来更新当前代数
#         def callback(algorithm):
#             problem.current_generation = algorithm.n_gen
        
#         # 运行优化
#         results = minimize(
#             problem,
#             algorithm,
#             ('n_gen', n_gen),
#             verbose=True,
#             save_history=True,
#             callback=callback
#         )
        
#         # 记录结束时间
#         end_time = time.time()
#         print(f"优化完成! 耗时: {end_time - start_time:.2f} 秒")
        
#         # 获取结果
#         X = results.X  # 决策变量
#         F = results.F  # 目标函数值
#         G = results.G  # 约束违反值
        
#         # 可视化Pareto前沿
#         visualize_pareto_front(F)
        
#         # 可视化约束违反情况
#         visualize_constraints(G)
        
#         # 可视化结构类型分布
#         visualize_structure_distribution(problem.structure_types)
        
#         # 找到最佳设计
#         # 首先根据约束违反值筛选出可行解
#         feasible_mask = np.all(G <= 0, axis=1)
#         feasible_indices = np.where(feasible_mask)[0]
        
#         if len(feasible_indices) > 0:
#             # 在可行解中找到最好的解
#             feasible_F = F[feasible_indices]
#             best_feasible_idx = feasible_indices[np.argmin(feasible_F[:, 0])]
#             print(f"\n找到 {len(feasible_indices)} 个满足所有约束的设计")
#             print(f"从中选择目标最优的设计 (索引: {best_feasible_idx})")
#             best_gene = X[best_feasible_idx]
#         else:
#             # 如果没有完全可行的解，找到约束违反最小的
#             # 计算每个解的总约束违反量
#             constraint_violation = np.sum(np.maximum(0, G), axis=1)
#             min_violation_idx = np.argmin(constraint_violation)
#             print("\n警告: 没有找到完全满足所有约束的设计")
#             print(f"选择约束违反最小的设计 (索引: {min_violation_idx})")
#             best_gene = X[min_violation_idx]
        
#         print("\n测试最佳设计...")
#         robot_config = decode_gene(best_gene)
#         robot_config = fix_prismatic_joints(robot_config)
#         robot_config = fix_connection_structure(robot_config)
#         print_robot_structure(robot_config)
        
#         # 如果启用了保存设计，将所有进化出的设计保存到文件
#         if save_designs:
#             save_evolved_designs(problem.evaluated_designs, X, F, G)
        
#         try:
#             test_robot_with_gene(best_gene)
#             # 保存最佳设计的URDF文件
#             best_urdf_file = save_best_robot_design(best_gene)
#         except Exception as e:
#             print(f"测试最佳设计时出错: {str(e)}")
#             print("尝试修复问题后重新测试...")
#             try:
#                 # 尝试修复基因问题
#                 fixed_gene = create_constrained_gene()
#                 # 复制最佳设计的部分特征
#                 fixed_gene[0] = best_gene[0]  # 连杆数量
#                 fixed_gene[1:6] = best_gene[1:6]  # 车身参数
#                 test_robot_with_gene(fixed_gene)
#                 # 保存修复后的设计
#                 save_best_robot_design(fixed_gene)
#             except Exception as e2:
#                 print(f"修复后测试仍然失败: {str(e2)}")
#                 print("请尝试使用默认基因测试。")
        
#         return best_gene
#     except Exception as e:
#         print(f"遗传算法优化过程中出错: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return None


def run_diverse_genetic_optimization(pop_size=10, n_gen=5, use_gui=True, verbose=False, 
                                     pause_after_eval=False, save_designs=True,
                                     checkpoint_interval=5, resume_from=None):
    """运行多样化结构的遗传算法优化机器人设计，使用约束优化
    
    Args:
        pop_size (int): 种群大小
        n_gen (int): 进化代数
        use_gui (bool): 是否显示可视化界面
        verbose (bool): 是否打印详细信息
        pause_after_eval (bool): 每次评估后是否暂停
        save_designs (bool): 是否保存设计
        checkpoint_interval (int): 检查点保存间隔(代数)
        resume_from (str): 从检查点恢复(文件名)
        
    Returns:
        np.ndarray: 最佳机器人设计基因
    """
    # 打印参数信息
    print("\n开始遗传算法优化机器人设计...")
    print(f"种群大小: {pop_size}, 进化代数: {n_gen}")
    print(f"使用结构约束: 是")
    print(f"增加结构多样性: 是 (使用多样性奖励机制)")
    print(f"使用约束优化: 是")
    print(f"显示模拟可视化: {'是' if use_gui else '否'}")
    print(f"打印详细结构信息: {'是' if verbose else '否'}")
    print(f"每次评估后暂停: {'是' if pause_after_eval else '否'}")
    print(f"保存机器人设计: {'是' if save_designs else '否'}")
    
    try:
        # 定义问题 - 启用多样性奖励
        problem = RobotDesignProblemWithDiversityReward(
            n_var=100, 
            use_gui=use_gui, 
            verbose=verbose, 
            pause_after_eval=pause_after_eval, 
            add_diversity=True,
            diversity_reward_weight=0.2  # 新增：多样性奖励权重
        )
        
        # 设置进化过程跟踪变量
        problem.total_generations = n_gen
        problem.total_populations = pop_size * n_gen
        
        # 从检查点恢复或创建新种群
        if resume_from and os.path.exists(resume_from):
            print(f"从检查点恢复: {resume_from}")
            checkpoint_data = load_checkpoint(resume_from)
            initial_pop = checkpoint_data["initial_pop"]
            problem.evaluated_designs = checkpoint_data["problem_data"]["evaluated_designs"]
            problem.structure_types = checkpoint_data["problem_data"]["structure_types"]
            problem.current_generation = checkpoint_data["problem_data"]["current_generation"]
            start_gen = problem.current_generation
            print(f"恢复从第 {start_gen} 代开始的优化")
        else:
            start_gen = 0
            # 创建初始种群 - 不再使用固定结构比例
            initial_pop = np.zeros((pop_size, 100))
            
            print("\n生成初始种群 - 使用多样性奖励而非固定结构比例")
            
            # 生成个体 - 均匀分布结构类型概率
            structure_type_counts = {'wheeled': 0, 'legged': 0, 'hybrid': 0, 'other': 0}
            
            for i in range(pop_size):
                # 随机选择结构类型 - 均匀概率分布
                r = np.random.random()
                if r < 0.33:
                    structure_type = "轮式结构"
                    gene_creator = create_constrained_gene
                elif r < 0.67:
                    structure_type = "腿式结构"
                    gene_creator = create_legged_gene
                else:
                    structure_type = "混合/多样结构"
                    gene_creator = create_diverse_gene
                    
                print(f"生成个体 {i+1}: {structure_type}")
                
                # 尝试生成符合约束的有效设计
                valid_design = False
                for attempt in range(10):
                    gene = gene_creator()  # 生成一个基因
                    
                    # 检查基本可行性
                    robot_config = decode_gene(gene)
                    robot_config = fix_prismatic_joints(robot_config)
                    
                    # 修复连接结构
                    robot_config = fix_connection_structure(robot_config, verbose=False)
                    
                    # 检查连接质量
                    connection_ok, issues = check_connection_quality(robot_config, verbose=False)
                    
                    # 检查稳定性
                    stability_score = problem.estimate_stability(robot_config)
                    stability_ok = (stability_score >= problem.min_stability)
                    
                    # 基本检查合格
                    if connection_ok and stability_ok:
                        initial_pop[i] = gene
                        valid_design = True
                        
                        # 记录结构类型
                        structure_type_name = problem.classify_structure_type(robot_config)
                        structure_type_counts[structure_type_name] += 1
                        break
                
                # 如果10次尝试都失败，使用替代方法
                if not valid_design:
                    print(f"警告: 个体 {i+1} 10次尝试后仍不符合基本要求，尝试替代方法")
                    
                    # 使用更简单的结构生成方法
                    success = False
                    
                    # 方法1: 尝试生成更简单的版本
                    simple_gene = generate_simple_design(structure_type)
                    
                    # 检查简化版本是否可行
                    try:
                        robot_config = decode_gene(simple_gene)
                        robot_config = fix_prismatic_joints(robot_config)
                        robot_config = fix_connection_structure(robot_config, verbose=False)
                        
                        connection_ok, _ = check_connection_quality(robot_config, verbose=False)
                        stability_score = problem.estimate_stability(robot_config)
                        stability_ok = (stability_score >= problem.min_stability)
                        
                        if connection_ok and stability_ok:
                            initial_pop[i] = simple_gene
                            success = True
                            print(f"  成功使用简化{structure_type}生成有效设计")
                            
                            # 记录结构类型
                            structure_type_name = problem.classify_structure_type(robot_config)
                            structure_type_counts[structure_type_name] += 1
                    except Exception as e:
                        if verbose:
                            print(f"  简化设计生成失败: {str(e)}")
                    
                    # 方法2: 随机生成多个设计并选择最佳的
                    if not success:
                        print(f"  尝试随机生成并筛选最佳设计")
                        best_gene, best_score = find_best_random_design(problem, 20)
                        
                        if best_gene is not None and best_score > 0:
                            initial_pop[i] = best_gene
                            success = True
                            print(f"  使用随机生成的最佳设计（可行性分数: {best_score:.2f}）")
                            
                            # 记录结构类型
                            try:
                                robot_config = decode_gene(best_gene)
                                robot_config = fix_prismatic_joints(robot_config)
                                robot_config = fix_connection_structure(robot_config, verbose=False)
                                structure_type_name = problem.classify_structure_type(robot_config)
                                structure_type_counts[structure_type_name] += 1
                            except:
                                pass
                    
                    # 方法3: 复制种群中已有的设计并添加变异
                    if not success and i > 0:
                        print(f"  无法生成有效设计，复制并修改现有设计")
                        # 查找已经创建的可行设计
                        for j in range(i):
                            try:
                                robot_config = decode_gene(initial_pop[j])
                                robot_config = fix_prismatic_joints(robot_config)
                                robot_config = fix_connection_structure(robot_config, verbose=False)
                                
                                connection_ok, _ = check_connection_quality(robot_config, verbose=False)
                                stability_score = problem.estimate_stability(robot_config)
                                
                                if connection_ok and stability_score >= problem.min_stability:
                                    # 复制基因并添加变异
                                    mutated_gene = initial_pop[j].copy()
                                    # 随机选择10个位置进行变异
                                    mutation_indices = np.random.choice(100, 10, replace=False)
                                    mutated_gene[mutation_indices] = np.random.random(10)
                                    
                                    initial_pop[i] = mutated_gene
                                    success = True
                                    print(f"  成功使用修改后的个体 {j+1} 作为替代设计")
                                    
                                    # 记录结构类型 - 可能与原始类型相同或不同
                                    robot_config = decode_gene(mutated_gene)
                                    robot_config = fix_prismatic_joints(robot_config)
                                    robot_config = fix_connection_structure(robot_config, verbose=False)
                                    structure_type_name = problem.classify_structure_type(robot_config)
                                    structure_type_counts[structure_type_name] += 1
                                    break
                            except:
                                continue
                    
                    # 方法4: 最后的后备选择 - 使用默认有效基因
                    if not success:
                        print(f"  所有尝试都失败，使用默认设计")
                        default_gene = get_default_valid_gene()
                        initial_pop[i] = default_gene
                        
                        try:
                            robot_config = decode_gene(default_gene)
                            robot_config = fix_prismatic_joints(robot_config)
                            robot_config = fix_connection_structure(robot_config, verbose=False)
                            structure_type_name = problem.classify_structure_type(robot_config)
                            structure_type_counts[structure_type_name] += 1
                        except:
                            structure_type_counts['other'] += 1
            
            # 打印最终初始种群结构分布
            print("\n初始种群结构分布:")
            for type_name, count in structure_type_counts.items():
                percentage = (count / pop_size) * 100 if pop_size > 0 else 0
                print(f"- {type_name}: {count} ({percentage:.1f}%)")
        
        # 使用自定义初始种群
        from pymoo.core.sampling import Sampling
        
        class CustomSampling(Sampling):
            def __init__(self, initial_pop):
                super().__init__()
                self.initial_pop = initial_pop
                
            def _do(self, problem, n_samples, **kwargs):
                return self.initial_pop
                
        sampling = CustomSampling(initial_pop)
        
        # 修改交叉算子和变异算子，增加多样性
        crossover = SBX(prob=0.9, eta=10)  # 降低eta增加多样性
        mutation = PM(prob=0.2, eta=15)    # 增加变异概率
        
        # 设置NSGA-II算法，显式指定约束处理方法
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True,
            constraint_handling="feasibility_first"  # 首先考虑可行性
        )
        
        # 记录开始时间
        start_time = time.time()
        
        # 创建结果文件夹
        results_dir = "optimization_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 定义回调函数来更新当前代数和处理中断
        def callback(algorithm):
            problem.current_generation = algorithm.n_gen
            
            # 每checkpoint_interval代保存检查点
            if checkpoint_interval > 0 and algorithm.n_gen % checkpoint_interval == 0:
                checkpoint_file = os.path.join(results_dir, f"checkpoint_gen_{algorithm.n_gen}.pkl")
                save_checkpoint(algorithm, problem, checkpoint_file)
            
            # 检查中断信号
            if os.path.exists("stop_optimization.flag"):
                print("检测到中断信号，正在保存检查点并退出...")
                checkpoint_file = os.path.join(results_dir, "interrupted_checkpoint.pkl")
                save_checkpoint(algorithm, problem, checkpoint_file)
                return True  # 停止优化
                
            return False
        
        # 运行优化
        results = minimize(
            problem,
            algorithm,
            ('n_gen', n_gen - start_gen),  # 考虑恢复的情况，只运行剩余的代数
            verbose=True,
            save_history=True,
            callback=callback
        )
        
        # 记录结束时间
        end_time = time.time()
        print(f"优化完成! 耗时: {end_time - start_time:.2f} 秒")
        
        # 获取结果
        X = results.X  # 决策变量
        F = results.F  # 目标函数值
        G = results.G  # 约束违反值
        
        # 可视化Pareto前沿
        visualize_pareto_front(F, save_path=os.path.join(results_dir, "pareto_front.png"))
        
        # 可视化约束违反情况
        visualize_constraints(G, save_path=os.path.join(results_dir, "constraints.png"))
        
        # 可视化结构类型分布
        visualize_structure_distribution(problem.structure_types, 
                                        save_path=os.path.join(results_dir, "structure_types.png"))
        
        # 可视化结构多样性演化 (新增)
        visualize_diversity_evolution(problem.generation_stats,
                                     save_path=os.path.join(results_dir, "diversity_evolution.png"))
        
        # 找到最佳设计
        # 首先根据约束违反值筛选出可行解
        feasible_mask = np.all(G <= 0, axis=1)
        feasible_indices = np.where(feasible_mask)[0]
        
        if len(feasible_indices) > 0:
            # 在可行解中找到最好的解
            feasible_F = F[feasible_indices]
            best_feasible_idx = feasible_indices[np.argmin(feasible_F[:, 0])]
            print(f"\n找到 {len(feasible_indices)} 个满足所有约束的设计")
            print(f"从中选择目标最优的设计 (索引: {best_feasible_idx})")
            best_gene = X[best_feasible_idx]
            
            # 分析最佳设计的结构类型
            robot_config = decode_gene(best_gene)
            robot_config = fix_prismatic_joints(robot_config)
            robot_config = fix_connection_structure(robot_config)
            structure_type = problem.classify_structure_type(robot_config)
            print(f"最佳设计结构类型: {structure_type}")
        else:
            # 如果没有完全可行的解，找到约束违反最小的
            # 计算每个解的总约束违反量
            constraint_violation = np.sum(np.maximum(0, G), axis=1)
            min_violation_idx = np.argmin(constraint_violation)
            print("\n警告: 没有找到完全满足所有约束的设计")
            print(f"选择约束违反最小的设计 (索引: {min_violation_idx})")
            best_gene = X[min_violation_idx]
        
        print("\n测试最佳设计...")
        robot_config = decode_gene(best_gene)
        robot_config = fix_prismatic_joints(robot_config)
        robot_config = fix_connection_structure(robot_config)
        print_robot_structure(robot_config)
        
        # 如果启用了保存设计，将所有进化出的设计保存到文件
        if save_designs:
            save_evolved_designs(problem.evaluated_designs, X, F, G, 
                                save_path=os.path.join(results_dir, "evolved_designs.pkl"))
        
        # 尝试测试最佳设计
        try:
            test_robot_with_gene(best_gene)
            # 保存最佳设计的URDF文件
            best_urdf_file = save_best_robot_design(best_gene, 
                                                    save_path=os.path.join(results_dir, "best_robot.urdf"))
            print(f"最佳设计已保存到: {best_urdf_file}")
        except Exception as e:
            print(f"测试最佳设计时出错: {str(e)}")
            print("尝试修复问题后重新测试...")
            try:
                # 获取结构类型
                robot_config = decode_gene(best_gene)
                robot_config = fix_prismatic_joints(robot_config)
                robot_config = fix_connection_structure(robot_config)
                
                # 根据结构类型生成新的基因
                structure_type = problem.classify_structure_type(robot_config)
                if structure_type == "wheeled":
                    fixed_gene = create_constrained_gene()
                elif structure_type == "legged":
                    fixed_gene = create_legged_gene()
                else:
                    fixed_gene = create_diverse_gene()
                
                # 保留部分原始设计特征
                fixed_gene[0] = best_gene[0]  # 连杆数量
                if len(robot_config['link_sizes']) > 0:
                    # 保留主体尺寸参数 (前5-10个参数)
                    params_to_preserve = min(10, len(best_gene)//10)
                    fixed_gene[1:params_to_preserve] = best_gene[1:params_to_preserve]
                    
                test_robot_with_gene(fixed_gene)
                # 保存修复后的设计
                fixed_urdf_file = save_best_robot_design(fixed_gene, 
                                                        save_path=os.path.join(results_dir, "fixed_best_robot.urdf"))
                print(f"修复后的设计已保存到: {fixed_urdf_file}")
                # 更新最佳基因
                best_gene = fixed_gene
            except Exception as e2:
                print(f"修复后测试仍然失败: {str(e2)}")
                print("请尝试使用默认基因测试。")
        
        # 保存最终结果摘要
        save_optimization_summary(problem, results, best_gene, 
                                 save_path=os.path.join(results_dir, "optimization_summary.json"))
        
        return best_gene
    except Exception as e:
        print(f"遗传算法优化过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# 新增：带有多样性奖励的机器人设计问题类
class RobotDesignProblemWithDiversityReward(RobotDesignProblem):
    """具有结构多样性奖励的机器人设计问题"""
    
    def __init__(self, *args, diversity_reward_weight=0.2, **kwargs):
        """初始化具有多样性奖励的问题
        
        Args:
            *args: 传递给父类的位置参数
            diversity_reward_weight: 多样性奖励权重 (0-1之间)
            **kwargs: 传递给父类的关键字参数
        """
        super().__init__(*args, **kwargs)
        self.diversity_reward_weight = diversity_reward_weight
        
        # 添加结构类型计数器 - 用于计算结构多样性奖励
        self.current_structure_counts = {
            'wheeled': 0,
            'legged': 0,
            'hybrid': 0,
            'other': 0
        }
    
    def calculate_structure_diversity_reward(self, structure_type):
        """计算结构多样性奖励分数
        
        计算给定结构类型的稀有度奖励，基于当前评估的结构类型分布。
        
        Args:
            structure_type (str): 结构类型
            
        Returns:
            float: 多样性奖励分数 (0-1之间，越高奖励越大)
        """
        # 更新当前结构类型计数
        self.current_structure_counts[structure_type] += 1
        
        # 计算总评估数
        total = sum(self.current_structure_counts.values())
        
        if total <= 1:
            return 0.0  # 第一个设计没有奖励
            
        # 计算结构类型的稀有度 - 越稀有奖励越高
        type_ratio = self.current_structure_counts[structure_type] / total
        
        # 转换为奖励值 (稀有结构获得更高奖励)
        # 使用非线性映射使得差异更明显
        rarity_score = 1.0 - (type_ratio**0.7)  # 指数<1使稀有类型获得更高奖励
        
        # 应用权重并确保在合理范围内
        return min(1.0, max(0.0, rarity_score * self.diversity_reward_weight))
    
    def _calculate_objectives(self, F, G, i, metrics, structure_type, robot_config):
        """重写目标函数计算，加入多样性奖励
        
        Args:
            F (numpy.ndarray): 目标函数值矩阵
            G (numpy.ndarray): 约束违反值矩阵
            i (int): 当前个体索引
            metrics (list): 模拟得到的性能指标 [距离, 直线性, 稳定性, 能耗]
            structure_type (str): 结构类型
            robot_config (dict): 机器人配置
        """
        # 计算基础目标函数值
        F[i, 0] = -metrics[0]  # 距离 (最大化，所以取负)
        F[i, 1] = -metrics[1]  # 路径直线性 (最大化，所以取负)
        F[i, 2] = metrics[2]   # 稳定性 (最小化)
        F[i, 3] = metrics[3]   # 能量消耗 (最小化)
        
        # 约束: 能耗约束
        G[i, 2] = metrics[3] - self.max_energy  # 正值表示违反约束
        
        # 计算结构多样性奖励
        diversity_reward = self.calculate_structure_diversity_reward(structure_type)
        
        # 应用多样性奖励到移动距离目标
        # 稀有结构类型获得更好的移动距离评分 (减少负值)
        F[i, 0] = F[i, 0] * (1.0 - diversity_reward)
        
        # 如果启用额外的多样性目标，设置其值
        if self.add_diversity:
            if len(self.evaluated_designs) > 1:
                # 使用增强的多样性计算
                diversity_score = self.calculate_enhanced_diversity(robot_config, structure_type, i)
                # 加入结构稀有度奖励
                diversity_score = diversity_score * (1.0 + diversity_reward)
                F[i, 4] = -diversity_score  # 最大化多样性，所以取负
            else:
                F[i, 4] = 0.0  # 第一个个体的多样性得分设为0


# 新增：可视化多样性演化的函数
def visualize_diversity_evolution(generation_stats, save_path=None):
    """可视化结构多样性随代数的演化
    
    Args:
        generation_stats (list): 每代的统计信息
        save_path (str, optional): 保存路径
    """
    if not generation_stats or len(generation_stats) == 0:
        print("没有可用的代际统计数据进行多样性演化可视化")
        return
    
    import matplotlib.pyplot as plt
    
    # 提取代数和结构类型数据
    generations = []
    type_data = {'wheeled': [], 'legged': [], 'hybrid': [], 'other': []}
    
    for stats in generation_stats:
        if 'generation' in stats and 'structure_types' in stats:
            generations.append(stats['generation'])
            for type_name in type_data.keys():
                if type_name in stats['structure_types']:
                    type_data[type_name].append(stats['structure_types'][type_name])
                else:
                    type_data[type_name].append(0)
    
    if not generations:
        print("无法从代际统计数据中提取多样性信息")
        return
    
    # 计算Shannon熵作为多样性指标
    diversity_indices = []
    
    for i in range(len(generations)):
        # 获取当前代的类型计数
        counts = [type_data[type_name][i] for type_name in type_data.keys()]
        total = sum(counts)
        
        # 计算Shannon熵
        entropy = 0
        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        # 归一化熵 (除以最大可能熵)
        max_entropy = np.log2(len(type_data))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        diversity_indices.append(normalized_entropy)
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制堆叠面积图
    plt.subplot(2, 1, 1)
    bottom = np.zeros(len(generations))
    for type_name, values in type_data.items():
        plt.fill_between(generations, bottom, bottom + values, label=type_name, alpha=0.7)
        bottom += values
    
    plt.title('结构类型分布随代数的变化')
    plt.xlabel('代数')
    plt.ylabel('数量')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 绘制多样性指标变化
    plt.subplot(2, 1, 2)
    plt.plot(generations, diversity_indices, 'o-', color='green', linewidth=2)
    plt.title('结构多样性指标随代数的变化')
    plt.xlabel('代数')
    plt.ylabel('归一化Shannon熵')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"多样性演化图表已保存到: {save_path}")
    
    plt.show()


# 辅助函数 (保持不变)
def generate_simple_design(structure_type):
    """生成简单的特定结构类型设计
    
    Args:
        structure_type (str): 结构类型 ("轮式结构", "腿式结构", "混合/多样结构")
        
    Returns:
        np.ndarray: 生成的基因
    """
    if "轮式" in structure_type:
        return generate_simple_wheeled_design()
    elif "腿式" in structure_type:
        return generate_simple_legged_design()
    else:
        return generate_simple_hybrid_design()


def generate_simple_wheeled_design():
    """生成简单的轮式设计，注重稳定性和可行性，不使用硬编码"""
    gene = np.random.random(100) * 0.5  # 使用较低的随机值作为基础
    
    # 设置主要参数为稳定范围
    gene[0] = 0.3 + np.random.random() * 0.2  # 连杆数量 (~4-6)
    
    # 使用随机方式设置2-4个轮子
    num_wheels = max(2, int(np.random.random() * 3) + 2)  # 2-4个轮子
    
    # 选择连杆作为轮子
    wheel_indices = []
    for i in range(1, int(gene[0] * 10)):  # 遍历所有潜在连杆
        if len(wheel_indices) < num_wheels and np.random.random() > 0.5:
            wheel_indices.append(i)
    
    # 确保至少有2个轮子
    if len(wheel_indices) < 2:
        wheel_indices = [1, 2]
    
    # 设置轮子标记和参数
    for idx in wheel_indices:
        wheel_gene_start = 10 + idx * 8  # 假设每个连杆参数占8个基因位置
        if wheel_gene_start + 5 < 100:  # 确保不越界
            gene[wheel_gene_start] = 0.8 + np.random.random() * 0.2  # 标记为轮子 (高概率)
            
            # 设置轮子位置，确保分布均匀
            angle = (idx / len(wheel_indices)) * 2 * np.pi
            gene[wheel_gene_start + 1] = 0.5 + 0.3 * np.cos(angle)  # x 位置
            gene[wheel_gene_start + 2] = 0.5 + 0.3 * np.sin(angle)  # y 位置
            gene[wheel_gene_start + 3] = 0.1  # z 位置 (接近地面)
    
    return gene


def generate_simple_legged_design():
    """生成简单的腿式设计，注重稳定性和可行性，不使用硬编码"""
    gene = np.random.random(100) * 0.6  # 使用较低的随机值作为基础
    
    # 设置连杆数量为合理范围，腿式结构通常需要更多连杆
    gene[0] = 0.4 + np.random.random() * 0.3  # 连杆数量 (~5-8)
    
    # 设置活动关节类型
    num_legs = min(4, max(2, int(gene[0] * 10) - 3))  # 2-4条腿
    
    # 为每条腿创建2个关节
    for leg_idx in range(num_legs):
        base_idx = 10 + (leg_idx * 2 + 1) * 8  # 每条腿从主体连接两个连杆
        
        # 确保不越界
        if base_idx + 16 < 100:
            # 第一个关节 - 连接到主体
            gene[base_idx] = 0.2  # 不是轮子
            gene[base_idx + 4] = 0.6 + np.random.random() * 0.4  # 关节类型 - 偏向旋转关节
            
            # 设置关节位置，使腿均匀分布
            angle = (leg_idx / num_legs) * 2 * np.pi
            gene[base_idx + 1] = 0.5 + 0.3 * np.cos(angle)  # x 位置
            gene[base_idx + 2] = 0.5 + 0.3 * np.sin(angle)  # y 位置
            gene[base_idx + 3] = 0.2  # z 位置
            
            # 第二个关节 - 连接到第一个关节
            gene[base_idx + 8] = 0.1  # 不是轮子
            gene[base_idx + 12] = 0.7 + np.random.random() * 0.3  # 关节类型 - 更高概率为旋转关节
            
            # 设置位置 - 沿腿的方向延伸
            gene[base_idx + 9] = 0.5 + 0.5 * np.cos(angle)   # x 位置
            gene[base_idx + 10] = 0.5 + 0.5 * np.sin(angle)  # y 位置
            gene[base_idx + 11] = 0.1  # z 位置 (接近地面)
    
    return gene


def generate_simple_hybrid_design():
    """生成简单的混合结构设计，结合轮式和腿式特点"""
    # 以50%概率基于轮式或腿式设计，然后添加另一类型的特征
    if np.random.random() > 0.5:
        gene = generate_simple_wheeled_design()
        
        # 添加1-2条腿
        num_legs = 1 + int(np.random.random() > 0.5)
        
        # 找到未使用的基因段落
        used_segments = set()
        for i in range(10, 90, 8):
            if i < 100 and gene[i] > 0.7:  # 这个位置已经是轮子
                used_segments.add(i // 8)
        
        # 为腿选择未使用的段落
        available_segments = [i for i in range(1, 10) if i not in used_segments]
        if len(available_segments) >= num_legs:
            leg_segments = np.random.choice(available_segments, num_legs, replace=False)
            
            for leg_idx, segment in enumerate(leg_segments):
                base_idx = segment * 8
                if base_idx + 5 < 100:
                    gene[base_idx] = 0.2  # 不是轮子
                    gene[base_idx + 4] = 0.7  # 关节类型 - 偏向旋转关节
                    
                    # 设置关节位置
                    angle = np.pi / 2 + leg_idx * np.pi  # 在轮子之间的位置
                    gene[base_idx + 1] = 0.5 + 0.4 * np.cos(angle)  # x 位置
                    gene[base_idx + 2] = 0.5 + 0.4 * np.sin(angle)  # y 位置
                    gene[base_idx + 3] = 0.2  # z 位置
    else:
        gene = generate_simple_legged_design()
        
        # 添加1-2个轮子
        num_wheels = 1 + int(np.random.random() > 0.5)
        
        # 找到未使用的基因段落
        used_segments = set()
        for i in range(10, 90, 8):
            if i < 100 and gene[i+4] > 0.6:  # 这个位置已经是活动关节
                used_segments.add(i // 8)
        
        # 为轮子选择未使用的段落
        available_segments = [i for i in range(1, 10) if i not in used_segments]
        if len(available_segments) >= num_wheels:
            wheel_segments = np.random.choice(available_segments, num_wheels, replace=False)
            
            for wheel_idx, segment in enumerate(wheel_segments):
                base_idx = segment * 8
                if base_idx + 5 < 100:
                    gene[base_idx] = 0.9  # 是轮子
                    gene[base_idx + 4] = 0.2  # 关节类型 - 偏向固定或简单关节
                    
                    # 设置轮子位置
                    angle = wheel_idx * np.pi
                    gene[base_idx + 1] = 0.5 + 0.4 * np.cos(angle)  # x 位置
                    gene[base_idx + 2] = 0.5 + 0.4 * np.sin(angle)  # y 位置
                    gene[base_idx + 3] = 0.1  # z 位置 (接近地面)
    
    return gene


def find_best_random_design(problem, num_attempts=20):
    """生成多个随机设计并选择最佳的
    
    Args:
        problem: 问题实例，用于评估稳定性
        num_attempts: 尝试次数
        
    Returns:
        tuple: (最佳基因, 最佳分数)
    """
    best_score = -float('inf')
    best_gene = None
    
    for _ in range(num_attempts):
        random_gene = np.random.random(100)
        
        try:
            # 快速检查基本可行性
            rc = decode_gene(random_gene)
            rc = fix_prismatic_joints(rc)
            rc = fix_connection_structure(rc, verbose=False)
            
            # 计算简单可行性分数
            conn_quality, _ = check_connection_quality(rc, verbose=False)
            stab_score = problem.estimate_stability(rc)
            
            # 综合分数 (连接质量 + 稳定性)
            feasibility_score = (1 if conn_quality else 0) + stab_score
            
            if feasibility_score > best_score:
                best_score = feasibility_score
                best_gene = random_gene.copy()
                
                # 如果找到完全可行的设计，立即退出
                if conn_quality and stab_score >= problem.min_stability:
                    break
        except:
            continue
    
    return best_gene, best_score


def get_default_valid_gene():
    """获取一个已知有效的默认基因，仅在其他所有方法都失败时使用"""
    # 创建一个简单的四轮车基因
    gene = np.zeros(100)
    
    # 基本参数
    gene[0] = 0.4  # 约5个连杆
    gene[1] = 0.1  # 第一个形状参数
    
    # 主体尺寸 - 扁平宽大
    gene[2] = 0.6  # 长
    gene[3] = 0.6  # 宽
    gene[4] = 0.2  # 高
    
    # 设置4个轮子在四个角落
    wheel_positions = [(0.7, 0.7), (0.7, 0.3), (0.3, 0.7), (0.3, 0.3)]
    
    for i, (x, y) in enumerate(wheel_positions):
        base_idx = 10 + (i+1) * 8
        if base_idx + 5 < 100:
            gene[base_idx] = 0.9      # 是轮子
            gene[base_idx + 1] = x    # x位置
            gene[base_idx + 2] = y    # y位置
            gene[base_idx + 3] = 0.1  # z位置
            gene[base_idx + 4] = 0.2  # 关节类型
    
    # 随机化其余参数以增加多样性
    for i in range(50, 100):
        if i % 8 not in [0, 1, 2, 3, 4]:  # 不是关键参数
            gene[i] = np.random.random()
    
    return gene


def save_checkpoint(algorithm, problem, filename):
    """保存优化过程检查点
    
    Args:
        algorithm: 优化算法实例
        problem: 问题实例
        filename: 保存文件名
        
    Returns:
        bool: 是否成功保存
    """
    import pickle
    
    # 提取算法状态
    algorithm_state = {}
    for k, v in algorithm.__dict__.items():
        if not k.startswith("_") and not callable(v):
            try:
                # 尝试让对象可序列化
                pickle.dumps(v)
                algorithm_state[k] = v
            except:
                pass
    
    # 准备问题数据
    problem_data = {
        "evaluated_designs": problem.evaluated_designs,
        "structure_types": problem.total_structure_types if hasattr(problem, "total_structure_types") else problem.structure_types,
        "current_generation": problem.current_generation,
        "generation_stats": problem.generation_stats if hasattr(problem, "generation_stats") else []
    }
    
    # 提取种群
    initial_pop = None
    if hasattr(algorithm, "pop") and hasattr(algorithm.pop, "get"):
        try:
            initial_pop = algorithm.pop.get("X")
        except:
            pass
    
    # 准备完整检查点数据
    data = {
        "algorithm_state": algorithm_state,
        "problem_data": problem_data,
        "initial_pop": initial_pop,
        "timestamp": time.time()
    }
    
    try:
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"检查点已保存到: {filename}")
        return True
    except Exception as e:
        print(f"保存检查点失败: {str(e)}")
        return False


def load_checkpoint(filename):
    """加载优化检查点
    
    Args:
        filename: 检查点文件名
        
    Returns:
        dict: 检查点数据
    """
    import pickle
    
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(f"成功加载检查点: {filename}")
        return data
    except Exception as e:
        print(f"加载检查点失败: {str(e)}")
        return None


def visualize_pareto_front(F, save_path=None):
    """可视化Pareto前沿
    
    Args:
        F (numpy.ndarray): 目标函数值矩阵
        save_path (str, optional): 保存路径
    """
    import matplotlib.pyplot as plt
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    
    # 找到非支配解
    nds = NonDominatedSorting()
    fronts = nds.do(F)
    pareto_front = fronts[0]
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制所有解
    plt.scatter(F[:, 0], F[:, 3], s=30, facecolors='none', edgecolors='lightgray', label='所有解')
    
    # 突出显示Pareto前沿
    plt.scatter(F[pareto_front, 0], F[pareto_front, 3], s=50, c='red', label='Pareto前沿')
    
    plt.title('目标空间中的Pareto前沿')
    plt.xlabel('移动距离 (最小化)')
    plt.ylabel('能耗 (最小化)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 翻转x轴（因为我们是最小化负的距离）
    plt.gca().invert_xaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Pareto前沿图表已保存到: {save_path}")
    
    plt.show()


def visualize_constraints(G, save_path=None):
    """可视化约束违反情况
    
    Args:
        G (numpy.ndarray): 约束违反值矩阵
        save_path (str, optional): 保存路径
    """
    import matplotlib.pyplot as plt
    
    if G is None or G.size == 0:
        print("没有可用的约束数据进行可视化")
        return
    
    # 计算每个约束的违反统计
    n_constraints = G.shape[1]
    violation_counts = np.sum(G > 0, axis=0)
    violation_rates = violation_counts / G.shape[0] * 100
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制约束违反率条形图
    constraint_names = ['连接质量约束', '稳定性约束', '能耗约束']
    if len(constraint_names) > n_constraints:
        constraint_names = constraint_names[:n_constraints]
    elif len(constraint_names) < n_constraints:
        constraint_names.extend([f'约束 {i+1}' for i in range(len(constraint_names), n_constraints)])
    
    bars = plt.bar(constraint_names, violation_rates, color='skyblue')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.title('约束违反率')
    plt.ylabel('违反率 (%)')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"约束违反图表已保存到: {save_path}")
    
    plt.show()


def visualize_structure_distribution(structure_types, save_path=None):
    """可视化结构类型分布
    
    Args:
        structure_types (dict): 结构类型计数
        save_path (str, optional): 保存路径
    """
    import matplotlib.pyplot as plt
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 按结构类型名称排序
    sorted_types = sorted(structure_types.items())
    names = [t[0] for t in sorted_types]
    counts = [t[1] for t in sorted_types]
    
    # 计算百分比
    total = sum(counts)
    percentages = [count/total*100 if total > 0 else 0 for count in counts]
    
    # 绘制饼图
    plt.pie(counts, labels=names, autopct='%1.1f%%', startangle=90, 
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}, 
            textprops={'fontsize': 12})
    
    plt.axis('equal')  # 使饼图为正圆形
    plt.title('机器人结构类型分布')
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"结构类型分布图表已保存到: {save_path}")
    
    plt.show()


def save_evolved_designs(evaluated_designs, X, F, G, save_path="evolved_designs.pkl"):
    """保存进化出的设计
    
    Args:
        evaluated_designs (list): 评估过的设计列表
        X (numpy.ndarray): 决策变量矩阵
        F (numpy.ndarray): 目标函数值矩阵
        G (numpy.ndarray): 约束违反值矩阵
        save_path (str): 保存路径
    """
    import pickle
    
    data = {
        "evaluated_designs": evaluated_designs,
        "final_population": {
            "X": X,
            "F": F,
            "G": G
        },
        "timestamp": time.time()
    }
    
    try:
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"进化设计数据已保存到: {save_path}")
    except Exception as e:
        print(f"保存进化设计失败: {str(e)}")


def save_optimization_summary(problem, results, best_gene, save_path="optimization_summary.json"):
    """保存优化结果摘要
    
    Args:
        problem: 问题实例
        results: 优化结果
        best_gene: 最佳基因
        save_path: 保存路径
    """
    import json
    from datetime import datetime
    
    # 准备摘要数据
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generations": problem.current_generation,
        "population_size": len(results.X) if results.X is not None else 0,
        "structure_type_counts": problem.total_structure_types if hasattr(problem, "total_structure_types") else problem.structure_types,
        "best_gene_found": best_gene.tolist() if best_gene is not None else None,
        "optimization_parameters": {
            "add_diversity": problem.add_diversity,
            "diversity_reward_weight": problem.diversity_reward_weight if hasattr(problem, "diversity_reward_weight") else 0,
            "min_stability": problem.min_stability,
            "max_energy": problem.max_energy,
            "terrain_type": problem.terrain_type,
            "sim_time": problem.sim_time
        }
    }
    
    try:
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"优化摘要已保存到: {save_path}")
    except Exception as e:
        print(f"保存优化摘要失败: {str(e)}")


def visualize_structure_distribution(structure_types):
    """可视化结构类型分布"""
    if not structure_types:
        print("没有结构类型数据可供可视化")
        return
        
    # 提取类型和数量
    types = list(structure_types.keys())
    counts = list(structure_types.values())
    total = sum(counts)
    
    if total == 0:
        print("没有有效的结构类型数据")
        return
    
    # 计算百分比
    percentages = [count / total * 100 for count in counts]
    
    # 设置颜色
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 创建条形图
    bars = plt.bar(types, counts, color=colors)
    
    # 添加标题和标签
    plt.title('机器人结构类型分布')
    plt.xlabel('结构类型')
    plt.ylabel('数量')
    
    # 添加数值标签
    for i, (bar, percentage) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{counts[i]} ({percentage:.1f}%)',
                ha='center', va='bottom')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('structure_distribution.png')
    print("已保存结构类型分布图表到 structure_distribution.png")
    plt.close()

def enhance_urdf_generation(robot_config, structure_type):
    """增强URDF生成，添加更好的关节限制和结构特性"""
    urdf = '<?xml version="1.0"?>\n<robot name="evolved_robot">\n'
    
    # 添加材质
    urdf += '''  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>\n'''
    
    # 特殊处理腿式结构
    if structure_type == 'legged':
        # 增强腿部关节的限制和动力学参数
        for i in range(1, robot_config['num_links']):
            if not robot_config['is_wheel'][i]:
                # 找到腿部关节
                joint_type = robot_config['joint_types'][i]
                if joint_type == p.JOINT_PRISMATIC:
                    # 修改棱柱关节的限制范围，增加伸缩能力
                    robot_config['joint_limits'][i] = [-0.2, 0.2]
                elif joint_type == p.JOINT_REVOLUTE:
                    # 修改旋转关节的限制范围，增加摆动能力
                    robot_config['joint_limits'][i] = [-0.7, 0.7]
                    
                # 增强关节动力学属性
                robot_config['joint_damping'][i] = 0.5  # 适中阻尼
    
    # 使用修改后的配置生成URDF (调用原有生成函数)
    # 这里需要根据你原有的generate_urdf函数进行适配
    # 假设原有函数能够使用修改后的robot_config
    
    return robot_config

def visualize_pareto_front(F):
    """可视化优化结果的Pareto前沿"""
    # 创建3D图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制散点图，颜色按第四个目标函数值
    scatter = ax.scatter(F[:, 0], F[:, 1], F[:, 2], c=F[:, 3], cmap='viridis', s=50)
    
    # 添加标签和标题
    ax.set_xlabel('距离 (-)')
    ax.set_ylabel('路径直线性 (-)')
    ax.set_zlabel('稳定性')
    ax.set_title('机器人设计多目标优化Pareto前沿')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('能量消耗')
    
    # 保存图表
    plt.savefig('robot_pareto_front.png')
    print("已保存Pareto前沿图到 robot_pareto_front.png")
    plt.close()

def visualize_constraints(G):
    """可视化约束违反情况"""
    # 创建约束违反统计图
    constraint_names = ["轮子数量", "连接质量", "稳定性", "能耗"]
    
    # 计算违反约束的比例
    n_individuals = G.shape[0]
    violation_rates = np.sum(G > 0, axis=0) / n_individuals * 100
    mean_violations = np.mean(G, axis=0)
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 违反比例条形图
    bars1 = ax1.bar(constraint_names, violation_rates)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('违反比例 (%)')
    ax1.set_title('约束违反比例')
    
    # 为每个条形添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 平均违反程度条形图
    bars2 = ax2.bar(constraint_names, mean_violations)
    ax2.set_ylabel('平均违反值')
    ax2.set_title('约束违反程度')
    
    # 为每个条形添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('constraint_violations.png')
    print("已保存约束违反分析图表到 constraint_violations.png")
    plt.close()

def save_evolved_designs(evaluated_designs, X, F, G=None):
    """保存所有进化出的机器人设计到文件"""
    # 创建保存目录
    designs_dir = "evolved_designs"
    if not os.path.exists(designs_dir):
        os.makedirs(designs_dir)
    
    # 创建子目录用于保存URDF文件
    urdf_dir = f"{designs_dir}/urdf_files"
    if not os.path.exists(urdf_dir):
        os.makedirs(urdf_dir)
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{designs_dir}/robot_designs_{timestamp}.json"
    
    # 准备保存的数据
    designs_data = []
    
    # 安全转换为JSON可序列化类型的辅助函数
    def safe_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [safe_for_json(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: safe_for_json(value) for key, value in obj.items()}
        else:
            return obj
    
    for i in range(len(evaluated_designs)):
        design = evaluated_designs[i]
        try:
            robot_config = decode_gene(design['gene'])
            
            # 确保robot_config中有必要的键
            if 'parent_indices' not in robot_config:
                num_links = robot_config['num_links']
                robot_config['parent_indices'] = [0] * num_links
                robot_config['parent_indices'][0] = -1
                
            if 'joint_positions' not in robot_config:
                num_links = robot_config['num_links']
                robot_config['joint_positions'] = []
                robot_config['joint_positions'].append([0, 0, 0])
                
                # 为其余连杆创建环形排列位置
                for j in range(1, num_links):
                    angle = 2 * np.pi * (j / float(num_links))
                    radius = 0.2
                    if j < len(robot_config['link_sizes']):
                        radius = max(0.2, np.mean(robot_config['link_sizes'][j]) * 2)
                    pos = [radius * np.cos(angle), radius * np.sin(angle), 0.0]
                    robot_config['joint_positions'].append(pos)
            
            # 应用修复函数确保结构合理
            robot_config = fix_prismatic_joints(robot_config)
            robot_config = fix_connection_structure(robot_config, verbose=False)
            
            # 生成并保存URDF文件
            urdf_content = generate_urdf(robot_config)
            urdf_filename = f"{urdf_dir}/robot_design_{timestamp}_{i+1}.urdf"
            with open(urdf_filename, 'w', encoding='utf-8') as urdf_file:
                urdf_file.write(urdf_content)
            
            # 计算性能指标的索引
            performance_idx = -1
            for j in range(len(X)):
                if np.array_equal(X[j], design['gene']):
                    performance_idx = j
                    break
            
            # 保存设计数据
            design_data = {
                'design_id': i+1,
                'gene': safe_for_json(design['gene']),
                'structure': {
                    'num_links': int(robot_config['num_links']),
                    'num_wheels': int(sum(robot_config['is_wheel'])),
                    'body_shape': ['盒子', '圆柱', '球体'][min(int(robot_config['shapes'][0] * 3), 2)],
                    'body_size': safe_for_json(robot_config['link_sizes'][0]),
                    'joint_types': safe_for_json(robot_config['joint_types']),
                    'is_wheel': safe_for_json(robot_config['is_wheel']),
                    'shapes': safe_for_json(robot_config['shapes']),
                    'link_materials': safe_for_json(robot_config['link_materials']),
                    'has_motor': safe_for_json(robot_config['has_motor'])
                },
                'urdf_file': urdf_filename,
            }
            
            # 添加性能指标（如果找到）
            if performance_idx >= 0:
                performance = {
                    'distance': float(-F[performance_idx, 0]),
                    'path_linearity': float(-F[performance_idx, 1]),
                    'stability': float(F[performance_idx, 2]),
                    'energy': float(F[performance_idx, 3])
                }
                if F.shape[1] > 4:  # 如果有多样性指标
                    performance['diversity'] = float(-F[performance_idx, 4])
                    
                design_data['performance'] = performance
                
                # 添加约束违反值（如果有）
                if G is not None:
                    constraint_violations = {
                        'wheels_count': float(G[performance_idx, 0]),
                        'connection_quality': float(G[performance_idx, 1]),
                        'stability': float(G[performance_idx, 2]),
                        'energy': float(G[performance_idx, 3])
                    }
                    design_data['constraint_violations'] = constraint_violations
            
            designs_data.append(design_data)
        except Exception as e:
            print(f"保存设计 {i+1} 时出错: {str(e)}")
    
    # 创建汇总信息
    summary = {
        'timestamp': timestamp,
        'total_designs': len(designs_data),
        'best_design_id': int(np.argmin(F[:, 0]) + 1) if len(F) > 0 else None,
        'designs': designs_data
    }
    
    # 保存到文件
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(safe_for_json(summary), f, ensure_ascii=False, indent=2)
    
    print(f"\n已保存 {len(designs_data)} 个机器人设计到文件: {filename}")
    print(f"URDF文件已保存到目录: {urdf_dir}")

def save_best_robot_design(best_gene):
    """保存最佳机器人设计的URDF文件"""
    # 创建保存目录
    designs_dir = "evolved_designs"
    if not os.path.exists(designs_dir):
        os.makedirs(designs_dir)
        
    # 解码基因为机器人配置
    robot_config = decode_gene(best_gene)
    
    # 确保robot_config中有必要的键
    if 'parent_indices' not in robot_config:
        num_links = robot_config['num_links']
        robot_config['parent_indices'] = [0] * num_links
        robot_config['parent_indices'][0] = -1
        
    if 'joint_positions' not in robot_config:
        num_links = robot_config['num_links']
        robot_config['joint_positions'] = []
        robot_config['joint_positions'].append([0, 0, 0])
        
        # 为其余连杆创建环形排列位置
        for j in range(1, num_links):
            angle = 2 * np.pi * (j / float(num_links))
            radius = 0.2
            if j < len(robot_config['link_sizes']):
                radius = max(0.2, np.mean(robot_config['link_sizes'][j]) * 2)
            pos = [radius * np.cos(angle), radius * np.sin(angle), 0.0]
            robot_config['joint_positions'].append(pos)
    
    robot_config = fix_prismatic_joints(robot_config)
    robot_config = fix_connection_structure(robot_config, verbose=False)
    
    # 生成URDF内容
    urdf_content = generate_urdf(robot_config)
    
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存URDF文件
    urdf_filename = f"{designs_dir}/best_robot_design_{timestamp}.urdf"
    with open(urdf_filename, 'w', encoding='utf-8') as f:
        f.write(urdf_content)
    
    print(f"\n已保存最佳机器人设计的URDF文件: {urdf_filename}")
    return urdf_filename

def create_constrained_gene():
    """创建带有结构约束的随机机器人基因"""
    gene = np.zeros(100)
    
    # 1. 连杆数量约束 (4-8个连杆)
    gene[0] = 0.3 + np.random.random() * 0.7  # 确保至少有4个连杆
    
    # 2. 车身参数约束 - 保证稳定性
    # 形状 - 偏向于盒子形状(更稳定)
    gene[1] = np.random.random() * 0.4  # 80%概率是盒子
    
    # 尺寸 - 确保合理的车身比例
    gene[2] = 0.4 + np.random.random() * 0.3  # 尺寸X - 中等 (0.4-0.7)
    gene[3] = 0.4 + np.random.random() * 0.3  # 尺寸Y - 中等 (0.4-0.7)
    gene[4] = 0.1 + np.random.random() * 0.2  # 尺寸Z - 偏小 (0.1-0.3)
    
    # 材质 - 随机
    gene[5] = np.random.random()
    
    # 3. 轮子参数约束 - 确保存在轮子
    # 约束轮子数量在2-6之间
    num_wheels = np.random.randint(2, 7) 
    wheel_indices = np.random.choice(range(1, 8), size=min(num_wheels, 7), replace=False)
    
    # 初始化所有潜在连杆
    for i in range(1, 8):
        # 每个连杆的基因起始位置
        idx = 7 + (i-1) * 13
        
        # 检查是否为轮子连杆
        if i in wheel_indices:
            # 是轮子 - 设置轮子特性
            gene[idx] = 0.3 + np.random.random() * 0.2  # 关节类型 - 倾向于旋转关节
            gene[idx+1] = 0.6 + np.random.random()
            gene[idx+1] = 0.6 + np.random.random() * 0.4  # 有电机
            gene[idx+2] = 0.3 + np.random.random() * 0.4  # 形状 - 倾向于圆柱
            gene[idx+3] = 0.6 + np.random.random() * 0.4  # 是轮子标志
            gene[idx+4] = np.random.random()  # 轮子类型
            
            # 轮子尺寸 - 确保合理的比例
            gene[idx+5] = 0.3 + np.random.random() * 0.4  # 轮半径 - 适中 (0.3-0.7)
            gene[idx+6] = 0.3 + np.random.random() * 0.4  # 轮宽度 - 适中 (0.3-0.7)
            gene[idx+7] = 0.0  # 不使用
            
            # 轮子材质 - 偏向橡胶
            gene[idx+8] = 0.7 + np.random.random() * 0.3
            
            # 关节轴 - 随机选择X轴、Y轴或Z轴作为主要旋转轴
            # 增加Z轴旋转的可能性
            axis_choice = np.random.random()
            if axis_choice < 0.33:  # 33%概率使用X轴为主要旋转轴
                gene[idx+9] = 0.8 + np.random.random() * 0.2  # X轴分量(大)
                gene[idx+10] = 0.1 + np.random.random() * 0.2  # Y轴分量(小)
                gene[idx+11] = 0.1 + np.random.random() * 0.2  # Z轴分量(小)
            elif axis_choice < 0.66:  # 33%概率使用Y轴为主要旋转轴
                gene[idx+9] = 0.1 + np.random.random() * 0.2  # X轴分量(小)
                gene[idx+10] = 0.8 + np.random.random() * 0.2  # Y轴分量(大)
                gene[idx+11] = 0.1 + np.random.random() * 0.2  # Z轴分量(小)
            else:  # 33%概率使用Z轴为主要旋转轴
                gene[idx+9] = 0.1 + np.random.random() * 0.2  # X轴分量(小)
                gene[idx+10] = 0.1 + np.random.random() * 0.2  # Y轴分量(小)
                gene[idx+11] = 0.8 + np.random.random() * 0.2  # Z轴分量(大)
            
            # 关节阻尼 - 适中
            gene[idx+12] = 0.2 + np.random.random() * 0.3
        else:
            # 非轮子连杆 - 可以是支架或机械臂
            # 关节类型和电机
            gene[idx] = 0.3 + np.random.random() * 0.4  # 关节类型 - 倾向于旋转或棱柱
            gene[idx+1] = 0.3 + np.random.random() * 0.4  # 电机概率适中
            
            # 形状和尺寸
            gene[idx+2] = np.random.random()  # 形状随机
            gene[idx+3] = np.random.random() * 0.3  # 不是轮子
            
            # 限制连杆尺寸在合理范围内
            gene[idx+4] = 0.3 + np.random.random() * 0.4  # 尺寸参数
            gene[idx+5] = 0.3 + np.random.random() * 0.4  # 尺寸参数
            gene[idx+6] = 0.3 + np.random.random() * 0.4  # 尺寸参数
            
            # 材质
            gene[idx+7] = np.random.random()
            gene[idx+8] = np.random.random()
            
            # 限制非轮子连杆的位置参数，确保不会远离主体
            gene[idx+9] = 0.3 + np.random.random() * 0.4  # 接近中间值
            gene[idx+10] = 0.3 + np.random.random() * 0.4
            gene[idx+11] = 0.3 + np.random.random() * 0.4
            gene[idx+12] = 0.2 + np.random.random() * 0.4  # 适中阻尼
    
    return gene 

def print_robot_structure(robot_config):
   """打印机器人的结构信息"""
   print("\n机器人结构信息:")
   print(f"连杆数量: {robot_config['num_links']}")
   print(f"轮子数量: {sum(robot_config['is_wheel'])}")
   
   # 打印主体信息
   shape_names = ['盒子', '圆柱', '球体']
   body_shape_idx = min(int(robot_config['shapes'][0] * 3), 2)
   body_shape = shape_names[body_shape_idx]
   print(f"主体形状: {body_shape}")
   print(f"主体尺寸: {robot_config['link_sizes'][0]}")
   
   # 打印连杆信息
   print("\n连杆信息:")
   for i in range(1, robot_config['num_links']):
       joint_type = robot_config['joint_types'][i]
       joint_type_name = "未知"
       if joint_type == p.JOINT_REVOLUTE:
           joint_type_name = "旋转关节"
       elif joint_type == p.JOINT_PRISMATIC:
           joint_type_name = "棱柱关节"
       elif joint_type == p.JOINT_FIXED:
           joint_type_name = "固定关节"
       elif joint_type == p.JOINT_SPHERICAL:
           joint_type_name = "球形关节"
           
       is_wheel = "是" if robot_config['is_wheel'][i] else "否"
       shape_idx = min(int(robot_config['shapes'][i] * 3), 2)
       shape = shape_names[shape_idx]
       has_motor = "是" if robot_config['has_motor'][i] else "否"
       
       # 获取连杆的父连杆索引
       parent_idx = robot_config['parent_indices'][i] if 'parent_indices' in robot_config else 0
       parent_name = "主体" if parent_idx == 0 else f"连杆{parent_idx}"
       
       # 获取关节位置
       joint_pos = "未知"
       if 'joint_positions' in robot_config and i < len(robot_config['joint_positions']):
           pos = robot_config['joint_positions'][i]
           joint_pos = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
       
       print(f"连杆 {i}:")
       print(f"  - 连接到: {parent_name}")
       print(f"  - 关节类型: {joint_type_name}")
       print(f"  - 关节位置: {joint_pos}")
       print(f"  - 是轮子: {is_wheel}")
       print(f"  - 形状: {shape}")
       print(f"  - 尺寸: {robot_config['link_sizes'][i]}")
       print(f"  - 有电机: {has_motor}")
       
       # 如果是轮子，显示轮子的旋转轴
       if robot_config['is_wheel'][i]:
           axis = robot_config['joint_axes'][i]
           print(f"  - 轮子旋转轴: [{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}]")

def test_robot_with_gene(gene):
   """测试使用基因参数生成的机器人"""
   # 解码基因为机器人配置
   robot_config = decode_gene(gene)
   
   # 确保robot_config中有必要的键
   if 'parent_indices' not in robot_config:
       num_links = robot_config['num_links']
       robot_config['parent_indices'] = [0] * num_links
       robot_config['parent_indices'][0] = -1
       
   if 'joint_positions' not in robot_config:
       num_links = robot_config['num_links']
       robot_config['joint_positions'] = []
       robot_config['joint_positions'].append([0, 0, 0])
       
       # 为其余连杆创建环形排列位置
       for j in range(1, num_links):
           angle = 2 * np.pi * (j / float(num_links))
           radius = 0.2
           if j < len(robot_config['link_sizes']):
               radius = max(0.2, np.mean(robot_config['link_sizes'][j]) * 2)
           pos = [radius * np.cos(angle), radius * np.sin(angle), 0.0]
           robot_config['joint_positions'].append(pos)
   
   # 应用修复
   robot_config = fix_prismatic_joints(robot_config)
   robot_config = fix_connection_structure(robot_config, verbose=True)
   
   # 打印机器人配置信息
   print_robot_structure(robot_config)
   
   # 生成URDF
   urdf = generate_urdf(robot_config)
   with open("gene_robot.urdf", "w") as f:
       f.write(urdf)
   print("\n已生成基于基因的机器人URDF")  
   
   # 初始化PyBullet
   p.connect(p.GUI)
   p.resetSimulation()
   p.setAdditionalSearchPath(pybullet_data.getDataPath())
   p.setGravity(0, 0, -9.8)
   
   # 加载地面
   p.loadURDF("plane.urdf")
   
   # 加载机器人
   robot_id = p.loadURDF("gene_robot.urdf", basePosition=[0, 0, 0.1])
   
   # 设置相机
   p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0, 0, 0])
   
   # 目标点
   goal_pos = [2.0, 0, 0.1]
   visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 0.7])
   p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=goal_pos)
   
   # 控制所有轮子
   wheel_joints = []
   
   # 识别轮子关节
   for i in range(p.getNumJoints(robot_id)):
       joint_info = p.getJointInfo(robot_id, i)
       joint_name = joint_info[1].decode('utf-8')
       joint_type = joint_info[2]
       
       # 根据关节名称或类型判断是否为轮子
       is_wheel = False
       if "wheel" in joint_name.lower() or (i < len(robot_config['is_wheel']) and robot_config['is_wheel'][i]):
           is_wheel = True
       elif joint_type == p.JOINT_REVOLUTE:
           # 如果是旋转关节但没有明确标记为轮子，查看关节轴是否主要在X或Y方向
           joint_axis = p.getJointInfo(robot_id, i)[13]
           max_axis = max(abs(joint_axis[0]), abs(joint_axis[1]), abs(joint_axis[2]))
           if abs(joint_axis[0]) == max_axis or abs(joint_axis[1]) == max_axis:
               is_wheel = True
       
       if is_wheel:
           wheel_joints.append(i)
   
   print(f"\n找到 {len(wheel_joints)} 个轮子关节")
   
   # 为轮子设置控制
   for i in wheel_joints:
       p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=10.0, force=10.0)
   
   # 模拟循环
   print("\n开始模拟 - 按Ctrl+C停止")
   try:
       for _ in range(10000):  # 运行约40秒
           p.stepSimulation()
           time.sleep(1/240.0)
   except KeyboardInterrupt:
       print("\n模拟被用户中断")
   finally:
       p.disconnect()
   
   print("\n模拟完成")

if __name__ == "__main__":
    print("增强版多样化机器人结构进化")
    print("1. 使用默认参数运行约束优化的多样化结构进化")
    print("2. 设置自定义参数运行约束优化的多样化结构进化")
    print("3. 测试随机腿式机器人设计")
    print("4. 测试随机多样化机器人设计")
    print("5. 查看已保存的约束优化机器人设计")
    
    choice = input("\n请选择(1-5): ")
   
    if choice == '1':
        # 使用默认参数运行
        best_gene = run_diverse_genetic_optimization(pop_size=5, n_gen=3)
        # 测试最佳基因
        if best_gene is not None:
            test_robot_with_gene(best_gene)
            
    elif choice == '2':
        # 设置自定义参数
        pop_size = int(input("请输入种群大小 (建议3-10): "))
        n_gen = int(input("请输入进化代数 (建议2-5): "))
        print_verbose = input("是否打印详细结构信息? (y/n): ").lower() == 'y'
        pause_after_eval = input("是否在每次评估后暂停? (y/n): ").lower() == 'y'
        save_designs = input("是否保存所有进化出的设计? (y/n): ").lower() == 'y'
        
        # 运行优化
        best_gene = run_diverse_genetic_optimization(
            pop_size=pop_size, 
            n_gen=n_gen, 
            verbose=print_verbose,
            pause_after_eval=pause_after_eval,
            save_designs=save_designs
        )
        
        # 测试最佳基因
        if best_gene is not None:
            test_robot_with_gene(best_gene)
    elif choice == '3':
            
            # 测试随机腿式机器人设计
            print("\n生成随机腿式机器人设计...")
            gene = create_legged_gene()
            test_robot_with_gene(gene)

            
    elif choice == '4':
        # 测试随机多样化机器人设计
        gene = create_diverse_gene()
        test_robot_with_gene(gene)
        
    elif choice == '5':
        # 查看已保存的多样化机器人设计
        designs_dir = "evolved_designs"
        if not os.path.exists(designs_dir):
            print("错误: 没有找到保存的设计目录")
        else:
            # 列出所有JSON文件
            json_files = [f for f in os.listdir(designs_dir) if f.endswith('.json')]
            
            if not json_files:
                print("错误: 没有找到保存的设计文件")
            else:
                print("\n可用的设计文件:")
                for i, f in enumerate(json_files):
                    print(f"{i+1}. {f}")
                
                file_idx = int(input("\n请选择文件序号: ")) - 1
                if 0 <= file_idx < len(json_files):
                    # 加载设计文件
                    with open(os.path.join(designs_dir, json_files[file_idx]), 'r') as f:
                        designs = json.load(f)
                    
                    print(f"\n找到 {designs['total_designs']} 个设计")
                    
                    if designs['total_designs'] > 0:
                        design_idx = int(input(f"请选择设计序号 (1-{designs['total_designs']}): ")) - 1
                        
                        if 0 <= design_idx < designs['total_designs']:
                            design = designs['designs'][design_idx]
                            
                            # 打印设计信息
                            print("\n设计详情:")
                            print(f"设计ID: {design['design_id']}")
                            print(f"连杆数量: {design['structure']['num_links']}")
                            print(f"轮子数量: {design['structure']['num_wheels']}")
                            print(f"主体形状: {design['structure']['body_shape']}")
                            
                            if 'performance' in design:
                                print("\n性能指标:")
                                print(f"移动距离: {design['performance']['distance']:.2f}")
                                print(f"路径直线性: {design['performance']['path_linearity']:.2f}")
                                print(f"稳定性指标: {design['performance']['stability']:.2f}")
                                print(f"能量消耗: {design['performance']['energy']:.2f}")
                                if 'diversity' in design['performance']:
                                    print(f"结构多样性: {design['performance']['diversity']:.2f}")
                            
                            if 'constraint_violations' in design:
                                print("\n约束违反情况:")
                                for name, value in design['constraint_violations'].items():
                                    status = "✓ 满足" if value <= 0 else "✗ 违反"
                                    print(f"{name}: {value:.2f} - {status}")
                            
                            # 询问是否测试该设计
                            if 'urdf_file' in design and os.path.exists(design['urdf_file']):
                                if input("\n是否测试该设计? (y/n): ").lower() == 'y':
                                    # 初始化PyBullet
                                    p.connect(p.GUI)
                                    p.resetSimulation()
                                    p.setAdditionalSearchPath(pybullet_data.getDataPath())
                                    p.setGravity(0, 0, -9.8)
                                    
                                    # 加载地面
                                    p.loadURDF("plane.urdf")
                                    
                                    # 加载机器人
                                    robot_id = p.loadURDF(design['urdf_file'], basePosition=[0, 0, 0.1])
                                    
                                    # 设置相机
                                    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0, 0, 0])
                                    
                                    # 目标点
                                    goal_pos = [2.0, 0, 0.1]
                                    visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 0.7])
                                    p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=goal_pos)
                                    
                                    # 控制所有轮子关节
                                    wheel_joints = []
                                    for i in range(p.getNumJoints(robot_id)):
                                        joint_info = p.getJointInfo(robot_id, i)
                                        joint_type = joint_info[2]
                                        
                                        if joint_type == p.JOINT_REVOLUTE:
                                            wheel_joints.append(i)
                                            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=10.0, force=10.0)
                                    
                                    print(f"找到 {len(wheel_joints)} 个轮子/旋转关节")
                                    
                                    # 模拟循环
                                    print("\n开始模拟 - 按Ctrl+C停止")
                                    try:
                                        for _ in range(10000):  # 运行约40秒
                                            p.stepSimulation()
                                            time.sleep(1/240.0)
                                    except KeyboardInterrupt:
                                        print("\n模拟被用户中断")
                                    finally:
                                        p.disconnect()
                                    
                                    print("\n模拟完成")
                            else:
                                print("错误: 找不到该设计的URDF文件")
                        else:
                            print("错误: 无效的设计序号")
                else:
                    print("错误: 无效的文件序号")
    else:
        print("无效选择，使用默认参数运行约束优化的多样化结构进化")
        best_gene = run_diverse_genetic_optimization(pop_size=5, n_gen=3)
        if best_gene is not None:
            test_robot_with_gene(best_gene)