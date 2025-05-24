import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import tempfile
from robot_evolution_fixed import decode_gene, generate_urdf, simulate_robot_multi
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover as SBX
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling as LHS
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import datetime

# u521bu5efau4e00u4e2au5b8cu5168u72ecu7acbu7684u673au5668u4ebau751fu6210u811au672c

def generate_simple_urdf():
    """(u751fu6210u4e00u4e2au7b80u5355u7684u56dbu8f6eu673au5668u4ebaURDF"""
    urdf = '<?xml version="1.0"?>\n<robot name="simple_robot">\n'
    
    # u6750u8d28u5b9au4e49
    urdf += '''  <material name="metal">
    <color rgba="0.7 0.7 0.7 1.0"/>
  </material>
  <material name="wheel_material">
    <color rgba="0.2 0.2 0.2 1.0"/>
  </material>\n'''
    
    # u8f66u8eabu8fdeu6746 - u76d2u5b50u5f62
    chassis_x, chassis_y, chassis_z = 0.3, 0.2, 0.05
    urdf += f'''  <link name="base_link">
    <visual>
      <geometry><box size="{chassis_x} {chassis_y} {chassis_z}"/></geometry>
      <material name="metal"/>
    </visual>
    <collision>
      <geometry><box size="{chassis_x} {chassis_y} {chassis_z}"/></geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.017" ixy="0" ixz="0" iyy="0.017" iyz="0" izz="0.017"/>
    </inertial>
  </link>\n'''
    
    # u8f6eu5b50u53c2u6570
    wheel_radius = 0.05
    wheel_width = 0.04
    
    # u8f6eu5b50u4f4du7f6e - u56dbu4e2au89d2u843du5904
    wheel_positions = [
        [chassis_x/2 - wheel_radius/2, chassis_y/2, -chassis_z/2],  # u5de6u524du8f6e
        [chassis_x/2 - wheel_radius/2, -chassis_y/2, -chassis_z/2],  # u53f3u524du8f6e
        [-chassis_x/2 + wheel_radius/2, chassis_y/2, -chassis_z/2],   # u5de6u540eu8f6e
        [-chassis_x/2 + wheel_radius/2, -chassis_y/2, -chassis_z/2],  # u53f3u540eu8f6e
    ]
    
    # u6dfbu52a0u56dbu4e2au8f6eu5b50
    for i in range(4):
        # u8f6eu5b50u8fdeu6746
        urdf += f'''  <link name="wheel{i+1}">
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
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>\n'''
        
        # u8f6eu5b50u5173u8282 - u8f6eu5b50u9700u8981u65cbu8f6c90u5ea6u4f7fu5176u4fa7u5411u653eu7f6e
        pos = wheel_positions[i]
        urdf += f'''  <joint name="wheel_joint{i+1}" type="continuous">
    <parent link="base_link"/>
    <child link="wheel{i+1}"/>
    <origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <dynamics damping="0.01" friction="0.01"/>
  </joint>\n'''
    
    urdf += '</robot>'
    return urdf

def test_robot():
    """(u6d4bu8bd5u751fu6210u7684u673au5668u4eba"""
    # u751fu6210URDF
    urdf = generate_simple_urdf()
    with open("simple_robot.urdf", "w") as f:
        f.write(urdf)
    print("\nu5df2u751fu6210u7b80u5355u56dbu8f6eu673au5668u4ebaURDF")  
    
    # u521du59cbu5316PyBullet
    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # u52a0u8f7du5730u9762
    p.loadURDF("plane.urdf")
    
    # u52a0u8f7du673au5668u4eba
    robot_id = p.loadURDF("simple_robot.urdf", basePosition=[0, 0, 0.1])
    
    # u8bbeu7f6eu76f8u673a
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0, 0, 0])
    
    # u76eeu6807u70b9
    goal_pos = [2.0, 0, 0.1]
    visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 0.7])
    p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=goal_pos)
    
    # u63a7u5236u6240u6709u8f6eu5b50
    num_joints = p.getNumJoints(robot_id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        if "wheel" in joint_name:
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=10.0, force=10.0)
    
    # u6a21u62dfu5faau73af
    print("\nu5f00u59cbu6a21u62df - u6309Ctrl+Cu505cu6b62")
    try:
        for _ in range(10000):  # u8fd0u884cu7ea6u516c40u79d2
            p.stepSimulation()
            time.sleep(1/240.0)
    except KeyboardInterrupt:
        print("\nu6a21u62dfu88abu7528u6237u4e2du65ad")
    finally:
        p.disconnect()
    
    print("\nu6a21u62dfu5b8cu6210")

def test_robot_with_gene(gene=None):
    """测试使用基因参数生成的机器人"""
    # 如果没有提供基因，创建一个默认基因
    if gene is None:
        gene = create_default_gene()
    
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
    robot_config = fix_connection_structure(robot_config, verbose=False)
    
    # 打印机器人配置信息
    print("\n机器人配置:")
    print(f"- 连杆数量: {robot_config['num_links']}")
    print(f"- 轮子数量: {sum(robot_config['is_wheel'][1:])}")
    print(f"- 车身尺寸: {robot_config['link_sizes'][0]}")
    
    # 打印轮子旋转轴信息
    print("\n轮子旋转轴信息:")
    wheel_count = 0
    for i in range(1, robot_config['num_links']):
        if robot_config['is_wheel'][i]:
            wheel_count += 1
            axis = robot_config['joint_axes'][i]
            main_axis = "未知"
            # 确定主轴方向
            max_axis = max(abs(axis[0]), abs(axis[1]), abs(axis[2]))
            if abs(axis[0]) == max_axis:
                main_axis = "X轴"
            elif abs(axis[1]) == max_axis:
                main_axis = "Y轴" 
            elif abs(axis[2]) == max_axis:
                main_axis = "Z轴"
                
            print(f"- 轮子 {wheel_count}: 旋转轴=[{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}] (主要是{main_axis}旋转)")
    
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
    x_axis_wheels = []
    y_axis_wheels = []
    other_axis_wheels = []
    
    # 识别并分类不同轴向的轮子
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
            # 获取关节轴
            joint_axis = p.getJointInfo(robot_id, i)[13]
            # 分类轮子
            max_axis = max(abs(joint_axis[0]), abs(joint_axis[1]), abs(joint_axis[2]))
            if abs(joint_axis[0]) == max_axis:
                x_axis_wheels.append(i)
            elif abs(joint_axis[1]) == max_axis:
                y_axis_wheels.append(i)
            else:
                other_axis_wheels.append(i)
    
    print(f"\n找到 {len(wheel_joints)} 个轮子关节:")
    print(f"- X轴旋转轮: {len(x_axis_wheels)} 个")
    print(f"- Y轴旋转轮: {len(y_axis_wheels)} 个")
    print(f"- 其他轴旋转轮: {len(other_axis_wheels)} 个")
    
    # 为不同轴的轮子设置不同的控制方式
    for i in wheel_joints:
        # 所有轮子都使用相同速度，但可以根据轴向设置不同速度模式
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

def create_default_gene():
    """创建一个默认的四轮机器人基因，确保稳定性和连接合理性"""
    gene = np.zeros(100)  # 创建足够长的基因数组
    
    # 设置连杆数量为5 (车身+4轮)
    gene[0] = 0.4  # 对应5个连杆
    
    # 车身参数 - 设置为更宽扁的形状增加稳定性
    gene[1] = 0.1  # 形状 - 盒子
    gene[2] = 0.7  # 尺寸X - 较大
    gene[3] = 0.8  # 尺寸Y - 更宽
    gene[4] = 0.2  # 尺寸Z - 扁平
    gene[5] = 0.1  # 材质 - 金属
    gene[6] = 0.5  # 其他参数保持中等
    
    # 为四个轮子设置更合理的位置参数
    # 轮子位置分别在车身四个角落
    wheel_positions = [
        [0.2, 0.2, 0.0],   # 右前
        [0.2, -0.2, 0.0],  # 左前
        [-0.2, 0.2, 0.0],  # 右后
        [-0.2, -0.2, 0.0]  # 左后
    ]
    
    # 设置不同轴向的旋转轮子
    # 包含X轴、Y轴和Z轴旋转轮，以探索更多的运动方式
    wheel_axes = [
        [0.1, 0.8, 0.1],  # 右前 - Y轴为主
        [0.1, 0.8, 0.1],  # 左前 - Y轴为主
        [0.8, 0.1, 0.1],  # 右后 - X轴为主
        [0.1, 0.1, 0.8]   # 左后 - Z轴为主(全向轮效果)
    ]
    
    # 轮子的基本参数
    wheel_params = [
        # 基本轮子参数 - 使用标准设置
        0.3,    # 关节类型 - 旋转关节
        0.9,    # 有电机 - 高概率
        0.4,    # 形状 - 圆柱形
        0.9,    # 是轮子标志 - 确保识别为轮子
        0.1,    # 轮子类型 - 普通轮
        0.5,    # 轮半径 - 中等
        0.4,    # 轮宽度 - 适中
        0.0,    # 不使用
        0.8,    # 材质 - 橡胶
        0.5,    # 关节轴X - 中值 (将被覆盖)
        0.8,    # 关节轴Y - 主要分量 (将被覆盖)
        0.5,    # 关节轴Z - 中值 (将被覆盖)
        0.3     # 关节阻尼 - 较低，减少摩擦
    ]
    
    # 填充轮子参数
    for i in range(4):
        start_idx = 7 + i * 13
        
        # 复制基本参数
        gene[start_idx:start_idx+13] = wheel_params
        
        # 设置位置参数 - 根据轮子位置修改
        # X轴位置
        gene[start_idx+9] = 0.5 + wheel_positions[i][0] * 0.5  # 转换到0-1范围
        # Y轴位置
        gene[start_idx+10] = 0.5 + wheel_positions[i][1] * 0.5  # 转换到0-1范围
        # Z轴位置保持不变
        
        # 设置旋转轴 - 使用预定义的轴向
        gene[start_idx+9] = wheel_axes[i][0]  # X轴分量
        gene[start_idx+10] = wheel_axes[i][1]  # Y轴分量
        gene[start_idx+11] = wheel_axes[i][2]  # Z轴分量
    
    return gene

def create_random_gene():
    """创建一个完全随机的机器人基因"""
    gene = np.random.random(100)
    return gene

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

def test_multiple_designs(num_designs=5):
    """测试多个随机机器人设计"""
    print(f"\n将测试 {num_designs} 个随机机器人设计")
    
    for i in range(num_designs):
        print(f"\n测试设计 #{i+1}")
        # 创建一个随机基因
        gene = create_random_gene()
        # 测试该基因生成的机器人
        test_robot_with_gene(gene)
        
        # 每次测试后暂停
        if i < num_designs - 1:
            input("\n按Enter继续测试下一个设计...")

def fix_prismatic_joints(robot_config):
    """修复棱柱关节的限制问题"""
    for i in range(robot_config['num_links']):
        # 检查是否为棱柱关节(PRISMATIC)
        if robot_config['joint_types'][i] == p.JOINT_PRISMATIC:
            # 确保关节限制是有效的
            limits = robot_config['joint_limits'][i]
            if limits[0] >= limits[1] or limits[0] == 0 and limits[1] == 0:
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
    
    def __init__(self, n_var=100, use_gui=False, verbose=False, pause_after_eval=False, add_diversity=False):
        super().__init__(
            n_var=n_var,         # 基因变量数量
            n_obj=5 if add_diversity else 4,  # 增加一个多样性目标
            n_constr=0,          # 约束条件数量
            xl=np.zeros(n_var),  # 基因下限
            xu=np.ones(n_var)    # 基因上限
        )
        self.use_gui = use_gui
        self.verbose = verbose
        self.pause_after_eval = pause_after_eval
        self.add_diversity = add_diversity
        self.evaluated_designs = []  # 记录已评估的设计，用于计算多样性
        
    def _evaluate(self, X, out, *args, **kwargs):
        """评估机器人设计的适应度"""
        n_individuals = X.shape[0]
        F = np.zeros((n_individuals, self.n_obj))
        
        for i in range(n_individuals):
            gene = X[i, :]
            print(f"\n评估个体 {i+1}/{n_individuals}")
            
            # 解码基因为机器人配置
            robot_config = decode_gene(gene)
            
            # 检查并修复棱柱关节的限制
            robot_config = fix_prismatic_joints(robot_config)
            
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
            
            # 修复零件连接问题，确保所有零件都合理连接到主体
            robot_config = fix_connection_structure(robot_config, verbose=self.verbose)
            
            # 检查修复后的连接质量，如果仍有问题，给予最差评分
            connection_ok, connection_issues = check_connection_quality(robot_config, verbose=self.verbose)
            
            if not connection_ok:
                print(f"个体 {i+1} 连接问题: {connection_issues}")
                # 给予最差评分
                F[i, 0] = 0.0    # 距离为0
                F[i, 1] = 0.0    # 直线性为0
                F[i, 2] = 3.14   # 最大稳定性问题
                F[i, 3] = 1000   # 最大能耗
                if self.add_diversity:
                    F[i, 4] = 0.0  # 多样性得分为0
                    
                # 跳过模拟
                print(f"跳过连接质量不合格的个体 {i+1} 的模拟")
                continue
            
            try:
                # 模拟机器人并获取性能指标
                metrics = simulate_robot_multi(
                    robot_config, 
                    gui=self.use_gui,  # 对每个个体都可以显示GUI
                    sim_time=10.0
                )
                
                # 记录性能指标
                F[i, 0] = -metrics[0]  # 距离 (最大化，所以取负)
                F[i, 1] = -metrics[1]  # 路径直线性 (最大化，所以取负)
                F[i, 2] = metrics[2]   # 稳定性 (最小化)
                F[i, 3] = metrics[3]   # 能量消耗 (最小化)
                
                # 添加结构信息 - 用于多样性计算
                self.evaluated_designs.append({
                    'gene': gene.copy(),
                    'config': {
                        'num_links': robot_config['num_links'],
                        'num_wheels': sum(robot_config['is_wheel']),
                        'shape_type': robot_config['shapes'][0],
                        'joint_types': robot_config['joint_types'].copy()
                    }
                })
                
                # 如果启用多样性目标，计算多样性得分
                if self.add_diversity and len(self.evaluated_designs) > 1:
                    # 计算与之前设计的结构差异度
                    diversity_score = self.calculate_diversity(robot_config, i)
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
            
            # 如果启用了暂停，让用户决定何时继续
            if self.use_gui and self.pause_after_eval:
                input("按Enter键继续评估下一个个体...")
        
        out["F"] = F
    
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

def run_genetic_optimization(pop_size=10, n_gen=5, use_gui=True, use_constraints=True, verbose=False, pause_after_eval=False, diverse_mode=False, save_designs=False):
    """运行遗传算法优化机器人设计"""
    print("\n开始遗传算法优化机器人设计...")
    print(f"种群大小: {pop_size}, 进化代数: {n_gen}")
    print(f"使用结构约束: {'是' if use_constraints else '否'}")
    print(f"增加结构多样性: {'是' if diverse_mode else '否'}")
    print(f"显示模拟可视化: {'是' if use_gui else '否'}")
    print(f"打印详细结构信息: {'是' if verbose else '否'}")
    print(f"每次评估后暂停: {'是' if pause_after_eval else '否'}")
    print(f"保存机器人设计: {'是' if save_designs else '否'}")
    print(f"启用零件连接修复: 是")  # 现在始终启用
    print(f"启用有问题设计过滤: 是")  # 新增
    
    try:
        # 定义问题
        problem = RobotDesignProblem(n_var=100, use_gui=use_gui, verbose=verbose, 
                                    pause_after_eval=pause_after_eval, add_diversity=diverse_mode)
        
        # 设置初始种群采样方法
        if use_constraints:
            # 创建带有约束的初始种群
            initial_pop = np.zeros((pop_size, 100))
            
            # 为每个个体生成基因
            for i in range(pop_size):
                # 最多尝试10次生成合格的设计
                max_attempts = 10
                design_ok = False
                
                for attempt in range(max_attempts):
                    if diverse_mode:
                        # 在多样性模式下，增加多样化基因生成的比例到70%
                        if np.random.random() > 0.3:  # 70%概率使用多样化基因
                            gene = create_diverse_gene()
                        else:
                            gene = create_constrained_gene()
                    else:
                        gene = create_constrained_gene()
                    
                    # 检查设计是否合格
                    robot_config = decode_gene(gene)
                    robot_config = fix_prismatic_joints(robot_config)
                    
                    # 确保robot_config中有必要的键
                    if 'parent_indices' not in robot_config:
                        num_links = robot_config['num_links']
                        robot_config['parent_indices'] = [0] * num_links
                        robot_config['parent_indices'][0] = -1
                        
                    if 'joint_positions' not in robot_config:
                        # 创建默认的关节位置
                        num_links = robot_config['num_links']
                        robot_config['joint_positions'] = []
                        # 主体关节位置为原点
                        robot_config['joint_positions'].append([0, 0, 0])
                        
                        # 为其余连杆创建更多样的3D位置分布
                        for j in range(1, num_links):
                            # 随机确定是使用球面分布还是平面分布
                            use_3d = np.random.random() > 0.3  # 70%概率使用3D位置
                            
                            if use_3d:
                                # 3D球面分布 - 允许Z轴有明显变化
                                # 随机角度
                                theta = np.random.random() * 2 * np.pi  # 水平角
                                phi = np.random.random() * np.pi - np.pi/2  # 垂直角
                                
                                # 根据连杆尺寸确定距离，保持在0.05米到0.5米范围内
                                radius = 0.2  # 默认值
                                if j < len(robot_config['link_sizes']):
                                    # 根据主体和连杆尺寸设置合理距离
                                    body_size = max(0.05, np.mean(robot_config['link_sizes'][0]))
                                    link_size = max(0.05, np.mean(robot_config['link_sizes'][j]))
                                    # 计算距离，确保在合理范围内
                                    radius = min(0.4, max(0.05 + body_size/2 + link_size/2, 0.15))
                                
                                # 球坐标转笛卡尔坐标
                                x = radius * np.cos(phi) * np.cos(theta)
                                y = radius * np.cos(phi) * np.sin(theta)
                                z = radius * np.sin(phi)
                                pos = [x, y, z]
                            else:
                                # 平面分布 - 传统环形排列
                                angle = 2 * np.pi * (j / float(num_links))
                                radius = 0.2
                                if j < len(robot_config['link_sizes']):
                                    body_size = max(0.05, np.mean(robot_config['link_sizes'][0]))
                                    link_size = max(0.05, np.mean(robot_config['link_sizes'][j]))
                                    radius = min(0.4, max(0.05 + body_size + link_size, body_size + 2 * link_size))
                                
                                pos = [radius * np.cos(angle), radius * np.sin(angle), 0.0]
                                
                            robot_config['joint_positions'].append(pos)
                    
                    # 应用连接修复
                    robot_config = fix_connection_structure(robot_config, verbose=False)
                    
                    # 检查修复后的连接质量
                    connection_ok, issues = check_connection_quality(robot_config, verbose=False)
                    
                    if connection_ok:
                        # 设计合格，保存并退出尝试循环
                        design_ok = True
                        initial_pop[i] = gene
                        if verbose:
                            print(f"个体 {i+1} 连接检查通过，尝试次数: {attempt+1}")
                        break
                    elif verbose and attempt == max_attempts - 1:
                        print(f"个体 {i+1} 连接问题未解决，尝试次数: {attempt+1}")
                        print(f"最后的问题: {issues}")
                
                # 如果无法生成合格设计，使用最基本的设计
                if not design_ok:
                    print(f"警告: 个体 {i+1} 无法生成合格设计，使用基本设计替代")
                    # 使用基本的四轮设计
                    initial_pop[i] = create_default_gene()
            
            # 使用自定义初始种群
            from pymoo.core.sampling import Sampling
            
            class CustomSampling(Sampling):
                def __init__(self, initial_pop):
                    super().__init__()
                    self.initial_pop = initial_pop
                    
                def _do(self, problem, n_samples, **kwargs):
                    return self.initial_pop
                    
            sampling = CustomSampling(initial_pop)
        else:
            # 使用标准拉丁超立方采样
            sampling = LHS()
        
        # 修改交叉算子，增加多样性
        if diverse_mode:
            crossover = SBX(prob=0.9, eta=10)  # 降低eta增加多样性
            mutation = PM(prob=0.2, eta=15)    # 增加变异概率
        else:
            crossover = SBX(prob=0.9, eta=15)
            mutation = PM(eta=20)
        
        # 设置NSGA-II算法
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
        )
        
        # 记录开始时间
        start_time = time.time()
        
        # 运行优化
        results = minimize(
            problem,
            algorithm,
            ('n_gen', n_gen),
            verbose=True,
            save_history=True
        )
        
        # 记录结束时间
        end_time = time.time()
        print(f"优化完成! 耗时: {end_time - start_time:.2f} 秒")
        
        # 获取结果
        X = results.X  # 决策变量
        F = results.F  # 目标函数值
        
        # 可视化Pareto前沿
        visualize_pareto_front(F)
        
        # 找到并测试最佳设计
        best_idx = np.argmin(F[:, 0])  # 最大距离(F是负的)
        best_gene = X[best_idx]
        
        print("\n测试最佳设计...")
        robot_config = decode_gene(best_gene)
        robot_config = fix_prismatic_joints(robot_config)
        robot_config = fix_connection_structure(robot_config)
        print_robot_structure(robot_config)
        
        # 如果启用了保存设计，将所有进化出的设计保存到文件
        if save_designs:
            save_evolved_designs(problem.evaluated_designs, X, F)
        
        try:
            test_robot_with_gene(best_gene)
            # 保存最佳设计的URDF文件
            best_urdf_file = save_best_robot_design(best_gene)
        except Exception as e:
            print(f"测试最佳设计时出错: {str(e)}")
            print("尝试修复问题后重新测试...")
            try:
                # 尝试修复基因问题
                fixed_gene = create_constrained_gene()
                # 复制最佳设计的部分特征
                fixed_gene[0] = best_gene[0]  # 连杆数量
                fixed_gene[1:6] = best_gene[1:6]  # 车身参数
                test_robot_with_gene(fixed_gene)
                # 保存修复后的设计
                save_best_robot_design(fixed_gene)
            except Exception as e2:
                print(f"修复后测试仍然失败: {str(e2)}")
                print("请尝试使用默认基因测试。")
        
        return best_gene
    except Exception as e:
        print(f"遗传算法优化过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

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

def save_evolved_designs(evaluated_designs, X, F):
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
                    'body_shape': ['盒子', '圆柱', '球体'][min(int(robot_config['shapes'][0]), 2)],
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

def load_and_test_robot_from_urdf(urdf_file):
    """从保存的URDF文件加载并测试机器人"""
    print(f"\n加载并测试URDF文件: {urdf_file}")
    
    # 检查文件是否存在
    if not os.path.exists(urdf_file):
        print(f"错误: URDF文件不存在: {urdf_file}")
        return
        
    # 初始化PyBullet
    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # 加载地面
    p.loadURDF("plane.urdf")
    
    # 加载机器人
    try:
        robot_id = p.loadURDF(urdf_file, basePosition=[0, 0, 0.1])
    except Exception as e:
        print(f"加载URDF文件失败: {str(e)}")
        p.disconnect()
        return
    
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
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        
        # 根据关节名称或类型判断是否为轮子
        if "wheel" in joint_name.lower() or joint_type == p.JOINT_REVOLUTE:
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

if __name__ == "__main__":
    print("机器人设计测试工具")
    print("1. 测试默认四轮机器人")
    print("2. 测试随机机器人设计")
    print("3. 测试多个随机设计")
    print("4. 使用遗传算法优化机器人设计")
    print("5. 使用结构约束的遗传算法优化")
    print("6. 测试带可视化的遗传算法优化")
    print("7. 测试多样化结构进化")
    print("8. 加载并测试URDF文件")
    print("9. 修复现有机器人URDF文件")
    print("10. 使用仅Z轴轮子的遗传算法优化")
    
    choice = input("\n请选择(1-10): ")
    
    if choice == '1':
        test_robot_with_gene()
    elif choice == '2':
        random_gene = create_random_gene()
        test_robot_with_gene(random_gene)
    elif choice == '3':
        num = int(input("请输入要测试的设计数量: "))
        test_multiple_designs(num)
    elif choice == '4':
        pop_size = int(input("请输入种群大小 (建议5-20): "))
        n_gen = int(input("请输入进化代数 (建议3-10): "))
        best_gene = run_genetic_optimization(pop_size, n_gen, use_gui=False, use_constraints=False)
    elif choice == '5':
        pop_size = int(input("请输入种群大小 (建议5-20): "))
        n_gen = int(input("请输入进化代数 (建议3-10): "))
        best_gene = run_genetic_optimization(pop_size, n_gen, use_gui=False, use_constraints=True)
    elif choice == '6':
        pop_size = int(input("请输入种群大小 (建议3-10): "))
        n_gen = int(input("请输入进化代数 (建议2-5): "))
        print_verbose = input("是否打印详细结构信息? (y/n): ").lower() == 'y'
        pause_after_eval = input("是否在每次评估后暂停? (y/n): ").lower() == 'y'
        best_gene = run_genetic_optimization(pop_size, n_gen, use_gui=True, 
                                           use_constraints=True, verbose=print_verbose,
                                           pause_after_eval=pause_after_eval)
    elif choice == '7':
        pop_size = int(input("请输入种群大小 (建议3-10): "))
        n_gen = int(input("请输入进化代数 (建议2-5): "))
        print_verbose = input("是否打印详细结构信息? (y/n): ").lower() == 'y'
        pause_after_eval = input("是否在每次评估后暂停? (y/n): ").lower() == 'y'
        save_designs = input("是否保存所有进化出的设计? (y/n): ").lower() == 'y'
        best_gene = run_genetic_optimization(pop_size, n_gen, use_gui=True, 
                                           use_constraints=True, verbose=print_verbose,
                                           pause_after_eval=pause_after_eval,
                                           diverse_mode=True, save_designs=save_designs)
    elif choice == '8':
        urdf_file = input("请输入URDF文件路径: ")
        load_and_test_robot_from_urdf(urdf_file)
    elif choice == '9':
        urdf_file = input("请输入要修复的URDF文件路径: ")
        if not os.path.exists(urdf_file):
            print(f"错误: 文件不存在: {urdf_file}")
        else:
            # 生成一个默认基因
            default_gene = create_constrained_gene()
            # 解码基因，修复连接问题
            robot_config = decode_gene(default_gene)
            robot_config = fix_prismatic_joints(robot_config)
            
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
            
            robot_config = fix_connection_structure(robot_config, verbose=True)
            
            # 生成修复后的URDF
            fixed_urdf = generate_urdf(robot_config)
            
            # 保存修复后的URDF
            fixed_file = urdf_file.replace('.urdf', '_fixed.urdf')
            with open(fixed_file, 'w', encoding='utf-8') as f:
                f.write(fixed_urdf)
                
            print(f"已生成修复后的URDF文件: {fixed_file}")
            
            # 询问是否测试修复后的文件
            if input("是否测试修复后的URDF文件? (y/n): ").lower() == 'y':
                load_and_test_robot_from_urdf(fixed_file)
    elif choice == '10':
        # 运行仅Z轴轮子的遗传算法优化
        try:
            from z_axis_wheel_override import run_z_axis_genetic_optimization
            run_z_axis_genetic_optimization()
        except ImportError:
            print("错误: 找不到z_axis_wheel_override模块，请确保您已创建该文件。")
    else:
        print("无效选择，使用默认设计")
        test_robot_with_gene() 