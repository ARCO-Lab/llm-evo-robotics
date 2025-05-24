import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import tempfile
from robot_evolution_fixed import generate_urdf as original_generate_urdf

def custom_decode_gene(x, max_links=8):
    """自定义基因解码函数，允许轮子以Z轴为主要旋转轴"""
    # 连杆数量
    num_links = max(4, min(int(x[0] * max_links) + 1, max_links))
    
    # 初始化数组
    joint_types, has_motor, shapes = [], [], []
    is_wheel, wheel_types, joint_axes = [], [], []
    link_sizes, link_materials = [], []
    joint_limits, joint_damping = [], []
    
    # 车身/底板 (总是第一个连杆)
    joint_types.append(p.JOINT_FIXED)  # 第一个关节总是固定的
    has_motor.append(False)
    
    # 从基因中解码车身参数
    base_idx = 1  # 基因中的起始索引
    
    # 车身形状: x[1]
    shape_val = x[base_idx]
    shapes.append(int(shape_val * 3)) 
    base_idx += 1
    
    is_wheel.append(False)  # 车身不是轮子
    wheel_types.append(0)
    
    # 车身尺寸: x[2], x[3], x[4]
    size_x = 0.1 + x[base_idx] * 0.4
    size_y = 0.1 + x[base_idx+1] * 0.4
    size_z = 0.05 + x[base_idx+2] * 0.2
    link_sizes.append([size_x, size_y, size_z])
    base_idx += 3
    
    # 车身材质: x[5]
    link_materials.append(int(x[base_idx] * 3))
    base_idx += 1
    
    joint_axes.append([0, 0, 1])  # 车身的固定关节方向
    joint_limits.append([0, 0])    # 固定关节没有限制
    joint_damping.append(0.1)
    
    # 创建其余连杆 (轮子或机械臂段)
    for i in range(1, num_links):
        # 为每个连杆预留13个基因参数
        gene_start = 1 + 6 + (i-1) * 13
        
        # 防止索引越界
        if gene_start + 12 >= len(x):
            # 如果剩余基因不足，就使用默认值
            joint_types.append(p.JOINT_REVOLUTE)
            has_motor.append(True)
            shapes.append(1)  # 圆柱形
            is_wheel.append(True)
            wheel_types.append(0)  # 普通轮
            joint_axes.append([0, 0, 1])  # Z轴 (修改为Z轴旋转)
            link_sizes.append([0.06, 0.06, 0.04])  # 默认轮子尺寸
            link_materials.append(2)  # 橡胶
            joint_limits.append([-3.14, 3.14])
            joint_damping.append(0.1)
            continue
        
        # 关节类型
        joint_type_val = x[gene_start]
        if joint_type_val < 0.25:
            joint_types.append(p.JOINT_FIXED)
        elif joint_type_val < 0.5:
            joint_types.append(p.JOINT_REVOLUTE)
        elif joint_type_val < 0.75:
            joint_types.append(p.JOINT_PRISMATIC)
        else:
            joint_types.append(p.JOINT_SPHERICAL)
        gene_start += 1
        
        # 是否有电机
        has_motor.append(x[gene_start] > 0.5)
        gene_start += 1
        
        # 连杆形状
        shape_val = x[gene_start]
        shapes.append(int(shape_val * 3))
        gene_start += 1
        
        # 是否是轮子
        is_wheel_val = x[gene_start] > 0.5
        is_wheel.append(is_wheel_val)
        gene_start += 1
        
        # 轮子类型 (如果是轮子)
        wheel_types.append(int(x[gene_start] > 0.5))
        gene_start += 1
        
        # 连杆尺寸
        if is_wheel_val:
            # 轮子尺寸
            wheel_radius = 0.03 + x[gene_start] * 0.07
            wheel_width = 0.02 + x[gene_start+1] * 0.06
            link_sizes.append([wheel_radius, wheel_radius, wheel_width])
        else:
            # 普通连杆尺寸
            size_x = 0.05 + x[gene_start] * 0.2
            size_y = 0.05 + x[gene_start+1] * 0.2
            size_z = 0.05 + x[gene_start+2] * 0.2
            link_sizes.append([size_x, size_y, size_z])
        gene_start += 3
        
        # 连杆材质
        link_materials.append(int(x[gene_start] * 3))
        gene_start += 1
        
        # 关节轴向 - 重要的修改：允许轮子使用指定的旋转轴
        if gene_start + 2 < len(x):
            # 读取基因中的轴向设置
            axis_x = x[gene_start]
            axis_y = x[gene_start+1]
            axis_z = x[gene_start+2]
            
            # 如果是轮子，并且指定了Z轴为主要旋转轴
            if is_wheel_val and axis_z > 0.8:
                # 强制使用纯Z轴旋转
                joint_axes.append([0, 0, 1])
            else:
                # 其他情况，使用标准化的向量
                norm = np.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
                if norm < 0.001:
                    # 防止零向量
                    if is_wheel_val:
                        joint_axes.append([0, 0, 1])  # 默认使用Z轴
                    else:
                        joint_axes.append([0, 1, 0])  # 非轮子默认使用Y轴
                else:
                    joint_axes.append([axis_x/norm, axis_y/norm, axis_z/norm])
        else:
            # 如果基因不足，使用默认轴向
            joint_axes.append([0, 0, 1] if is_wheel_val else [0, 1, 0])
        
        # 关节限制
        if not is_wheel_val and joint_types[-1] != p.JOINT_FIXED:
            if gene_start + 4 < len(x):
                limit_lower = -3.14 * x[gene_start+3]
                limit_upper = 3.14 * x[gene_start+4]
                joint_limits.append([limit_lower, limit_upper])
            else:
                joint_limits.append([-3.14, 3.14])
        else:
            joint_limits.append([-3.14, 3.14])
        
        # 关节阻尼
        if gene_start + 5 < len(x):
            damping = 0.1 + x[gene_start+5] * 0.9
            joint_damping.append(damping)
        else:
            joint_damping.append(0.1)
    
    # 确保至少有4个连杆以形成稳定的机器人
    while len(joint_types) < 4:
        # 添加默认轮子（使用Z轴旋转）
        joint_types.append(p.JOINT_REVOLUTE)
        has_motor.append(True)
        shapes.append(1)  # 圆柱形
        is_wheel.append(True)
        wheel_types.append(0)  # 普通轮
        joint_axes.append([0, 0, 1])  # Z轴旋转
        link_sizes.append([0.06, 0.06, 0.04])  # 默认轮子尺寸
        link_materials.append(2)  # 橡胶
        joint_limits.append([-3.14, 3.14])
        joint_damping.append(0.1)
    
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

def custom_generate_urdf(gene):
    """自定义URDF生成函数，确保轮子旋转轴正确设置"""
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
    
    # 基础连杆 (车身)
    base_size = link_sizes[0]
    base_material = material_names[link_materials[0]]
    
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
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>\n'''
    elif shapes[0] == 1:  # 圆柱
        urdf += f'''  <link name="base_link">
    <visual>
      <geometry><cylinder radius="{base_size[0]/2}" length="{base_size[2]}"/></geometry>
      <material name="{base_material}"/>
    </visual>
    <collision>
      <geometry><cylinder radius="{base_size[0]/2}" length="{base_size[2]}"/></geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>\n'''
    else:  # 球体
        urdf += f'''  <link name="base_link">
    <visual>
      <geometry><sphere radius="{base_size[0]/2}"/></geometry>
      <material name="{base_material}"/>
    </visual>
    <collision>
      <geometry><sphere radius="{base_size[0]/2}"/></geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>\n'''
    
    # 添加其他连杆
    for i in range(1, num_links):
        link_name = f"link{i}"
        link_size = link_sizes[i]
        link_material = material_names[link_materials[i]] if link_materials[i] < len(material_names) else "rubber"
        if is_wheel[i]:
            link_material = "wheel_material"
            
        # 形状类型
        if shapes[i] == 0:  # 盒子
            urdf += f'''  <link name="{link_name}">
    <visual>
      <geometry><box size="{link_size[0]} {link_size[1]} {link_size[2]}"/></geometry>
      <material name="{link_material}"/>
    </visual>
    <collision>
      <geometry><box size="{link_size[0]} {link_size[1]} {link_size[2]}"/></geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>\n'''
        elif shapes[i] == 1:  # 圆柱
            urdf += f'''  <link name="{link_name}">
    <visual>
      <geometry><cylinder radius="{link_size[0]}" length="{link_size[2]}"/></geometry>
      <material name="{link_material}"/>
    </visual>
    <collision>
      <geometry><cylinder radius="{link_size[0]}" length="{link_size[2]}"/></geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>\n'''
        else:  # 球体
            radius = min(link_size) / 2
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
            joint_type_str = "floating"
        
        # 关节位置和方向
        pos = [0.1 * i, 0, 0]  # 简单线性排列
        joint_axis = joint_axes[i]
        axis_str = f"{joint_axis[0]} {joint_axis[1]} {joint_axis[2]}"
        
        # 对于轮子，设置合适的方向
        rpy = "0 0 0"
        if is_wheel[i]:
            # 对于Z轴旋转轮，保持默认方向
            if abs(joint_axis[2]) > 0.9:  # 确认是Z轴轮子
                rpy = "0 0 0"  # 不需要特殊旋转
            else:
                # 其他轴的轮子使用传统设置
                rpy = "1.5708 0 0"  # 90度旋转，使轮子垂直于地面
        
        # 创建关节
        urdf += f'''  <joint name="joint{i}" type="{joint_type_str}">
    <parent link="base_link"/>
    <child link="{link_name}"/>
    <origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="{rpy}"/>
    <axis xyz="{axis_str}"/>'''
        
        # 添加关节限制和阻尼
        if joint_type_str in ["revolute", "prismatic"]:
            lower, upper = joint_limits[i]
            urdf += f'''
    <limit lower="{lower}" upper="{upper}" effort="10" velocity="10"/>'''
        
        urdf += f'''
    <dynamics damping="{joint_damping[i]}" friction="0.1"/>
  </joint>\n'''
    
    urdf += '</robot>'
    return urdf

def create_z_axis_wheels_gene():
    """创建一个所有轮子都绕Z轴旋转的机器人基因"""
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
    
    # 为每个轮子参数设置首个位置的索引
    wheel_indices = [7, 20, 33, 46]
    
    # 对每个轮子进行单独设置
    for i, idx in enumerate(wheel_indices):
        # 基本轮子参数
        gene[idx] = 0.35    # 关节类型 - 旋转关节
        gene[idx+1] = 0.9   # 有电机 - 高概率
        gene[idx+2] = 0.4   # 形状 - 圆柱形
        gene[idx+3] = 0.9   # 是轮子标志 - 确保识别为轮子
        gene[idx+4] = 0.1   # 轮子类型 - 普通轮
        gene[idx+5] = 0.5   # 轮半径 - 中等
        gene[idx+6] = 0.4   # 轮宽度 - 适中
        gene[idx+7] = 0.0   # 不使用
        gene[idx+8] = 0.8   # 材质 - 橡胶
        
        # 轮子位置 - 转换到0-1范围
        pos = wheel_positions[i]
        gene[idx+9] = 0.5 + pos[0] * 0.5   # X轴位置
        gene[idx+10] = 0.5 + pos[1] * 0.5  # Y轴位置
        
        # 关键部分：设置旋转轴 - 强制使用Z轴为主要旋转轴
        gene[idx+9] = 0.0     # X轴分量 - 设为0
        gene[idx+10] = 0.0    # Y轴分量 - 设为0
        gene[idx+11] = 1.0    # Z轴分量 - 设为最大，确保为纯Z轴旋转
        
        gene[idx+12] = 0.3    # 关节阻尼 - 较低，减少摩擦
    
    return gene

def test_robot_with_gene(gene=None):
    """测试使用基因参数生成的机器人"""
    # 如果没有提供基因，创建一个默认基因
    if gene is None:
        gene = create_z_axis_wheels_gene()
    
    # 使用自定义函数解码基因为机器人配置
    robot_config = custom_decode_gene(gene)
    
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
    
    # 生成URDF - 使用自定义URDF生成函数
    urdf = custom_generate_urdf(robot_config)
    with open("z_axis_wheel_robot.urdf", "w") as f:
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
    robot_id = p.loadURDF("z_axis_wheel_robot.urdf", basePosition=[0, 0, 0.1])
    
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
    z_axis_wheels = []
    
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
            # 如果是旋转关节但没有明确标记为轮子，通过轴向判断
            joint_axis = p.getJointInfo(robot_id, i)[13]
            max_axis = max(abs(joint_axis[0]), abs(joint_axis[1]), abs(joint_axis[2]))
            if max_axis > 0.01:  # 确保有明确的轴向
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
            elif abs(joint_axis[2]) == max_axis:
                z_axis_wheels.append(i)
    
    print(f"\n找到 {len(wheel_joints)} 个轮子关节:")
    print(f"- X轴旋转轮: {len(x_axis_wheels)} 个")
    print(f"- Y轴旋转轮: {len(y_axis_wheels)} 个")
    print(f"- Z轴旋转轮: {len(z_axis_wheels)} 个")
    
    # 为不同轴的轮子设置不同的控制方式
    for i in wheel_joints:
        # 所有轮子都使用相同速度
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
    print("创建并测试Z轴旋转轮子的机器人设计")
    z_axis_gene = create_z_axis_wheels_gene()
    test_robot_with_gene(z_axis_gene) 