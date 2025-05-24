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

# --- 基因解码函数 ---
def decode_gene(x, max_links=8):
    num_links = max(4, int(x[0] * max_links) + 1)
    joint_types, has_motor, shapes = [], [], []
    is_wheel, wheel_types, joint_axes = [], [], []
    link_sizes, link_materials = [], []  # 新增：连杆尺寸和材质
    joint_limits, joint_damping = [], []  # 新增：关节限制和阻尼
    
    # 简化机器人结构，创建一个简单的车轮式机器人
    
    # 第一个连杆是主体
    joint_types.append(p.JOINT_FIXED)
    has_motor.append(False)
    shapes.append(0)  # 盒子形状
    is_wheel.append(False)
    wheel_types.append(0)
    joint_axes.append([0, 0, 1])
    link_sizes.append([0.3, 0.2, 0.05])  # 扁平的盒子
    link_materials.append(0)  # 金属材质
    joint_limits.append([0, 0])
    joint_damping.append(0.5)
    
    # 第二个连杆是左前轮
    joint_types.append(p.JOINT_REVOLUTE)
    has_motor.append(True)
    shapes.append(1)  # 圆柱形状
    is_wheel.append(True)
    wheel_types.append(0)
    joint_axes.append([0, 1, 0])
    link_sizes.append([0.1, 0.1, 0.05])
    link_materials.append(2)  # 橡胶材质
    joint_limits.append([-3.14, 3.14])
    joint_damping.append(0.1)
    
    # 第三个连杆是右前轮
    joint_types.append(p.JOINT_REVOLUTE)
    has_motor.append(True)
    shapes.append(1)  # 圆柱形状
    is_wheel.append(True)
    wheel_types.append(0)
    joint_axes.append([0, 1, 0])
    link_sizes.append([0.1, 0.1, 0.05])
    link_materials.append(2)  # 橡胶材质
    joint_limits.append([-3.14, 3.14])
    joint_damping.append(0.1)
    
    # 第四个连杆是左后轮
    joint_types.append(p.JOINT_REVOLUTE)
    has_motor.append(True)
    shapes.append(1)  # 圆柱形状
    is_wheel.append(True)
    wheel_types.append(0)
    joint_axes.append([0, 1, 0])
    link_sizes.append([0.1, 0.1, 0.05])
    link_materials.append(2)  # 橡胶材质
    joint_limits.append([-3.14, 3.14])
    joint_damping.append(0.1)
    
    # 第五个连杆是右后轮
    joint_types.append(p.JOINT_REVOLUTE)
    has_motor.append(True)
    shapes.append(1)  # 圆柱形状
    is_wheel.append(True)
    wheel_types.append(0)
    joint_axes.append([0, 1, 0])
    link_sizes.append([0.1, 0.1, 0.05])
    link_materials.append(2)  # 橡胶材质
    joint_limits.append([-3.14, 3.14])
    joint_damping.append(0.1)
    
    return {
        'num_links': 5,  # 固定为5个连杆
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

# --- URDF 构造函数 ---
def generate_urdf(gene):
    urdf = '<?xml version="1.0"?>\n<robot name="robot">\n'
    
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
  <material name="joint_material">
    <color rgba="1.0 0.0 0.0 1.0"/>
  </material>
  <material name="wheel_material">
    <color rgba="0.2 0.2 0.2 1.0"/>
  </material>\n'''
    
    # 基础连杆 (车身)
    chassis_size = gene['link_sizes'][0]
    urdf += f'''  <link name="base_link">
    <visual>
      <geometry><box size="{chassis_size[0]} {chassis_size[1]} {chassis_size[2]}"/></geometry>
      <material name="metal"/>
    </visual>
    <collision>
      <geometry><box size="{chassis_size[0]} {chassis_size[1]} {chassis_size[2]}"/></geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
  </link>\n'''
    
    # 轮子尺寸
    wheel_radius = 0.06
    wheel_width = 0.04
    
    # 轮子位置 (相对于车身中心)
    wheel_positions = [
        [chassis_size[0]/2 - wheel_radius/2, chassis_size[1]/2, -chassis_size[2]/2],  # 左前轮
        [chassis_size[0]/2 - wheel_radius/2, -chassis_size[1]/2, -chassis_size[2]/2],  # 右前轮
        [-chassis_size[0]/2 + wheel_radius/2, chassis_size[1]/2, -chassis_size[2]/2],  # 左后轮
        [-chassis_size[0]/2 + wheel_radius/2, -chassis_size[1]/2, -chassis_size[2]/2],  # 右后轮
    ]
    
    # 添加四个轮子
    for i in range(1, 5):
        # 轮子连杆
        urdf += f'''  <link name="wheel{i}">
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
        
        # 轮子关节 - 注意轮子方向
        pos = wheel_positions[i-1]
        urdf += f'''  <joint name="wheel_joint{i}" type="continuous">
    <parent link="base_link"/>
    <child link="wheel{i}"/>
    <origin xyz="{pos[0]} {pos[1]} {pos[2]}" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <dynamics damping="0.01" friction="0.01"/>
  </joint>\n'''
        
        # 添加连接器可视化 - 使用更大、更明显的连接器
        urdf += f'''  <link name="connector{i}">
    <visual>
      <geometry>
        <cylinder radius="0.03" length="0.08"/>
      </geometry>
      <material name="joint_material"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.03" length="0.08"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>\n'''
        
        # 连接器与车身的固定关节
        urdf += f'''  <joint name="connector_joint{i}" type="fixed">
    <parent link="base_link"/>
    <child link="connector{i}"/>
    <origin xyz="{pos[0]} {pos[1]} {pos[2] + wheel_radius/2}" rpy="0 0 0"/>
  </joint>\n'''
    
    urdf += '</robot>'
    return urdf

# --- 仿真函数 ---
def simulate_robot_multi(gene, gui=False, sim_time=5.0, use_self_collision=True, terrain_type="flat"):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # 加载不同类型的地形
    if terrain_type == "flat":
        plane_id = p.loadURDF("plane.urdf")
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
    
    # 设置目标点 - 放得更远
    goal_pos = [5.0, 0, 0.1]  # 从2.0增加到5.0
    if gui:
        visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 0.7])
        p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=goal_pos)
    
    # 生成并加载机器人URDF
    urdf_string = generate_urdf(gene)
    with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as f:
        f.write(urdf_string.encode('utf-8'))
        urdf_path = f.name
    
    try:
        flags = p.URDF_USE_SELF_COLLISION if use_self_collision else 0
        robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1], flags=flags)
        
        # 设置相机位置，便于观察
        if gui:
            p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[1.0, 0, 0])
        
        # 设置关节电机控制
        total_energy = 0.0  # 跟踪能量消耗
        
        # 为所有轮子设置相同的速度，使机器人向前移动
        for i in range(p.getNumJoints(robot_id)):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            
            if "wheel_joint" in joint_name:
                # 所有轮子都向前转动
                p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=-10.0, force=100.0)
        
        # 记录初始位置和轨迹
        start_pos, _ = p.getBasePositionAndOrientation(robot_id)
        trajectory = [start_pos]
        max_roll_pitch = 0
        
        # 仿真循环
        for step in range(int(sim_time / (1./240.))):
            p.stepSimulation()
            
            # 每100步记录一次数据
            if step % 100 == 0:
                pos, orn = p.getBasePositionAndOrientation(robot_id)
                trajectory.append(pos)
                
                # 计算姿态稳定性
                euler = p.getEulerFromQuaternion(orn)
                max_roll_pitch = max(max_roll_pitch, abs(euler[0]), abs(euler[1]))
                
                # 计算能量消耗
                for i in range(p.getNumJoints(robot_id)):
                    joint_name = p.getJointInfo(robot_id, i)[1].decode('utf-8')
                    if "wheel_joint" in joint_name:
                        joint_state = p.getJointState(robot_id, i)
                        velocity = joint_state[1]  # 关节速度
                        force = joint_state[3]     # 关节力/扭矩
                        power = abs(velocity * force)  # 功率 = 速度 * 力
                        total_energy += power * (1./240.)  # 能量 = 功率 * 时间
            
            if gui:
                time.sleep(1./240.)
        
        # 获取最终位置和计算指标
        end_pos, end_orn = p.getBasePositionAndOrientation(robot_id)
        
        # 1. 距离目标的距离 - 我们希望最小化这个值
        dist_to_goal = np.linalg.norm(np.array(end_pos) - np.array(goal_pos))
        
        # 2. 路径直线性 - 我们希望最大化这个值，所以返回负值
        path_linearity = 1.0
        if len(trajectory) > 2:
            total_dist = sum(np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i-1])) for i in range(1, len(trajectory)))
            direct_dist = np.linalg.norm(np.array(trajectory[-1]) - np.array(trajectory[0]))
            if total_dist > 0:
                path_linearity = direct_dist / total_dist
        
        # 3. 稳定性 (roll/pitch) - 我们希望最小化这个值
        stability = max_roll_pitch
        
        # 4. 能量效率 (归一化) - 我们希望最小化这个值
        distance_traveled = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        energy_efficiency = total_energy / max(0.1, distance_traveled)
        
        # 5. 前进距离 - 我们希望最大化这个值，所以返回负值
        forward_distance = end_pos[0] - start_pos[0]
        
        # 计算综合得分 - 越低越好
        # 如果机器人向前移动了，我们给予奖励；如果向后移动，我们给予惩罚
        if forward_distance > 0:
            # 奖励向前移动
            dist_score = max(0, dist_to_goal - forward_distance)
        else:
            # 惩罚向后移动
            dist_score = dist_to_goal + abs(forward_distance)
        
        return dist_score, -path_linearity, stability, energy_efficiency
        
    except Exception as e:
        if gui:
            print(f"Simulation error: {e}")
        # 如果仿真失败，返回较差的性能指标
        return 999.0, -0.0, 3.14, 1000.0
        
    finally:
        os.unlink(urdf_path)
        p.disconnect()

# --- 定义多目标优化问题 ---
class RobotMultiObjectiveProblem(Problem):
    def __init__(self, max_links=8, use_self_collision=True):
        # 计算变量数量：基础变量 + 每个连杆的参数数量
        n_vars = 1 + max_links * 13  # 13个参数/连杆
        super().__init__(n_var=n_vars, n_obj=4, n_constr=0, xl=0.0, xu=1.0)
        self.max_links = max_links
        self.use_self_collision = use_self_collision
        self.terrain_type = "flat"  # 可以是 "flat", "stairs", "rough"
        
    def _evaluate(self, X, out, *args, **kwargs):
        f1, f2, f3, f4 = [], [], [], []  # 距离、路径直线性、稳定性、能量效率
        
        for x in X:
            gene = decode_gene(x, self.max_links)
            d, lin, roll, energy = simulate_robot_multi(
                gene, 
                gui=False, 
                use_self_collision=self.use_self_collision,
                terrain_type=self.terrain_type
            )
            f1.append(d)
            f2.append(-lin)  # 注意：最大化路径直线性 → 最小化负值
            f3.append(roll)
            f4.append(energy)
            
        out["F"] = np.column_stack([f1, f2, f3, f4])

# --- 设置 NSGA-II 并优化 ---
problem = RobotMultiObjectiveProblem(max_links=8, use_self_collision=True)
algorithm = NSGA2(
    pop_size=50,  # 增加种群大小以适应更复杂的搜索空间
    crossover=SBX(prob=0.9, eta=20),  # 调整交叉算子参数
    mutation=PM(prob=0.2, eta=20),    # 调整变异算子参数
    eliminate_duplicates=True,
    sampling=LHS()  # 使用拉丁超立方采样进行初始化
)
res = minimize(problem,
               algorithm,
               termination=('n_gen', 30),  # 增加迭代次数
               seed=1,
               verbose=True)

# --- 绘制 3D 帕累托前沿 ---
F = res.F
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(F[:, 0], -F[:, 1], F[:, 2], c=F[:, 3], cmap='viridis', marker='o')
ax.set_xlabel('Distance to Goal (minimize)')
ax.set_ylabel('Path Linearity (maximize)')
ax.set_zlabel('Max Roll/Pitch (minimize)')
ax.set_title('Pareto Front of Robot Morphology Optimization')
cbar = plt.colorbar(scatter)
cbar.set_label('Energy Efficiency (minimize)')
plt.savefig('pareto_front.png')
print("Optimization completed. Pareto front saved to 'pareto_front.png'")
plt.close()

# --- 保存最佳机器人设计 ---
# 找到帕累托前沿上的最佳解
X = res.X
best_distance_idx = np.argmin(F[:, 0])  # 最小距离
best_linearity_idx = np.argmin(F[:, 1])  # 最大直线性（注意：F中是负值）
best_stability_idx = np.argmin(F[:, 2])  # 最小翻滚/俯仰
best_energy_idx = np.argmin(F[:, 3])     # 最小能量消耗

# 计算综合评分 - 归一化每个指标并加权求和
normalized_F = np.zeros_like(F)
for j in range(F.shape[1]):
    min_val = np.min(F[:, j])
    max_val = np.max(F[:, j])
    if max_val > min_val:
        normalized_F[:, j] = (F[:, j] - min_val) / (max_val - min_val)
    else:
        normalized_F[:, j] = 0

# 权重 - 距离最重要，其次是稳定性，再次是直线性，最后是能量效率
weights = np.array([0.7, 0.1, 0.1, 0.1])
scores = np.sum(normalized_F * weights, axis=1)
best_overall_idx = np.argmin(scores)

best_designs = {
    'best_overall': decode_gene(X[best_overall_idx])
}

# 保存最佳设计的URDF
for name, design in best_designs.items():
    urdf = generate_urdf(design)
    with open(f"{name}.urdf", "w") as f:
        f.write(urdf)
    print(f"Saved {name}.urdf")

# 打印最佳设计的性能
print("\nBest overall design performance:")
print(f"Distance={F[best_overall_idx, 0]:.3f}, Linearity={-F[best_overall_idx, 1]:.3f}, Stability={F[best_overall_idx, 2]:.3f}, Energy={F[best_overall_idx, 3]:.3f}")

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
    
    # 加载机器人
    robot_id = p.loadURDF(urdf_file, basePosition=[0, 0, 0.1])
    
    # 打印机器人信息
    num_joints = p.getNumJoints(robot_id)
    print(f"Robot has {num_joints} joints")
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        print(f"Joint {i}: {joint_info[1].decode('utf-8')}, Type: {joint_info[2]}")
    
    # 为所有轮子设置相同的速度，使机器人向前移动
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        
        if "wheel_joint" in joint_name:
            # 所有轮子都向前转动
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=-10.0, force=100.0)
    
    # 设置相机
    p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[1.0, 0, 0])
    
    # 设置目标点 - 更远
    goal_pos = [5.0, 0, 0.1]
    visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 0.7])
    p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=goal_pos)
    
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
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        p.disconnect()

# 自动可视化最佳设计
print("\n自动可视化最佳设计中...")
visualize_robot("best_overall.urdf", sim_time=20.0)
