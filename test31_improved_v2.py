import numpy as np
import random
import pybullet as p
import pybullet_data
import time
import json
import os
import tempfile

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover as SBX
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling as LHS


def convert_to_builtin(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin(i) for i in obj]
    else:
        return obj


def decode_gene(x, max_links=8):
    # 增加最大链接数为8
    num_links = max(4, int(x[0] * max_links) + 1)
    joint_types, has_motor, shapes = [], [], []
    is_wheel = []
    wheel_types = []  # 新增：轮子类型（0=普通轮，1=大轮，2=窄轮）
    joint_axes = []   # 新增：关节轴向
    idx = 1
    
    for i in range(num_links):
        # 关节类型：0=revolute, 1=fixed, 2=prismatic, 3=continuous
        joint_type_val = x[idx] * 4
        if joint_type_val < 1:
            jt = p.JOINT_REVOLUTE
        elif joint_type_val < 2:
            jt = p.JOINT_FIXED
        elif joint_type_val < 3:
            jt = p.JOINT_PRISMATIC
        else:
            jt = p.JOINT_REVOLUTE  # 连续关节用revolute模拟
            
        # 是否有马达
        motor = jt != p.JOINT_FIXED and x[idx + 1] < 0.8
        
        # 形状类型：0=box, 1=cylinder, 2=sphere, 3=capsule, 4=large_wheel
        shape = int(x[idx + 2] * 5)
        
        # 是否为轮子
        wheel = shape == 1 and x[idx + 3] < 0.6  # 圆柱体有40%概率成为轮子
        
        # 轮子类型
        wheel_type = int(x[idx + 4] * 3) if wheel else 0
        
        # 关节轴向：随机选择x、y、z轴或其组合
        axis_val = x[idx + 5] * 3
        if axis_val < 1:
            axis = [1, 0, 0]  # x轴
        elif axis_val < 2:
            axis = [0, 1, 0]  # y轴
        else:
            axis = [0, 0, 1]  # z轴
            
        joint_types.append(jt)
        has_motor.append(motor)
        shapes.append(shape)
        is_wheel.append(wheel)
        wheel_types.append(wheel_type)
        joint_axes.append(axis)
        idx += 8  # 增加基因长度

    # 确保至少有一个马达
    if not any(has_motor):
        for i, jt in enumerate(joint_types):
            if jt != p.JOINT_FIXED:
                has_motor[i] = True
                break

    return {
        'num_links': num_links,
        'joint_types': joint_types,
        'has_motor': has_motor,
        'shapes': shapes,
        'is_wheel': is_wheel,
        'wheel_types': wheel_types,
        'joint_axes': joint_axes
    }


def generate_urdf(gene):
    """Generate a URDF string from a gene."""
    urdf = '<?xml version="1.0"?>\n<robot name="evolved_robot">\n'
    
    # Add base link
    shape_id = gene['shapes'][0]
    if shape_id == 0:  # Box
        urdf += '''  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.15 0.1"/>
      </geometry>
      <material name="base_color">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.15 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.0025" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.005"/>
    </inertial>
  </link>\n'''
    elif shape_id == 1:  # Cylinder
        is_wheel = gene['is_wheel'][0]
        wheel_type = gene['wheel_types'][0] if is_wheel else 0
        
        # 根据轮子类型设置不同参数
        if wheel_type == 0:  # 普通轮
            radius, length = 0.06, 0.05
            color = "0.1 0.1 0.1 1"
            mass = 0.8
        elif wheel_type == 1:  # 大轮
            radius, length = 0.1, 0.05
            color = "0.1 0.1 0.1 1"
            mass = 1.2
        else:  # 窄轮
            radius, length = 0.06, 0.02
            color = "0.1 0.1 0.1 1"
            mass = 0.5
            
        if not is_wheel:
            color = "0.8 0.2 0.2 1"
            mass = 1.0
            
        urdf += f'''  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="{radius}" length="{length}"/>
      </geometry>
      <material name="base_color">
        <color rgba="{color}"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="{radius}" length="{length}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="{mass*(3*radius**2+length**2)/12}" ixy="0" ixz="0" 
               iyy="{mass*(3*radius**2+length**2)/12}" iyz="0" 
               izz="{mass*radius**2/2}"/>
    </inertial>
  </link>\n'''
    elif shape_id == 2:  # Sphere
        urdf += '''  <link name="base_link">
    <visual>
      <geometry>
        <sphere radius="0.07"/>
      </geometry>
      <material name="base_color">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.07"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.0028" ixy="0" ixz="0" iyy="0.0028" iyz="0" izz="0.0028"/>
    </inertial>
  </link>\n'''
    elif shape_id == 3:  # Capsule (使用两个球体和一个圆柱体组合)
        urdf += '''  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="base_color">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.2"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.0015"/>
    </inertial>
  </link>\n'''
    else:  # Large wheel (大型轮子)
        urdf += '''  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.12" length="0.05"/>
      </geometry>
      <material name="wheel_color">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.12" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.0054" ixy="0" ixz="0" iyy="0.0054" iyz="0" izz="0.0108"/>
    </inertial>
  </link>\n'''
    
    # Add child links and joints
    for i in range(1, gene['num_links']):
        shape_id = gene['shapes'][i]
        is_wheel = gene['is_wheel'][i]
        wheel_type = gene['wheel_types'][i] if is_wheel else 0
        
        # 确定关节类型
        if gene['joint_types'][i] == p.JOINT_REVOLUTE:
            joint_type = "revolute"
        elif gene['joint_types'][i] == p.JOINT_FIXED:
            joint_type = "fixed"
        elif gene['joint_types'][i] == p.JOINT_PRISMATIC:
            joint_type = "prismatic"
        else:
            joint_type = "continuous"
            
        # 确定关节轴向
        axis = gene['joint_axes'][i]
        axis_str = f"{axis[0]} {axis[1]} {axis[2]}"
        
        # Add link based on shape type
        if shape_id == 0:  # Box
            urdf += f'''  <link name="link{i}">
    <visual>
      <geometry>
        <box size="0.2 0.15 0.1"/>
      </geometry>
      <material name="link_color">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.15 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.0017" ixy="0" ixz="0" iyy="0.0033" iyz="0" izz="0.0033"/>
    </inertial>
  </link>\n'''
        elif shape_id == 1:  # Cylinder
            # 根据轮子类型设置不同参数
            if wheel_type == 0:  # 普通轮
                radius, length = 0.06, 0.05
                color = "0.1 0.1 0.1 1"
                mass = 0.8
            elif wheel_type == 1:  # 大轮
                radius, length = 0.1, 0.05
                color = "0.1 0.1 0.1 1"
                mass = 1.2
            else:  # 窄轮
                radius, length = 0.06, 0.02
                color = "0.1 0.1 0.1 1"
                mass = 0.5
                
            if not is_wheel:
                color = "0.8 0.2 0.2 1"
                mass = 1.0
                
            urdf += f'''  <link name="link{i}">
    <visual>
      <geometry>
        <cylinder radius="{radius}" length="{length}"/>
      </geometry>
      <material name="link_color">
        <color rgba="{color}"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="{radius}" length="{length}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="{mass*(3*radius**2+length**2)/12}" ixy="0" ixz="0" 
               iyy="{mass*(3*radius**2+length**2)/12}" iyz="0" 
               izz="{mass*radius**2/2}"/>
    </inertial>
  </link>\n'''
        elif shape_id == 2:  # Sphere
            urdf += f'''  <link name="link{i}">
    <visual>
      <geometry>
        <sphere radius="0.07"/>
      </geometry>
      <material name="link_color">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.07"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.0028" ixy="0" ixz="0" iyy="0.0028" iyz="0" izz="0.0028"/>
    </inertial>
  </link>\n'''
        elif shape_id == 3:  # Capsule
            urdf += f'''  <link name="link{i}">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="link_color">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.2"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.0015"/>
    </inertial>
  </link>\n'''
        else:  # Large wheel
            urdf += f'''  <link name="link{i}">
    <visual>
      <geometry>
        <cylinder radius="0.12" length="0.05"/>
      </geometry>
      <material name="wheel_color">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.12" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.0054" ixy="0" ixz="0" iyy="0.0054" iyz="0" izz="0.0108"/>
    </inertial>
  </link>\n'''
        
        # 添加关节，根据链接形状调整关节位置
        parent = f"link{i-1}" if i > 1 else "base_link"
        
        # 根据形状调整关节位置，使结构更合理
        if shape_id == 1 and is_wheel:  # 轮子应该在侧面
            urdf += f'''  <joint name="joint{i}" type="{joint_type}">
    <parent link="{parent}"/>
    <child link="link{i}"/>
    <origin xyz="0 0.15 0" rpy="0 0 0"/>
    <axis xyz="{axis_str}"/>
    <limit lower="-3.14" upper="3.14" effort="30" velocity="20"/>
  </joint>\n'''
        elif shape_id == 4:  # 大轮子
            urdf += f'''  <joint name="joint{i}" type="{joint_type}">
    <parent link="{parent}"/>
    <child link="link{i}"/>
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
    <axis xyz="{axis_str}"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="15"/>
  </joint>\n'''
        else:  # 其他形状
            urdf += f'''  <joint name="joint{i}" type="{joint_type}">
    <parent link="{parent}"/>
    <child link="link{i}"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
    <axis xyz="{axis_str}"/>
    <limit lower="-3.14" upper="3.14" effort="30" velocity="20"/>
  </joint>\n'''
    
    urdf += '</robot>'
    return urdf


def simulate_robot(gene, gui=False, show_goal=True, sim_time=20.0, use_self_collision=True):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    goal_pos = [2.0, 0, 0.1]
    if show_goal:
        goal_id = p.loadURDF("cube_small.urdf", basePosition=goal_pos, globalScaling=1.5)
        p.changeVisualShape(goal_id, -1, rgbaColor=[0, 1, 0, 1])

    if gui:
        p.resetDebugVisualizerCamera(cameraDistance=3.0,
                                     cameraYaw=90,
                                     cameraPitch=-30,
                                     cameraTargetPosition=[1.0, 0, 0.0])

    # Generate URDF for the robot
    urdf_string = generate_urdf(gene)
    
    # Save the URDF to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as f:
        f.write(urdf_string.encode('utf-8'))
        urdf_path = f.name
    
    try:
        # Load the robot from the URDF with self-collision flag if enabled
        flags = p.URDF_USE_SELF_COLLISION if use_self_collision else 0
        if gui:
            print(f"使用标志: {flags} ({'启用自碰撞' if use_self_collision else '不启用自碰撞'})")
        robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.3], flags=flags)
        
        # Apply motors to joints
        num_joints = p.getNumJoints(robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_index = joint_info[0]
            joint_type = joint_info[2]
            
            # Check if this joint should have a motor
            link_idx = i + 1  # +1 because joint indices start at 0 but our links start at 1
            if link_idx < len(gene['has_motor']) and gene['has_motor'][link_idx]:
                if joint_type == p.JOINT_REVOLUTE:
                    # 为轮子设置更高的速度和力量
                    if link_idx < len(gene['is_wheel']) and gene['is_wheel'][link_idx]:
                        p.setJointMotorControl2(robot_id, joint_index, p.VELOCITY_CONTROL,
                                               targetVelocity=15.0, force=500.0)
                    else:
                        p.setJointMotorControl2(robot_id, joint_index, p.VELOCITY_CONTROL,
                                               targetVelocity=10.0, force=300.0)
                elif joint_type == p.JOINT_PRISMATIC:
                    # 为伸缩关节设置位置控制
                    p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL,
                                           targetPosition=0.2, force=200.0)
        
        # 检测初始状态是否有碰撞
        if gui:
            contact_points = p.getContactPoints(robot_id, robot_id)
            if contact_points:
                print(f"初始状态检测到 {len(contact_points)} 个自碰撞点")
                for i, cp in enumerate(contact_points):
                    link_index_a = cp[3]
                    link_index_b = cp[4]
                    link_name_a = "base_link" if link_index_a == -1 else f"link{link_index_a}"
                    link_name_b = "base_link" if link_index_b == -1 else f"link{link_index_b}"
                    print(f"  碰撞 {i+1}: {link_name_a} 与 {link_name_b} 之间")
        
        start_pos, _ = p.getBasePositionAndOrientation(robot_id)
        
        # 记录运动轨迹
        trajectory = [start_pos]
        stability_score = 0
        last_orientation = None
        
        for step in range(int(sim_time / (1. / 240.))):
            p.stepSimulation()
            
            # 每500步检测一次碰撞（仅在GUI模式下）
            if gui and step % 500 == 0:
                contact_points = p.getContactPoints(robot_id, robot_id)
                if contact_points:
                    print(f"第 {step} 步检测到 {len(contact_points)} 个自碰撞点")
            
            # 每100步记录一次位置
            if step % 100 == 0:
                pos, orientation = p.getBasePositionAndOrientation(robot_id)
                trajectory.append(pos)
                
                # 计算稳定性分数（方向变化不大为佳）
                if last_orientation is not None:
                    orientation_diff = sum(abs(a-b) for a, b in zip(p.getEulerFromQuaternion(orientation), 
                                                                   p.getEulerFromQuaternion(last_orientation)))
                    if orientation_diff < 0.1:  # 方向变化小，稳定性好
                        stability_score += 0.01
                last_orientation = orientation
            
            if gui:
                time.sleep(1. / 240.)

        end_pos, end_orientation = p.getBasePositionAndOrientation(robot_id)
        
        # 计算总移动距离
        base_move = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        
        # 计算到目标的距离
        dist_to_goal = np.linalg.norm(np.array(end_pos) - np.array(goal_pos))
        
        # 计算路径直线性（越直越好）
        path_linearity = 1.0
        if len(trajectory) > 2:
            total_dist = 0
            direct_dist = np.linalg.norm(np.array(trajectory[-1]) - np.array(trajectory[0]))
            for i in range(1, len(trajectory)):
                total_dist += np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i-1]))
            if total_dist > 0:
                path_linearity = direct_dist / total_dist  # 1.0表示完全直线
        
        # 检查机器人是否翻倒
        euler = p.getEulerFromQuaternion(end_orientation)
        tipped_over = abs(euler[0]) > 0.5 or abs(euler[1]) > 0.5
        
        # 如果翻倒，降低适应度
        if tipped_over:
            base_move *= 0.5
            
    except Exception as e:
        if gui:
            print(f"Error in simulation: {e}")
        base_move = 0
        dist_to_goal = float('inf')
        path_linearity = 0
        stability_score = 0
    
    finally:
        # Clean up the temporary URDF file
        if os.path.exists(urdf_path):
            os.unlink(urdf_path)
        p.disconnect()
    
    # 如果没有马达，给予很低的适应度
    if not any(gene['has_motor']):
        return 0.01
        
    # 计算最终适应度（距离目标近、移动距离长、路径直、稳定性好）
    # 添加结构复杂度奖励
    complexity_bonus = gene["num_links"] * 0.2  # 每个链接增加0.2的适应度
    wheel_bonus = sum(gene["is_wheel"]) * 0.3  # 每个轮子增加0.3的适应度
    
    # 计算最终适应度
    fitness = (1.0 / (1e-3 + dist_to_goal)) + base_move + path_linearity + stability_score + complexity_bonus + wheel_bonus
    
    return fitness


def render_gene(gene, use_self_collision=True):
    print("👁️ 正在展示机器人...")
    fitness = simulate_robot(gene, gui=True, use_self_collision=use_self_collision)
    print(f"机器人适应度: {fitness:.4f}")
    return fitness


class RobotMoveToTargetProblem(Problem):
    def __init__(self, max_links=8, use_self_collision=True):
        # 增加每个链接的基因长度为8
        super().__init__(n_var=1 + max_links * 8,
                         n_obj=1,
                         n_constr=0,
                         xl=0.0,
                         xu=1.0)
        self.max_links = max_links
        self.use_self_collision = use_self_collision

    def _evaluate(self, X, out, *args, **kwargs):
        f = []
        for idx, x in enumerate(X):
            gene = decode_gene(x, self.max_links)
            try:
                fitness = simulate_robot(gene, gui=False, use_self_collision=self.use_self_collision)
            except Exception as e:
                print(f"Error evaluating gene: {e}")
                fitness = 0.0
            
            # 打印评估信息
            wheel_count = sum(gene['is_wheel'])
            motor_count = sum(gene['has_motor'])
            print(f"[Eval {idx}] Links={gene['num_links']} Motors={motor_count} Wheels={wheel_count} Fitness={fitness:.4f}")
            
            f.append(-fitness)  # 最小化负适应度 = 最大化适应度
        out["F"] = np.column_stack([f])


if __name__ == "__main__":
    # 设置是否使用自碰撞检测
    use_self_collision = True
    print(f"使用自碰撞检测: {'是' if use_self_collision else '否'}")
    
    # 增加最大链接数为8
    max_links = 8
    problem = RobotMoveToTargetProblem(max_links=max_links, use_self_collision=use_self_collision)

    # 增加种群大小和进化代数
    algorithm = NSGA2(
        pop_size=40,  # 增加种群大小
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.25, eta=15),  # 增加变异概率
        eliminate_duplicates=True,
        sampling=LHS()
    )

    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 40),  # 增加进化代数
                   seed=1,
                   verbose=True,
                   save_history=True)

    print("\n🎬 可视化每代最优个体：")
    best_fitness = -float('inf')
    best_gene = None
    
    # 只展示每5代的最优个体，节省时间
    for i, gen in enumerate(res.history):
        if i % 5 == 0 or i == len(res.history) - 1:  # 每5代或最后一代
            opt = gen.opt[0]
            gene = decode_gene(opt.X, max_links)
            fitness = -opt.F[0]
            print(f"Generation {i+1}, fitness: {fitness:.4f}")
            
            # 记录最佳个体
            if fitness > best_fitness:
                best_fitness = fitness
                best_gene = gene
                
            render_gene(gene, use_self_collision=use_self_collision)

    # 如果最后一代不是最优的，再展示一次全局最优个体
    if best_gene is not None and best_gene != gene:
        print(f"\n🏆 展示全局最优个体 (fitness: {best_fitness:.4f}):")
        render_gene(best_gene, use_self_collision=use_self_collision)

    # 保存最优基因
    with open("best_gene_improved.json", "w") as f:
        json.dump(convert_to_builtin(best_gene), f, indent=2)

    print("\n✅ 最优结构基因已保存为 best_gene_improved.json") 