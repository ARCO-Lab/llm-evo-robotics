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


def decode_gene(x, max_links=6):
    num_links = max(2, int(x[0] * max_links) + 1)
    joint_types, has_motor, shapes = [], [], []
    is_wheel = []
    idx = 1
    for i in range(num_links):
        jt = p.JOINT_REVOLUTE if x[idx + 1] < 0.5 else p.JOINT_FIXED
        motor = jt == p.JOINT_REVOLUTE and x[idx + 2] < 0.7
        shape = int(x[idx + 3] * 3)
        joint_types.append(jt)
        has_motor.append(motor)
        shapes.append(shape)
        is_wheel.append(x[idx + 4] < 0.3 and shape == 1)
        idx += 7

    if not any(has_motor):
        for i, jt in enumerate(joint_types):
            if jt == p.JOINT_REVOLUTE:
                has_motor[i] = True
                break

    return {
        'num_links': num_links,
        'joint_types': joint_types,
        'has_motor': has_motor,
        'shapes': shapes,
        'is_wheel': is_wheel,
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
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.0017" ixy="0" ixz="0" iyy="0.0033" iyz="0" izz="0.0033"/>
    </inertial>
  </link>\n'''
    elif shape_id == 1:  # Cylinder
        is_wheel = gene['is_wheel'][0]
        color = "0.1 0.1 0.1 1" if is_wheel else "0.8 0.2 0.2 1"
        urdf += f'''  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="base_color">
        <color rgba="{color}"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>\n'''
    else:  # Sphere
        urdf += '''  <link name="base_link">
    <visual>
      <geometry>
        <sphere radius="0.07"/>
      </geometry>
      <material name="red">
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
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>\n'''
    
    # Add child links and joints
    for i in range(1, gene['num_links']):
        shape_id = gene['shapes'][i]
        is_wheel = gene['is_wheel'][i]
        joint_type = "revolute" if gene['joint_types'][i] == p.JOINT_REVOLUTE else "fixed"
        
        # Add link
        if shape_id == 0:  # Box
            urdf += f'''  <link name="link{i}">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
      <material name="link_color">
        <color rgba="0.8 0.2 0.2 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.0017" ixy="0" ixz="0" iyy="0.0033" iyz="0" izz="0.0033"/>
    </inertial>
  </link>\n'''
        elif shape_id == 1:  # Cylinder
            color = "0.1 0.1 0.1 1" if is_wheel else "0.8 0.2 0.2 1"
            urdf += f'''  <link name="link{i}">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="link_color">
        <color rgba="{color}"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>\n'''
        else:  # Sphere
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
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>\n'''
        
        # Add joint
        parent = f"link{i-1}" if i > 1 else "base_link"
        urdf += f'''  <joint name="joint{i}" type="{joint_type}">
    <parent link="{parent}"/>
    <child link="link{i}"/>
    <origin xyz="0.2 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
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
        p.resetDebugVisualizerCamera(cameraDistance=1.0,
                                     cameraYaw=90,
                                     cameraPitch=-89,
                                     cameraTargetPosition=[0.3, 0, 0.0])

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
            if link_idx < len(gene['has_motor']) and gene['has_motor'][link_idx] and joint_type == p.JOINT_REVOLUTE:
                p.setJointMotorControl2(robot_id, joint_index, p.VELOCITY_CONTROL,
                                       targetVelocity=10.0, force=300.0)
        
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
        for step in range(int(sim_time / (1. / 240.))):
            p.stepSimulation()
            
            # 每500步检测一次碰撞（仅在GUI模式下）
            if gui and step % 500 == 0:
                contact_points = p.getContactPoints(robot_id, robot_id)
                if contact_points:
                    print(f"第 {step} 步检测到 {len(contact_points)} 个自碰撞点")
            
            if gui:
                time.sleep(1. / 240.)

        end_pos, _ = p.getBasePositionAndOrientation(robot_id)
        base_move = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        dist_to_goal = np.linalg.norm(np.array(end_pos) - np.array(goal_pos))
        
    except Exception as e:
        if gui:
            print(f"Error in simulation: {e}")
        base_move = 0
        dist_to_goal = float('inf')
    
    finally:
        # Clean up the temporary URDF file
        if os.path.exists(urdf_path):
            os.unlink(urdf_path)
        p.disconnect()
    
    if not any(gene['has_motor']):
        return 0.01
    return 1.0 / (1e-3 + dist_to_goal) + base_move


def render_gene(gene, use_self_collision=True):
    print("👁️ 正在展示机器人...")
    simulate_robot(gene, gui=True, use_self_collision=use_self_collision)


class RobotMoveToTargetProblem(Problem):
    def __init__(self, max_links=6, use_self_collision=True):
        super().__init__(n_var=1 + max_links * 7,
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
            print(f"[Eval {idx}] Links={gene['num_links']}  Motors={gene['has_motor']}  Fitness={fitness:.4f}")
            f.append(-fitness)
        out["F"] = np.column_stack([f])


if __name__ == "__main__":
    # 设置是否使用自碰撞检测
    use_self_collision = True
    print(f"使用自碰撞检测: {'是' if use_self_collision else '否'}")
    
    problem = RobotMoveToTargetProblem(max_links=6, use_self_collision=use_self_collision)

    algorithm = NSGA2(
        pop_size=20,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=0.1, eta=20),
        eliminate_duplicates=True,
        sampling=LHS()
    )

    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 30),
                   seed=1,
                   verbose=True,
                   save_history=True)

    print("\n🎬 可视化每代最优个体：")
    for i, gen in enumerate(res.history):
        opt = gen.opt[0]
        gene = decode_gene(opt.X)
        print(f"Generation {i+1}, fitness: {-opt.F[0]:.4f}")
        render_gene(gene, use_self_collision=use_self_collision)

    if res.X.ndim == 1:
        best_gene = decode_gene(res.X)
    else:
        best_idx = np.argmin(res.F)
        best_gene = decode_gene(res.X[best_idx])

    with open("best_gene.json", "w") as f:
        json.dump(convert_to_builtin(best_gene), f, indent=2)

    print("\n✅ 最优结构基因已保存为 best_gene.json")
    render_gene(best_gene, use_self_collision=use_self_collision) 