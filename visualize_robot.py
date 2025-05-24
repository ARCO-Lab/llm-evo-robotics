import pybullet as p
import pybullet_data
import time
import argparse
import numpy as np

def create_terrain(terrain_type="flat"):
    """创建不同类型的地形"""
    if terrain_type == "flat":
        plane_id = p.loadURDF("plane.urdf")
        return plane_id
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
        return terrain
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
        return terrain
    elif terrain_type == "obstacles":
        # 创建有障碍物的平地
        plane_id = p.loadURDF("plane.urdf")
        
        # 添加一些随机障碍物
        obstacles = []
        for i in range(10):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            size = np.random.uniform(0.1, 0.3)
            height = np.random.uniform(0.1, 0.5)
            
            shape = np.random.choice([p.GEOM_BOX, p.GEOM_CYLINDER, p.GEOM_SPHERE])
            if shape == p.GEOM_BOX:
                collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, height/2])
                visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, height/2], rgbaColor=[0.5, 0.5, 0.5, 1.0])
            elif shape == p.GEOM_CYLINDER:
                collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=size, height=height)
                visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=size, length=height, rgbaColor=[0.5, 0.5, 0.5, 1.0])
            else:  # GEOM_SPHERE
                collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=size)
                visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=[0.5, 0.5, 0.5, 1.0])
            
            obstacle = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[x, y, height/2]
            )
            obstacles.append(obstacle)
        return plane_id
    else:
        return p.loadURDF("plane.urdf")

def visualize_robot(urdf_file, sim_time=30.0, terrain_type="flat"):
    """可视化机器人设计并进行简单仿真"""
    print(f"Visualizing {urdf_file} on {terrain_type} terrain...")
    cid = p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # 创建地形
    terrain_id = create_terrain(terrain_type)
    
    # 加载机器人
    robot_id = p.loadURDF(urdf_file, basePosition=[0, 0, 0.3])
    
    # 打印机器人信息
    num_joints = p.getNumJoints(robot_id)
    print(f"Robot has {num_joints} joints")
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        print(f"Joint {i}: {joint_info[1].decode('utf-8')}, Type: {joint_info[2]}")
    
    # 为所有关节设置电机控制
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[2] != p.JOINT_FIXED:  # 如果不是固定关节
            if joint_info[2] == p.JOINT_REVOLUTE:
                p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=1.0, force=10.0)
            elif joint_info[2] == p.JOINT_PRISMATIC:
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=0.5, force=10.0)
    
    # 设置相机
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0, 0, 0])
    
    # 设置目标点
    goal_pos = [2.0, 0, 0.1]
    visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 0.7])
    p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=goal_pos)
    
    # 添加调试线条，显示轨迹
    line_id = None
    prev_pos = None
    
    # 运行仿真
    print("Running simulation... Press Ctrl+C to stop.")
    start_time = time.time()
    total_energy = 0.0
    try:
        while time.time() - start_time < sim_time:
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
            
            # 计算能量消耗
            for i in range(p.getNumJoints(robot_id)):
                joint_state = p.getJointState(robot_id, i)
                velocity = joint_state[1]  # 关节速度
                force = joint_state[3]     # 关节力/扭矩
                power = abs(velocity * force)  # 功率 = 速度 * 力
                total_energy += power * (1./240.)  # 能量 = 功率 * 时间
            
            # 每0.5秒打印一次状态
            if int((time.time() - start_time) * 2) % 1 == 0:
                dist_to_goal = np.linalg.norm(np.array(pos) - np.array(goal_pos))
                print(f"Time: {time.time() - start_time:.1f}s, Pos: {pos}, "
                      f"Roll/Pitch: {euler[0]:.2f}/{euler[1]:.2f}, "
                      f"Dist to goal: {dist_to_goal:.2f}, "
                      f"Energy: {total_energy:.2f}")
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        p.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize robot URDF files")
    parser.add_argument("urdf_file", help="Path to the URDF file to visualize")
    parser.add_argument("--time", type=float, default=30.0, help="Simulation time in seconds")
    parser.add_argument("--terrain", type=str, default="flat", 
                        choices=["flat", "stairs", "rough", "obstacles"],
                        help="Type of terrain to use")
    args = parser.parse_args()
    
    visualize_robot(args.urdf_file, args.time, args.terrain) 