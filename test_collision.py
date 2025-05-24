import pybullet as p
import pybullet_data
import time
import numpy as np
import tempfile
import os

def generate_simple_urdf():
    """生成一个简单的机器人URDF，包含三个链接，它们的位置会导致碰撞"""
    urdf = '''<?xml version="1.0"?>
<robot name="collision_test_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.1"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.0017" ixy="0" ixz="0" iyy="0.0033" iyz="0" izz="0.0033"/>
    </inertial>
  </link>

  <link name="link1">
    <visual>
      <geometry>
        <box size="0.3 0.1 0.1"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.0017" ixy="0" ixz="0" iyy="0.0033" iyz="0" izz="0.0033"/>
    </inertial>
  </link>

  <link name="link2">
    <visual>
      <geometry>
        <box size="0.3 0.1 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.0017" ixy="0" ixz="0" iyy="0.0033" iyz="0" izz="0.0033"/>
    </inertial>
  </link>

  <!-- 将link1连接到base_link -->
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>

  <!-- 将link2连接到link1 -->
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0.3 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="10"/>
  </joint>
</robot>
'''
    return urdf

def test_collision(use_self_collision_flag=False):
    """测试机器人零件之间的碰撞检测
    
    Args:
        use_self_collision_flag: 是否使用URDF_USE_SELF_COLLISION标志
    """
    # 连接到PyBullet
    physicsClient = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    # 设置相机视角
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=60,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.3]
    )
    
    # 生成URDF并保存到临时文件
    urdf_string = generate_simple_urdf()
    with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as f:
        f.write(urdf_string.encode('utf-8'))
        urdf_path = f.name
    
    try:
        # 加载机器人
        flags = p.URDF_USE_SELF_COLLISION if use_self_collision_flag else 0
        print(f"使用标志: {flags} ({'启用自碰撞' if use_self_collision_flag else '不启用自碰撞'})")
        robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.3], flags=flags)
        
        # 获取关节信息
        num_joints = p.getNumJoints(robot_id)
        print(f"机器人有 {num_joints} 个关节")
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            print(f"关节 {i}: {joint_info[1].decode('utf-8')}")
        
        # 添加调试文本
        text_id = p.addUserDebugText(
            "使用" + ("自碰撞" if use_self_collision_flag else "无自碰撞"), 
            [0, 0, 0.8], 
            textColorRGB=[1, 1, 1],
            textSize=1.5
        )
        
        # 添加调试线，显示链接的位置和方向
        p.addUserDebugLine([0, 0, 0], [0.3, 0, 0], [1, 0, 0], parentObjectUniqueId=robot_id, parentLinkIndex=0)
        p.addUserDebugLine([0, 0, 0], [0, 0.3, 0], [0, 1, 0], parentObjectUniqueId=robot_id, parentLinkIndex=0)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.3], [0, 0, 1], parentObjectUniqueId=robot_id, parentLinkIndex=0)
        
        # 运行模拟
        collision_detected = False
        max_steps = 500
        
        print("开始模拟...")
        
        # 首先让机器人稳定下来
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1/240)
        
        # 设置关节角度，使零件之间产生碰撞
        print("设置关节角度，使零件之间产生碰撞...")
        p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL, targetPosition=np.pi/2, force=500)
        time.sleep(0.5)  # 等待第一个关节转动
        
        for step in range(max_steps):
            # 第二个关节逐渐转动，使其与base_link碰撞
            if step > 100 and step < 300:
                angle = (step - 100) * (np.pi/2) / 200
                p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=angle, force=500)
            
            p.stepSimulation()
            
            # 检查碰撞
            contact_points = p.getContactPoints(robot_id, robot_id)
            if contact_points and not collision_detected:
                collision_detected = True
                print(f"\n第 {step} 步检测到 {len(contact_points)} 个碰撞点:")
                for i, cp in enumerate(contact_points):
                    link_index_a = cp[3]
                    link_index_b = cp[4]
                    link_name_a = "base_link" if link_index_a == -1 else f"link{link_index_a}"
                    link_name_b = "base_link" if link_index_b == -1 else f"link{link_index_b}"
                    contact_distance = cp[8]
                    normal_force = cp[9]
                    print(f"  碰撞 {i+1}: {link_name_a} 与 {link_name_b} 之间")
                    print(f"    距离: {contact_distance:.6f}, 法向力: {normal_force:.6f}")
                    
                    # 添加碰撞点可视化
                    p.addUserDebugLine(
                        cp[5],  # 接触点A
                        cp[6],  # 接触点B
                        [1, 0, 0],  # 红色
                        lineWidth=5,
                        lifeTime=5
                    )
            
            # 每100步显示一次关节角度
            if step % 100 == 0:
                joint1_state = p.getJointState(robot_id, 0)
                joint2_state = p.getJointState(robot_id, 1)
                print(f"步骤 {step}: 关节1角度 = {joint1_state[0]:.2f}, 关节2角度 = {joint2_state[0]:.2f}")
            
            time.sleep(1/240)
        
        if not collision_detected:
            print("模拟结束，未检测到碰撞")
        else:
            print("模拟结束，检测到碰撞")
        
        # 保持窗口打开一段时间以便观察
        print("保持窗口打开5秒以便观察...")
        time.sleep(5)
    
    except Exception as e:
        print(f"错误: {e}")
    
    finally:
        # 清理临时文件
        if os.path.exists(urdf_path):
            os.unlink(urdf_path)
        p.disconnect()

if __name__ == "__main__":
    print("=== 测试不带自碰撞标志的情况 ===")
    test_collision(use_self_collision_flag=False)
    
    print("\n=== 测试带自碰撞标志的情况 ===")
    test_collision(use_self_collision_flag=True) 