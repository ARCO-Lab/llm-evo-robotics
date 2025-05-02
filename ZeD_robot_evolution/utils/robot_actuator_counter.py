import xml.etree.ElementTree as ET

def parse_mujoco_xml_string(xml_string):
    """
    解析 MuJoCo XML 字符串，统计每个机器人包含的 actuator 数量
    """
    root = ET.fromstring(xml_string)  # ✅ 解析 XML 字符串

    # 存储每个机器人包含的 actuator 数量
    robot_actuator_counts = {}

    # 1️⃣ 找到所有机器人 base（假设每个机器人有一个 `baseX` 命名）
    for body in root.find("worldbody").findall("body"):
        robot_name = body.get("name")  # 获取机器人名称（如 base1, base2）

        # 确保 robot_name 是有效的
        if not robot_name:
            continue
        
        # 2️⃣ 统计该机器人包含的 joint 名称
        joints = []
        for sub_body in body.findall(".//joint"):
            joint_name = sub_body.get("name")
            if joint_name:
                joints.append(joint_name)
        
        # 3️⃣ 统计 actuator 绑定的 joint
        actuator_count = 0
        actuators = root.find("actuator")
        if actuators is not None:
            for motor in actuators.findall("motor"):
                joint_name = motor.get("joint")
                if joint_name in joints:  # 只统计当前机器人拥有的 joint
                    actuator_count += 1

        # 4️⃣ 记录每个机器人的 actuator 数量
        robot_actuator_counts[robot_name] = actuator_count

    return robot_actuator_counts

if __name__ == "__main__":
    # ✅ 示例 XML 直接作为字符串
    mujoco_xml_string = """<mujoco model="multi_robot">
        <compiler angle="radian" />
        <option timestep="0.002" />

        <asset>
            <texture name="checker" type="2d" builtin="checker" width="512" height="512" />
            <material name="gray" texture="checker" rgba="0.7 0.7 0.7 1" />
        </asset>

        <worldbody>
            <geom name="ground" type="plane" size="10 10 0.1" rgba="0.2 0.6 0.2 1"/>
            <light name="main_light" pos="0 0 2" dir="0 0 -1" diffuse="1 1 1"/>

            <!-- 第一个机器人 -->
            <body name="base1" pos="0 0 0.58">
                <freejoint />
                <geom type="box" size="0.2 0.2 0.1" rgba="0.8 0.8 0.8 1"/>

                <body name="LF_HIP" pos="0.277 0.116 0">
                    <joint name="LF_HAA" type="hinge" axis="1 0 0" pos="0.277 0.116 0" range="-0.5 0.5"/>
                    <geom type="cylinder" size="0.05 0.1" rgba="1 0 0 1"/>
                </body>

                <body name="RF_HIP" pos="0.277 -0.116 0">
                    <joint name="RF_HAA" type="hinge" axis="1 0 0" pos="0.277 -0.116 0" range="-0.5 0.5"/>
                    <geom type="cylinder" size="0.05 0.1" rgba="0 0 1 1"/>
                </body>
            </body>

            <!-- 第二个机器人 -->
            <body name="base2" pos="1.0 0 0.58"> <!-- 位移到 (1.0, 0, 0.58) 以避免碰撞 -->
                <freejoint />
                <geom type="box" size="0.2 0.2 0.1" rgba="0.7 0.7 0.7 1"/>

                <body name="LB_HIP" pos="0.277 0.116 0">
                    <joint name="LB_HAA" type="hinge" axis="1 0 0" pos="0.277 0.116 0" range="-0.5 0.5"/>
                    <geom type="cylinder" size="0.05 0.1" rgba="0 1 0 1"/>
                </body>

                <body name="RB_HIP" pos="0.277 -0.116 0">
                    <joint name="RB_HAA" type="hinge" axis="1 0 0" pos="0.277 -0.116 0" range="-0.5 0.5"/>
                    <geom type="cylinder" size="0.05 0.1" rgba="1 1 0 1"/>
                </body>
            </body>
        </worldbody>

        <actuator>
            <!-- 第一个机器人的执行器 -->
            <motor name="LF_HAA_motor" joint="LF_HAA" ctrlrange="-1 1" gear="10"/>
            <motor name="RF_HAA_motor" joint="RF_HAA" ctrlrange="-1 1" gear="10"/>

            <!-- 第二个机器人的执行器 -->
            <motor name="LB_HAA_motor" joint="LB_HAA" ctrlrange="-1 1" gear="10"/>
            <motor name="RB_HAA_motor" joint="RB_HAA" ctrlrange="-1 1" gear="10"/>
        </actuator>
    </mujoco>
    """

    robot_actuators = parse_mujoco_xml_string(mujoco_xml_string)

    print("📢 机器人 actuator 数量统计：")
    for robot, count in robot_actuators.items():
        print(f"🤖 {robot}: {count} 个 actuators")
