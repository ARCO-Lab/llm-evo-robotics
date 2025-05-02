import xml.etree.ElementTree as ET
import numpy as np

def add_walls_and_goals(xml_string):
    """
    解析 XML 文件，自动为每个机器人添加围墙，并确保机器人位于围墙中心。
    同时，在围墙内部随机生成目标点。
    """
    root = ET.fromstring(xml_string)
    worldbody = root.find("worldbody")

    if worldbody is None:
        raise ValueError("XML 文件中缺少 <worldbody> 标签")

    robot_bodies = [body for body in worldbody.findall("body") if "base" in body.attrib.get("name", "")]

    new_elements = []  # 存储新增的墙体和目标点
    region_size = 2.0  # 每个机器人的围墙区域大小
    wall_thickness = 0.1  # 围墙厚度

    for i, robot in enumerate(robot_bodies):
        robot_name = robot.attrib["name"]
        pos = robot.attrib.get("pos", "0 0 0").split()
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

        # ✅ 将机器人放入围墙中心
        robot.set("pos", f"{x} {y} {z}")

        # ✅ 计算围墙位置
        walls = [
            {"name": f"wall_left_{i}", "pos": f"{x - region_size} {y} {z}", "size": f"{wall_thickness} {region_size} 0.2"},
            {"name": f"wall_right_{i}", "pos": f"{x + region_size} {y} {z}", "size": f"{wall_thickness} {region_size} 0.2"},
            {"name": f"wall_back_{i}", "pos": f"{x} {y + region_size} {z}", "size": f"{region_size} {wall_thickness} 0.2"},
            {"name": f"wall_front_{i}", "pos": f"{x} {y - region_size} {z}", "size": f"{region_size} {wall_thickness} 0.2"},
        ]

        for wall in walls:
            wall_elem = ET.Element("geom", attrib={
                "name": wall["name"],
                "type": "box",
                "pos": wall["pos"],
                "size": wall["size"],
                "rgba": "0.3 0.3 0.3 1"
            })
            new_elements.append(wall_elem)

        # ✅ 生成围墙内部的随机目标点
        goal_x = np.random.uniform(x - region_size + 0.2, x + region_size - 0.2)
        goal_y = np.random.uniform(y - region_size + 0.2, y + region_size - 0.2)
        goal_elem = ET.Element("geom", attrib={
            "name": f"goal_{i}",
            "type": "sphere",
            "pos": f"{goal_x} {goal_y} {z}",
            "size": "0.05",
            "rgba": "1 0 0 1"
        })
        new_elements.append(goal_elem)

    # ✅ 把新创建的围墙和目标点添加到 worldbody 里
    for elem in new_elements:
        worldbody.append(elem)

    return ET.tostring(root, encoding="unicode")

# 测试 XML 解析和修改
if __name__ == "__main__":
    original_xml = """<mujoco model="multi_robot">
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
        </body>

        <!-- 第二个机器人 -->
        <body name="base2" pos="3 0 0.58">
            <freejoint />
            <geom type="box" size="0.2 0.2 0.1" rgba="0.7 0.7 0.7 1"/>
        </body>
    </worldbody>

    <actuator>
        <motor name="motor1" joint="LF_HAA" ctrlrange="-1 1" gear="10"/>
        <motor name="motor2" joint="RF_HAA" ctrlrange="-1 1" gear="10"/>
    </actuator>
</mujoco>"""

    new_xml = add_walls_and_goals(original_xml)
    print(new_xml)  # ✅ 输出带有围墙和随机目标点的 XML
