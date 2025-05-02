import xml.etree.ElementTree as ET
import copy

def generate_robot_xml(individuals, components, actuators):
    """生成机器人 XML 并引用 `planes.xml`（避免重复创建 plane）"""
    root = ET.Element("mujoco", model="robot_simulation")

    # 🔥 引入 `planes.xml`
    ET.SubElement(root, "include", file="planes.xml")

    worldbody = ET.SubElement(root, "worldbody")
    actuators_section = ET.SubElement(root, "actuator")

    for i, binary_encoding in enumerate(individuals):
        robot_body = ET.SubElement(worldbody, "body", name=f"robot_{i}", pos=f"{i * 5} 0 0.5")

        # 选择组件
        for idx, (name, body) in enumerate(components.items()):
            if binary_encoding[idx] == 1:
                robot_body.append(copy.deepcopy(body))

        # 选择 actuator
        actuator_offset = len(components)
        for idx, (name, actuator) in enumerate(actuators.items()):
            if binary_encoding[actuator_offset + idx] == 1:
                actuators_section.append(copy.deepcopy(actuator))

    return ET.tostring(root, encoding="unicode")

if __name__ == "__main__":
    print("✅ 测试机器人 XML 生成...")
    from xml_parser import load_components, load_actuators
    import numpy as np

    components = load_components("../basic_components/basic_components.xml")
    actuators = load_actuators("../basic_actuators/basic_actuators.xml")

    dummy_individual = np.random.randint(0, 2, len(components) + len(actuators))
    robot_xml = generate_robot_xml([dummy_individual], components, actuators)

    print("测试完成，生成的 XML 片段:")
    print(robot_xml[:500])  # 只打印前 500 字符

