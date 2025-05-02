import xml.etree.ElementTree as ET

def load_components(filename):
    """解析 basic_components.xml"""
    tree = ET.parse(filename)
    root = tree.getroot()
    components = {body.get("name"): body for body in root.findall(".//body")}
    return components

def load_actuators(filename):
    """解析 basic_actuators.xml"""
    tree = ET.parse(filename)
    root = tree.getroot()
    actuators = {actuator.get("name"): actuator for actuator in root.findall(".//actuator/*")}
    return actuators

if __name__ == "__main__":
    print("✅ 测试 XML 解析...")
    components = load_components("../basic_components/basic_components.xml")
    actuators = load_actuators("../basic_actuators/basic_actuators.xml")

    print(f"解析成功，找到 {len(components)} 个组件, {len(actuators)} 个驱动器")

