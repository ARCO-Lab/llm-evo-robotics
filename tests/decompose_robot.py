import mujoco
import os
import xml.etree.ElementTree as ET
import time

def load_and_parse_xml(xml_path):
    """加载并解析XML文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return root

def extract_robot_components(root):
    """提取机器人的组件信息"""
    components = []
    
    # 递归提取所有机器人组件
    def extract_body(element, parent_path=""):
        if element.tag == 'body':
            # 记录body的位置和属性
            body_info = {
                'name': element.get('name', 'unnamed'),
                'path': f"{parent_path}/{element.get('name', 'unnamed')}",
                'attributes': element.attrib,
                'geoms': [],
                'joints': []
            }
            
            # 提取该body下的所有geom和joint
            for child in element:
                if child.tag == 'geom':
                    body_info['geoms'].append({
                        'type': child.tag,
                        'attributes': child.attrib
                    })
                elif child.tag == 'joint':
                    body_info['joints'].append({
                        'type': child.tag,
                        'attributes': child.attrib
                    })
            
            components.append(body_info)
            
            # 递归处理子body
            for child in element:
                if child.tag == 'body':
                    extract_body(child, body_info['path'])
    
    # 从worldbody开始递归
    worldbody = root.find('.//worldbody')
    if worldbody is not None:
        for child in worldbody:
            if child.tag == 'body':
                extract_body(child)
    
    return components

def create_component_model(component):
    """为单个机器人组件创建MuJoCo模型"""
    model_xml = """
    <mujoco>
        <asset>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" 
                    rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" 
                    width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" 
                     texrepeat="5 5" reflectance="0.2"/>
        </asset>
        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20"/>
        </visual>
        <worldbody>
            <light pos="0 0 3" dir="0 0 -1" directional="true"/>
            <geom name="ground" type="plane" size="2 2 0.1" rgba="0.8 0.9 0.8 1" material="groundplane"/>
            <body name="component_root" pos="0 0 0.5">
    """
    
    # 添加body属性
    for key, value in component['attributes'].items():
        if key not in ['pos', 'quat']:  # 忽略位置和方向，使组件居中显示
            model_xml += f' {key}="{value}"'
    
    # 添加geoms
    for geom in component['geoms']:
        model_xml += f'\n                <geom'
        for key, value in geom['attributes'].items():
            model_xml += f' {key}="{value}"'
        model_xml += '/>'
    
    # 添加joints
    for joint in component['joints']:
        model_xml += f'\n                <joint'
        for key, value in joint['attributes'].items():
            model_xml += f' {key}="{value}"'
        model_xml += '/>'
    
    model_xml += """
            </body>
        </worldbody>
    </mujoco>
    """
    
    return model_xml

def visualize_component(model_xml, component_name):
    """可视化单个组件"""
    # 保存临时XML文件
    temp_file = f"temp_{component_name}.xml"
    with open(temp_file, "w") as f:
        f.write(model_xml)
    
    try:
        # 加载模型
        model = mujoco.MjModel.from_xml_path(temp_file)
        data = mujoco.MjData(model)
        
        # 使用 launch_passive 显示场景
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print(f"\n显示组件: {component_name}")
            print("按 Ctrl+C 继续下一个组件...")
            while viewer.is_running():
                step_start = time.time()
                
                # 步进模拟
                mujoco.mj_step(model, data)
                
                # 更新查看器
                viewer.sync()
                
                # 控制模拟步频
                time_until_next_step = step_start + 0.001 - time.time()
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    
    except KeyboardInterrupt:
        print("\n继续下一个组件")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def create_combined_xml(components):
    """创建包含所有分解组件的组合XML文件"""
    model_xml = """
    <mujoco>
        <asset>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" 
                    rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" 
                    width="300" height="300"/>
            <material name="groundplane" texture="groundplane" texuniform="true" 
                     texrepeat="5 5" reflectance="0.2"/>
        </asset>
        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20"/>
        </visual>
        <worldbody>
            <light pos="0 0 3" dir="0 0 -1" directional="true"/>
            <geom name="ground" type="plane" size="5 5 0.1" rgba="0.8 0.9 0.8 1" material="groundplane"/>
    """
    
    # 为每个组件创建独立的body
    for i, component in enumerate(components):
        # 计算网格布局位置
        row = i // 4  # 每行4个组件
        col = i % 4
        x_pos = (col - 1.5) * 1.0  # 水平间距1.0
        y_pos = (row - 1.5) * 1.0  # 垂直间距1.0
        
        model_xml += f"""
            <body name="{component['name']}" pos="{x_pos} {y_pos} 0.5">"""
        
        # 添加geoms
        for geom in component['geoms']:
            model_xml += '\n                <geom'
            for key, value in geom['attributes'].items():
                model_xml += f' {key}="{value}"'
            model_xml += '/>'
        
        # 添加joints
        for joint in component['joints']:
            model_xml += '\n                <joint'
            for key, value in joint['attributes'].items():
                model_xml += f' {key}="{value}"'
            model_xml += '/>'
        
        model_xml += '\n            </body>'
    
    model_xml += """
        </worldbody>
    </mujoco>
    """
    
    return model_xml

def main():
    # 加载机器人XML文件
    robot_path = "cassie.xml"
    root = load_and_parse_xml(robot_path)
    
    # 提取机器人组件
    components = extract_robot_components(root)
    
    # 显示组件信息
    print(f"\n=== 发现 {len(components)} 个机器人组件 ===")
    
    for component in components:
        print(f"\n组件: {component['name']}")
        print(f"路径: {component['path']}")
        
        if component['geoms']:
            print("几何体:")
            for geom in component['geoms']:
                for key, value in geom['attributes'].items():
                    print(f"  {key}: {value}")
        
        if component['joints']:
            print("关节:")
            for joint in component['joints']:
                for key, value in joint['attributes'].items():
                    print(f"  {key}: {value}")
    
    # 创建组合XML并保存
    combined_xml = create_combined_xml(components)
    output_file = "robot_components.xml"
    with open(output_file, "w") as f:
        f.write(combined_xml)
    print(f"\n所有组件已保存到 {output_file}")

if __name__ == "__main__":
    main()
