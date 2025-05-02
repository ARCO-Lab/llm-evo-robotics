import mujoco
import mujoco.viewer
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import time
import copy
import os

class CassiePartsViewer:
    def __init__(self, xml_path):
        """初始化CassiePartsViewer并解析XML"""
        self.xml_path = xml_path
        print(f"xml_path: {xml_path}")
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.parts = self._get_body_parts()

        # 确保 compiler 存在并设置 meshdir
        self._ensure_meshdir()

    def _get_body_parts(self):

        root_text = ET.tostring(self.root, encoding='unicode')
        self.save_xml_to_file(root_text, "data/root.xml")

        """提取所有身体部件名称"""
        parts = []
        worldbody = self.root.find('.//worldbody')

        def extract_bodies(element, parts_list):
            for body in element.findall('.//body'):
                parts_list.append(body.get('name'))

        extract_bodies(worldbody, parts)

        part_text = ET.tostring(self.root, encoding='unicode')
        self.save_xml_to_file(part_text, "data/part.xml")
        return parts

    def _ensure_meshdir(self):

        """确保 compiler 存在，并设置 meshdir="assets" """
        compiler = self.root.find('compiler')
       
        if compiler is None:
            # 如果没有 compiler，创建并添加到 root 的最前面
            compiler = ET.Element('compiler')
            compiler.set('meshdir', 'materials/test1/assets')
            self.root.insert(0, compiler)
        else:
            # 如果已有 compiler，更新 meshdir
            compiler.set('meshdir', 'materials/test1/assets')
            compiler.set("texturedir", "materials/test1/assets")
        print(f"compiler: {self.xml_path}")
       
      
        # 保存修改后的 XML
        self.tree.write(self.xml_path)

    def create_single_part_xml(self, part_name):
        """为单个部件创建XML"""
        new_root = ET.Element('mujoco')
        new_root.set('model', f'cassie-{part_name}')

        # 复制必要的 sections
        for child in self.root:
            if child.tag in ['compiler', 'option', 'default', 'asset']:
                new_root.append(copy.deepcopy(child))

        # 创建新的 worldbody
        new_worldbody = ET.SubElement(new_root, 'worldbody')

        # 添加光源
        light = ET.SubElement(new_worldbody, 'light')
        light.set('name', 'spotlight')
        light.set('pos', '0 -1 2')

        # 找到并复制目标部件
        original_body = self.root.find(f'.//body[@name="{part_name}"]')
        if original_body is not None:
            new_body = copy.deepcopy(original_body)
            new_body.set('pos', '0 0 0')  # 重置位置到原点
            new_worldbody.append(new_body)

        return ET.tostring(new_root, encoding='unicode')

    def view_part(self, part_name):
        """查看单个部件"""
        print(f"Viewing part: {part_name}")
        xml_string = self.create_single_part_xml(part_name)
        # print(f"xml_string: {xml_string}")
        # 加载模型
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)

        # 显示模型
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Press Ctrl+C to view next part")
            try:
                while viewer.is_running():
                    step_start = time.time()
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))
            except KeyboardInterrupt:
                print("\nMoving to next part...")

    def view_all_parts(self):
        """查看所有部件"""
        print(f"Total parts to view: {len(self.parts)}")
        for part in self.parts:
            self.view_part(part)

    def create_exploded_view_xml(self):
        """创建分解视图 XML，所有零件平铺展示"""
        new_root = ET.Element('mujoco')
        new_root.set('model', 'cassie-exploded')

      
        

        # 复制必要的 sections
        for child in self.root:
            if child.tag in ['compiler', 'option', 'default', 'asset']:
                new_root.append(copy.deepcopy(child))

       

        # 创建新的 worldbody
        new_worldbody = ET.SubElement(new_root, 'worldbody')
      
        # # 添加光源
        light = ET.SubElement(new_worldbody, 'light')
        light.set('name', 'spotlight')
        light.set('pos', '0 -1 2')

  

        # 计算零件布局
        n_parts = len(self.parts)
        grid_size = int(np.ceil(np.sqrt(n_parts)))  # 计算网格大小
        spacing = 0.5  # 零件之间的间距

        for i, part_name in enumerate(self.parts):
            original_body = self.root.find(f'.//body[@name="{part_name}"]')
            if original_body is not None:
                new_body = copy.deepcopy(original_body)

                # 修改 body 名称，确保唯一
                base_name = new_body.get("name")
                new_name = f"{base_name}_{i}"  # 添加索引，确保唯一
                new_body.set("name", new_name)

                # 确保 body 内部的所有 joint、geom、site 也唯一
                for elem in new_body.iter():
                    if "name" in elem.attrib:
                        elem.set("name", f"{elem.get('name')}_{i}")

                # 计算网格位置
                x_pos = (i % grid_size) * spacing
                y_pos = (i // grid_size) * spacing
                new_body.set("pos", f"{x_pos} {y_pos} 0.1")  # 轻微抬高

                new_worldbody.append(new_body)


        test = ET.tostring(new_root, encoding='unicode')

        self.save_xml_to_file(test, "data/data1.xml")

        # return ET.tostring(new_root, encoding='unicode')

    def view_exploded(self):
        """查看分解视图"""
        xml_string = self.create_exploded_view_xml()
        model = mujoco.MjModel.from_xml_string(xml_string)
        print(f"xml_string: {xml_string}")
        data = mujoco.MjData(model)
        # self.save_xml_to_file(data, "data/data1.xml")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Press Ctrl+C to exit")
            try:
                while viewer.is_running():
                    step_start = time.time()
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))
            except KeyboardInterrupt:
                print("\nExiting...")

    def save_xml_to_file(self, xml_string, output_path):
        """
        将XML字符串保存到文件中
        
        Args:
            xml_string (str): 要保存的XML字符串
            output_path (str): 输出文件路径
            
        Returns:
            None
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 写入文件
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_string)
            print(f"XML已成功保存到: {output_path}")
        except Exception as e:
            print(f"保存XML文件时出错: {str(e)}")


def find_xml_and_assets(start_folder):
    """
    从指定文件夹开始搜索所有的xml文件和assets文件夹
    
    Args:
        start_folder (str): 起始搜索文件夹路径
        
    Returns:
        tuple: (xml_files, assets_folders)
            - xml_files: 所有.xml文件的完整路径列表
            - assets_folders: 所有assets文件夹的完整路径列表
    """
    # 初始化结果列表
    xml_files = []
    assets_folders = []
    
    # 遍历所有子文件夹
    for root, dirs, files in os.walk(start_folder):
        # 查找xml文件
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
                
        # 查找assets文件夹
        if 'assets' in dirs:
            assets_folders.append(os.path.join(root, 'assets'))
            
    return xml_files, assets_folders


def main():
    print("testsetstetse")   
    """主函数，加载多个 XML 并显示"""
    xml_files, assets_folders = find_xml_and_assets("materials")
    print(f"xml_files: {xml_files}")
    print(f"assets_folders: {assets_folders}")

    viewers = []
    for xml_path in xml_files:
        if not Path(xml_path).exists():
            print(f"Error: Cannot find {xml_path}")
            continue

        viewer = CassiePartsViewer(xml_path)
        print(f"Available parts in {xml_path}:", viewer.parts)
        viewers.append(viewer)

    if not viewers:
        print("No valid robot models found")
        return

    # 使用第一个查看器作为主查看器
    main_viewer = viewers[0]

    # 选择是否查看所有部件或分解视图
    while True:
        print("\nOptions:")
        print("1. View all parts one by one")
        print("2. View exploded view")
        print("3. Exit")
        choice = input("Select an option: ")

        if choice == "1":
            main_viewer.view_all_parts()
        elif choice == "2":
            main_viewer.view_exploded()
        elif choice == "3":
            break
        else:
            print("Invalid option, please select again.")

if __name__ == "__main__":
    main()
