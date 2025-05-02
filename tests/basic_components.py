import mujoco
import mujoco.viewer
import numpy as np
import os

def create_simple_model():
    # 创建一个简单的 XML 模型字符串
    model_xml = """
    <mujoco>
        <worldbody>
            <!-- 地面 -->
            <geom name="ground" type="plane" size="2 2 0.1" rgba="0.8 0.9 0.8 1"/>
            
            <!-- 一个立方体 -->
            <body name="cube" pos="0 0 1">
                <joint name="free_joint" type="free"/>
                <geom name="cube_geom" type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
            </body>
            
            <!-- 一个球体 -->
            <body name="sphere" pos="0.5 0 1">
                <joint name="ball_joint" type="ball"/>
                <geom name="sphere_geom" type="sphere" size="0.1" rgba="0 0 1 1"/>
            </body>
            
            <!-- 一个圆柱体 -->
            <body name="cylinder" pos="-0.5 0 1">
                <joint name="hinge" type="hinge" axis="0 1 0"/>
                <geom name="cylinder_geom" type="cylinder" size="0.1 0.1" rgba="0 1 0 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    
    # 将 XML 字符串保存为临时文件
    with open("temp_model.xml", "w") as f:
        f.write(model_xml)
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path("temp_model.xml")
    data = mujoco.MjData(model)
    
    # 删除临时文件
    os.remove("temp_model.xml")
    
    return model, data

def main():
    # 创建模型和数据
    model, data = create_simple_model()
    
    # 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 模拟循环
        while viewer.is_running():
            # 步进模拟
            mujoco.mj_step(model, data)
            
            # 更新查看器
            viewer.sync()

if __name__ == "__main__":
    main()
