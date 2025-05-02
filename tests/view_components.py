import mujoco
import time

def view_robot_components():
    # 加载组件模型
    model = mujoco.MjModel.from_xml_path("robot_components.xml")
    data = mujoco.MjData(model)
    
    # 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n=== 查看机器人组件 ===")
        print("使用鼠标控制视角：")
        print("- 左键拖动：旋转视角")
        print("- 右键拖动：平移视角")
        print("- 滚轮：缩放")
        print("\n按 Ctrl+C 退出查看器")
        
        try:
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
            print("\n退出查看器")

if __name__ == "__main__":
    view_robot_components() 