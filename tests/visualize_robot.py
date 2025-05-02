import mujoco
import mujoco.viewer
import time

def visualize_robot():
    # 加载 XML 文件
    model = mujoco.MjModel.from_xml_path('robot_components.xml')
    data = mujoco.MjData(model)

    # 创建查看器并显示模型
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 更新查看器
        viewer.cam.distance = 5.0  # 设置相机距离
        viewer.cam.azimuth = 120   # 设置方位角
        viewer.cam.elevation = -20  # 设置仰角

        # 保持窗口打开
        while viewer.is_running():
            step_start = time.time()
            
            # 步进模拟
            mujoco.mj_step(model, data)
            
            # 更新查看器
            viewer.sync()
            
            # 控制模拟速度
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    visualize_robot() 