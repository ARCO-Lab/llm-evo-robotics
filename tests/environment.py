import pybullet as p
import pybullet_data
import numpy as np
from typing import List, Dict, Optional
import time

class ModularRobotEnv:
    def __init__(self, render: bool = True):
        """
        初始化模块化机器人环境
        Initialize modular robot environment
        
        Args:
            render: 是否渲染环境 / Whether to render the environment
        """
        self.render_mode = render
        self.components = {}  # 存储加载的组件 / Store loaded components
        
        # 添加默认相机参数 / Add default camera parameters
        self.camera_distance = 2.0
        self.camera_yaw = 50.0
        self.camera_pitch = -35.0
        self.camera_target = [0.0, 0.0, 0.0]
        
        # 初始化模拟环境 / Initialize simulation environment
        self._init_simulation()
        
    def _init_simulation(self):
        """初始化PyBullet模拟环境 / Initialize PyBullet simulation"""
        if self.render_mode:
            p.connect(p.GUI)  # 图形界面模式 / GUI mode
            # 设置初始相机视角 / Set initial camera view
            self.reset_camera()
        else:
            p.connect(p.DIRECT)  # 无渲染模式 / Headless mode
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")  # 加载地面 / Load ground plane
        
    def add_component(self, component_name: str, urdf_path: str, 
                     position: List[float], orientation: List[float]) -> int:
        """
        添加机器人组件
        Add robot component
        
        Args:
            component_name: 组件名称 / Component name
            urdf_path: URDF文件路径 / Path to URDF file
            position: [x, y, z] 位置 / Position
            orientation: [x, y, z, w] 四元数方向 / Quaternion orientation
            
        Returns:
            组件ID / Component ID
        """
        component_id = p.loadURDF(urdf_path, position, orientation)
        self.components[component_name] = {
            'id': component_id,
            'urdf_path': urdf_path,
            'position': position,
            'orientation': orientation
        }
        return component_id
    
    def remove_component(self, component_name: str):
        """
        移除机器人组件
        Remove robot component
        """
        if component_name in self.components:
            p.removeBody(self.components[component_name]['id'])
            del self.components[component_name]
            
    def get_component_state(self, component_name: str) -> Dict:
        """
        获取组件状态
        Get component state
        """
        if component_name not in self.components:
            raise ValueError(f"Component {component_name} not found")
            
        component_id = self.components[component_name]['id']
        position, orientation = p.getBasePositionAndOrientation(component_id)
        linear_vel, angular_vel = p.getBaseVelocity(component_id)
        
        return {
            'position': position,
            'orientation': orientation,
            'linear_velocity': linear_vel,
            'angular_velocity': angular_vel
        }
    
    def step(self):
        """模拟一步 / Simulate one step"""
        p.stepSimulation()
        
    def reset(self):
        """重置环境 / Reset environment"""
        p.resetSimulation()
        self._init_simulation()
        self.components.clear()
        
    def close(self):
        """关闭环境 / Close environment"""
        p.disconnect()

    def reset_camera(self):
        """
        重置相机到默认视角
        Reset camera to default view
        """
        p.resetDebugVisualizerCamera(
            cameraDistance=self.camera_distance,
            cameraYaw=self.camera_yaw,
            cameraPitch=self.camera_pitch,
            cameraTargetPosition=self.camera_target
        )

    def visualize(self, show_frames: bool = True, show_names: bool = True):
        """
        可视化环境中的组件
        Visualize components in the environment

        Args:
            show_frames: 是否显示坐标系 / Whether to show coordinate frames
            show_names: 是否显示组件名称 / Whether to show component names
        """
        if not self.render_mode:
            return

        # 清除之前的可视化效果 / Clear previous visualization effects
        p.removeAllUserDebugItems()

        for name, component in self.components.items():
            pos = component['position']
            orn = component['orientation']

            if show_frames:
                # 显示坐标系，长度0.2米 / Show coordinate frame, length 0.2 meters
                length = 0.2
                p.addUserDebugLine(pos, [pos[0] + length, pos[1], pos[2]], [1, 0, 0])  # X轴红色 / X-axis red
                p.addUserDebugLine(pos, [pos[0], pos[1] + length, pos[2]], [0, 1, 0])  # Y轴绿色 / Y-axis green
                p.addUserDebugLine(pos, [pos[0], pos[1], pos[2] + length], [0, 0, 1])  # Z轴蓝色 / Z-axis blue

            if show_names:
                # 在组件上方显示名称 / Show name above component
                p.addUserDebugText(
                    text=name,
                    textPosition=[pos[0], pos[1], pos[2] + 0.2],
                    textColorRGB=[1, 1, 1],
                    textSize=1.5
                )

    def adjust_camera(self, distance: float = None, yaw: float = None, 
                     pitch: float = None, target: List[float] = None):
        """
        调整相机视角
        Adjust camera view

        Args:
            distance: 相机距离 / Camera distance
            yaw: 相机偏航角 / Camera yaw
            pitch: 相机俯仰角 / Camera pitch
            target: 相机目标点 / Camera target position
        """
        if not self.render_mode:
            return

        if distance is not None:
            self.camera_distance = distance
        if yaw is not None:
            self.camera_yaw = yaw
        if pitch is not None:
            self.camera_pitch = pitch
        if target is not None:
            self.camera_target = target

        self.reset_camera()

def main():
    """
    演示环境和可视化功能的主函数
    Main function to demonstrate environment and visualization features
    """
    # 创建环境
    env = ModularRobotEnv(render=True)
    
    # 调整相机视角以获得更好的视角
    env.adjust_camera(
        distance=3.0,
        yaw=45,
        pitch=-30,
        target=[0, 0, 0]
    )
    
    # 加载机器人组件
    # 加载一个基座 (使用默认的URDF文件作为示例)
    env.add_component(
        component_name="base",
        urdf_path="r2d2.urdf",  # 使用pybullet自带的模型作为示例
        position=[0, 0, 0.1],
        orientation=[0, 0, 0, 1]
    )
    
    # 加载一个物块 (使用husky作为另一个示例)
    env.add_component(
        component_name="robot2",
        urdf_path="husky/husky.urdf",  # 使用husky机器人代替cube
        position=[1, 0, 0.1],
        orientation=[0, 0, 0, 1]
    )
    
    # 模拟循环
    try:
        while True:
            env.step()
            # 更新可视化
            env.visualize(show_frames=True, show_names=True)
            # 控制模拟速度
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n用户中断模拟 / Simulation interrupted by user")
    finally:
        env.close()

if __name__ == "__main__":
    import time
    main()
