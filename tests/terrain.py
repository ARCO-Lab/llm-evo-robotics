import pybullet as p
import pybullet_data
import numpy as np
import time
from typing import List, Tuple

class TerrainGenerator:
    def __init__(self, render: bool = True):
        """
        初始化地形生成器
        Initialize terrain generator
        """
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 设置相机
        p.resetDebugVisualizerCamera(
            cameraDistance=15.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0]
        )

    def create_flat_terrain(self, size: Tuple[float, float] = (10, 10)):
        """
        创建平坦地形
        Create flat terrain
        
        Args:
            size: (length, width) 地形大小
        """
        shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[size[0]/2, size[1]/2, 0.1]
        )
        
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape,
            basePosition=[0, 0, -0.1],
            baseOrientation=[0, 0, 0, 1]
        )

    def create_slope_terrain(self, 
                           size: Tuple[float, float] = (10, 10),
                           angle: float = 15.0):
        """
        创建斜坡地形
        Create slope terrain
        
        Args:
            size: (length, width) 地形大小
            angle: 斜坡角度(度)
        """
        # 将角度转换为弧度
        angle_rad = np.deg2rad(angle)
        length, width = size
        height = length * np.sin(angle_rad)
        
        # 方法1：使用两个盒子组合创建斜坡
        # 创建底座
        base_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[length/2, width/2, 0.1]
        )
        
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=base_shape,
            basePosition=[0, 0, -0.1],
            baseOrientation=[0, 0, 0, 1]
        )
        
        # 创建斜坡部分
        slope_length = length
        slope_height = height
        slope_width = width
        
        # 计算斜坡的旋转角度和位置
        slope_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[slope_length/2, slope_width/2, 0.1]
        )
        
        # 计算斜坡的位置和方向
        slope_position = [0, 0, height/2]
        slope_orientation = p.getQuaternionFromEuler([0, -angle_rad, 0])
        
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=slope_shape,
            basePosition=slope_position,
            baseOrientation=slope_orientation
        )

    def create_random_terrain(self, 
                            size: Tuple[float, float] = (10, 10),
                            resolution: int = 20,
                            height_range: Tuple[float, float] = (-0.5, 0.5)):
        """
        创建随机凹凸地形
        Create random bumpy terrain
        
        Args:
            size: (length, width) 地形大小
            resolution: 网格分辨率
            height_range: (min_height, max_height) 高度范围
        """
        length, width = size
        
        # 创建高度场数据
        # 注意：heightfieldData需要是一个一维数组
        heightfieldData = np.zeros(resolution * resolution)
        for i in range(resolution):
            for j in range(resolution):
                # 使用柏林噪声或其他方法生成更自然的地形
                height = np.random.uniform(height_range[0], height_range[1])
                heightfieldData[i * resolution + j] = height
        
        # 创建高度场形状
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[length/resolution, width/resolution, 1],
            heightfieldData=heightfieldData.tolist(),  # 转换为列表
            numHeightfieldRows=resolution,
            numHeightfieldColumns=resolution
        )
        
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            basePosition=[-length/2, -width/2, 0],
            baseOrientation=[0, 0, 0, 1]
        )

    def add_test_object(self, position: List[float]):
        """
        添加测试物体（球体）
        Add test object (sphere)
        """
        sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        sphere_body = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=sphere_shape,
            basePosition=position,
        )
        return sphere_body

    def close(self):
        """关闭环境"""
        p.disconnect(self.client)

def main():
    """测试不同地形"""
    # 创建单个环境来依次展示不同地形
    terrain = TerrainGenerator(render=True)
    
    try:
        # 1. 平地
        print("\n创建平地地形...")
        terrain.create_flat_terrain()
        terrain.add_test_object([0, 0, 3])
        
        # 模拟一段时间
        for _ in range(300):  # 模拟3秒
            p.stepSimulation()
            time.sleep(0.01)
            
        # 重置环境
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 2. 斜坡
        print("\n创建斜坡地形...")
        terrain.create_slope_terrain(angle=15)
        terrain.add_test_object([0, 0, 3])
        
        # 模拟一段时间
        for _ in range(300):  # 模拟3秒
            p.stepSimulation()
            time.sleep(0.01)
            
        # 重置环境
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # 3. 随机地形
        print("\n创建随机地形...")
        terrain.create_random_terrain()
        terrain.add_test_object([0, 0, 3])
        
        # 模拟一段时间
        for _ in range(300):  # 模拟3秒
            p.stepSimulation()
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n用户中断模拟")
    finally:
        terrain.close()

if __name__ == "__main__":
    main() 