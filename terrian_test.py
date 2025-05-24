import pybullet as p
import pybullet_data as pd
import math
import time
import random

# 连接到PyBullet物理引擎（GUI模式，可视化）
p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

# 选择高度场来源
# 0: 程序化生成
# 1: 从PNG文件导入
# 2: 从CSV/TXT文件导入
heightfieldSource = 1  # 默认使用程序化生成

# 方法1：程序化生成高度场
if heightfieldSource == 0:
    # 设置高度场的尺寸
    numHeightfieldRows = 256
    numHeightfieldColumns = 256
    
    # 创建高度场数据数组（初始全为0）
    heightfieldData = [0] * numHeightfieldRows * numHeightfieldColumns
    
    # 设置随机种子以便结果可重现
    random.seed(10)
    
    # 高度扰动范围
    heightPerturbationRange = 0.05
    
    # 生成随机高度数据
    for j in range(int(numHeightfieldColumns/2)):
        for i in range(int(numHeightfieldRows/2)):
            height = random.uniform(0, heightPerturbationRange)
            # 为了平滑，我们将2x2的区域设置为相同的高度
            heightfieldData[2*i+2*j*numHeightfieldRows] = height
            heightfieldData[2*i+1+2*j*numHeightfieldRows] = height
            heightfieldData[2*i+(2*j+1)*numHeightfieldRows] = height
            heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows] = height
    
    # 创建地形碰撞形状
    terrainShape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[0.05, 0.05, 1],
        heightfieldTextureScaling=(numHeightfieldRows-1)/2,
        heightfieldData=heightfieldData,
        numHeightfieldRows=numHeightfieldRows,
        numHeightfieldColumns=numHeightfieldColumns
    )
    
    # 创建地形多体对象
    terrain = p.createMultiBody(0, terrainShape)
    
    # 重置地形位置和方向
    p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])

# 方法2：从PNG图像文件导入
elif heightfieldSource == 1:
    # 从PNG文件创建地形
    # 注意：需要提供有效的PNG文件路径
    terrainShape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[0.1, 0.1, 24],
        fileName="heightmaps/your_height_map.png"  # 替换为你的高度图文件
    )
    
    # 可选：加载纹理
    textureId = p.loadTexture("heightmaps/your_texture.png")  # 替换为你的纹理文件
    
    # 创建地形多体对象
    terrain = p.createMultiBody(0, terrainShape)
    
    # 应用纹理
    p.changeVisualShape(terrain, -1, textureUniqueId=textureId)
    
    # 重置地形位置和方向
    p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])

# 方法3：从CSV/TXT文件导入
elif heightfieldSource == 2:
    # 从CSV/TXT文件创建地形
    terrainShape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[0.5, 0.5, 2.5],
        fileName="heightmaps/your_height_data.txt",  # 替换为你的高度数据文件
        heightfieldTextureScaling=128
    )
    
    # 创建地形多体对象
    terrain = p.createMultiBody(0, terrainShape)
    
    # 重置地形位置和方向
    p.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])

# 设置地形颜色（可选）
p.changeVisualShape(terrain, -1, rgbaColor=[1, 1, 1, 1])

# 启用渲染
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

# 添加一些物体以便测试地形碰撞
# 创建球体碰撞形状
sphereRadius = 0.05
colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)

# 创建多个球体并让它们落在地形上
for i in range(3):
    for j in range(3):
        for k in range(3):
            basePosition = [
                i * 5 * sphereRadius, 
                j * 5 * sphereRadius, 
                1 + k * 5 * sphereRadius + 1
            ]
            baseOrientation = [0, 0, 0, 1]
            # 修正：使用位置参数而非关键字参数
            sphereUid = p.createMultiBody(
                1,  # mass
                colSphereId,  # collisionShapeIndex
                -1,  # visualShapeIndex
                basePosition,  # basePosition
                baseOrientation  # baseOrientation
            )

# 设置重力
p.setGravity(0, 0, -10)

# 设置仿真参数
timeStep = 1./240.
p.setTimeStep(timeStep)

# 仿真循环
while True:
    p.stepSimulation()
    time.sleep(timeStep)