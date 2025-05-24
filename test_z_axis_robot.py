import numpy as np
import pybullet as p
from fix_robot_model import test_robot_with_gene

def create_z_axis_wheels_gene():
    """创建一个所有轮子都绕Z轴旋转的机器人基因"""
    gene = np.zeros(100)  # 创建足够长的基因数组
    
    # 设置连杆数量为5 (车身+4轮)
    gene[0] = 0.4  # 对应5个连杆
    
    # 车身参数 - 设置为更宽扁的形状增加稳定性
    gene[1] = 0.1  # 形状 - 盒子
    gene[2] = 0.7  # 尺寸X - 较大
    gene[3] = 0.8  # 尺寸Y - 更宽
    gene[4] = 0.2  # 尺寸Z - 扁平
    gene[5] = 0.1  # 材质 - 金属
    gene[6] = 0.5  # 其他参数保持中等
    
    # 为四个轮子设置更合理的位置参数
    # 轮子位置分别在车身四个角落
    wheel_positions = [
        [0.2, 0.2, 0.0],   # 右前
        [0.2, -0.2, 0.0],  # 左前
        [-0.2, 0.2, 0.0],  # 右后
        [-0.2, -0.2, 0.0]  # 左后
    ]
    
    # 为每个轮子参数设置首个位置的索引
    wheel_indices = [7, 20, 33, 46]
    
    # 对每个轮子进行单独设置
    for i, idx in enumerate(wheel_indices):
        # 基本轮子参数
        gene[idx] = 0.3    # 关节类型 - 旋转关节
        gene[idx+1] = 0.9  # 有电机 - 高概率
        gene[idx+2] = 0.4  # 形状 - 圆柱形
        gene[idx+3] = 0.9  # 是轮子标志 - 确保识别为轮子
        gene[idx+4] = 0.1  # 轮子类型 - 普通轮
        gene[idx+5] = 0.5  # 轮半径 - 中等
        gene[idx+6] = 0.4  # 轮宽度 - 适中
        gene[idx+7] = 0.0  # 不使用
        gene[idx+8] = 0.8  # 材质 - 橡胶
        
        # 轮子位置 - 转换到0-1范围
        pos = wheel_positions[i]
        gene[idx+9] = 0.5 + pos[0] * 0.5   # X轴位置
        gene[idx+10] = 0.5 + pos[1] * 0.5  # Y轴位置
        
        # 关键部分：设置旋转轴 - 确保Z轴是主要旋转轴
        gene[idx+9] = 0.05   # X轴分量 - 很小
        gene[idx+10] = 0.05  # Y轴分量 - 很小
        gene[idx+11] = 0.95  # Z轴分量 - 很大，确保为主要旋转轴
        
        gene[idx+12] = 0.3  # 关节阻尼 - 较低，减少摩擦
    
    return gene

if __name__ == "__main__":
    print("创建并测试Z轴旋转轮子的机器人设计")
    z_axis_gene = create_z_axis_wheels_gene()
    test_robot_with_gene(z_axis_gene) 