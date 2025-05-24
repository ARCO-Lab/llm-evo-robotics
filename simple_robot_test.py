import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import tempfile
from robot_evolution_fixed import decode_gene, generate_urdf, visualize_robot


def test_simple_robot():
    print("\n测试简单的四轮机器人配置")
    
    # 创建一个简单的四轮机器人基因
    # 基本的基因结构：
    # - x[0]: 连杆数量 (设为0.5表示约4-5个连杆)
    # - x[1:7]: 车身参数 (形状、尺寸x/y/z、材质等)
    # - 其余: 每个连杆的参数
    
    # 创建一个简单的手动配置 - 四轮机器人
    gene = np.zeros(100)  # 创建足够长的基因数组
    
    # 设置连杆数量为5 (车身+4轮)
    gene[0] = 0.5  # 约4-5个连杆
    
    # 车身参数
    gene[1] = 0.1  # 形状 - 盒子
    gene[2] = 0.6  # 尺寸X - 较大
    gene[3] = 0.8  # 尺寸Y - 较宽
    gene[4] = 0.2  # 尺寸Z - 扁平
    gene[5] = 0.0  # 材质 - 金属
    
    # 为四个轮子设置参数 - 每个轮子占13个参数
    wheel_params = [
        # 连杆1 - 左前轮
        [0.3, 1, 0.4, 1, 0, 0.5, 0.5, 0.3, 0.9, 0, 1, 0, 0.5],
        # 连杆2 - 右前轮
        [0.3, 1, 0.4, 1, 0, 0.5, 0.5, 0.3, 0.9, 0, 1, 0, 0.5],
        # 连杆3 - 左后轮
        [0.3, 1, 0.4, 1, 0, 0.5, 0.5, 0.3, 0.9, 0, 1, 0, 0.5],
        # 连杆4 - 右后轮
        [0.3, 1, 0.4, 1, 0, 0.5, 0.5, 0.3, 0.9, 0, 1, 0, 0.5]
    ]
    
    # 填充轮子参数
    for i, params in enumerate(wheel_params):
        start_idx = 7 + i * 13
        gene[start_idx:start_idx+13] = params
    
    # 解码基因并生成机器人配置
    robot_config = decode_gene(gene)
    
    # 打印机器人配置信息
    print("\n机器人配置:")
    print(f"- 连杆数量: {robot_config['num_links']}")
    print(f"- 轮子数量: {sum(robot_config['is_wheel'][1:])}")
    print(f"- 车身尺寸: {robot_config['link_sizes'][0]}")
    
    # 生成URDF
    urdf = generate_urdf(robot_config)
    with open("simple_robot.urdf", "w") as f:
        f.write(urdf)
    print("\n已生成机器人URDF文件: simple_robot.urdf")
    
    # 可视化并模拟机器人
    print("\n开始模拟机器人...")
    visualize_robot("simple_robot.urdf", sim_time=20.0, terrain_type="flat")


if __name__ == "__main__":
    test_simple_robot() 