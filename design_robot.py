import numpy as np
import os
import matplotlib.pyplot as plt
from robot_evolution_fixed import decode_gene, generate_urdf, simulate_robot_multi
from fix_robot_model import test_robot_with_gene

def create_specialized_gene(design_type="balanced"):
    """
    创建具有特定特性的机器人基因
    
    参数:
    design_type - 设计类型:
        "balanced" - 平衡型四轮设计
        "speed" - 注重速度的轻型设计
        "stability" - 注重稳定性的宽轮距设计
        "terrain" - 适应复杂地形的大轮设计
        "creative" - 创新型混合设计
    """
    gene = np.zeros(100)  # 创建基因数组
    
    if design_type == "balanced":
        # 平衡型四轮设计 - 正常的底盘和轮子尺寸
        gene[0] = 0.5  # 5个连杆 (1个底盘 + 4个轮子)
        
        # 底盘 - 中等尺寸矩形
        gene[1] = 0.1  # 形状 - 盒子
        gene[2] = 0.6  # 尺寸X - 中等长度
        gene[3] = 0.7  # 尺寸Y - 中等宽度
        gene[4] = 0.3  # 尺寸Z - 中等高度
        gene[5] = 0.0  # 材质 - 金属
        
        # 四个轮子 - 中等尺寸
        wheel_params = [
            # 左前轮
            [0.3, 1, 0.4, 1, 0, 0.5, 0.5, 0.3, 0.9, 0, 1, 0, 0.5],
            # 右前轮
            [0.3, 1, 0.4, 1, 0, 0.5, 0.5, 0.3, 0.9, 0, 1, 0, 0.5],
            # 左后轮
            [0.3, 1, 0.4, 1, 0, 0.5, 0.5, 0.3, 0.9, 0, 1, 0, 0.5],
            # 右后轮
            [0.3, 1, 0.4, 1, 0, 0.5, 0.5, 0.3, 0.9, 0, 1, 0, 0.5]
        ]
    
    elif design_type == "speed":
        # 速度型设计 - 窄小底盘，大轮子
        gene[0] = 0.5  # 5个连杆
        
        # 底盘 - 细长形
        gene[1] = 0.1  # 形状 - 盒子
        gene[2] = 0.8  # 尺寸X - 较长
        gene[3] = 0.4  # 尺寸Y - 较窄
        gene[4] = 0.2  # 尺寸Z - 较扁
        gene[5] = 0.0  # 材质 - 金属
        
        # 四个轮子 - 大尺寸轮子
        wheel_params = [
            # 左前轮
            [0.3, 1, 0.4, 1, 0, 0.8, 0.8, 0.3, 0.9, 0, 1, 0, 0.3],  # 大轮子
            # 右前轮
            [0.3, 1, 0.4, 1, 0, 0.8, 0.8, 0.3, 0.9, 0, 1, 0, 0.3],
            # 左后轮
            [0.3, 1, 0.4, 1, 0, 0.8, 0.8, 0.3, 0.9, 0, 1, 0, 0.3],
            # 右后轮
            [0.3, 1, 0.4, 1, 0, 0.8, 0.8, 0.3, 0.9, 0, 1, 0, 0.3]
        ]
    
    elif design_type == "stability":
        # 稳定型设计 - 宽底盘，宽轮距
        gene[0] = 0.5  # 5个连杆
        
        # 底盘 - 宽平形
        gene[1] = 0.1  # 形状 - 盒子
        gene[2] = 0.6  # 尺寸X - 中等长度
        gene[3] = 0.9  # 尺寸Y - 非常宽
        gene[4] = 0.2  # 尺寸Z - a扁平
        gene[5] = 0.0  # 材质 - 金属
        
        # 四个轮子 - 宽轮子
        wheel_params = [
            # 左前轮
            [0.3, 1, 0.4, 1, 0, 0.5, 0.5, 0.6, 0.9, 0, 1, 0, 0.7],  # 宽轮子
            # 右前轮
            [0.3, 1, 0.4, 1, 0, 0.5, 0.5, 0.6, 0.9, 0, 1, 0, 0.7],
            # 左后轮
            [0.3, 1, 0.4, 1, 0, 0.5, 0.5, 0.6, 0.9, 0, 1, 0, 0.7],
            # 右后轮
            [0.3, 1, 0.4, 1, 0, 0.5, 0.5, 0.6, 0.9, 0, 1, 0, 0.7]
        ]
    
    elif design_type == "terrain":
        # 越野型设计 - 高底盘，大轮子
        gene[0] = 0.5  # 5个连杆
        
        # 底盘 - 高底盘
        gene[1] = 0.1  # 形状 - 盒子
        gene[2] = 0.6  # 尺寸X - 中等长度
        gene[3] = 0.6  # 尺寸Y - 中等宽度
        gene[4] = 0.4  # 尺寸Z - 较高
        gene[5] = 0.0  # 材质 - 金属
        
        # 四个轮子 - 大而宽的轮子
        wheel_params = [
            # 左前轮
            [0.3, 1, 0.4, 1, 0, 0.9, 0.9, 0.6, 0.9, 0, 1, 0, 0.3],  # 大而宽的轮子
            # 右前轮
            [0.3, 1, 0.4, 1, 0, 0.9, 0.9, 0.6, 0.9, 0, 1, 0, 0.3],
            # 左后轮
            [0.3, 1, 0.4, 1, 0, 0.9, 0.9, 0.6, 0.9, 0, 1, 0, 0.3],
            # 右后轮
            [0.3, 1, 0.4, 1, 0, 0.9, 0.9, 0.6, 0.9, 0, 1, 0, 0.3]
        ]
    
    elif design_type == "creative":
        # 创新型混合设计 - 六轮设计
        gene[0] = 0.7  # 7个连杆 (1个底盘 + 6个轮子)
        
        # 底盘 - 较长底盘
        gene[1] = 0.1  # 形状 - 盒子
        gene[2] = 0.9  # 尺寸X - 很长
        gene[3] = 0.6  # 尺寸Y - 中等宽度
        gene[4] = 0.3  # 尺寸Z - 中等高度
        gene[5] = 0.0  # 材质 - 金属
        
        # 六个轮子 - 中前大后小
        wheel_params = [
            # 左前轮
            [0.3, 1, 0.4, 1, 0, 0.6, 0.6, 0.4, 0.9, 0, 1, 0, 0.5],
            # 右前轮
            [0.3, 1, 0.4, 1, 0, 0.6, 0.6, 0.4, 0.9, 0, 1, 0, 0.5],
            # 左中轮
            [0.3, 1, 0.4, 1, 0, 0.8, 0.8, 0.4, 0.9, 0, 1, 0, 0.3],
            # 右中轮
            [0.3, 1, 0.4, 1, 0, 0.8, 0.8, 0.4, 0.9, 0, 1, 0, 0.3],
            # 左后轮
            [0.3, 1, 0.4, 1, 0, 0.4, 0.4, 0.3, 0.9, 0, 1, 0, 0.6],
            # 右后轮
            [0.3, 1, 0.4, 1, 0, 0.4, 0.4, 0.3, 0.9, 0, 1, 0, 0.6]
        ]
    
    # 填充轮子参数
    for i, params in enumerate(wheel_params):
        start_idx = 7 + i * 13
        if start_idx + len(params) <= len(gene):  # 确保不超出基因长度
            gene[start_idx:start_idx+len(params)] = params
    
    return gene

def evaluate_designs():
    """评估不同设计类型的性能"""
    design_types = ["balanced", "speed", "stability", "terrain", "creative"]
    results = {}
    
    for design_type in design_types:
        print(f"\n评估 {design_type} 设计...")
        gene = create_specialized_gene(design_type)
        robot_config = decode_gene(gene)
        
        # 生成URDF文件用于查看
        urdf = generate_urdf(robot_config)
        with open(f"{design_type}_robot.urdf", "w") as f:
            f.write(urdf)
        
        # 模拟并评估机器人
        fitness = simulate_robot_multi(robot_config, gui=False, sim_time=10.0)
        results[design_type] = fitness
        
        print(f"  - 距离: {fitness[0]:.2f}")
        print(f"  - 路径直线性: {fitness[1]:.2f}")
        print(f"  - 稳定性: {fitness[2]:.2f}")
        print(f"  - 能量消耗: {fitness[3]:.2f}")
    
    # 可视化结果
    plot_results(results)
    return results

def plot_results(results):
    """绘制不同设计的性能对比图"""
    metrics = ["距离", "路径直线性", "稳定性", "能量消耗"]
    design_types = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[dt][i] for dt in design_types]
        axes[i].bar(design_types, values)
        axes[i].set_title(metric)
        axes[i].set_ylabel("值")
        axes[i].set_xticklabels(design_types, rotation=45)
    
    plt.tight_layout()
    plt.savefig("design_comparison.png")
    plt.close()

def main():
    print("机器人设计生成器")
    print("1. 创建平衡型四轮设计")
    print("2. 创建速度型设计")
    print("3. 创建稳定型设计")
    print("4. 创建越野型设计")
    print("5. 创建创新型混合设计")
    print("6. 评估所有设计")
    
    choice = input("\n请选择(1-6): ")
    
    if choice == '1':
        gene = create_specialized_gene("balanced")
        test_robot_with_gene(gene)
    elif choice == '2':
        gene = create_specialized_gene("speed")
        test_robot_with_gene(gene)
    elif choice == '3':
        gene = create_specialized_gene("stability")
        test_robot_with_gene(gene)
    elif choice == '4':
        gene = create_specialized_gene("terrain")
        test_robot_with_gene(gene)
    elif choice == '5':
        gene = create_specialized_gene("creative")
        test_robot_with_gene(gene)
    elif choice == '6':
        evaluate_designs()
    else:
        print("无效选择，使用平衡型设计")
        gene = create_specialized_gene("balanced")
        test_robot_with_gene(gene)

if __name__ == "__main__":
    main() 