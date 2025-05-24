import numpy as np
import pybullet as p
from z_axis_wheel_override import test_robot_with_gene, modify_gene_for_z_axis_wheels
from fix_robot_model import create_default_gene, create_constrained_gene, create_diverse_gene

def test_z_axis_wheels():
    """测试Z轴轮子约束功能"""
    print("\n===== 测试Z轴轮子约束功能 =====")
    
    # 1. 测试默认四轮设计（强制所有轮子都沿Z轴旋转）
    print("\n==== 测试默认四轮设计（Z轴轮子） ====")
    default_gene = create_default_gene()
    z_axis_default_gene = modify_gene_for_z_axis_wheels(default_gene)
    test_robot_with_gene(z_axis_default_gene)
    
    # 询问是否继续
    input("\n按Enter键继续测试随机约束设计...")
    
    # 2. 测试随机约束设计（强制所有轮子都沿Z轴旋转）
    print("\n==== 测试随机约束设计（Z轴轮子） ====")
    constrained_gene = create_constrained_gene()
    z_axis_constrained_gene = modify_gene_for_z_axis_wheels(constrained_gene)
    test_robot_with_gene(z_axis_constrained_gene)
    
    # 询问是否继续
    input("\n按Enter键继续测试多样化设计...")
    
    # 3. 测试多样化设计（强制所有轮子都沿Z轴旋转）
    print("\n==== 测试多样化设计（Z轴轮子） ====")
    diverse_gene = create_diverse_gene()
    z_axis_diverse_gene = modify_gene_for_z_axis_wheels(diverse_gene)
    test_robot_with_gene(z_axis_diverse_gene)

if __name__ == "__main__":
    test_z_axis_wheels() 