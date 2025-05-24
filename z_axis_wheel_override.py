import numpy as np
import pybullet as p
from robot_evolution_fixed import decode_gene as original_decode_gene
from robot_evolution_fixed import generate_urdf

# 重写decode_gene函数，确保所有轮子只能沿Z轴旋转
def z_axis_wheels_decode_gene(x, max_links=8):
    """
    修改后的基因解码函数，确保所有轮子只能沿Z轴旋转
    这个函数基于原始的decode_gene函数，但强制将所有轮子的旋转轴设为Z轴
    """
    # 先使用原始函数解码基因
    robot_config = original_decode_gene(x, max_links)
    
    # 修改所有轮子的旋转轴为Z轴
    for i in range(robot_config['num_links']):
        if i > 0 and robot_config['is_wheel'][i]:
            # 如果是轮子，强制设置为Z轴旋转
            robot_config['joint_axes'][i] = [0, 0, 1]
    
    return robot_config

# 修改基因生成函数，确保轮子的旋转轴参数始终指向Z轴
def modify_gene_for_z_axis_wheels(gene):
    """修改基因，确保所有潜在轮子的旋转轴参数都指向Z轴"""
    # 获取可能的轮子数量
    num_links = int(gene[0] * 8) + 1
    num_links = max(4, min(num_links, 8))
    
    # 遍历所有可能是轮子的连杆基因位置
    for i in range(1, num_links):
        idx = 7 + (i-1) * 13  # 计算基因起始位置
        
        # 检查是否超出基因长度
        if idx + 12 >= len(gene):
            continue
            
        # 检查是否可能是轮子
        is_wheel_val = gene[idx+3] > 0.5
        
        if is_wheel_val:
            # 是轮子，则将旋转轴设置为Z轴
            gene[idx+9] = 0.0   # X轴分量为0
            gene[idx+10] = 0.0  # Y轴分量为0
            gene[idx+11] = 1.0  # Z轴分量为1
    
    return gene

# 自定义生成URDF的函数，确保轮子使用Z轴旋转
def generate_z_axis_urdf(robot_config):
    """自定义URDF生成函数，确保轮子使用Z轴旋转"""
    # 先修改所有轮子的旋转轴为Z轴
    for i in range(robot_config['num_links']):
        if i > 0 and robot_config['is_wheel'][i]:
            robot_config['joint_axes'][i] = [0, 0, 1]
    
    # 然后使用原始函数生成URDF
    return generate_urdf(robot_config)

# 装饰原始函数以确保Z轴约束
def create_z_axis_constrained_gene(original_gene_func):
    """装饰器函数，确保任何基因生成函数都会产生Z轴轮子"""
    def wrapper(*args, **kwargs):
        # 获取原始基因
        gene = original_gene_func(*args, **kwargs)
        # 应用Z轴约束
        return modify_gene_for_z_axis_wheels(gene)
    return wrapper

# 主要测试功能
def test_robot_with_gene(gene=None):
    """测试使用基因参数生成的机器人，确保轮子只沿Z轴旋转"""
    from fix_robot_model import test_robot_with_gene as original_test_func
    
    # 如果提供了基因，确保它符合Z轴约束
    if gene is not None:
        gene = modify_gene_for_z_axis_wheels(gene)
    
    # 创建一个修补过的测试函数
    def patched_test_func(test_gene=None):
        """内部修补函数，确保使用Z轴轮子"""
        import robot_evolution_fixed
        import fix_robot_model
        
        # 保存原始函数
        original_decode = robot_evolution_fixed.decode_gene
        original_generate = robot_evolution_fixed.generate_urdf
        
        try:
            # 替换函数
            robot_evolution_fixed.decode_gene = z_axis_wheels_decode_gene
            robot_evolution_fixed.generate_urdf = generate_z_axis_urdf
            
            # 调用原始测试函数
            original_test_func(test_gene)
        finally:
            # 恢复原始函数
            robot_evolution_fixed.decode_gene = original_decode
            robot_evolution_fixed.generate_urdf = original_generate
    
    # 使用修补后的测试函数
    patched_test_func(gene)

# 导出修改后的函数，用于遗传算法优化
def run_z_axis_genetic_optimization():
    """运行只使用Z轴轮子的遗传算法优化"""
    from fix_robot_model import run_genetic_optimization, create_constrained_gene, create_diverse_gene, create_default_gene
    
    # 修改基因生成函数，确保Z轴轮子
    z_constrained_gene = create_z_axis_constrained_gene(create_constrained_gene)
    z_diverse_gene = create_z_axis_constrained_gene(create_diverse_gene)
    z_default_gene = create_z_axis_constrained_gene(create_default_gene)
    
    # 替换原始函数
    import fix_robot_model
    import robot_evolution_fixed
    
    # 保存原始函数
    original_constrained = fix_robot_model.create_constrained_gene
    original_diverse = fix_robot_model.create_diverse_gene
    original_default = fix_robot_model.create_default_gene
    original_decode = robot_evolution_fixed.decode_gene
    original_generate = robot_evolution_fixed.generate_urdf
    
    try:
        # 替换函数
        fix_robot_model.create_constrained_gene = z_constrained_gene
        fix_robot_model.create_diverse_gene = z_diverse_gene
        fix_robot_model.create_default_gene = z_default_gene
        robot_evolution_fixed.decode_gene = z_axis_wheels_decode_gene
        robot_evolution_fixed.generate_urdf = generate_z_axis_urdf
        
        # 运行遗传算法优化
        print("运行Z轴轮子约束的遗传算法优化...")
        
        # 获取用户输入
        pop_size = int(input("请输入种群大小 (建议5-20): "))
        n_gen = int(input("请输入进化代数 (建议3-10): "))
        print_verbose = input("是否打印详细结构信息? (y/n): ").lower() == 'y'
        pause_after_eval = input("是否在每次评估后暂停? (y/n): ").lower() == 'y'
        save_designs = input("是否保存所有进化出的设计? (y/n): ").lower() == 'y'
        
        # 运行优化
        run_genetic_optimization(
            pop_size=pop_size, 
            n_gen=n_gen, 
            use_gui=True,
            use_constraints=True, 
            verbose=print_verbose,
            pause_after_eval=pause_after_eval,
            diverse_mode=True, 
            save_designs=save_designs
        )
    finally:
        # 恢复原始函数
        fix_robot_model.create_constrained_gene = original_constrained
        fix_robot_model.create_diverse_gene = original_diverse
        fix_robot_model.create_default_gene = original_default
        robot_evolution_fixed.decode_gene = original_decode
        robot_evolution_fixed.generate_urdf = original_generate

if __name__ == "__main__":
    # 运行Z轴遗传算法优化
    run_z_axis_genetic_optimization() 