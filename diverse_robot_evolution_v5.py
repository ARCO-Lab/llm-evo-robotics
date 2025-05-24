
import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import datetime
import json
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover as SBX
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize

# 导入关键函数
from robot_evolution_test import decode_gene, generate_urdf, simulate_robot_multi
from robot_evolution_test import update_adaptation_state

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from visualize_robot_structure import visualize_robot_structure


    

def fix_prismatic_joints(robot_config):
    """修复棱柱关节的限制问题"""
    for i in range(robot_config['num_links']):
        if robot_config['joint_types'][i] == p.JOINT_PRISMATIC:
            limits = robot_config['joint_limits'][i]
            if (limits[0] >= limits[1]) or (limits[0] == 0 and limits[1] == 0):
                robot_config['joint_limits'][i] = [-0.5, 0.5]
            if i > 0 and robot_config['is_wheel'][i]:
                robot_config['joint_types'][i] = p.JOINT_REVOLUTE
    return robot_config


def fix_connection_structure(robot_config, verbose=False):
    """修复零件连接结构问题，支持树形结构"""
    num_links = robot_config['num_links']
    if num_links <= 1:
        return robot_config
    
    # 检查是否存在父节点索引数组
    if 'parent_indices' not in robot_config:
        # 创建默认的父节点索引 - 所有节点连接到底座
        robot_config['parent_indices'] = [-1] + [0] * (num_links - 1)
    
    # 确保父节点索引有效
    for i in range(1, num_links):  # 跳过底座
        parent_idx = robot_config['parent_indices'][i]
        # 如果父节点索引无效，连接到底座
        if parent_idx < 0 or parent_idx >= i:  # 父节点必须在当前节点之前
            robot_config['parent_indices'][i] = 0
    
    # 检查是否存在连接点位置数组
    if 'connection_points' not in robot_config:
        robot_config['connection_points'] = [0.5] * num_links
    
    # 确保连接点位置数组长度足够
    while len(robot_config['connection_points']) < num_links:
        robot_config['connection_points'].append(0.5)
    
    # 检查是否存在节点深度数组
    if 'node_depths' not in robot_config:
        # 计算节点深度
        robot_config['node_depths'] = [0]  # 底座深度为0
        for i in range(1, num_links):
            parent_idx = robot_config['parent_indices'][i]
            if parent_idx >= 0 and parent_idx < i:
                robot_config['node_depths'].append(robot_config['node_depths'][parent_idx] + 1)
            else:
                robot_config['node_depths'].append(1)  # 默认深度为1
    
    # 确保节点深度数组长度足够
    while len(robot_config['node_depths']) < num_links:
        robot_config['node_depths'].append(1)
    
    # 如果有joint_positions字段但它与树结构不兼容，移除它
    if 'joint_positions' in robot_config:
        # 检查是否已经通过树结构计算了位置
        has_tree_positions = False
        
        if verbose:
            print("移除joint_positions字段，将通过树结构动态计算位置")
        
        robot_config.pop('joint_positions', None)
    
    return robot_config

def check_connection_quality(robot_config, verbose=False):
    """检查机器人连接质量"""
    num_links = robot_config['num_links']
    if num_links <= 1:
        return True, ""
    
    if 'parent_indices' not in robot_config or 'joint_positions' not in robot_config:
        return False, "缺少parent_indices或joint_positions参数"
    
    # 为简洁起见，省略具体计算过程
    return True, ""

class RobotDesignProblem(Problem):
    """机器人设计多目标优化问题"""
    
    def __init__(self, n_var=100, use_gui=False, verbose=False, pause_after_eval=False, add_diversity=True):
        # 定义约束条件数量
        n_constraints = 4
        
        super().__init__(
            n_var=n_var,
            n_obj=5 if add_diversity else 4,
            n_constr=n_constraints,
            xl=np.zeros(n_var),
            xu=np.ones(n_var)
        )
        
        self.use_gui = use_gui
        self.verbose = verbose
        self.pause_after_eval = pause_after_eval
        self.add_diversity = add_diversity
        self.evaluated_designs = []
        
        self.min_stability = 0.75
        self.max_energy = 2000
        self.connection_quality_threshold = 0.8

        self.structure_types = {
            'wheeled': 0, 'legged': 0, 'hybrid': 0, 'other': 0
        }
        
        self.current_generation = 0
        self.total_generations = 0
        self.total_populations = 0
        
        # 添加自适应状态
        self.adaptation_state = None
        self.adaptation_history = []

    def _evaluate(self, X, out, *args, **kwargs):
        """评估机器人设计的适应度和约束条件"""
        n_individuals = X.shape[0]
        F = np.zeros((n_individuals, self.n_obj))
        G = np.zeros((n_individuals, self.n_constr))
        
        current_population_types = {
            'wheeled': 0, 'legged': 0, 'hybrid': 0, 'other': 0
        }
        
        # 打印当前进化状态
        print(f"\n当前进化状态:")
        print(f"- 当前代数: {self.current_generation + 1}/{self.total_generations}")
        print(f"- 当前种群: {n_individuals} 个个体")
        
        # 创建当前代的评估设计列表，用于更新自适应状态
        generation_designs = []
        
        for i in range(n_individuals):
            gene = X[i, :]
            print(f"\n评估个体 {i+1}/{n_individuals}")
            
            # 使用自适应状态解码基因
            robot_config = decode_gene(gene, adaptation_state=self.adaptation_state)
            robot_config = fix_prismatic_joints(robot_config)
            # robot_config = fix_connection_structure(robot_config, verbose=self.verbose)
            
            # 评估约束条件
            connection_ok, _ = check_connection_quality(robot_config, verbose=False)
            G[i, 0] = 0 if connection_ok else 1
            
            stability_score = self.estimate_stability(robot_config)
            G[i, 1] = self.min_stability - stability_score
            
            structure_type = self.classify_structure_type(robot_config)
            current_population_types[structure_type] += 1
            
            try:
                config = {"gui":self.use_gui,
                          "sim_time": 10.0,
                          "terrain_type": "rough"}

                # 模拟机器人获取性能指标
                metrics = simulate_robot_multi(
                    robot_config, 
                    config=config
                )
                
                # 记录性能指标
                F[i, 0] = -metrics[0]  # 距离
                F[i, 1] = -metrics[1]  # 路径直线性
                F[i, 2] = metrics[2]   # 稳定性
                F[i, 3] = metrics[3]   # 能量消耗
                
                # 能耗约束
                G[i, 2] = metrics[3] - self.max_energy
                
                # 为腿式结构提供奖励
                if structure_type == 'legged':
                    distance_bonus = 0.1 * metrics[0]
                    F[i, 0] -= distance_bonus
                
                # 记录设计信息
                design_data = {
                    'gene': gene.copy(),
                    'config': robot_config,
                    'performance': metrics[0],
                    'structure_type': structure_type
                }
                generation_designs.append(design_data)
                
                # 记录到评估历史
                self.evaluated_designs.append({
                    'gene': gene.copy(),
                    'config': {
                        'num_links': robot_config['num_links'],
                        'num_wheels': sum(robot_config['is_wheel']),
                        'shape_type': robot_config['shapes'][0],
                        'joint_types': robot_config['joint_types'].copy(),
                        'structure_type': structure_type
                    },
                    'performance': metrics[0]
                })
                
                self.structure_types[structure_type] += 1
                
                # 多样性计算
                if self.add_diversity and len(self.evaluated_designs) > 1:
                    diversity_score = self.calculate_diversity(robot_config)
                    F[i, 4] = -diversity_score
                elif self.add_diversity:
                    F[i, 4] = 0.0
                
            except Exception as e:
                print(f"模拟过程出错: {str(e)}")
                F[i, 0] = 0.0
                F[i, 1] = 0.0
                F[i, 2] = 3.14
                F[i, 3] = 1000
                G[i, 3] = 500.0
                
                if self.add_diversity:
                    F[i, 4] = 0.0
            
            # 打印该个体的评估结果
            print(f"个体评估结果:")
            print(f"- 移动距离: {-F[i, 0]:.2f}")
            print(f"- 路径直线性: {-F[i, 1]:.2f}")
            print(f"- 稳定性指标: {F[i, 2]:.2f}")
            print(f"- 能量消耗: {F[i, 3]:.2f}")
            
            if self.use_gui and self.pause_after_eval:
                input("按Enter键继续评估下一个个体...")

        # 更新自适应状态
        if generation_designs:
            if self.adaptation_state:
                self.adaptation_history.append(self.adaptation_state.copy())
            
            self.adaptation_state = update_adaptation_state(
                self.adaptation_state, 
                generation_designs, 
                self.current_generation
            )
        
        out["F"] = F
        out["G"] = G

    def classify_structure_type(self, robot_config):
        """识别机器人的结构类型"""
        num_wheels = sum(robot_config['is_wheel'])
        num_links = robot_config['num_links']
        num_non_wheels = num_links - 1 - num_wheels
        
        joint_types = robot_config['joint_types']
        prismatic_joints = sum(1 for jt in joint_types if jt == p.JOINT_PRISMATIC)
        
        # 分类逻辑
        if num_wheels == 0:
            return 'legged' if prismatic_joints > 0 else 'other'
        elif num_wheels <= 2 and num_non_wheels >= 2:
            return 'legged'
        elif num_wheels >= 3 and num_non_wheels >= 2:
            return 'hybrid'
        elif num_wheels >= 1 and num_non_wheels >= 3:
            return 'hybrid'
        else:
            return 'wheeled'

    def calculate_diversity(self, robot_config):
        """计算当前设计与之前设计的差异度，包含树形结构特征"""
        # 提取基本特征
        current_features = np.array([
            robot_config['num_links'],
            sum(robot_config['is_wheel']),
            np.mean(robot_config['joint_types'])
        ])
        
        # 提取树形结构特征
        if 'node_depths' in robot_config:
            max_depth = max(robot_config['node_depths'])
            current_features = np.append(current_features, max_depth)
        
        # 计算分支因子 - 父节点的平均子节点数
        branching_factor = 1.0
        if 'parent_indices' in robot_config:
            parent_counts = {}
            for parent_idx in robot_config['parent_indices'][1:]:  # 跳过底座
                if parent_idx not in parent_counts:
                    parent_counts[parent_idx] = 0
                parent_counts[parent_idx] += 1
            
            if parent_counts:
                branching_factor = sum(parent_counts.values()) / len(parent_counts)
            
            current_features = np.append(current_features, branching_factor)
        
        # 计算轮子分布 - 轮子在不同深度上的分布
        wheel_depth_dist = [0] * 5  # 最多考虑5个深度层次
        if 'is_wheel' in robot_config and 'node_depths' in robot_config:
            for i in range(min(len(robot_config['is_wheel']), len(robot_config['node_depths']))):
                if robot_config['is_wheel'][i]:
                    depth = min(robot_config['node_depths'][i], 4)
                    wheel_depth_dist[depth] += 1
            
            # 归一化
            if sum(wheel_depth_dist) > 0:
                wheel_depth_dist = [x / sum(wheel_depth_dist) for x in wheel_depth_dist]
            
            current_features = np.append(current_features, wheel_depth_dist)
        
        # 计算与历史设计的差异度
        diversity_scores = []
        for design in self.evaluated_designs:
            prev_config = design['config']
            
            # 提取相同的特征
            prev_features = np.array([
                prev_config['num_links'],
                prev_config['num_wheels'],
                np.mean(prev_config['joint_types'])
            ])
            
            # 如果有相同的额外特征，也加入比较
            if len(current_features) > 3:
                # 如果历史设计没有相应的特征，使用默认值
                if 'node_depths' in prev_config:
                    prev_max_depth = max(prev_config['node_depths'])
                else:
                    prev_max_depth = 1.0
                
                prev_features = np.append(prev_features, prev_max_depth)
            
            if len(current_features) > 4:
                # 分支因子
                prev_branching_factor = 1.0
                prev_features = np.append(prev_features, prev_branching_factor)
            
            if len(current_features) > 5:
                # 轮子分布
                prev_wheel_depth_dist = [0] * 5
                prev_features = np.append(prev_features, prev_wheel_depth_dist)
            
            # 计算欧氏距离
            # 根据特征长度可能不同，只使用共同的部分
            min_length = min(len(current_features), len(prev_features))
            distance = np.linalg.norm(current_features[:min_length] - prev_features[:min_length])
            diversity_scores.append(distance)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0

    def estimate_stability(self, robot_config):
        """估计机器人结构的稳定性"""
        # 简化版稳定性估计
        num_wheels = sum(robot_config['is_wheel'])
        
        # 简单启发式规则
        if num_wheels >= 3:
            return 0.8  # 三轮或以上通常稳定
        elif num_wheels == 2:
            return 0.6  # 两轮适中稳定
        else:
            return 0.4  # 少于两轮较不稳定

def create_diverse_gene():
    """创建多样化的机器人基因，支持树形结构和不同关节类型"""
    gene = np.zeros(100)
    
    # 基本参数，确保合理的连杆数量和形状
    gene[0] = 0.3 + np.random.random() * 0.7  # 连杆数量
    gene[1:6] = np.random.random(5)  # 车身参数
    
    # 随机填充其余基因
    gene[6:] = np.random.random(94)
    
    # 强制某些位置的值，增加多样性
    # 关节类型位置
    joint_positions = [7, 20, 33, 46, 59, 72, 85]
    
    for pos in joint_positions:
        if pos < len(gene):
            # 随机决定关节类型
            joint_type_val = np.random.random()
            
            # 增加REVOLUTE和PRISMATIC的概率
            if joint_type_val < 0.5:  # 50%概率是REVOLUTE
                gene[pos] = 0.15  # 确保小于0.3
            elif joint_type_val < 0.8:  # 30%概率是PRISMATIC
                gene[pos] = 0.5  # 在0.3-0.7之间
            else:  # 20%概率是FIXED
                gene[pos] = 0.85  # 大于0.7
    
    # 父节点位置 - 在每个连杆参数组的最后两个位置
    parent_positions = [19, 32, 45, 58, 71, 84, 97]
    
    # 确保形成合理的树结构
    for i, pos in enumerate(parent_positions):
        if pos < len(gene):
            # 随机选择父节点 - 限制在当前节点之前的节点中
            max_parent = min(i + 1, 3)  # 最多选择前3个节点作为父节点
            gene[pos] = np.random.random() * max_parent / 3.0  # 使得父节点更可能是较早的节点
    
    # 确保至少有一些轮子
    wheel_positions = [10, 23, 36, 49]
    wheel_count = 0
    
    for pos in wheel_positions:
        if pos < len(gene) and wheel_count < 2:  # 确保至少有2个轮子
            if np.random.random() < 0.7:  # 70%概率是轮子
                gene[pos] = 0.9  # 确保是轮子
                wheel_count += 1
    
    # 如果还没有足够的轮子，强制添加
    while wheel_count < 2 and wheel_positions and wheel_positions[0] < len(gene):
        gene[wheel_positions[0]] = 0.9
        wheel_positions.pop(0)
        wheel_count += 1
    
    return gene

class CustomSampling(Sampling):
    """自定义采样类，用于使用指定的初始种群"""
    def __init__(self, initial_pop):
        super().__init__()
        self.initial_pop = initial_pop
        
    def _do(self, problem, n_samples, **kwargs):
        return self.initial_pop

def test_robot_with_gene(gene, adaptation_state=None):
    """测试使用基因参数生成的机器人"""
    # 解码基因为机器人配置
    robot_config = decode_gene(gene, adaptation_state=adaptation_state)
    robot_config = fix_prismatic_joints(robot_config)
    robot_config = fix_connection_structure(robot_config, verbose=True)
    
    # 生成URDF
    urdf = generate_urdf(robot_config)
    with open("gene_robot.urdf", "w") as f:
        f.write(urdf)
    print("\n已生成基于基因的机器人URDF")  
    
    # 初始化PyBullet模拟
    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # 加载地面和机器人
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("gene_robot.urdf", basePosition=[0, 0, 0.1])
    
    # 设置相机
    p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0, 0, 0])
    
    # 简单控制：为所有关节设置速度控制
    for i in range(p.getNumJoints(robot_id)):
        joint_type = p.getJointInfo(robot_id, i)[2]
        if joint_type != p.JOINT_FIXED:
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=5.0, force=5.0)
    
    # 模拟循环
    print("\n开始模拟 - 按Ctrl+C停止")
    try:
        for _ in range(1000):  # 缩短模拟时间
            p.stepSimulation()
            time.sleep(1/240.0)
    except KeyboardInterrupt:
        print("\n模拟被用户中断")
    finally:
        p.disconnect()
    
    print("\n模拟完成")

# 修改save_best_robot_design函数
def save_best_robot_design(best_gene, adaptation_state=None):
    """保存最佳机器人设计的URDF文件"""
    # 创建保存目录
    if not os.path.exists("results"):
        os.makedirs("results")
        
    # 解码基因为机器人配置
    robot_config = decode_gene(best_gene, adaptation_state=adaptation_state)
    robot_config = fix_prismatic_joints(robot_config)
    robot_config = fix_connection_structure(robot_config)
    
    # 可视化机器人结构
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    vis_path = f"results/best_robot_structure_{timestamp}.png"
    visualize_robot_structure(robot_config, save_path=vis_path)
    print(f"\n已保存最佳机器人结构可视化: {vis_path}")
    
    # 生成URDF内容
    urdf_content = generate_urdf(robot_config)
    
    # 保存URDF文件
    save_path = f"results/best_robot_{timestamp}.urdf"
    with open(save_path, 'w') as f:
        f.write(urdf_content)
    
    print(f"\n已保存最佳机器人设计的URDF文件: {save_path}")
    return save_path

def run_diverse_genetic_optimization(pop_size=5, n_gen=3, use_gui=True, verbose=False, pause_after_eval=False):
    """Run genetic algorithm optimization for diverse robot designs"""
    print("\nStarting genetic algorithm optimization for robot design...")
    print(f"Population size: {pop_size}, Generations: {n_gen}")
    
    try:
        # Initialize adaptation state
        initial_adaptation_state = {
            'max_links': 8,
            'joint_type_ranges': [0.25, 0.5, 0.75],
            'wheel_probability': 0.4,
            'min_wheel_prob': 0.0,
            'generation': 0
        }
        
        # Define problem
        problem = RobotDesignProblem(
            n_var=100, 
            use_gui=use_gui,
            verbose=verbose, 
            pause_after_eval=pause_after_eval, 
            add_diversity=True
        )
        
        # Initialize problem's adaptation state
        problem.adaptation_state = initial_adaptation_state
        
        # Set evolution process tracking variables
        problem.total_generations = n_gen
        problem.total_populations = pop_size * n_gen
        
        # Create initial population
        initial_pop = np.zeros((pop_size, 100))
        for i in range(pop_size):
            initial_pop[i] = create_diverse_gene()
        
        # Configure algorithm
        sampling = CustomSampling(initial_pop)
        crossover = SBX(prob=0.9, eta=10)
        mutation = PM(prob=0.2, eta=15)
        
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
        )
        
        # Define callback function to update current generation
        generation_data = []
        
        def callback(algorithm):
            problem.current_generation = algorithm.n_gen
            print(f"Completed generation {algorithm.n_gen}")
            
            # Save current population in case the final result is None
            if hasattr(algorithm, 'pop'):
                if hasattr(algorithm.pop, 'get') and callable(algorithm.pop.get):
                    X = algorithm.pop.get("X")
                    F = algorithm.pop.get("F")
                    
                    if X is not None and F is not None:
                        generation_data.append({
                            'gen': algorithm.n_gen,
                            'X': X.copy(),
                            'F': F.copy()
                        })
                        print(f"Saved intermediate population: {X.shape[0]} individuals")
            
            return False
        
        print("Starting optimization process...")
        
        # Run optimization
        results = minimize(
            problem,
            algorithm,
            ('n_gen', n_gen),
            verbose=True,
            save_history=True,
            callback=callback
        )
        
        print("\nOptimization completed")
        
        # Check if results are valid
        if results is None:
            print("Error: optimization returned None")
            
            # Try to use the last saved generation
            if generation_data:
                last_gen = generation_data[-1]
                print(f"Using data from generation {last_gen['gen']} instead")
                
                X = last_gen['X']
                F = last_gen['F']
                
                # Find best design
                if F.shape[0] > 0:
                    best_idx = np.argmin(F[:, 0]) 
                    best_gene = X[best_idx]
                    
                    print("\nTesting best design from saved generations...")
                    test_robot_with_gene(best_gene, adaptation_state=problem.adaptation_state)
                    
                    # Save best design's URDF file
                    save_best_robot_design(best_gene, adaptation_state=problem.adaptation_state)
                    
                    return best_gene
            
            return None
            
        if not hasattr(results, 'X') or results.X is None:
            print("Error: optimization did not return any solutions (X is None)")
            
            # Try to use the last saved generation
            if generation_data:
                last_gen = generation_data[-1]
                print(f"Using data from generation {last_gen['gen']} instead")
                
                X = last_gen['X']
                F = last_gen['F']
                
                # Find best design
                if F.shape[0] > 0:
                    best_idx = np.argmin(F[:, 0]) 
                    best_gene = X[best_idx]
                    
                    print("\nTesting best design from saved generations...")
                    test_robot_with_gene(best_gene, adaptation_state=problem.adaptation_state)
                    
                    # Save best design's URDF file
                    save_best_robot_design(best_gene, adaptation_state=problem.adaptation_state)
                    
                    return best_gene
            
            return None
            
        if not hasattr(results, 'F') or results.F is None:
            print("Error: optimization did not return objective values (F is None)")
            
            # Try to use the last saved generation
            if generation_data:
                last_gen = generation_data[-1]
                print(f"Using data from generation {last_gen['gen']} instead")
                
                X = last_gen['X']
                F = last_gen['F']
                
                # Find best design
                if F.shape[0] > 0:
                    best_idx = np.argmin(F[:, 0]) 
                    best_gene = X[best_idx]
                    
                    print("\nTesting best design from saved generations...")
                    test_robot_with_gene(best_gene, adaptation_state=problem.adaptation_state)
                    
                    # Save best design's URDF file
                    save_best_robot_design(best_gene, adaptation_state=problem.adaptation_state)
                    
                    return best_gene
            
            return None
        
        # Print shape information for debugging
        print(f"Solutions shape: {results.X.shape}")
        print(f"Objective values shape: {results.F.shape}")
        
        # Get results
        X = results.X  # Decision variables
        F = results.F  # Objective function values
        
        # Find best design (minimize movement distance objective)
        try:
            best_idx = np.argmin(F[:, 0])
            best_gene = X[best_idx]
            
            print("\nTesting best design...")
            test_robot_with_gene(best_gene, adaptation_state=problem.adaptation_state)
            
            # Save best design's URDF file
            save_best_robot_design(best_gene, adaptation_state=problem.adaptation_state)
            
            return best_gene
            
        except Exception as e:
            print(f"Error processing results: {str(e)}")
            
            # Try to find any valid solution
            if len(X) > 0:
                print("Returning first valid solution instead")
                return X[0]
            else:
                return None
        
    except KeyboardInterrupt:
        print("\nOptimization process interrupted by user")
        return None
    except Exception as e:
        print(f"\nError during optimization process: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 设置自定义参数
    print("===== 机器人多样化进化优化系统 =====")
    print("请设置优化参数:")
    
    try:
        pop_size = int(input("请输入种群大小 (建议3-10): ") or "5")
        n_gen = int(input("请输入进化代数 (建议2-5): ") or "3")
        show_gui = input("是否显示模拟可视化? (y/n): ").lower() != 'n'  # 默认显示
        
        # 运行优化
        best_gene = run_diverse_genetic_optimization(
            pop_size=pop_size, 
            n_gen=n_gen, 
            use_gui=show_gui
        )
        
        if best_gene is not None:
            print("\n优化成功完成!")
        else:
            print("\n优化过程中出错。")
    
    except KeyboardInterrupt:
        print("\n\n用户中断了优化过程。")
    except Exception as e:
        print(f"\n设置参数时出错: {str(e)}")