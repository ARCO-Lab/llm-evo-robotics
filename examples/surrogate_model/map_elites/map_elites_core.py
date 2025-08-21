import numpy as np
import torch
import os
import pickle
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class RobotGenotype:
    """机器人基因型 - 包含所有可变的设计参数"""
    # 🤖 机器人形态参数
    num_links: int = 4
    link_lengths: List[float] = None
    
    # 🧠 SAC超参数
    lr: float = 1e-4
    alpha: float = 0.2
    tau: float = 0.005
    gamma: float = 0.99
    batch_size: int = 64
    buffer_capacity: int = 10000
    
    # 🎯 训练参数
    warmup_steps: int = 1000
    target_entropy_factor: float = 0.8
    
    def __post_init__(self):
        if self.link_lengths is None:
            self.link_lengths = [80.0] * self.num_links
        assert len(self.link_lengths) == self.num_links


@dataclass 
class RobotPhenotype:
    """机器人表型 - 行为特征"""
    # 🎯 性能特征
    avg_reward: float = 0.0
    success_rate: float = 0.0
    min_distance_to_goal: float = float('inf')
    
    # 🏗️ 形态特征
    total_reach: float = 0.0
    complexity_score: float = 0.0
    
    # 🎮 行为特征
    trajectory_smoothness: float = 0.0
    collision_frequency: float = 0.0
    exploration_coverage: float = 0.0  # 修复拼写错误: converage -> coverage
    
    # 🧠 控制特征
    action_variance: float = 0.0
    learning_efficiency: float = 0.0  # 修复拼写错误: leanring -> learning


@dataclass
class Individual:
    """MAP-Elites个体"""
    genotype: RobotGenotype
    phenotype: RobotPhenotype
    fitness: float = 0.0
    generation: int = 0
    parent_id: Optional[str] = None  # 修复类型错误: int -> str
    individual_id: str = ""
    
    def __post_init__(self):
        if not self.individual_id:
            self.individual_id = f"gen_{self.generation}_{int(time.time() * 1000) % 100000}"


class FeatureExtractor:
    """从训练过程中提取行为特征"""
    
    def __init__(self):
        pass
    
    def extract_from_training_data(self, training_metrics: Dict, robot_config: Dict) -> RobotPhenotype:
        """从训练指标中提取表型特征"""
        phenotype = RobotPhenotype()
        
        # 🎯 性能特征
        phenotype.avg_reward = training_metrics.get('avg_reward', 0.0)
        phenotype.success_rate = training_metrics.get('success_rate', 0.0)
        phenotype.min_distance_to_goal = training_metrics.get('min_distance', float('inf'))
        
        # 🏗️ 形态特征
        link_lengths = robot_config.get('link_lengths', [80] * 4)
        phenotype.total_reach = sum(link_lengths)
        phenotype.complexity_score = len(link_lengths) + np.var(link_lengths) / 100.0
        
        # 🎮 行为特征
        phenotype.trajectory_smoothness = training_metrics.get('trajectory_smoothness', 0.0)
        phenotype.collision_frequency = training_metrics.get('collision_rate', 0.0)
        phenotype.exploration_coverage = training_metrics.get('exploration_area', 0.0)
        
        # 🧠 控制特征
        phenotype.action_variance = training_metrics.get('action_variance', 0.0)
        phenotype.learning_efficiency = training_metrics.get('learning_rate', 0.0)
        
        return phenotype
    
    def discretize_features(self, phenotype: RobotPhenotype) -> Tuple[int, ...]:
        """将连续特征离散化为网格坐标"""
        # 🎯 奖励 (10个区间)
        reward_bin = min(9, max(0, int((phenotype.avg_reward + 100) / 20)))  # 修复语法错误
        
        # 🏗️ 复杂度 (5个等级)
        complexity_bin = min(4, max(0, int(phenotype.complexity_score / 2)))
        
        # 🎮 平滑度 (5个等级)
        smoothness_bin = min(4, max(0, int(phenotype.trajectory_smoothness * 5)))
        
        # 🧠 碰撞频率 (5个等级)
        collision_bin = min(4, max(0, int(phenotype.collision_frequency * 5)))
        
        # 📏 伸展范围 (6个等级)
        reach_bin = min(5, max(0, int((phenotype.total_reach - 200) / 50)))
        
        return (reward_bin, complexity_bin, smoothness_bin, collision_bin, reach_bin)


class MAPElitesArchive:
    """MAP-Elites存档"""
    
    def __init__(self, feature_dimensions: Tuple[int, ...] = (10, 5, 5, 5, 6), 
                 save_dir: str = "map_elites_archive"):
        self.feature_dimensions = feature_dimensions
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 存档网格：坐标 -> Individual
        self.archive: Dict[Tuple[int, ...], Individual] = {}
        
        # 统计信息
        self.generation = 0
        self.total_evaluations = 0
        self.improvement_count = 0
        
        print(f"🗂️ MAP-Elites存档初始化")
        print(f"📐 网格维度: {feature_dimensions}")
        print(f"📊 总单元数: {np.prod(feature_dimensions)}")
    
    def add_individual(self, individual: Individual) -> bool:
        """尝试将个体添加到存档"""
        # 将表型特征离散化
        extractor = FeatureExtractor()
        coords = extractor.discretize_features(individual.phenotype)
        
        self.total_evaluations += 1
        
        # 检查是否应该替换现有个体
        if coords not in self.archive or individual.fitness > self.archive[coords].fitness:
            self.archive[coords] = individual  # 修复：实际添加个体到存档
            self.improvement_count += 1
            print(f"🆕 个体添加到位置 {coords}, 适应度: {individual.fitness:.2f}")
            return True
        
        return False
    
    def get_random_elite(self) -> Optional[Individual]:
        """随机选择一个精英个体"""
        if not self.archive:
            return None
        
        # 修复：正确处理字典键的随机选择
        coords_list = list(self.archive.keys())
        if not coords_list:
            return None
        
        # 使用random.choice而不是np.random.choice来处理元组
        import random
        selected_coords = random.choice(coords_list)
        return self.archive[selected_coords]
    
    def get_best_individual(self) -> Optional[Individual]:
        """获取最佳个体"""
        if not self.archive:
            return None
        return max(self.archive.values(), key=lambda x: x.fitness)
    
    def get_statistics(self) -> Dict:
        """获取存档统计信息"""
        if not self.archive:
            return {'size': 0, 'coverage': 0.0, 'best_fitness': -float('inf')}  # 修复语法错误
        
        fitnesses = [ind.fitness for ind in self.archive.values()]
        coverage = len(self.archive) / np.prod(self.feature_dimensions)  # 修复语法错误
        
        return {
            'size': len(self.archive),
            'coverage': coverage,
            'best_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses) if len(fitnesses) > 1 else 0.0,
            'total_evaluations': self.total_evaluations,
            'improvement_rate': self.improvement_count / max(1, self.total_evaluations)
        }
    
    def save_archive(self, filename: Optional[str] = None):
        """保存存档"""
        if filename is None:
            filename = f"archive_gen_{self.generation}.pkl"
        
        filepath = os.path.join(self.save_dir, filename)
        archive_data = {
            'archive': self.archive,
            'feature_dimensions': self.feature_dimensions,
            'generation': self.generation,
            'total_evaluations': self.total_evaluations,
            'improvement_count': self.improvement_count
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(archive_data, f)
        print(f"💾 存档已保存: {filepath}")


class RobotMutator:  # 修复类名拼写错误: RobotMulator -> RobotMutator
    """机器人基因型变异器"""
    
    def __init__(self, mutation_rates: Dict[str, float] = None):
        self.mutation_rates = mutation_rates or {
            'link_length': 0.1,    # 链节长度变异概率
            'num_links': 0.05,     # 链节数量变异概率  
            'sac_params': 0.3,     # SAC超参数变异概率
            'training_params': 0.2  # 训练参数变异概率
        }
    
    def mutate(self, parent: RobotGenotype) -> RobotGenotype:
        """对基因型进行变异"""
        # 深拷贝父代
        mutant = RobotGenotype(
            num_links=parent.num_links,
            link_lengths=parent.link_lengths.copy(),
            lr=parent.lr,
            alpha=parent.alpha,
            tau=parent.tau,
            gamma=parent.gamma,
            batch_size=parent.batch_size,
            buffer_capacity=parent.buffer_capacity,
            warmup_steps=parent.warmup_steps,
            target_entropy_factor=parent.target_entropy_factor
        )
        
        # 🤖 形态变异
        if np.random.random() < self.mutation_rates['link_length']:
            # 随机修改一个链节长度
            idx = np.random.randint(len(mutant.link_lengths))
            mutant.link_lengths[idx] *= np.random.uniform(0.8, 1.2)
            mutant.link_lengths[idx] = max(20, min(120, mutant.link_lengths[idx]))
        
        if np.random.random() < self.mutation_rates['num_links']:
            # 增加或减少链节数量
            if np.random.random() < 0.5 and mutant.num_links > 2:
                # 减少链节
                mutant.num_links -= 1
                mutant.link_lengths = mutant.link_lengths[:-1]
            elif mutant.num_links < 6:
                # 增加链节
                mutant.num_links += 1
                new_length = np.random.uniform(40, 80)
                mutant.link_lengths.append(new_length)
        
        # 🧠 SAC参数变异
        # if np.random.random() < self.mutation_rates['sac_params']:
        #     # 学习率变异
        #     mutant.lr *= np.random.lognormal(0, 0.3)
        #     mutant.lr = np.clip(mutant.lr, 1e-6, 1e-3)
            
        #     # Alpha变异
        #     mutant.alpha += np.random.normal(0, 0.1)
        #     mutant.alpha = np.clip(mutant.alpha, 0.01, 2.0)
            
        #     # Tau变异
        #     mutant.tau *= np.random.uniform(0.8, 1.2)
        #     mutant.tau = np.clip(mutant.tau, 0.001, 0.01)
        
        # # 🎯 训练参数变异
        # if np.random.random() < self.mutation_rates['training_params']:
        #     # Batch size变异 (在2的幂中选择)
        #     batch_sizes = [32, 64, 128, 256]
        #     mutant.batch_size = np.random.choice(batch_sizes)
            
        #     # Warmup steps变异
        #     mutant.warmup_steps = int(mutant.warmup_steps * np.random.uniform(0.5, 2.0))
        #     mutant.warmup_steps = max(100, min(5000, mutant.warmup_steps))
        
        return mutant  # 修复：取消注释并修正缩进
    
    def random_genotype(self) -> RobotGenotype:  # 修复缩进错误
        """生成随机基因型"""
        num_links = np.random.randint(3, 7)
        link_lengths = [np.random.uniform(40, 100) for _ in range(num_links)]
        
        return RobotGenotype(
            num_links=num_links,
            link_lengths=link_lengths,
            lr=10 ** np.random.uniform(-6, -3),  # 1e-6 到 1e-3
            alpha=np.random.uniform(0.1, 1.0),
            tau=np.random.uniform(0.001, 0.01),
            gamma=np.random.uniform(0.95, 0.999),
            batch_size=np.random.choice([32, 64, 128, 256]),
            buffer_capacity=np.random.choice([5000, 10000, 20000]),
            warmup_steps=np.random.randint(500, 3000),
            target_entropy_factor=np.random.uniform(0.5, 1.2)
        )


# 🧪 测试函数
def test_map_elites_core():
    """测试MAP-Elites核心组件"""
    print("🧪 开始测试MAP-Elites核心组件\n")
    
    # 1. 测试基因型和表型创建
    print("📊 测试1: 基因型和表型创建")
    genotype = RobotGenotype(num_links=4, link_lengths=[80, 60, 40, 30])
    phenotype = RobotPhenotype(avg_reward=50.0, success_rate=0.8, total_reach=210)
    print(f"✅ 基因型: {genotype.num_links}链节, 长度={genotype.link_lengths}")
    print(f"✅ 表型: 奖励={phenotype.avg_reward}, 成功率={phenotype.success_rate}\n")
    
    # 2. 测试个体创建
    print("📊 测试2: 个体创建")
    individual = Individual(genotype=genotype, phenotype=phenotype, fitness=50.0)
    print(f"✅ 个体ID: {individual.individual_id}")
    print(f"✅ 适应度: {individual.fitness}\n")
    
    # 3. 测试特征提取器
    print("📊 测试3: 特征提取器")
    extractor = FeatureExtractor()
    
    # 模拟训练指标
    training_metrics = {
        'avg_reward': 45.0,
        'success_rate': 0.75,
        'min_distance': 25.0,
        'trajectory_smoothness': 0.6,
        'collision_rate': 0.1,
        'exploration_area': 300.0,
        'action_variance': 0.3,
        'learning_rate': 0.8
    }
    
    robot_config = {
        'num_links': 4,
        'link_lengths': [80, 60, 40, 30]
    }
    
    extracted_phenotype = extractor.extract_from_training_data(training_metrics, robot_config)
    print(f"✅ 提取的表型: 奖励={extracted_phenotype.avg_reward}, 伸展={extracted_phenotype.total_reach}")
    
    # 特征离散化
    coords = extractor.discretize_features(extracted_phenotype)
    print(f"✅ 离散化坐标: {coords}\n")
    
    # 4. 测试存档
    print("📊 测试4: MAP-Elites存档")
    archive = MAPElitesArchive()
    
    # 添加个体到存档
    test_individual = Individual(
        genotype=genotype, 
        phenotype=extracted_phenotype, 
        fitness=extracted_phenotype.avg_reward
    )
    
    success = archive.add_individual(test_individual)
    print(f"✅ 个体添加成功: {success}")
    
    # 获取统计信息
    stats = archive.get_statistics()
    print(f"✅ 存档统计: {stats}\n")
    
    # 5. 测试变异器
    print("📊 测试5: 变异器")
    mutator = RobotMutator()
    
    # 生成随机基因型
    random_genotype = mutator.random_genotype()
    print(f"✅ 随机基因型: {random_genotype.num_links}链节, lr={random_genotype.lr:.2e}")
    
    # 变异基因型
    mutant_genotype = mutator.mutate(genotype)
    print(f"✅ 变异基因型: {mutant_genotype.num_links}链节, 长度={[f'{x:.1f}' for x in mutant_genotype.link_lengths]}")
    
    # 6. 测试多个个体
    print("\n📊 测试6: 多个个体存档")
    for i in range(5):
        rand_genotype = mutator.random_genotype()
        rand_phenotype = RobotPhenotype(
            avg_reward=np.random.uniform(-50, 100),
            success_rate=np.random.uniform(0, 1),
            total_reach=sum(rand_genotype.link_lengths)
        )
        rand_individual = Individual(
            genotype=rand_genotype,
            phenotype=rand_phenotype,
            fitness=rand_phenotype.avg_reward,
            generation=1
        )
        archive.add_individual(rand_individual)
    
    final_stats = archive.get_statistics()
    print(f"✅ 最终存档统计: {final_stats}")
    
    # 测试随机精英选择
    print("\n📊 测试7: 随机精英选择")
    for i in range(3):
        elite = archive.get_random_elite()
        if elite:
            print(f"   随机精英{i+1}: ID={elite.individual_id}, 适应度={elite.fitness:.2f}")
        else:
            print(f"   随机精英{i+1}: 无精英个体")
    
    # 获取最佳个体
    best = archive.get_best_individual()
    if best:
        print(f"✅ 最佳个体: ID={best.individual_id}, 适应度={best.fitness:.2f}")
    
    # 7. 测试存档保存
    print("\n📊 测试8: 存档保存")
    archive.save_archive("test_archive.pkl")
    
    print("\n🎉 所有测试完成!")


if __name__ == "__main__":
    test_map_elites_core()