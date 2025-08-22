import sys
import os
import numpy as np
import torch
from typing import Dict, Any, Optional
import time

# 添加路径以便导入现有模块
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from map_elites_core import Individual, RobotGenotype, RobotPhenotype, FeatureExtractor

# 导入真实训练接口
try:
    from enhanced_train_interface import MAPElitesTrainingInterface
    REAL_TRAINING_AVAILABLE = True
except ImportError:
    print("⚠️  真实训练接口不可用，将使用模拟训练")
    REAL_TRAINING_AVAILABLE = False


class MAPElitesTrainingAdapter:
    """MAP-Elites与SAC训练的适配器 - 优化版"""
    
    def __init__(self, base_args, base_save_dir: str = "./map_elites_experiments", 
                 use_real_training: bool = True,
                 enable_rendering: bool = False,  # 🆕 控制是否显示可视化
                 silent_mode: bool = True):       # 🆕 控制是否静默
        self.base_args = base_args
        self.base_save_dir = base_save_dir
        self.use_real_training = use_real_training and REAL_TRAINING_AVAILABLE
        
        os.makedirs(base_save_dir, exist_ok=True)
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor()
        
        # 🔧 优化：可配置的训练接口
        if self.use_real_training:
            self.training_interface = MAPElitesTrainingInterface(
                silent_mode=silent_mode,
                enable_rendering=enable_rendering
            )
            print(f"🔧 MAP-Elites训练适配器已初始化 (使用enhanced_train.py)")
            print(f"   🎨 渲染: {'启用' if enable_rendering else '禁用'}")
            print(f"   🔇 静默: {'启用' if silent_mode else '禁用'}")
        else:
            print("🔧 MAP-Elites训练适配器已初始化 (使用模拟训练)")
    
    def evaluate_individual(self, individual: Individual, training_steps: int = 5000) -> Individual:
        """评估单个个体 - 增强版"""
        print(f"\n🧬 评估个体 {individual.individual_id}")
        print(f"🤖 基因型: num_links={individual.genotype.num_links}, "
              f"link_lengths={[f'{x:.1f}' for x in individual.genotype.link_lengths]}")
        print(f"🧠 SAC参数: lr={individual.genotype.lr:.2e}, alpha={individual.genotype.alpha:.3f}")
        
        # 1. 根据基因型创建训练参数
        training_args = self._genotype_to_training_args(individual.genotype, training_steps)
        
        # 2. 运行训练
        start_time = time.time()
        if self.use_real_training:
            print(f"   🎯 使用enhanced_train.py进行真实训练 ({training_steps} steps)")
            try:
                training_metrics = self.training_interface.train_individual(training_args)
            except Exception as e:
                print(f"   ❌ 真实训练失败: {e}")
                print(f"   🔄 回退到模拟训练")
                training_metrics = self._run_simulated_training(training_args)
        else:
            print(f"   🎲 使用模拟训练 ({training_steps} steps)")
            training_metrics = self._run_simulated_training(training_args)
        
        training_time = time.time() - start_time
        
        # 3. 提取表型特征
        robot_config = {
            'num_links': individual.genotype.num_links,
            'link_lengths': individual.genotype.link_lengths
        }
        
        phenotype = self.feature_extractor.extract_from_training_data(training_metrics, robot_config)
        
        # 4. 更新个体
        individual.phenotype = phenotype
        individual.fitness = phenotype.avg_reward  # 使用平均奖励作为适应度
        
        print(f"✅ 评估完成: 适应度={individual.fitness:.2f}, 成功率={phenotype.success_rate:.2f}, 耗时={training_time:.1f}s")
        
        return individual
    
    def _genotype_to_training_args(self, genotype: RobotGenotype, training_steps: int):
        """将基因型转换为训练参数"""
        # 创建参数对象
        args = type('Args', (), {})()
        
        # 复制基础参数（如果存在的话）
        if hasattr(self.base_args, '__dict__'):
            for key, value in vars(self.base_args).items():
                setattr(args, key, value)
        
        # 🤖 设置机器人形态参数
        args.num_joints = genotype.num_links
        args.link_lengths = genotype.link_lengths
        
        # 🧠 设置SAC超参数
        args.lr = genotype.lr
        args.alpha = genotype.alpha
        args.tau = genotype.tau
        args.gamma = genotype.gamma
        args.batch_size = genotype.batch_size
        args.buffer_capacity = genotype.buffer_capacity
        args.warmup_steps = genotype.warmup_steps
        args.target_entropy_factor = genotype.target_entropy_factor
        
        # 🎯 设置训练步数
        args.total_steps = training_steps
        
        # 📁 设置保存目录
        individual_dir = os.path.join(self.base_save_dir, f"individual_{int(time.time() * 1000) % 100000}")
        args.save_dir = individual_dir
        
        # 🔧 设置其他训练参数
        args.update_frequency = getattr(self.base_args, 'update_frequency', 1)
        args.num_processes = 1  # MAP-Elites使用单进程
        args.seed = getattr(self.base_args, 'seed', 42)
        
        return args
    
    def _run_simulated_training(self, args) -> Dict[str, Any]:
        """运行模拟训练（备用方案）"""
        # 基于基因型预测大致性能
        num_links = getattr(args, 'num_joints', 4)
        link_lengths = getattr(args, 'link_lengths', [80.0] * num_links)
        
        # 简单的启发式评估
        total_reach = sum(link_lengths)
        complexity_penalty = abs(num_links - 4) * 10  # 4链节最优
        length_variance_penalty = np.var(link_lengths) / 10
        
        # 基于超参数的性能预测
        lr_factor = 1.0 if 1e-5 <= args.lr <= 1e-3 else 0.5
        alpha_factor = 1.0 if 0.1 <= args.alpha <= 0.5 else 0.7
        
        base_reward = min(100, total_reach / 5) - complexity_penalty - length_variance_penalty
        base_reward *= lr_factor * alpha_factor
        
        # 添加随机性
        noise = np.random.normal(0, 15)
        final_reward = base_reward + noise
        
        return {
            'avg_reward': final_reward,
            'success_rate': max(0, min(1, (final_reward + 50) / 150)),
            'min_distance': max(10, 200 - final_reward * 2),
            'trajectory_smoothness': np.random.uniform(0.3, 0.8),
            'collision_rate': np.random.uniform(0.0, 0.3),
            'exploration_area': np.random.uniform(100, 500),
            'action_variance': np.random.uniform(0.1, 0.5),
            'learning_rate': np.random.uniform(0.1, 0.9),
            'final_critic_loss': np.random.uniform(0.1, 5.0),
            'final_actor_loss': np.random.uniform(0.1, 2.0),
            'training_stability': np.random.uniform(0.3, 0.9)
        }


# 🧪 测试函数
def test_training_adapter():
    """测试训练适配器"""
    print("🧪 开始测试MAP-Elites训练适配器\n")
    
    # 1. 创建模拟的基础参数
    print("📊 测试1: 创建基础参数")
    import argparse
    base_args = argparse.Namespace()
    base_args.env_type = 'reacher2d'
    base_args.num_processes = 1
    base_args.seed = 42
    base_args.save_dir = './test_map_elites_results'
    base_args.lr = 1e-4
    base_args.alpha = 0.2
    base_args.tau = 0.005
    base_args.gamma = 0.99
    base_args.update_frequency = 1
    print(f"✅ 基础参数创建完成: {base_args.env_type}")
    
    # 2. 测试两种模式
    for use_real in [False, True]:
        print(f"\n{'='*30}")
        print(f"📊 测试模式: {'真实训练(enhanced_train.py)' if use_real else '模拟训练'}")
        print(f"{'='*30}")
        
        adapter = MAPElitesTrainingAdapter(
            base_args, 
            "./test_adapter_results", 
            use_real_training=use_real
        )
        
        # 创建测试个体
        from map_elites_core import RobotGenotype, RobotPhenotype, Individual
        
        test_genotype = RobotGenotype(
            num_links=3,
            link_lengths=[70, 50, 30],
            lr=2e-4,
            alpha=0.25
        )
        
        test_individual = Individual(
            genotype=test_genotype,
            phenotype=RobotPhenotype()
        )
        
        # 评估个体（使用较少步数进行快速测试）
        test_steps = 500 if use_real else 100
        start_time = time.time()
        
        evaluated = adapter.evaluate_individual(test_individual, training_steps=test_steps)
        
        end_time = time.time()
        
        print(f"✅ 评估结果 (耗时: {end_time - start_time:.1f}秒):")
        print(f"   适应度: {evaluated.fitness:.2f}")
        print(f"   成功率: {evaluated.phenotype.success_rate:.2f}")
        print(f"   总伸展: {evaluated.phenotype.total_reach:.1f}")
        print(f"   训练稳定性: {evaluated.phenotype.learning_efficiency:.2f}")
        
        if use_real and REAL_TRAINING_AVAILABLE:
            print(f"✅ 真实训练模式工作正常!")
        elif use_real:
            print(f"⚠️  真实训练不可用，已回退到模拟模式")


if __name__ == "__main__":
    test_training_adapter()