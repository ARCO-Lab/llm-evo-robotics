import sys
import os
import numpy as np
import torch
from typing import Dict, Any, Optional
import time

# 添加路径以便导入现有模块
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from map_elites_core import Individual, RobotGenotype, RobotPhenotype, FeatureExtractor

# 🆕 导入新的遗传算法fitness评估器
from genetic_fitness_evaluator import GeneticFitnessEvaluator
from fitness_manager import FitnessManager
# 导入真实训练接口
try:
    from enhanced_train_interface import MAPElitesTrainingInterface
    REAL_TRAINING_AVAILABLE = True
except ImportError:
    print("⚠️  真实训练接口不可用，将使用模拟训练")
    REAL_TRAINING_AVAILABLE = False


class MAPElitesTrainingAdapter:
    """MAP-Elites与SAC训练的适配器 - 集成遗传算法Fitness评估系统"""
    
    def __init__(self, base_args, base_save_dir: str = "./map_elites_experiments", 
                 use_real_training: bool = True,
                 enable_rendering: bool = False,  # 🆕 控制是否显示可视化
                 silent_mode: bool = True,        # 🆕 控制是否静默
                 use_genetic_fitness: bool = True, # 🆕 控制是否使用新fitness系统
                 shared_ppo_trainer=None):        # 🆕 共享PPO训练器
        self.base_args = base_args
        self.base_save_dir = base_save_dir
        self.use_real_training = use_real_training and REAL_TRAINING_AVAILABLE
        self.enable_rendering = enable_rendering
        self.silent_mode = silent_mode
        self.use_genetic_fitness = use_genetic_fitness
        self.shared_ppo_trainer = shared_ppo_trainer  # 🆕 保存共享PPO训练器引用
        
        os.makedirs(base_save_dir, exist_ok=True)
        
        # 🆕 初始化fitness评估系统
        if self.use_genetic_fitness:
            try:
                self.fitness_evaluator = GeneticFitnessEvaluator()
                print("🎯 使用遗传算法分层Fitness评估系统")
            except Exception as e:
                print(f"⚠️  遗传算法Fitness评估器初始化失败: {e}")
                print("🔄 回退到传统Fitness评估")
                self.fitness_evaluator = None
                self.use_genetic_fitness = False
        else:
            self.fitness_evaluator = None
            print("🎯 使用传统平均奖励Fitness评估")
        
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


        self.fitness_manager = FitnessManager(
            use_genetic_fitness=use_genetic_fitness,
            primary_strategy='episodes' if use_real_training else 'genetic'
        )


    def evaluate_individual(self, individual: Individual, training_steps: int = 5000) -> Individual:
        """评估单个个体 - 使用统一的FitnessManager"""
        print(f"\n🧬 评估个体 {individual.individual_id}")
        print(f"🤖 基因型: num_links={individual.genotype.num_links}, "
            f"link_lengths={[f'{x:.1f}' for x in individual.genotype.link_lengths]}")
        print(f"🧠 SAC参数: lr={individual.genotype.lr:.2e}, alpha={individual.genotype.alpha:.3f}")
        print(f"   总长度: {sum(individual.genotype.link_lengths):.1f}px")
        
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
        
        # 3. 提取表型特征（保留原有逻辑，用于兼容性）
        robot_config = {
            'num_links': individual.genotype.num_links,
            'link_lengths': individual.genotype.link_lengths,
            'lr': individual.genotype.lr,
            'alpha': individual.genotype.alpha
        }
        
        phenotype = self.feature_extractor.extract_from_training_data(training_metrics, robot_config)
        
        # 4. 🆕 使用统一的FitnessManager计算fitness
        if not hasattr(self, 'fitness_manager'):
            # 延迟初始化FitnessManager
            from fitness_manager import FitnessManager
            self.fitness_manager = FitnessManager(
                use_genetic_fitness=self.use_genetic_fitness,
                primary_strategy='episodes' if self.use_real_training else 'genetic'
            )
        
        try:
            # 🎯 准备完整的训练结果数据
            complete_training_data = self._prepare_training_data_for_fitness(
                training_metrics, phenotype, training_time
            )
            
            # 🎯 使用FitnessManager统一计算fitness
            fitness_result = self.fitness_manager.calculate_fitness(
                individual=individual, 
                training_results=complete_training_data
            )
            # 更新个体
            individual.phenotype = phenotype
            individual.fitness = fitness_result['fitness']
            individual.fitness_details = fitness_result['details']
            
            # 🎯 统一的结果显示
            print(f"✅ 评估完成:")
            print(f"   Fitness方法: {fitness_result['details']['method']}")
            print(f"   最终Fitness: {individual.fitness:.3f}")
            print(f"   评估类别: {fitness_result['details']['category']}")
            print(f"   评估策略: {fitness_result['details']['strategy']}")
            print(f"   原因: {fitness_result['details']['reason']}")
            
            # 🎯 显示对比信息（如果有的话）
            if 'avg_reward' in fitness_result['details']:
                print(f"   传统fitness (avg_reward): {fitness_result['details']['avg_reward']:.2f}")
            if 'success_rate' in fitness_result['details']:
                print(f"   成功率: {fitness_result['details']['success_rate']:.1%}")
            
        except Exception as e:
            print(f"❌ FitnessManager计算失败: {e}")
            print(f"🔄 使用传统备用方案")
            
            # 备用方案：使用传统fitness
            individual.phenotype = phenotype
            individual.fitness = phenotype.avg_reward
            individual.fitness_details = {
                'method': 'fallback',
                'category': 'error_recovery',
                'strategy': 'avg_reward_fallback',
                'reason': f'FitnessManager失败，使用avg_reward: {phenotype.avg_reward:.2f}',
                'error': str(e)
            }
            
            print(f"✅ 备用评估完成:")
            print(f"   Fitness (avg_reward): {individual.fitness:.2f}")
        
        print(f"   训练耗时: {training_time:.1f}s")
        
        return individual

    def _prepare_training_data_for_fitness(self, training_metrics, phenotype, training_time):
        """准备用于fitness计算的完整训练数据"""
        
        # 🎯 检查是否有episodes结果（来自enhanced_train.py的新格式）
        if isinstance(training_metrics, dict) and 'episode_results' in training_metrics:
            # 新的episodes-based数据格式
            return {
                'episodes_completed': training_metrics.get('episodes_completed', 0),
                'success_rate': training_metrics.get('success_rate', 0.0),
                'avg_best_distance': training_metrics.get('avg_best_distance', float('inf')),
                'learning_progress': training_metrics.get('learning_progress', 0.0),
                'avg_steps_to_best': training_metrics.get('avg_steps_to_best', 120000),
                'total_training_time': training_metrics.get('total_training_time', training_time),
                'episode_details': training_metrics.get('episode_details', []),
                'episode_results': training_metrics['episode_results'],
                # 兼容性数据
                'avg_reward': phenotype.avg_reward,
                'phenotype': phenotype
            }
        
        # 🎯 检查是否有详细的训练指标（模拟训练或旧格式）
        elif isinstance(training_metrics, dict):
            return {
                'success_rate': training_metrics.get('success_rate', phenotype.success_rate),
                'avg_reward': training_metrics.get('avg_reward', phenotype.avg_reward),
                'max_distance': training_metrics.get('max_distance', 0),
                'efficiency': training_metrics.get('efficiency', 0),
                'near_success_rate': training_metrics.get('near_success_rate', phenotype.success_rate + 0.1),
                'training_time': training_time,
                # 原始数据
                'raw_training_metrics': training_metrics,
                'phenotype': phenotype
            }
        
        # 🎯 最简单的数据格式（只有phenotype）
        else:
            return {
                'avg_reward': phenotype.avg_reward,
                'success_rate': phenotype.success_rate,
                'training_time': training_time,
                'phenotype': phenotype
            }
    
    # def evaluate_individual(self, individual: Individual, training_steps: int = 5000) -> Individual:
    #     """评估单个个体 - 支持新旧两种fitness评估系统"""
    #     print(f"\n🧬 评估个体 {individual.individual_id}")
    #     print(f"🤖 基因型: num_links={individual.genotype.num_links}, "
    #           f"link_lengths={[f'{x:.1f}' for x in individual.genotype.link_lengths]}")
    #     print(f"🧠 SAC参数: lr={individual.genotype.lr:.2e}, alpha={individual.genotype.alpha:.3f}")
    #     print(f"   总长度: {sum(individual.genotype.link_lengths):.1f}px")
        
    #     # 1. 根据基因型创建训练参数
    #     training_args = self._genotype_to_training_args(individual.genotype, training_steps)
        
    #     # 2. 运行训练
    #     start_time = time.time()
    #     if self.use_real_training:
    #         print(f"   🎯 使用enhanced_train.py进行真实训练 ({training_steps} steps)")
    #         try:
    #             training_metrics = self.training_interface.train_individual(training_args)
    #         except Exception as e:
    #             print(f"   ❌ 真实训练失败: {e}")
    #             print(f"   🔄 回退到模拟训练")
    #             training_metrics = self._run_simulated_training(training_args)
    #     else:
    #         print(f"   🎲 使用模拟训练 ({training_steps} steps)")
    #         training_metrics = self._run_simulated_training(training_args)
        
    #     training_time = time.time() - start_time
        
    #     # 3. 提取表型特征
    #     robot_config = {
    #         'num_links': individual.genotype.num_links,
    #         'link_lengths': individual.genotype.link_lengths,
    #         'lr': individual.genotype.lr,
    #         'alpha': individual.genotype.alpha
    #     }
        
    #     phenotype = self.feature_extractor.extract_from_training_data(training_metrics, robot_config)
        
    #     # 4. 🆕 Fitness评估 - 新旧系统选择
    #     if self.use_genetic_fitness and self.fitness_evaluator:
    #         # 🎯 使用新的遗传算法fitness评估系统
    #         training_performance = {
    #             'success_rate': phenotype.success_rate,
    #             'avg_reward': phenotype.avg_reward,
    #             'max_distance_covered': training_metrics.get('max_distance', 0),
    #             'efficiency': training_metrics.get('efficiency', 0),
    #             'near_success_rate': training_metrics.get('near_success_rate', 0)
    #         }
            
    #         try:
    #             fitness_result = self.fitness_evaluator.evaluate_fitness(
    #                 link_lengths=individual.genotype.link_lengths,
    #                 training_performance=training_performance
    #             )
                
    #             # 更新个体
    #             individual.phenotype = phenotype
    #             individual.fitness = fitness_result['fitness']
    #             individual.fitness_details = fitness_result  # 🆕 保存详细分析
                
    #             print(f"✅ 评估完成 (新系统):")
    #             print(f"   旧fitness (avg_reward): {phenotype.avg_reward:.2f}")
    #             print(f"   新fitness (分层评估): {individual.fitness:.3f}")
    #             print(f"   评估类别: {fitness_result['category']}")
    #             print(f"   评估策略: {fitness_result['strategy']}")
    #             print(f"   原因: {fitness_result['reason']}")
                
    #         except Exception as e:
    #             print(f"   ⚠️ 新fitness系统评估失败: {e}")
    #             print(f"   🔄 回退到传统fitness评估")
    #             individual.phenotype = phenotype
    #             individual.fitness = phenotype.avg_reward
                
    #     else:
    #         # 🔄 使用原有的简单fitness评估
    #         individual.phenotype = phenotype
    #         individual.fitness = phenotype.avg_reward
            
    #         print(f"✅ 评估完成 (传统系统):")
    #         print(f"   Fitness (avg_reward): {individual.fitness:.2f}")
        
    #     print(f"   成功率: {phenotype.success_rate:.2f}, 耗时: {training_time:.1f}s")
        
    #     return individual
    
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
        args.buffer_size = genotype.buffer_capacity  # 🔧 修复：添加缺失的buffer_size参数
        args.warmup_steps = genotype.warmup_steps
        args.target_entropy_factor = genotype.target_entropy_factor
        
        # 🎯 设置训练步数
        args.total_steps = training_steps
        
        # 📁 设置保存目录
        individual_dir = os.path.join(self.base_save_dir, f"individual_{int(time.time() * 1000) % 100000}")
        args.save_dir = individual_dir
        
        # 🔧 设置其他训练参数
        args.update_frequency = getattr(self.base_args, 'update_frequency', 1)
        args.num_processes = 1  # 🔧 强制单进程，避免多进程通信问题
        args.seed = getattr(self.base_args, 'seed', 42)
        
        # 🔧 添加更多必需参数
        args.ppo_epochs = 10
        args.clip_epsilon = 0.2
        args.entropy_coef = 0.01
        args.value_coef = 0.5
        args.max_grad_norm = 0.5
        
        # 🆕 渲染和静默控制
        args.render = self.enable_rendering
        args.silent = self.silent_mode
        
        return args
    
    def _run_simulated_training(self, args) -> Dict[str, Any]:
        """运行模拟训练（备用方案）- 增强版，支持新fitness系统"""
        # 基于基因型预测大致性能
        num_links = getattr(args, 'num_joints', 4)
        link_lengths = getattr(args, 'link_lengths', [80.0] * num_links)
        
        # 简单的启发式评估
        total_reach = sum(link_lengths)
        complexity_penalty = abs(num_links - 4) * 10  # 4链节最优
        length_variance_penalty = np.var(link_lengths) / 10 if len(link_lengths) > 1 else 0
        
        # 基于超参数的性能预测
        lr_factor = 1.0 if 1e-5 <= args.lr <= 1e-3 else 0.5
        alpha_factor = 1.0 if 0.1 <= args.alpha <= 0.5 else 0.7
        
        base_reward = min(100, total_reach / 5) - complexity_penalty - length_variance_penalty
        base_reward *= lr_factor * alpha_factor
        
        # 添加随机性
        noise = np.random.normal(0, 15)
        final_reward = base_reward + noise
        
        # 🆕 为新fitness系统添加更多指标
        success_rate = max(0, min(1, (final_reward + 50) / 150))
        
        # 基于机器人长度估算最大距离覆盖
        max_distance = total_reach * success_rate * np.random.uniform(0.6, 0.9)
        
        # 估算效率（基于成功率和复杂度）
        efficiency = success_rate * (1.0 - complexity_penalty / 50) * np.random.uniform(0.7, 1.0)
        efficiency = max(0, min(1, efficiency))
        
        # 估算接近成功率（通常比成功率高一些）
        near_success_rate = min(1.0, success_rate + np.random.uniform(0.1, 0.3))
        
        return {
            'avg_reward': final_reward,
            'success_rate': success_rate,
            'min_distance': max(10, 200 - final_reward * 2),
            'trajectory_smoothness': np.random.uniform(0.3, 0.8),
            'collision_rate': np.random.uniform(0.0, 0.3),
            'exploration_area': np.random.uniform(100, 500),
            'action_variance': np.random.uniform(0.1, 0.5),
            'learning_rate': np.random.uniform(0.1, 0.9),
            'final_critic_loss': np.random.uniform(0.1, 5.0),
            'final_actor_loss': np.random.uniform(0.1, 2.0),
            'training_stability': np.random.uniform(0.3, 0.9),
            # 🆕 新fitness系统需要的指标
            'max_distance': max_distance,
            'efficiency': efficiency,
            'near_success_rate': near_success_rate
        }


# 🧪 测试函数
def test_training_adapter():
    """测试训练适配器 - 包括新旧fitness系统"""
    print("🧪 开始测试MAP-Elites训练适配器 (新旧fitness系统对比)\n")
    
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
    
    # 2. 测试新旧两种fitness系统
    fitness_modes = [
        (False, "传统Fitness (avg_reward)"),
        (True, "遗传算法分层Fitness")
    ]
    
    from map_elites_core import RobotGenotype, RobotPhenotype, Individual
    
    # 创建测试个体
    test_genotypes = [
        RobotGenotype(num_links=2, link_lengths=[40, 40], lr=2e-4, alpha=0.25),    # 短机器人
        RobotGenotype(num_links=3, link_lengths=[60, 60, 60], lr=2e-4, alpha=0.25), # 中等机器人  
        RobotGenotype(num_links=4, link_lengths=[80, 80, 80, 80], lr=2e-4, alpha=0.25) # 长机器人
    ]
    
    results = []
    
    for use_genetic, mode_name in fitness_modes:
        print(f"\n{'='*50}")
        print(f"📊 测试模式: {mode_name}")
        print(f"{'='*50}")
        
        try:
            adapter = MAPElitesTrainingAdapter(
                base_args, 
                "./test_adapter_results", 
                use_real_training=False,  # 使用模拟训练进行快速测试
                use_genetic_fitness=use_genetic
            )
            
            mode_results = []
            
            for i, genotype in enumerate(test_genotypes):
                print(f"\n🤖 测试机器人 {i+1}: {genotype.num_links}链节, 总长{sum(genotype.link_lengths):.1f}px")
                
                individual = Individual(
                    genotype=genotype,
                    phenotype=RobotPhenotype()
                )
                
                # 评估个体
                start_time = time.time()
                evaluated = adapter.evaluate_individual(individual, training_steps=100)
                end_time = time.time()
                
                result = {
                    'genotype': genotype,
                    'fitness': evaluated.fitness,
                    'success_rate': evaluated.phenotype.success_rate,
                    'total_reach': evaluated.phenotype.total_reach,
                    'evaluation_time': end_time - start_time,
                    'fitness_details': getattr(evaluated, 'fitness_details', None)
                }
                
                mode_results.append(result)
                
                print(f"   适应度: {evaluated.fitness:.3f}")
                print(f"   成功率: {evaluated.phenotype.success_rate:.2f}")
                if hasattr(evaluated, 'fitness_details') and evaluated.fitness_details:
                    print(f"   类别: {evaluated.fitness_details['category']}")
                    print(f"   策略: {evaluated.fitness_details['strategy']}")
            
            results.append((mode_name, mode_results))
            
        except Exception as e:
            print(f"❌ {mode_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. 对比分析
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print("📊 新旧Fitness系统对比分析")
        print(f"{'='*60}")
        
        print("机器人配置 | 传统Fitness | 新Fitness | 新系统类别")
        print("-" * 55)
        
        for i in range(len(test_genotypes)):
            genotype = test_genotypes[i]
            old_result = results[0][1][i]  # 传统系统结果
            new_result = results[1][1][i]  # 新系统结果
            
            config = f"{genotype.num_links}链节,{sum(genotype.link_lengths):.0f}px"
            old_fitness = old_result['fitness']
            new_fitness = new_result['fitness']
            category = new_result['fitness_details']['category'] if new_result['fitness_details'] else 'N/A'
            
            print(f"{config:12} | {old_fitness:10.2f} | {new_fitness:9.3f} | {category}")
    
    print(f"\n✅ 训练适配器测试完成!")
    if len(results) >= 2:
        print(f"🎯 新的遗传算法Fitness系统已成功集成!")


if __name__ == "__main__":
    test_training_adapter()