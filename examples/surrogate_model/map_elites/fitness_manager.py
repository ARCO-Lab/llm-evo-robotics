"""
统一的Fitness计算管理器
支持多种fitness计算策略：episodes-based, genetic, traditional
"""

import numpy as np
from typing import Dict, Any, List, Optional
from genetic_fitness_evaluator import GeneticFitnessEvaluator


class FitnessManager:
    """统一的Fitness计算管理器"""
    
    def __init__(self, use_genetic_fitness: bool = True, 
                 primary_strategy: str = 'episodes'):
        """
        初始化FitnessManager
        
        Args:
            use_genetic_fitness: 是否启用遗传算法fitness作为备选
            primary_strategy: 主要策略 ('episodes', 'genetic', 'traditional')
        """
        self.use_genetic_fitness = use_genetic_fitness
        self.primary_strategy = primary_strategy
        
        # 初始化各种fitness评估器
        self.genetic_evaluator = None
        if use_genetic_fitness:
            try:
                self.genetic_evaluator = GeneticFitnessEvaluator()
                print("🎯 遗传算法Fitness评估器已初始化")
            except Exception as e:
                print(f"⚠️ 遗传算法Fitness评估器初始化失败: {e}")
        
        print(f"🏆 FitnessManager初始化完成，主策略: {primary_strategy}")
    
    def calculate_fitness(self, individual, training_results, 
                         training_type: Optional[str] = None) -> Dict[str, Any]:
        """
        统一的fitness计算入口
        
        Args:
            individual: 个体对象
            training_results: 训练结果数据
            training_type: 强制指定的训练类型，None则自动检测
        
        Returns:
            包含fitness和详细信息的字典
        """
        # 自动检测训练结果类型
        if training_type is None:
            training_type = self._detect_training_type(training_results)
        
        print(f"🎯 计算Fitness: 类型={training_type}")
        
        try:
            if training_type == 'episodes':
                return self._calculate_episodes_fitness(individual, training_results)
            elif training_type == 'genetic':
                return self._calculate_genetic_fitness(individual, training_results)
            elif training_type == 'traditional':
                return self._calculate_traditional_fitness(individual, training_results)
            else:
                print(f"⚠️ 未知的训练类型: {training_type}，使用传统方法")
                return self._calculate_traditional_fitness(individual, training_results)
                
        except Exception as e:
            print(f"❌ Fitness计算失败: {e}")
            # 回退到最简单的fitness计算
            return self._calculate_fallback_fitness(individual, training_results)
    
    def _detect_training_type(self, training_results) -> str:
        """自动检测训练结果的类型"""
        print(f"🔍 检测训练结果类型:")
        print(f"   数据类型: {type(training_results)}")
        
        if isinstance(training_results, dict):
            print(f"   包含键: {list(training_results.keys())}")
            
            # 检查是否包含episodes结果
            if ('episode_results' in training_results or 
                'episodes_completed' in training_results or
                'success_rate' in training_results):
                print(f"   ✅ 检测为episodes类型")
                return 'episodes'
            
            # 检查是否包含genetic fitness需要的数据
            if ('success_rate' in training_results or 
                'avg_reward' in training_results):
                print(f"   ✅ 检测为genetic类型")
                return 'genetic'
        
        print(f"   ✅ 检测为traditional类型")
        return 'traditional'
    
    def _calculate_episodes_fitness(self, individual, training_results) -> Dict[str, Any]:
        """
        基于episodes结果的fitness计算
        这是最详细和准确的fitness计算方法
        """
        print("📊 使用Episodes-based Fitness计算")
        
        # 提取episodes数据
        episodes_completed = training_results.get('episodes_completed', 0)
        success_rate = training_results.get('success_rate', 0.0)
        avg_best_distance = training_results.get('avg_best_distance', float('inf'))
        learning_progress = training_results.get('learning_progress', 0.0)
        avg_steps_to_best = training_results.get('avg_steps_to_best', 120000)
        total_training_time = training_results.get('total_training_time', 0.0)
        episode_details = training_results.get('episode_details', [])
        
        # 🎯 多维度fitness组件计算
        fitness_components = {
            # 1. 完成度评分 (0-1)
            'completion_rate': min(1.0, episodes_completed / 2.0),
            
            # 2. 成功率评分 (0-1) - 最重要
            'success_rate': success_rate,
            
            # 3. 距离性能评分 (0-1)
            'distance_score': self._calculate_distance_score(avg_best_distance),
            
            # 4. 学习能力评分 (0-1)
            'learning_score': self._calculate_learning_score(learning_progress),
            
            # 5. 效率评分 (0-1)
            'efficiency_score': self._calculate_efficiency_score(avg_steps_to_best),
            
            # 6. 稳定性评分 (0-1)
            'stability_score': self._calculate_stability_score(episode_details),
            
            # 7. 时间效率评分 (0-1)
            'time_efficiency_score': self._calculate_time_efficiency_score(total_training_time)
        }
        
        # 🎯 加权计算最终fitness
        weights = {
            'completion_rate': 0.15,    # 完成训练很重要
            'success_rate': 0.35,       # 成功率最重要
            'distance_score': 0.25,     # 距离性能很重要
            'learning_score': 0.10,     # 学习能力加分
            'efficiency_score': 0.08,   # 效率加分
            'stability_score': 0.05,    # 稳定性加分
            'time_efficiency_score': 0.02  # 时间效率小加分
        }
        
        final_fitness = sum(
            fitness_components[component] * weight 
            for component, weight in weights.items()
        )
        
        # 🎯 分类个体表现
        category, strategy, reason = self._classify_episodes_performance(
            success_rate, avg_best_distance, learning_progress, episodes_completed
        )
        
        return {
            'fitness': final_fitness,
            'details': {
                'method': 'episodes_based',
                'category': category,
                'strategy': strategy,
                'reason': reason,
                'episodes_completed': episodes_completed,
                'success_rate': success_rate,
                'avg_best_distance': avg_best_distance,
                'learning_progress': learning_progress,
                'fitness_components': fitness_components,
                'component_weights': weights,
                'training_time_minutes': total_training_time / 60,
                'raw_data': training_results
            }
        }
    
    def _calculate_genetic_fitness(self, individual, training_results) -> Dict[str, Any]:
        """使用遗传算法fitness评估器"""
        print("🧬 使用Genetic Fitness计算")
        
        if not self.genetic_evaluator:
            return self._calculate_traditional_fitness(individual, training_results)
        
        # 准备遗传算法评估器需要的数据格式
        if hasattr(training_results, 'success_rate'):
            # 来自phenotype的数据
            training_performance = {
                'success_rate': training_results.success_rate,
                'avg_reward': training_results.avg_reward,
                'max_distance_covered': getattr(training_results, 'total_reach', 0),
                'efficiency': getattr(training_results, 'learning_efficiency', 0),
                'near_success_rate': getattr(training_results, 'success_rate', 0) + 0.1
            }
        elif isinstance(training_results, dict):
            # 来自dict的数据
            training_performance = {
                'success_rate': training_results.get('success_rate', 0),
                'avg_reward': training_results.get('avg_reward', 0),
                'max_distance_covered': training_results.get('max_distance', 0),
                'efficiency': training_results.get('efficiency', 0),
                'near_success_rate': training_results.get('near_success_rate', 0)
            }
        else:
            # 没有有效数据，使用默认值
            training_performance = {
                'success_rate': 0.0,
                'avg_reward': 0.0,
                'max_distance_covered': 0.0,
                'efficiency': 0.0,
                'near_success_rate': 0.0
            }
        
        try:
            fitness_result = self.genetic_evaluator.evaluate_fitness(
                link_lengths=individual.genotype.link_lengths,
                training_performance=training_performance
            )
            
            # 添加方法标识
            fitness_result['details'] = fitness_result.copy()
            fitness_result['details']['method'] = 'genetic_algorithm'
            
            return fitness_result
            
        except Exception as e:
            print(f"⚠️ 遗传算法fitness计算失败: {e}")
            return self._calculate_traditional_fitness(individual, training_results)
    
    def _calculate_traditional_fitness(self, individual, training_results) -> Dict[str, Any]:
        """传统的简单fitness计算"""
        print("📈 使用Traditional Fitness计算")
        
        # 尝试提取avg_reward
        if hasattr(training_results, 'avg_reward'):
            fitness = training_results.avg_reward
        elif isinstance(training_results, dict) and 'avg_reward' in training_results:
            fitness = training_results['avg_reward']
        else:
            fitness = 0.0
        
        return {
            'fitness': fitness,
            'details': {
                'method': 'traditional',
                'category': 'simple_reward',
                'strategy': 'maximize_avg_reward',
                'reason': f'平均奖励: {fitness:.2f}',
                'avg_reward': fitness
            }
        }
    
    def _calculate_fallback_fitness(self, individual, training_results) -> Dict[str, Any]:
        """最后的回退fitness计算"""
        print("🆘 使用Fallback Fitness计算")
        
        return {
            'fitness': 0.0,
            'details': {
                'method': 'fallback',
                'category': 'error_recovery',
                'strategy': 'minimal_score',
                'reason': 'Fitness计算失败，使用最小分数',
                'error': True
            }
        }
    
    # === 辅助计算方法 ===
    
    def _calculate_distance_score(self, avg_best_distance: float) -> float:
        """计算距离性能得分 (0-1)"""
        if avg_best_distance == float('inf'):
            return 0.0
        
        # 目标阈值是20px，200px以内给部分分数
        if avg_best_distance <= 20:
            return 1.0
        elif avg_best_distance <= 200:
            return max(0, (200 - avg_best_distance) / 180)
        else:
            return 0.0
    
    def _calculate_learning_score(self, learning_progress: float) -> float:
        """计算学习能力得分 (0-1)"""
        # 学习进步 > 0.2 算很好，< -0.2 算很差
        if learning_progress >= 0.2:
            return 1.0
        elif learning_progress >= 0:
            return 0.5 + learning_progress * 2.5  # 0到0.2映射到0.5到1.0
        elif learning_progress >= -0.2:
            return 0.5 + learning_progress * 2.5  # -0.2到0映射到0到0.5
        else:
            return 0.0
    
    def _calculate_efficiency_score(self, avg_steps_to_best: float) -> float:
        """计算效率得分 (0-1)"""
        if avg_steps_to_best <= 0:
            return 0.0
        
        # 越早达到最佳距离越好，120000步是上限
        return max(0, min(1, (120000 - avg_steps_to_best) / 120000))
    
    def _calculate_stability_score(self, episode_details: List[Dict]) -> float:
        """计算稳定性得分 (0-1)"""
        if not episode_details or len(episode_details) < 2:
            return 0.5  # 默认中等稳定性
        
        # 计算两个episodes之间的性能方差
        scores = [ep.get('score', 0) for ep in episode_details]
        if len(scores) >= 2:
            variance = np.var(scores)
            # 方差越小越稳定，0.1是一个合理的阈值
            return max(0, min(1, 1 - variance / 0.1))
        
        return 0.5
    
    def _calculate_time_efficiency_score(self, total_training_time: float) -> float:
        """计算时间效率得分 (0-1)"""
        if total_training_time <= 0:
            return 1.0
        
        # 假设理想训练时间是30分钟，2小时是上限
        ideal_time = 30 * 60  # 30分钟
        max_time = 120 * 60   # 2小时
        
        if total_training_time <= ideal_time:
            return 1.0
        elif total_training_time <= max_time:
            return max(0, (max_time - total_training_time) / (max_time - ideal_time))
        else:
            return 0.0
    
    def _classify_episodes_performance(self, success_rate: float, avg_best_distance: float, 
                                     learning_progress: float, episodes_completed: int) -> tuple:
        """分类episodes性能表现"""
        
        if episodes_completed < 2:
            return ('INCOMPLETE_TRAINING', 'complete_training', '训练未完成')
        
        if success_rate >= 0.8 and avg_best_distance < 30:
            return ('EXCELLENT_PERFORMER', 'maintain_excellence', 
                   f'卓越表现: 成功率{success_rate:.1%}, 距离{avg_best_distance:.1f}px')
        
        elif success_rate >= 0.5 and avg_best_distance < 50:
            return ('GOOD_PERFORMER', 'optimize_consistency', 
                   f'良好表现: 成功率{success_rate:.1%}, 距离{avg_best_distance:.1f}px')
        
        elif avg_best_distance < 100 and learning_progress > 0.1:
            return ('IMPROVING_PERFORMER', 'encourage_learning', 
                   f'进步中: 距离{avg_best_distance:.1f}px, 学习进步{learning_progress:+.2f}')
        
        elif avg_best_distance < 100:
            return ('CLOSE_PERFORMER', 'distance_optimization', 
                   f'接近成功: 距离{avg_best_distance:.1f}px')
        
        elif learning_progress > 0.2:
            return ('LEARNING_PERFORMER', 'learning_potential', 
                   f'学习能力强: 进步{learning_progress:+.2f}')
        
        else:
            return ('POOR_PERFORMER', 'needs_improvement', 
                   f'需要改进: 成功率{success_rate:.1%}, 距离{avg_best_distance:.1f}px')


# === 测试函数 ===
def test_fitness_manager():
    """测试FitnessManager的各种功能"""
    print("🧪 开始测试FitnessManager")
    
    # 创建FitnessManager
    fm = FitnessManager(use_genetic_fitness=True, primary_strategy='episodes')
    
    # 模拟个体
    class MockGenotype:
        def __init__(self):
            self.link_lengths = [80, 70, 60, 50]
            self.lr = 3e-4
            self.alpha = 0.2
    
    class MockIndividual:
        def __init__(self):
            self.genotype = MockGenotype()
            self.individual_id = "test_001"
    
    individual = MockIndividual()
    
    # 测试不同类型的训练结果
    test_cases = [
        {
            'name': 'Episodes结果',
            'data': {
                'episodes_completed': 2,
                'success_rate': 0.8,
                'avg_best_distance': 25.0,
                'learning_progress': 0.3,
                'avg_steps_to_best': 5000,
                'total_training_time': 1800,  # 30分钟
                'episode_details': [
                    {'score': 0.7, 'success': False, 'best_distance': 35},
                    {'score': 0.9, 'success': True, 'best_distance': 15}
                ]
            }
        },
        {
            'name': 'Genetic结果',
            'data': {
                'success_rate': 0.6,
                'avg_reward': 50.0,
                'max_distance': 150,
                'efficiency': 0.7
            }
        },
        {
            'name': '传统结果',
            'data': {
                'avg_reward': 30.0
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*50}")
        print(f"测试: {test_case['name']}")
        print(f"{'='*50}")
        
        result = fm.calculate_fitness(individual, test_case['data'])
        
        print(f"Fitness: {result['fitness']:.3f}")
        print(f"方法: {result['details']['method']}")
        print(f"类别: {result['details']['category']}")
        print(f"策略: {result['details']['strategy']}")
        print(f"原因: {result['details']['reason']}")
    
    print("\n🎉 FitnessManager测试完成!")


if __name__ == "__main__":
    test_fitness_manager()