import time
import argparse
from typing import List, Optional
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from map_elites_core import MAPElitesArchive, RobotMutator, Individual, RobotGenotype, RobotPhenotype
from training_adapter import MAPElitesTrainingAdapter
import os
import sys
import pickle
import traceback
import torch
import numpy as np

# 导入可视化工具
try:
    from map_elites_visualizer import MAPElitesVisualizer
    from network_loss_visualizer import NetworkLossVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  可视化工具导入失败: {e}")
    VISUALIZATION_AVAILABLE = False

# 导入成功记录系统
from success_logger import (
    SuccessLogger, 
    create_robot_structure, 
    create_training_params, 
    create_performance_metrics
)

import argparse
from map_elites_core import Individual , RobotGenotype, RobotPhenotype
from training_adapter import MAPElitesTrainingAdapter

# 🔇 全局静默模式控制
GLOBAL_SILENT_MODE = False

def silent_print(*args, **kwargs):
    """静默模式下的打印函数"""
    if not GLOBAL_SILENT_MODE:
        print(*args, **kwargs)

def init_worker_process():
    import signal
    import os
    
    # 🔧 设置强力信号处理，让子进程能够立即响应Ctrl+C
    def force_signal_handler(signum, frame):
        print(f"\n🛑 子进程 {mp.current_process().pid} 收到中断信号，立即退出...")
        os._exit(1)  # 使用_exit强制退出，不执行清理
    
    signal.signal(signal.SIGINT, force_signal_handler)
    signal.signal(signal.SIGTERM, force_signal_handler)
    
    process_id = mp.current_process().pid
    torch.manual_seed(42 + process_id)
    np.random.seed(42 + process_id)

    torch.set_num_threads(1)
    if torch.cuda.is_available():
        device = (process_id % torch.cuda.device_count())
        torch.cuda.set_device(device)
    print(f"🔄 进程 {process_id} 初始化完成")

def evaluate_individual_isolated(individual_data, base_args_dict, training_steps):
    """在独立进程中评估个体"""

    try:

        process_id = os.getpid()
        enable_rendering = base_args_dict.get('enable_rendering', False)
        silent_mode = base_args_dict.get('silent_mode', False)  # 🔧 修复：默认不静默
        
        silent_print(f"🎨 进程 {process_id} 接收参数: rendering={enable_rendering}, silent={silent_mode}")
        silent_print(f"进程 {process_id}开始训练个体 {individual_data['individual_id']}")
        base_args = argparse.Namespace(**base_args_dict)

        genotype = RobotGenotype(
            num_links = individual_data['num_links'],
            link_lengths = individual_data['link_lengths'],
            lr = individual_data['lr'],
            alpha = individual_data['alpha']
        )
        individual = Individual(
            individual_id = individual_data['individual_id'],
            genotype = genotype,
            phenotype = RobotPhenotype(),
            generation = individual_data['generation'],
            parent_id = individual_data['parent_id']
        )

        adapter = MAPElitesTrainingAdapter(
            base_args,
            enable_rendering = base_args_dict.get('enable_rendering', False),  # 🔧 使用传递的参数
            silent_mode = base_args_dict.get('silent_mode', False),            # 🔧 修复：默认不静默
            use_genetic_fitness = True
        )
        result = adapter.evaluate_individual(individual, training_steps)
        silent_print(f"✅ 进程 {process_id} 完成训练个体 {individual_data['individual_id']}, fitness: {result.fitness:.3f}")

        return {
            'individual_id': result.individual_id,
            'fitness': result.fitness,
            'fitness_details': getattr(result, 'fitness_details', {}),
            'generation': result.generation,
            'parent_id': result.parent_id,
            'genotype': {
                'num_links': result.genotype.num_links,
                'link_lengths': result.genotype.link_lengths,
                'lr': result.genotype.lr,
                'alpha': result.genotype.alpha
            },
            'phenotype': {
                'avg_reward': result.phenotype.avg_reward,
                'success_rate': getattr(result.phenotype, 'success_rate', 0.0),
                'min_distance': getattr(result.phenotype, 'min_distance', float('inf'))
            }
        }
        
    except Exception as e:
        silent_print(f"❌ 进程 {os.getpid()} 训练个体 {individual_data['individual_id']} 失败: {e}")
        traceback.print_exc()
        return None


class MAPElitesEvolutionTrainer:
    """MAP-Elites进化训练器 - 支持遗传算法Fitness评估系统"""
    
    def __init__(self, base_args, num_initial_random: int = 20, 
                 training_steps_per_individual: int = 3000,
                 enable_rendering: bool = False,    # 🆕 是否启用渲染
                 silent_mode: bool = False,         # 🔧 修复：默认不静默
                 use_genetic_fitness: bool = True,  # 🆕 是否使用遗传算法fitness
                 enable_multiprocess: bool = False,
                 max_workers: int = 4,
                 use_shared_ppo: bool = False,      # 🆕 是否使用共享PPO训练
                 success_threshold: float = 0.7,   # 🆕 成功判定阈值
                 enable_success_logging: bool = True, # 🆕 是否启用成功记录
                 enable_visualization: bool = True,  # 🆕 是否启用可视化
                 visualization_interval: int = 5):  # 🆕 可视化更新间隔
        
        # 初始化基本属性
        self.num_initial_random = num_initial_random
        self.training_steps_per_individual = training_steps_per_individual
        self.use_genetic_fitness = use_genetic_fitness
        self.enable_multiprocess = enable_multiprocess
        self.max_workers = min(max_workers, mp.cpu_count()) if enable_multiprocess else 1
        self.base_args = base_args
        self.use_shared_ppo = use_shared_ppo  # 🆕 共享PPO设置
        self.success_threshold = success_threshold  # 🆕 成功阈值
        self.enable_success_logging = enable_success_logging  # 🆕 成功记录开关
        self.enable_visualization = enable_visualization and VISUALIZATION_AVAILABLE  # 🆕 可视化开关
        self.visualization_interval = visualization_interval  # 🆕 可视化更新间隔

        # 🆕 初始化成功记录器
        self.success_logger = None
        if enable_success_logging:
            print(f"📊 初始化实验成功记录器 (阈值: {success_threshold})")
            self.success_logger = SuccessLogger(
                base_dir="./experiment_results",
                success_threshold=success_threshold
            )

        # 🆕 初始化共享PPO训练器
        self.shared_ppo_trainer = None
        if use_shared_ppo:
            print("🚀 初始化共享PPO训练器...")
            try:
                # 🔧 启用共享PPO训练器
                print("🤖 正在导入共享PPO训练器...")
                
                # 导入共享PPO训练器
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, current_dir)
                
                from shared_ppo_trainer import SharedPPOTrainer
                
                model_config = {
                    'observation_dim': 14,  # reacher2d观察维度
                    'action_dim': 3,        # reacher2d动作维度
                    'hidden_dim': 256
                }
                
                training_config = {
                    'lr': getattr(base_args, 'lr', 2e-4),
                    'buffer_size': 20000,
                    'min_batch_size': 100,  # 🔧 减少批次大小
                    'model_path': f'{base_args.save_dir}/shared_ppo_model.pth',
                    'update_interval': 50   # 🔧 减少更新间隔
                }
                
                self.shared_ppo_trainer = SharedPPOTrainer(model_config, training_config)
                self.shared_ppo_trainer.start_training()
                print("✅ 共享PPO训练器启动成功")
                
            except ImportError as e:
                print(f"⚠️ 无法导入共享PPO训练器，回退到独立训练: {e}")
                self.use_shared_ppo = False
            except Exception as e:
                print(f"⚠️ 共享PPO训练器初始化失败，回退到独立训练: {e}")
                self.use_shared_ppo = False

        # 🆕 初始化可视化工具
        self.map_elites_visualizer = None
        self.loss_visualizer = None
        if self.enable_visualization:
            print("🎨 初始化可视化工具...")
            try:
                self.map_elites_visualizer = MAPElitesVisualizer(
                    output_dir=os.path.join(base_args.save_dir, 'visualizations')
                )
                self.loss_visualizer = NetworkLossVisualizer(
                    log_dir=os.path.join(base_args.save_dir, 'training_logs'),
                    output_dir=os.path.join(base_args.save_dir, 'visualizations')
                )
                print("✅ 可视化工具初始化成功")
            except Exception as e:
                print(f"⚠️  可视化工具初始化失败: {e}")
                self.enable_visualization = False

        # 初始化组件（在共享PPO训练器之后）
        self.archive = MAPElitesArchive()
        self.mutator = RobotMutator()
        self.adapter = MAPElitesTrainingAdapter(
            base_args, 
            enable_rendering=enable_rendering,  # 🆕 传递渲染设置
            silent_mode=silent_mode,           # 🆕 传递静默设置
            use_genetic_fitness=use_genetic_fitness,  # 🆕 传递fitness设置
            shared_ppo_trainer=self.shared_ppo_trainer  # 🆕 传递共享PPO训练器
        )

        if enable_multiprocess:
            silent_print(f"🔄 启用多进程训练 (最大进程数: {self.max_workers})")
        else:
            silent_print("🔄 使用单进程训练")

        silent_print("🧬 MAP-Elites进化训练器已初始化")
        silent_print(f"🎯 Fitness评估: {'遗传算法分层系统' if use_genetic_fitness else '传统平均奖励'}")
        silent_print(f"🎨 环境渲染: {'启用' if enable_rendering else '禁用'}")
        silent_print(f"📊 数据可视化: {'启用' if self.enable_visualization else '禁用'}")
        silent_print(f"🤝 PPO训练: {'共享模式' if self.use_shared_ppo else '独立模式'}")

    
    def run_evolution(self, num_generations: int = 50, individuals_per_generation: int = 10):
        """运行MAP-Elites进化"""
        print(f"🚀 开始MAP-Elites进化")
        print(f"📊 参数: {num_generations}代, 每代{individuals_per_generation}个个体")
        print(f"🎯 Fitness系统: {'遗传算法分层评估' if self.use_genetic_fitness else '传统平均奖励'}")
        
        # 第0代：随机初始化
        print(f"\n🎲 第0代: 随机初始化")
        self._initialize_random_population()
        
        # 进化循环
        for generation in range(1, num_generations + 1):
            print(f"\n🧬 第{generation}代")
            
            # 生成新个体
            new_individuals = []
            for i in range(individuals_per_generation):
                if np.random.random() < 0.1:  # 10% 概率生成随机个体
                    individual = self._create_random_individual(generation)
                else:  # 90% 概率变异现有个体
                    individual = self._create_mutant_individual(generation)
                
                if individual:
                    new_individuals.append(individual)
            
            # 评估新个体
            # for i, individual in enumerate(new_individuals):
            #     print(f"  个体 {i+1}/{len(new_individuals)}")
            #     evaluated_individual = self.adapter.evaluate_individual(
            #         individual, self.training_steps_per_individual
            #     )
            #     self.archive.add_individual(evaluated_individual)

            if len(new_individuals) > 0:
                print(f"📦 第{generation}代创建了 {len(new_individuals)} 个新个体，开始评估...")
                evaluated_individuals = self._evaluate_individuals_parallel(new_individuals)
                
                # 添加到存档
                for individual in evaluated_individuals:
                    self.archive.add_individual(individual)
            else:
                print(f"⚠️ 第{generation}代没有创建新个体")
            
            # 输出代际统计
            self._print_generation_stats(generation)
            
            # 保存存档
            if generation % 5 == 0:
                self.archive.generation = generation
                self.archive.save_archive()
            
            # 🆕 生成可视化（根据间隔）
            if self.enable_visualization and generation % self.visualization_interval == 0:
                self._generate_visualizations(generation)
        
        # 最终结果打印
        print(f"\n🎉 进化完成!")
        self._print_final_results()
        
        # 🆕 生成实验总结并关闭成功记录器
        if self.success_logger:
            self.success_logger.close()
        
        # 🆕 生成最终可视化报告
        if self.enable_visualization:
            self._generate_final_visualization_report()
        
        # 🆕 清理共享PPO训练器
        self._cleanup_shared_ppo()
    def _initialize_random_population(self):
        """初始化随机种群 - 支持并行评估"""
        print(f"🎲 创建 {self.num_initial_random} 个随机个体...")
        
        # 批量创建个体
        individuals = []
        for i in range(self.num_initial_random):
            individual = self._create_random_individual(0)
            individuals.append(individual)
        
        print(f"📦 个体创建完成，开始评估...")
        
        # 并行或顺序评估
        evaluated_individuals = self._evaluate_individuals_parallel(individuals)
        
        # 添加调试信息
        for i, individual in enumerate(evaluated_individuals):
            print(f"🔍 个体 {i+1} 评估结果:")
            print(f"   ID: {individual.individual_id}")
            print(f"   Fitness: {individual.fitness:.3f}")
            if hasattr(individual, 'fitness_details') and individual.fitness_details:
                print(f"   类别: {individual.fitness_details.get('category', 'N/A')}")
            
            # 添加到存档
            self.archive.add_individual(individual)
        
        stats = self.archive.get_statistics()
        print(f"📊 初始化完成: 存档大小={stats['size']}, 最佳适应度={stats['best_fitness']:.3f}")
    # def _initialize_random_population(self):
    #     """初始化随机种群"""
    #     for i in range(self.num_initial_random):
    #         print(f"  初始化个体 {i+1}/{self.num_initial_random}")
    #         individual = self._create_random_individual(0)
    #         evaluated_individual = self.adapter.evaluate_individual(
    #             individual, self.training_steps_per_individual
    #         )
    #          # 🆕 添加这些调试信息
    #         print(f"🔍 调试 - 个体 {i+1} 评估结果:")
    #         print(f"   Fitness: {evaluated_individual.fitness}")
    #         if hasattr(evaluated_individual, 'fitness_details'):
    #             print(f"   Fitness详情: {evaluated_individual.fitness_details}")
    #         else:
    #             print(f"   ⚠️ 没有fitness_details属性")

    #         self.archive.add_individual(evaluated_individual)
        
    #     stats = self.archive.get_statistics()
    #     print(f"📊 初始化完成: 存档大小={stats['size']}, 最佳适应度={stats['best_fitness']:.3f}")
    
    def _create_random_individual(self, generation: int) -> Individual:
        """创建随机个体"""
        genotype = self.mutator.random_genotype()
        # 🔧 确保每个个体都有唯一的ID
        import random
        unique_id = f"gen_{generation}_{int(time.time() * 1000000) % 1000000}_{random.randint(1000, 9999)}"
        return Individual(
            genotype=genotype,
            phenotype=RobotPhenotype(),
            generation=generation,
            individual_id=unique_id
        )
    
    def _create_mutant_individual(self, generation: int) -> Optional[Individual]:
        """创建变异个体"""
        parent = self.archive.get_random_elite()
        if parent is None:
            return self._create_random_individual(generation)
        
        mutant_genotype = self.mutator.mutate(parent.genotype)
        # 🔧 确保变异个体也有唯一的ID
        import random
        unique_id = f"gen_{generation}_{int(time.time() * 1000000) % 1000000}_{random.randint(1000, 9999)}"
        return Individual(
            genotype=mutant_genotype,
            phenotype=RobotPhenotype(),
            generation=generation,
            parent_id=parent.individual_id,
            individual_id=unique_id
        )
    def _evaluate_individuals_parallel(self, individuals):
        """并行评估多个个体 - 支持多进程渲染"""
        if not self.enable_multiprocess or len(individuals) <= 1:
            # 单进程模式
            return self._evaluate_individuals_sequential(individuals)
        
        print(f"🔄 开始并行评估 {len(individuals)} 个个体 (使用 {self.max_workers} 个进程)")
        if self.adapter.enable_rendering:
            print("🎨 多进程渲染模式：每个进程将显示独立的渲染窗口")
        
        # 准备可序列化的数据
        individual_data_list = []
        for individual in individuals:
            individual_data = {
                'individual_id': individual.individual_id,
                'num_links': individual.genotype.num_links,
                'link_lengths': individual.genotype.link_lengths,
                'lr': individual.genotype.lr,
                'alpha': individual.genotype.alpha,
                'generation': individual.generation,
                'parent_id': individual.parent_id
            }
            individual_data_list.append(individual_data)
        
        # 准备可序列化的参数
        base_args_dict = {
            'env_type': self.base_args.env_type,
            'num_processes': 1,
            'seed': self.base_args.seed,
            'save_dir': self.base_args.save_dir,
            'lr': self.base_args.lr,
            'alpha': self.base_args.alpha,
            'tau': self.base_args.tau,
            'gamma': self.base_args.gamma,
            'update_frequency': getattr(self.base_args, 'update_frequency', 1),
            'enable_rendering': self.adapter.enable_rendering,  # 🆕 从主训练器传递
            'silent_mode': self.adapter.silent_mode            # 🆕 从主训练器传递
        }
        
        # 使用进程池并行评估
        results = []
        # 🔧 声明全局变量
        global global_executor
        
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=init_worker_process
        ) as executor:
            # 🔧 设置全局引用以便信号处理
            global_executor = executor
            # 提交所有任务
            future_to_data = {
                executor.submit(
                    evaluate_individual_isolated, 
                    data, 
                    base_args_dict, 
                    self.training_steps_per_individual
                ): data
                for data in individual_data_list
            }
            
            # 收集结果 - 🔧 添加KeyboardInterrupt处理
            completed = 0
            try:
                for future in as_completed(future_to_data):
                    data = future_to_data[future]
                    try:
                        result = future.result(timeout=7200)  # 2小时超时
                        if result:
                            results.append(result)
                            completed += 1
                            print(f"✅ 完成 {completed}/{len(individuals)} 个个体")
                        else:
                            print(f"❌ 个体 {data['individual_id']} 评估失败")
                            
                    except Exception as e:
                        print(f"❌ 个体 {data['individual_id']} 异常: {e}")
            except KeyboardInterrupt:
                print(f"\n⚠️ 检测到中断信号，正在清理进程...")
                # 取消所有未完成的任务
                for future in future_to_data:
                    future.cancel()
                print(f"🛑 已取消剩余任务，完成了 {completed}/{len(individuals)} 个个体")
                raise  # 重新抛出KeyboardInterrupt
            finally:
                # 🔧 清理全局引用
                global_executor = None
        
        # 重建Individual对象
        evaluated_individuals = self._reconstruct_individuals_from_results(results, individuals)
        
        print(f"🎉 并行评估完成: {len(evaluated_individuals)}/{len(individuals)} 个个体成功")
        return evaluated_individuals

    def _evaluate_individuals_sequential(self, individuals):
        """顺序评估多个个体（原有逻辑）"""
        evaluated_individuals = []
        for i, individual in enumerate(individuals):
            print(f"🔄 评估个体 {i+1}/{len(individuals)}")
            evaluated_individual = self.adapter.evaluate_individual(
                individual, self.training_steps_per_individual
            )
            
            # 🆕 在无渲染模式下打印个体训练结果
            if not self.adapter.enable_rendering:
                self._print_individual_training_result(evaluated_individual, i+1, len(individuals))
            
            # 🆕 记录实验结果
            self._log_experiment_result(evaluated_individual)
            
            evaluated_individuals.append(evaluated_individual)
        return evaluated_individuals

    def _reconstruct_individuals_from_results(self, results, original_individuals):
        """从结果重建Individual对象"""
        result_map = {r['individual_id']: r for r in results if r}
        
        evaluated = []
        for individual in original_individuals:
            if individual.individual_id in result_map:
                result = result_map[individual.individual_id]
                
                # 重建个体对象
                from map_elites_core import Individual, RobotGenotype, RobotPhenotype
                
                genotype = RobotGenotype(
                    num_links=result['genotype']['num_links'],
                    link_lengths=result['genotype']['link_lengths'],
                    lr=result['genotype']['lr'],
                    alpha=result['genotype']['alpha']
                )
                
                phenotype = RobotPhenotype()
                phenotype.avg_reward = result['phenotype']['avg_reward']
                phenotype.success_rate = result['phenotype']['success_rate']
                phenotype.min_distance = result['phenotype']['min_distance']
                
                new_individual = Individual(
                    individual_id=result['individual_id'],
                    genotype=genotype,
                    phenotype=phenotype,
                    generation=result['generation'],
                    parent_id=result['parent_id']
                )
                
                new_individual.fitness = result['fitness']
                new_individual.fitness_details = result['fitness_details']
                
                # 🆕 记录实验结果
                self._log_experiment_result(new_individual)
                
                # 🆕 在无渲染模式下打印个体训练结果
                if not self.adapter.enable_rendering:
                    self._print_individual_training_result(new_individual, len(evaluated)+1, len(original_individuals))
                
                evaluated.append(new_individual)
            else:
                # 评估失败，设置默认fitness
                individual.fitness = 0.0
                print(f"⚠️ 个体 {individual.individual_id} 使用默认fitness")
                evaluated.append(individual)
        
        return evaluated
    
    def _print_individual_training_result(self, individual, current_idx, total_count):
        """在无渲染模式下打印个体训练结果"""
        print(f"\n{'='*50}")
        print(f"✅ 个体 {current_idx}/{total_count} 训练完成")
        print(f"{'='*50}")
        print(f"🤖 个体信息:")
        print(f"   ID: {individual.individual_id}")
        print(f"   关节数: {individual.genotype.num_links}")
        print(f"   总长度: {sum(individual.genotype.link_lengths):.1f}px")
        print(f"   学习率: {individual.genotype.lr:.2e}")
        print(f"   Alpha: {individual.genotype.alpha:.3f}")
        
        print(f"📊 训练结果:")
        print(f"   🎯 适应度: {individual.fitness:.4f}")
        if hasattr(individual, 'fitness_details') and individual.fitness_details:
            print(f"   📋 类别: {individual.fitness_details.get('category', 'N/A')}")
            print(f"   🎯 策略: {individual.fitness_details.get('strategy', 'N/A')}")
        
        print(f"🏆 性能指标:")
        if hasattr(individual.phenotype, 'success_rate'):
            print(f"   ✅ 成功率: {individual.phenotype.success_rate:.1%}")
        if hasattr(individual.phenotype, 'avg_reward'):
            print(f"   🎁 平均奖励: {individual.phenotype.avg_reward:.2f}")
        if hasattr(individual.phenotype, 'min_distance'):
            print(f"   📏 最佳距离: {individual.phenotype.min_distance:.1f}px")
        
        # 如果有训练指标，也显示
        if hasattr(individual, 'training_metrics') and individual.training_metrics:
            print(f"🔧 训练指标:")
            if 'final_loss' in individual.training_metrics:
                print(f"   📉 最终Loss: {individual.training_metrics['final_loss']:.4f}")
            if 'training_time' in individual.training_metrics:
                print(f"   ⏱️  训练时间: {individual.training_metrics['training_time']:.1f}s")
        
        print(f"{'='*50}")
    
    def _print_training_performance_stats(self, individuals, generation: int):
        """打印训练性能统计信息 - 包括loss和成功率"""
        if not individuals:
            return
        
        # 收集训练指标
        success_rates = []
        avg_rewards = []
        min_distances = []
        training_losses = []  # 如果有的话
        
        for ind in individuals:
            # 成功率统计
            if hasattr(ind.phenotype, 'success_rate'):
                success_rates.append(ind.phenotype.success_rate)
            
            # 奖励统计
            if hasattr(ind.phenotype, 'avg_reward'):
                avg_rewards.append(ind.phenotype.avg_reward)
            
            # 距离统计
            if hasattr(ind.phenotype, 'min_distance'):
                min_distances.append(ind.phenotype.min_distance)
            
            # Loss统计（如果个体有training_metrics）
            if hasattr(ind, 'training_metrics') and ind.training_metrics:
                if 'final_loss' in ind.training_metrics:
                    training_losses.append(ind.training_metrics['final_loss'])
        
        # 打印性能统计
        print(f"\n🎯 训练性能统计:")
        
        # 成功率统计
        if success_rates:
            avg_success_rate = np.mean(success_rates)
            max_success_rate = np.max(success_rates)
            success_individuals = sum(1 for sr in success_rates if sr > 0.5)
            print(f"   ✅ 成功率: 平均 {avg_success_rate:.1%}, 最高 {max_success_rate:.1%}")
            print(f"   🏆 成功个体: {success_individuals}/{len(individuals)} ({success_individuals/len(individuals):.1%})")
        
        # 奖励统计
        if avg_rewards:
            mean_reward = np.mean(avg_rewards)
            best_reward = np.max(avg_rewards)
            print(f"   🎁 奖励: 平均 {mean_reward:.2f}, 最佳 {best_reward:.2f}")
        
        # 距离统计
        if min_distances:
            valid_distances = [d for d in min_distances if d != float('inf')]
            if valid_distances:
                avg_distance = np.mean(valid_distances)
                best_distance = np.min(valid_distances)
                print(f"   📏 目标距离: 平均 {avg_distance:.1f}px, 最佳 {best_distance:.1f}px")
        
        # Loss统计
        if training_losses:
            avg_loss = np.mean(training_losses)
            min_loss = np.min(training_losses)
            print(f"   📉 训练Loss: 平均 {avg_loss:.4f}, 最低 {min_loss:.4f}")
        
        # 🆕 添加当前代的改进情况
        if generation > 0:
            print(f"\n📈 第{generation}代改进情况:")
            # 比较当前代与历史最佳
            current_best_fitness = np.max([ind.fitness for ind in individuals])
            if hasattr(self, '_previous_best_fitness'):
                improvement = current_best_fitness - self._previous_best_fitness
                if improvement > 0:
                    print(f"   🚀 适应度提升: +{improvement:.3f}")
                elif improvement < 0:
                    print(f"   📉 适应度下降: {improvement:.3f}")
                else:
                    print(f"   ➡️  适应度保持: {current_best_fitness:.3f}")
            self._previous_best_fitness = current_best_fitness
        
        # 🆕 添加训练效率分析
        if success_rates and len(success_rates) > 1:
            print(f"\n⚡ 训练效率分析:")
            successful_count = sum(1 for sr in success_rates if sr > 0.3)
            efficiency = successful_count / len(success_rates)
            print(f"   🎯 训练效率: {efficiency:.1%} ({successful_count}/{len(success_rates)} 个体达到30%+成功率)")
            
            if efficiency >= 0.7:
                print(f"   💪 训练效果优秀！大部分个体表现良好")
            elif efficiency >= 0.4:
                print(f"   👍 训练效果良好，还有提升空间")
            else:
                print(f"   ⚠️  训练效果需要改进，考虑调整参数")

    def _print_generation_stats(self, generation: int):
        """打印代际统计信息 - 增强fitness分析和训练指标"""
        stats = self.archive.get_statistics()
        
        print(f"\n{'='*70}")
        print(f"🧬 第{generation}代训练报告")
        print(f"{'='*70}")
        
        # 基础统计
        print(f"📊 MAP-Elites存档统计:")
        print(f"   存档大小: {stats['size']} 个个体")
        print(f"   覆盖率: {stats['coverage']:.3f}")
        print(f"   最佳适应度: {stats['best_fitness']:.3f}")
        print(f"   平均适应度: {stats['avg_fitness']:.3f}")
        print(f"   改善率: {stats['improvement_rate']:.3f}")
        
        # 🆕 添加训练性能统计
        if self.archive.archive:
            individuals = list(self.archive.archive.values())
            self._print_training_performance_stats(individuals, generation)
        
        # 🆕 遗传算法fitness类别分析
        if self.use_genetic_fitness and self.archive.archive:
            individuals = list(self.archive.archive.values())
            fitness_categories = {}
            strategy_count = {}
            
            for ind in individuals:
                if hasattr(ind, 'fitness_details') and ind.fitness_details:
                    category = ind.fitness_details['category']
                    strategy = ind.fitness_details['strategy']
                    
                    fitness_categories[category] = fitness_categories.get(category, 0) + 1
                    strategy_count[strategy] = strategy_count.get(strategy, 0) + 1
            
            if fitness_categories:
                print(f"🎯 Fitness类别分布:")
                for category, count in fitness_categories.items():
                    percentage = count / len(individuals) * 100
                    print(f"   {category}: {count}个 ({percentage:.1f}%)")
                
                print(f"🎯 优化策略分布:")
                for strategy, count in strategy_count.items():
                    percentage = count / len(individuals) * 100
                    print(f"   {strategy}: {count}个 ({percentage:.1f}%)")
        
        # 🆕 添加个体详情
        if self.archive.archive:
            individuals = list(self.archive.archive.values())
            print(f"🤖 形态多样性:")
            
            # 关节数统计
            joint_counts = {}
            length_ranges = {'短(<150px)': 0, '中(150-250px)': 0, '长(>250px)': 0}
            
            for ind in individuals:
                joints = ind.genotype.num_links
                joint_counts[joints] = joint_counts.get(joints, 0) + 1
                
                total_length = sum(ind.genotype.link_lengths)
                if total_length < 150:
                    length_ranges['短(<150px)'] += 1
                elif total_length < 250:
                    length_ranges['中(150-250px)'] += 1
                else:
                    length_ranges['长(>250px)'] += 1
            
            for joints, count in sorted(joint_counts.items()):
                percentage = count / len(individuals) * 100
                print(f"   {joints}关节: {count}个 ({percentage:.1f}%)")
            
            for length_type, count in length_ranges.items():
                percentage = count / len(individuals) * 100
                print(f"   {length_type}: {count}个 ({percentage:.1f}%)")
            
            # 前5名个体
            sorted_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
            print(f"🏆 前5名个体:")
            for i, ind in enumerate(sorted_individuals[:5]):
                fitness_info = f"适应度={ind.fitness:.3f}"
                if hasattr(ind, 'fitness_details') and ind.fitness_details:
                    fitness_info += f" ({ind.fitness_details['category']})"
                
                print(f"   #{i+1}: {fitness_info}, "
                    f"{ind.genotype.num_links}关节, "
                    f"总长{sum(ind.genotype.link_lengths):.0f}px, "
                    f"lr={ind.genotype.lr:.2e}")
        
        print(f"{'='*70}")
        
    def _print_final_results(self):
        """打印最终结果"""
        stats = self.archive.get_statistics()
        best_individual = self.archive.get_best_individual()
        
        print(f"\n🏆 最终结果:")
        print(f"📊 存档统计:")
        print(f"   总个体数: {stats['total_evaluations']}")
        print(f"   存档大小: {stats['size']}")
        print(f"   覆盖率: {stats['coverage']:.3f}")
        print(f"   最佳适应度: {stats['best_fitness']:.3f}")
        
        if best_individual:
            print(f"🥇 最佳个体:")
            print(f"   ID: {best_individual.individual_id}")
            print(f"   适应度: {best_individual.fitness:.3f}")
            print(f"   链节数: {best_individual.genotype.num_links}")
            print(f"   链节长度: {[f'{x:.1f}' for x in best_individual.genotype.link_lengths]}")
            print(f"   总长度: {sum(best_individual.genotype.link_lengths):.1f}px")
            print(f"   学习率: {best_individual.genotype.lr:.2e}")
            print(f"   Alpha: {best_individual.genotype.alpha:.3f}")
            
            # 🆕 显示遗传算法fitness详情
            if hasattr(best_individual, 'fitness_details') and best_individual.fitness_details:
                details = best_individual.fitness_details
                print(f"🎯 Fitness详情:")
                print(f"   类别: {details['category']}")
                print(f"   策略: {details['strategy']}")
                print(f"   原因: {details['reason']}")
                print(f"   可达性: {details.get('reachable', 'N/A')}")

    def _log_experiment_result(self, individual):
        """记录实验结果到成功日志"""
        if not self.success_logger:
            return
        
        try:
            # 创建机器人结构信息
            robot_structure = create_robot_structure(
                num_links=individual.genotype.num_links,
                link_lengths=individual.genotype.link_lengths
            )
            
            # 创建训练参数信息
            training_params = create_training_params(
                lr=individual.genotype.lr,
                alpha=individual.genotype.alpha,
                training_steps=self.training_steps_per_individual,
                buffer_capacity=getattr(self.base_args, 'buffer_capacity', 10000),
                batch_size=getattr(self.base_args, 'batch_size', 64)
            )
            
            # 创建性能指标信息
            performance = create_performance_metrics(
                fitness=individual.fitness,
                success_rate=individual.phenotype.success_rate,
                avg_reward=individual.phenotype.avg_reward,
                training_time=getattr(individual, 'training_time', 0.0),
                episodes_completed=getattr(individual, 'episodes_completed', 0),
                final_distance_to_target=individual.phenotype.min_distance,
                path_efficiency=getattr(individual, 'path_efficiency', None)
            )
            
            # 记录结果
            is_successful = self.success_logger.log_result(
                individual_id=individual.individual_id,
                robot_structure=robot_structure,
                training_params=training_params,
                performance=performance,
                generation=individual.generation,
                parent_id=individual.parent_id,
                notes=f"MAP-Elites 第{individual.generation}代"
            )
            
            if is_successful:
                print(f"🎉 发现成功结构: {individual.individual_id} (fitness: {individual.fitness:.3f})")
                
        except Exception as e:
            print(f"⚠️ 记录实验结果失败: {e}")
    
    def _cleanup_shared_ppo(self):
        """清理共享PPO训练器"""
        if self.shared_ppo_trainer:
            print("🧹 清理共享PPO训练器...")
            try:
                self.shared_ppo_trainer.stop_training()
                print("✅ 共享PPO训练器已停止")
            except Exception as e:
                print(f"⚠️ 清理共享PPO训练器时出错: {e}")
    
    def _generate_visualizations(self, generation: int):
        """生成当前代的可视化"""
        if not self.enable_visualization:
            return
        
        try:
            print(f"🎨 正在生成第{generation}代可视化...")
            
            # 保存当前存档用于可视化
            temp_archive_path = os.path.join(self.base_args.save_dir, f'temp_archive_gen_{generation}.pkl')
            self.archive.generation = generation
            self.archive.save_archive(temp_archive_path)
            
            # 加载到可视化器并生成热力图
            if self.map_elites_visualizer:
                self.map_elites_visualizer.load_archive(temp_archive_path)
                heatmap_path = self.map_elites_visualizer.create_fitness_heatmap(
                    save_path=os.path.join(
                        self.base_args.save_dir, 'visualizations', 
                        f'fitness_heatmap_gen_{generation}.png'
                    )
                )
                if heatmap_path:
                    print(f"✅ 第{generation}代热力图: {heatmap_path}")
                
                # 清理临时文件
                if os.path.exists(temp_archive_path):
                    os.remove(temp_archive_path)
            
            # 生成训练loss可视化（如果有训练日志）
            if self.loss_visualizer:
                training_log_dir = os.path.join(self.base_args.save_dir, 'training_logs')
                if os.path.exists(training_log_dir):
                    if self.loss_visualizer.load_training_logs(training_log_dir):
                        loss_curves_path = self.loss_visualizer.create_loss_curves(
                            save_path=os.path.join(
                                self.base_args.save_dir, 'visualizations',
                                f'loss_curves_gen_{generation}.png'
                            )
                        )
                        if loss_curves_path:
                            print(f"✅ 第{generation}代Loss曲线: {loss_curves_path}")
            
        except Exception as e:
            print(f"⚠️ 生成第{generation}代可视化时出错: {e}")
    
    def _generate_final_visualization_report(self):
        """生成最终可视化报告"""
        if not self.enable_visualization:
            return
        
        try:
            print("🎨 正在生成最终可视化报告...")
            
            # 生成MAP-Elites综合报告
            if self.map_elites_visualizer and self.archive.archive:
                # 保存最终存档
                final_archive_path = os.path.join(self.base_args.save_dir, 'final_archive.pkl')
                self.archive.save_archive(final_archive_path)
                
                # 加载并生成综合报告
                self.map_elites_visualizer.load_archive(final_archive_path)
                map_elites_report = self.map_elites_visualizer.generate_comprehensive_report()
                if map_elites_report:
                    print(f"✅ MAP-Elites综合报告: {map_elites_report}")
            
            # 生成训练Loss综合报告
            if self.loss_visualizer:
                training_log_dir = os.path.join(self.base_args.save_dir, 'training_logs')
                if os.path.exists(training_log_dir):
                    if self.loss_visualizer.load_training_logs(training_log_dir):
                        loss_report = self.loss_visualizer.generate_comprehensive_loss_report()
                        if loss_report:
                            print(f"✅ 训练Loss综合报告: {loss_report}")
            
            print("🎉 所有可视化报告生成完成!")
            
        except Exception as e:
            print(f"⚠️ 生成最终可视化报告时出错: {e}")


def start_real_training():
    """启动真实的MAP-Elites训练"""
    print("🚀 MAP-Elites + 遗传算法Fitness 真实训练")
    print("=" * 60)
    
    # 创建基础参数
    base_args = argparse.Namespace()
    
    # === 环境设置 ===
    base_args.env_type = 'reacher2d'
    base_args.num_processes = 1
    base_args.seed = 42
    base_args.save_dir = './map_elites_training_results'
    base_args.use_real_training = True  # 🆕 启用真实训练
    
    # === 学习参数 ===
    base_args.lr = 3e-4
    base_args.alpha = 0.2
    base_args.tau = 0.005
    base_args.gamma = 0.99
    base_args.update_frequency = 1
    
    print(f"📊 训练配置:")
    print(f"   初始种群: 10个个体")
    print(f"   每个体训练步数: 2000步")
    print(f"   进化代数: 20代")
    print(f"   每代新个体: 5个")
    print(f"   可视化: 启用")
    print(f"   Fitness系统: 遗传算法分层评估")
    print(f"   保存目录: {base_args.save_dir}")
    
    # 创建训练器
    trainer = MAPElitesEvolutionTrainer(
        base_args=base_args,
        num_initial_random=8,                # 🔧 增加到8个个体以充分利用多进程
        training_steps_per_individual=2000,  # 🔧 减少训练步数以便快速测试
        enable_rendering=True,               # 🎨 启用环境渲染
        silent_mode=False,                   # 🔊 显示详细输出
        use_genetic_fitness=True,             # 🎯 使用遗传算法fitness
        enable_multiprocess=True,             # 🆕 启用多进程
        max_workers=4,                       # 🔧 修复：使用4个工作进程
        enable_visualization=True,            # 🎨 启用数据可视化
        visualization_interval=5              # 🎨 每5代生成可视化
    )
    
    try:
        # 开始进化
        trainer.run_evolution(
            num_generations=200,              # 运行20代
            individuals_per_generation=50    # 每代5个新个体
        )
        
        print("\n🎉 训练完成!")
        print(f"📁 结果保存在: {base_args.save_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        print("📊 当前进度已保存")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def start_advanced_training():
    """启动高级配置的MAP-Elites训练"""
    print("🚀 MAP-Elites高级训练配置")
    print("=" * 60)
    
    # 交互式配置
    try:
        print("请配置训练参数 (按回车使用默认值):")
        
        # 获取用户输入
        generations = input("进化代数 [默认: 20]: ").strip()
        generations = int(generations) if generations else 20
        
        individuals_per_gen = input("每代个体数 [默认: 5]: ").strip()
        individuals_per_gen = int(individuals_per_gen) if individuals_per_gen else 5
        
        initial_pop = input("初始种群大小 [默认: 10]: ").strip()
        initial_pop = int(initial_pop) if initial_pop else 10
        
        training_steps = input("每个体训练步数 [默认: 2000]: ").strip()
        training_steps = int(training_steps) if training_steps else 2000
        
        render_choice = input("启用可视化? [y/N]: ").strip().lower()
        enable_render = render_choice in ['y', 'yes']
        
        fitness_choice = input("使用遗传算法Fitness? [Y/n]: ").strip().lower()
        use_genetic = fitness_choice not in ['n', 'no']
        
        save_dir = input("保存目录 [默认: ./map_elites_advanced_results]: ").strip()
        save_dir = save_dir if save_dir else './map_elites_advanced_results'
        
    except KeyboardInterrupt:
        print("\n❌ 配置被取消")
        return
    except ValueError:
        print("❌ 输入格式错误，使用默认配置")
        generations = 20
        individuals_per_gen = 5
        initial_pop = 10
        training_steps = 2000
        enable_render = True
        use_genetic = True
        save_dir = './map_elites_advanced_results'
    
    # 创建基础参数
    base_args = argparse.Namespace()
    base_args.env_type = 'reacher2d'
    base_args.num_processes = 1
    base_args.seed = 42
    base_args.save_dir = save_dir
    base_args.use_real_training = True
    base_args.lr = 3e-4
    base_args.alpha = 0.2
    base_args.tau = 0.005
    base_args.gamma = 0.99
    base_args.update_frequency = 1
    
    print(f"\n📊 最终配置:")
    print(f"   进化代数: {generations}")
    print(f"   每代个体数: {individuals_per_gen}")
    print(f"   初始种群: {initial_pop}")
    print(f"   每个体训练步数: {training_steps}")
    print(f"   可视化: {'启用' if enable_render else '禁用'}")
    print(f"   Fitness系统: {'遗传算法分层' if use_genetic else '传统avg_reward'}")
    print(f"   保存目录: {save_dir}")
    
    confirm = input("\n开始训练? [Y/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ 训练取消")
        return
    
    # 创建训练器
    trainer = MAPElitesEvolutionTrainer(
        base_args=base_args,
        num_initial_random=initial_pop,
        training_steps_per_individual=training_steps,
        enable_rendering=enable_render,
        silent_mode=False,
        use_genetic_fitness=use_genetic
    )
    
    try:
        # 开始进化
        trainer.run_evolution(
            num_generations=generations,
            individuals_per_generation=individuals_per_gen
        )
        
        print("\n🎉 训练完成!")
        print(f"📁 结果保存在: {save_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        print("📊 当前进度已保存")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def start_multiprocess_rendering_training():
    """启动4进程+渲染的MAP-Elites训练"""
    print("🚀 MAP-Elites多进程渲染训练")
    print("=" * 60)
    
    # 创建基础参数
    base_args = argparse.Namespace()
    
    # === 环境设置 ===
    base_args.env_type = 'reacher2d'
    base_args.num_processes = 1
    base_args.seed = 42
    base_args.save_dir = './map_elites_multiprocess_render_results'
    base_args.use_real_training = True
    
    # === 学习参数 ===
    base_args.lr = 2e-4
    base_args.alpha = 0.2
    base_args.tau = 0.005
    base_args.gamma = 0.99
    base_args.update_frequency = 1
    
    # 🎨 解析命令行参数控制渲染和静默模式
    enable_rendering = True   # 🎨 默认启用渲染
    silent_mode = False       # 🔇 默认启用详细输出
    
    # 检查命令行参数
    if '--no-render' in sys.argv:
        enable_rendering = False
        print("🔧 检测到 --no-render 参数，禁用渲染")
    if '--silent' in sys.argv:
        silent_mode = True
        # 🔇 设置全局静默模式
        global GLOBAL_SILENT_MODE
        GLOBAL_SILENT_MODE = True
        # 在设置静默模式前最后一次输出
        print("🔇 启用静默模式 - 后续将无输出")
    
    # 🚀 多进程设置 - 4个进程
    enable_multiprocess = True
    max_workers = 4
    
    print(f"📊 多进程渲染训练配置:")
    print(f"   🎨 渲染模式: {'每个进程显示独立窗口' if enable_rendering else '禁用渲染'}")
    print(f"   🔊 输出模式: {'详细输出' if not silent_mode else '静默模式'}")
    print(f"   🚀 多进程: {max_workers}个并行工作进程")
    print(f"   🤝 共享PPO: 启用 - 所有individual共享同一个PPO模型")
    print(f"   🤖 初始种群: 8个个体")
    print(f"   ⏱️  每个体训练步数: 20000步")
    print(f"   🧬 进化代数: 200代")
    print(f"   👶 每代新个体: 50个")
    print(f"   🎯 遗传算法Fitness: 启用")
    print(f"   📊 成功记录: 启用")
    print(f"   💾 保存目录: {base_args.save_dir}")
    
    # 创建训练器
    trainer = MAPElitesEvolutionTrainer(
        base_args=base_args,
        num_initial_random=8,                # 🔧 8个初始个体，确保能充分利用4进程
        training_steps_per_individual=20000,  # 🔧 适中的训练步数
        enable_rendering=enable_rendering,   # 🎨 强制启用渲染
        silent_mode=silent_mode,             # 🔊 显示详细输出
        use_genetic_fitness=True,            # 🎯 使用遗传算法fitness
        enable_multiprocess=enable_multiprocess,  # 🚀 启用多进程
        max_workers=max_workers,             # 🔧 4个工作进程
        use_shared_ppo=True,                 # 🆕 启用共享PPO - 所有individual共享同一个PPO
        success_threshold=0.7,               # 🎯 成功阈值
        enable_success_logging=True,         # 📊 启用实验成功记录
        enable_visualization=True,           # 🎨 启用数据可视化
        visualization_interval=2             # 🎨 每2代生成可视化
    )
    
    try:
        print("\n🎬 准备启动多进程共享PPO训练...")
        if enable_rendering:
            print("💡 提示: 将会同时打开4个渲染窗口，每个显示不同机器人的训练过程")
            print("⚠️  注意: 请确保您的显示器足够大以容纳多个窗口")
        else:
            print("💡 提示: 无渲染模式，4个进程将在后台并行训练")
        print("🤝 共享PPO: 所有机器人将共同训练同一个PPO模型，互相学习经验")
        
        # 开始进化
        trainer.run_evolution(
            num_generations=200,               # 🔧 5代进化
            individuals_per_generation=50    # 🔧 每代4个新个体
        )
        
        print("\n🎉 多进程渲染训练完成!")
        print(f"📁 结果保存在: {base_args.save_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        print("📊 当前进度已保存")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def start_shared_ppo_training():
    """启动共享PPO的MAP-Elites训练"""
    print("🚀 MAP-Elites + 共享PPO训练")
    print("=" * 60)
    
    # 创建基础参数
    base_args = argparse.Namespace()
    
    # === 环境设置 ===
    base_args.env_type = 'reacher2d'
    base_args.num_processes = 1
    base_args.seed = 42
    base_args.save_dir = './map_elites_shared_ppo_results'
    base_args.use_real_training = True
    
    # === 学习参数 ===
    base_args.lr = 2e-4
    base_args.alpha = 0.2
    base_args.tau = 0.005
    base_args.gamma = 0.99
    base_args.update_frequency = 1
    
    # 🔧 解析命令行参数控制可视化和模型加载
    enable_rendering = True   # 🎨 默认启用可视化
    silent_mode = False       # 🔇 默认启用详细输出
    resume_training = False   # 🔄 默认不恢复训练
    
    # 检查命令行参数
    if '--no-render' in sys.argv:
        enable_rendering = False
        print("🔧 检测到 --no-render 参数，禁用可视化")
    if '--silent' in sys.argv:
        silent_mode = True
        # 🔇 设置全局静默模式
        global GLOBAL_SILENT_MODE
        GLOBAL_SILENT_MODE = True
        # 在设置静默模式前最后一次输出
        print("🔇 启用静默模式 - 后续将无输出")
    if '--resume' in sys.argv:
        resume_training = True
        print("🔧 检测到 --resume 参数，将尝试加载已保存的模型继续训练")
    
    # 🔧 检查是否存在已保存的模型
    model_path = f'{base_args.save_dir}/shared_ppo_model.pth'
    if os.path.exists(model_path) and not resume_training:
        print(f"⚠️ 发现已保存的模型: {model_path}")
        print("💡 如果要继续之前的训练，请使用 --resume 参数")
        print("💡 当前将重新开始训练（会覆盖已有模型）")
    elif os.path.exists(model_path) and resume_training:
        print(f"🔄 将从已保存的模型继续训练: {model_path}")
    elif resume_training and not os.path.exists(model_path):
        print(f"⚠️ 使用了 --resume 参数但未找到模型文件: {model_path}")
        print("🆕 将开始新的训练")
    
    # 🔧 多进程设置
    enable_multiprocess = True   # 🚀 启用多进程以支持多个individual
    max_workers = 4              # 🔧 4个并行工作进程
    
    # 🔧 如果是测试模式，减少训练步数
    test_mode = '--test-quick' in sys.argv
    training_steps = 50 if test_mode else 500
    
    silent_print(f"📊 共享PPO训练配置:")
    silent_print(f"   初始种群: 4个个体 (支持并行可视化)")
    silent_print(f"   每个体训练步数: {training_steps}步")
    silent_print(f"   进化代数: 3代")
    silent_print(f"   每代新个体: 2个")
    silent_print(f"   多进程: {'启用' if enable_multiprocess else '禁用'} ({max_workers}个工作进程)")
    silent_print(f"   共享PPO: 启用")
    silent_print(f"   可视化: {'启用' if enable_rendering else '禁用'}")
    silent_print(f"   详细输出: {'启用' if not silent_mode else '禁用'}")
    silent_print(f"   保存目录: {base_args.save_dir}")
    
    # 创建训练器
    trainer = MAPElitesEvolutionTrainer(
        base_args=base_args,
        num_initial_random=4,                # 🔧 增加到4个个体
        training_steps_per_individual=training_steps,   # 🔧 使用动态训练步数
        enable_rendering=enable_rendering,   # 🎨 启用可视化
        silent_mode=silent_mode,             # 🔇 启用详细输出
        use_genetic_fitness=True,            # 🎯 使用遗传算法fitness
        enable_multiprocess=enable_multiprocess,  # 🚀 启用多进程
        max_workers=max_workers,             # 🔧 4个工作进程
        use_shared_ppo=True,                 # 🆕 启用共享PPO
        success_threshold=0.6,               # 🎯 成功阈值设为0.6 (适合长时间训练)
        enable_success_logging=True          # 📊 启用实验成功记录
    )
    
    try:
        # 开始进化
        trainer.run_evolution(
            num_generations=3,               # 🔧 减少到3代
            individuals_per_generation=2    # 🔧 减少到每代2个新个体
        )
        
        print("\n🎉 共享PPO训练完成!")
        print(f"📁 结果保存在: {base_args.save_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        print("📊 当前进度已保存")
        # 确保清理共享PPO训练器
        trainer._cleanup_shared_ppo()
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        # 确保清理共享PPO训练器
        if 'trainer' in locals():
            trainer._cleanup_shared_ppo()


# 🧪 测试函数
def test_map_elites_trainer():
    """测试MAP-Elites训练器 - 包括新fitness系统"""
    print("🧪 开始测试MAP-Elites训练器 (遗传算法Fitness集成)\n")
    
    # 1. 创建基础参数
    print("📊 测试1: 创建基础参数")
    base_args = argparse.Namespace()
    base_args.env_type = 'reacher2d'
    base_args.num_processes = 1
    base_args.seed = 42
    base_args.save_dir = './test_trainer_results'
    base_args.lr = 1e-4
    base_args.alpha = 0.2
    base_args.tau = 0.005
    base_args.gamma = 0.99
    print(f"✅ 基础参数创建完成")
    
    # 2. 测试新旧两种fitness系统
    fitness_systems = [
        (False, "传统Fitness系统"),
        (True, "遗传算法Fitness系统")
    ]
    
    for use_genetic, system_name in fitness_systems:
        print(f"\n{'='*50}")
        print(f"📊 测试: {system_name}")
        print(f"{'='*50}")
        
        try:
            trainer = MAPElitesEvolutionTrainer(
                base_args=base_args,
                num_initial_random=3,  # 使用更少的个体进行快速测试
                training_steps_per_individual=2000,  # 🔧 减少训练步数以便快速测试
                use_genetic_fitness=use_genetic
            )
            print("✅ 训练器创建成功")
            
            # 3. 测试个体创建方法
            print("\n📊 测试个体创建方法")
            
            # 测试随机个体创建
            random_individual = trainer._create_random_individual(0)
            print(f"✅ 随机个体创建成功: ID={random_individual.individual_id}")
            print(f"   基因型: {random_individual.genotype.num_links}链节")
            
            # 手动添加一个个体到存档以便测试变异
            evaluated_random = trainer.adapter.evaluate_individual(random_individual, 50)
            trainer.archive.add_individual(evaluated_random)
            
            # 测试变异个体创建
            mutant_individual = trainer._create_mutant_individual(1)
            if mutant_individual:
                print(f"✅ 变异个体创建成功: ID={mutant_individual.individual_id}")
                print(f"   父代ID: {mutant_individual.parent_id}")
            else:
                print("⚠️  变异个体创建返回None（这可能是正常的）")
            
            # 4. 测试统计信息
            print("\n📊 测试统计信息")
            trainer._print_generation_stats(0)
            
            # 5. 测试小规模进化
            print("\n📊 小规模进化测试")
            print("⚠️  这将运行一个小规模的进化过程...")
            
            trainer.run_evolution(
                num_generations=2,  # 只运行2代
                individuals_per_generation=2  # 每代只有2个个体
            )
            print(f"✅ {system_name}测试成功完成")
            
            # 6. 测试最终结果
            print(f"\n📊 {system_name}最终结果:")
            trainer._print_final_results()
            
        except Exception as e:
            print(f"❌ {system_name}测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 MAP-Elites训练器测试完成!")
    return True


def test_trainer_components():
    """测试训练器的各个组件"""
    print("🧪 开始测试训练器组件\n")
    
    # 创建基础参数
    base_args = argparse.Namespace()
    base_args.env_type = 'reacher2d'
    base_args.num_processes = 1
    base_args.seed = 42
    base_args.save_dir = './test_components'
    
    # 测试各个组件
    print("📊 测试组件初始化")
    archive = MAPElitesArchive()
    mutator = RobotMutator()
    adapter = MAPElitesTrainingAdapter(base_args, use_genetic_fitness=True)
    print("✅ 所有组件初始化成功")
    
    # 测试组件交互
    print("\n📊 测试组件交互")
    genotype = mutator.random_genotype()
    individual = Individual(
        genotype=genotype,
        phenotype=RobotPhenotype(),
        generation=0
    )
    
    evaluated_individual = adapter.evaluate_individual(individual, 50)
    success = archive.add_individual(evaluated_individual)
    
    print(f"✅ 组件交互测试成功: 个体添加={success}")
    
    stats = archive.get_statistics()
    print(f"✅ 存档统计: {stats}")
    
    return True


def main():
    """主函数"""
    print("🚀 开始MAP-Elites训练器完整测试 (遗传算法Fitness集成)\n")
    
    # 测试组件
    print("=" * 50)
    print("测试1: 组件测试")
    print("=" * 50)
    test_trainer_components()
    
    # 测试训练器
    print("\n" + "=" * 50)
    print("测试2: 训练器测试")
    print("=" * 50)
    test_map_elites_trainer()
    
    print("\n" + "=" * 50)
    print("🎉 所有测试完成!")
    print("=" * 50)


if __name__ == "__main__":
    # 🔧 设置主进程信号处理 - 强力版本
    import signal
    import sys
    import os
    import atexit
    
    # 全局变量存储进程池引用
    global_executor = None
    
    def emergency_cleanup():
        """紧急清理函数"""
        if global_executor is not None:
            print("🚨 执行紧急清理...")
            try:
                global_executor.shutdown(wait=False)
            except:
                pass
    
    def force_exit_handler(signum, frame):
        print(f"\n🛑 收到强制中断信号 (信号{signum})，立即终止所有进程...")
        emergency_cleanup()
        
        # 强制终止所有子进程
        try:
            import psutil
            current_process = psutil.Process(os.getpid())
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    print(f"🔪 强制终止子进程: {child.pid}")
                    child.terminate()
                except:
                    pass
            # 等待一下让子进程终止
            gone, alive = psutil.wait_procs(children, timeout=1)
            # 如果还有进程没有终止，强制杀死
            for p in alive:
                try:
                    print(f"💀 强制杀死顽固进程: {p.pid}")
                    p.kill()
                except:
                    pass
        except ImportError:
            # 如果没有psutil，使用系统命令
            try:
                print("🔪 使用系统命令终止子进程...")
                os.system(f"pkill -9 -P {os.getpid()}")
            except:
                pass
        
        print("💀 强制退出")
        os._exit(1)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, force_exit_handler)
    signal.signal(signal.SIGTERM, force_exit_handler)
    atexit.register(emergency_cleanup)
    
    # 可以选择运行完整测试或者真实训练
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            # 运行一个简单的演示
            print("🎬 运行MAP-Elites + 遗传算法Fitness演示")
            base_args = argparse.Namespace()
            base_args.env_type = 'reacher2d'
            base_args.num_processes = 1
            base_args.seed = 42
            base_args.save_dir = './demo_results'
            
            trainer = MAPElitesEvolutionTrainer(
                base_args=base_args,
                num_initial_random=50,
                training_steps_per_individual=20000,  # 🔧 减少训练步数
                use_genetic_fitness=True  # 🆕 使用遗传算法fitness
            )
            
            trainer.run_evolution(
                num_generations=200,
                individuals_per_generation=50
            )
            
        elif sys.argv[1] == '--test':
            # 运行完整测试
            main()
            
        elif sys.argv[1] == '--train':
            # 🆕 启动真实训练
            silent_print("🚀 启动MAP-Elites真实训练")
            start_real_training()
            
        elif sys.argv[1] == '--train-advanced':
            # 🆕 启动高级训练
            silent_print("🚀 启动MAP-Elites高级训练")
            start_advanced_training()
            
        elif sys.argv[1] == '--train-shared':
            # 🆕 启动共享PPO训练
            silent_print("🚀 启动MAP-Elites共享PPO训练")
            start_shared_ppo_training()
            
        elif sys.argv[1] == '--train-multiprocess':
            # 🆕 启动4进程+渲染训练
            silent_print("🚀 启动MAP-Elites多进程渲染训练")
            start_multiprocess_rendering_training()
            
        else:
            print("❌ 未知参数. 可用选项:")
            print("   --demo: 快速演示")
            print("   --test: 运行测试")
            print("   --train: 真实训练")
            print("   --train-advanced: 高级训练")
            print("   --train-shared: 共享PPO训练")
            print("   --train-multiprocess: 4进程+渲染训练")
            print("")
            print("🎨 可视化选项 (用于 --train-shared 和 --train-multiprocess):")
            print("   --no-render: 禁用可视化渲染")
            print("   --silent: 启用静默模式")
            print("   --resume: 从已保存的模型继续训练 (仅限 --train-shared)")
            print("")
            print("📝 使用示例:")
            print("   python map_elites_trainer.py --train-shared")
            print("   python map_elites_trainer.py --train-shared --no-render")
            print("   python map_elites_trainer.py --train-shared --silent")
            print("   python map_elites_trainer.py --train-shared --resume")
            print("   python map_elites_trainer.py --train-multiprocess  # 4进程+渲染")
            print("   python map_elites_trainer.py --train-multiprocess --no-render  # 4进程无渲染")
            print("   python map_elites_trainer.py --train-multiprocess --silent  # 4进程静默模式")
    else:
        # 默认运行真实训练
        silent_print("🚀 启动MAP-Elites真实训练 (默认模式)")
        start_real_training()