import time
import argparse
from typing import List, Optional
import numpy as np
import multiprocessing as mp

from map_elites_core import MAPElitesArchive, RobotMutator, Individual, RobotGenotype, RobotPhenotype
from training_adapter import MAPElitesTrainingAdapter


class MAPElitesEvolutionTrainer:
    """MAP-Elites进化训练器 - 支持遗传算法Fitness评估系统"""
    
    def __init__(self, base_args, num_initial_random: int = 20, 
                 training_steps_per_individual: int = 3000,
                 enable_rendering: bool = False,    # 🆕 是否启用渲染
                 silent_mode: bool = True,          # 🆕 是否静默模式
                 use_genetic_fitness: bool = True): # 🆕 是否使用遗传算法fitness
        
        # 初始化组件
        self.archive = MAPElitesArchive()
        self.mutator = RobotMutator()
        self.adapter = MAPElitesTrainingAdapter(
            base_args, 
            enable_rendering=enable_rendering,  # 🆕 传递渲染设置
            silent_mode=silent_mode,           # 🆕 传递静默设置
            use_genetic_fitness=use_genetic_fitness  # 🆕 传递fitness设置
        )
        
        self.num_initial_random = num_initial_random
        self.training_steps_per_individual = training_steps_per_individual
        self.use_genetic_fitness = use_genetic_fitness
        
        print("🧬 MAP-Elites进化训练器已初始化")
        print(f"🎯 Fitness评估: {'遗传算法分层系统' if use_genetic_fitness else '传统平均奖励'}")
        print(f"🎨 可视化: {'启用' if enable_rendering else '禁用'}")
    
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
            for i, individual in enumerate(new_individuals):
                print(f"  个体 {i+1}/{len(new_individuals)}")
                evaluated_individual = self.adapter.evaluate_individual(
                    individual, self.training_steps_per_individual
                )
                self.archive.add_individual(evaluated_individual)
            
            # 输出代际统计
            self._print_generation_stats(generation)
            
            # 保存存档
            if generation % 5 == 0:
                self.archive.generation = generation
                self.archive.save_archive()
        
        # 最终结果打印
        print(f"\n🎉 进化完成!")
        self._print_final_results()
    
    def _initialize_random_population(self):
        """初始化随机种群"""
        for i in range(self.num_initial_random):
            print(f"  初始化个体 {i+1}/{self.num_initial_random}")
            individual = self._create_random_individual(0)
            evaluated_individual = self.adapter.evaluate_individual(
                individual, self.training_steps_per_individual
            )
             # 🆕 添加这些调试信息
            print(f"🔍 调试 - 个体 {i+1} 评估结果:")
            print(f"   Fitness: {evaluated_individual.fitness}")
            if hasattr(evaluated_individual, 'fitness_details'):
                print(f"   Fitness详情: {evaluated_individual.fitness_details}")
            else:
                print(f"   ⚠️ 没有fitness_details属性")

            self.archive.add_individual(evaluated_individual)
        
        stats = self.archive.get_statistics()
        print(f"📊 初始化完成: 存档大小={stats['size']}, 最佳适应度={stats['best_fitness']:.3f}")
    
    def _create_random_individual(self, generation: int) -> Individual:
        """创建随机个体"""
        genotype = self.mutator.random_genotype()
        return Individual(
            genotype=genotype,
            phenotype=RobotPhenotype(),
            generation=generation
        )
    
    def _create_mutant_individual(self, generation: int) -> Optional[Individual]:
        """创建变异个体"""
        parent = self.archive.get_random_elite()
        if parent is None:
            return self._create_random_individual(generation)
        
        mutant_genotype = self.mutator.mutate(parent.genotype)
        return Individual(
            genotype=mutant_genotype,
            phenotype=RobotPhenotype(),
            generation=generation,
            parent_id=parent.individual_id
        )
    
    def _print_generation_stats(self, generation: int):
        """打印代际统计信息 - 增强fitness分析"""
        stats = self.archive.get_statistics()
        
        print(f"\n🧬 第{generation}代详细分析:")
        print(f"📊 基础统计:")
        print(f"   存档大小: {stats['size']}")
        print(f"   覆盖率: {stats['coverage']:.3f}")
        print(f"   最佳适应度: {stats['best_fitness']:.3f}")  # 🆕 增加精度
        print(f"   平均适应度: {stats['avg_fitness']:.3f}")
        print(f"   改善率: {stats['improvement_rate']:.3f}")
        
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
        num_initial_random=10,               # 初始随机个体数
        training_steps_per_individual=2000,  # 每个个体的训练步数
        enable_rendering=True,               # 🎨 启用可视化
        silent_mode=False,                   # 🔊 显示详细输出
        use_genetic_fitness=True             # 🎯 使用遗传算法fitness
    )
    
    try:
        # 开始进化
        trainer.run_evolution(
            num_generations=20,              # 运行20代
            individuals_per_generation=5    # 每代5个新个体
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
                training_steps_per_individual=100,  # 使用更少的训练步数
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
    # 可以选择运行完整测试或者真实训练
    import sys
    
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
                num_initial_random=5,
                training_steps_per_individual=200,
                use_genetic_fitness=True  # 🆕 使用遗传算法fitness
            )
            
            trainer.run_evolution(
                num_generations=3,
                individuals_per_generation=3
            )
            
        elif sys.argv[1] == '--test':
            # 运行完整测试
            main()
            
        elif sys.argv[1] == '--train':
            # 🆕 启动真实训练
            print("🚀 启动MAP-Elites真实训练")
            start_real_training()
            
        elif sys.argv[1] == '--train-advanced':
            # 🆕 启动高级训练
            print("🚀 启动MAP-Elites高级训练")
            start_advanced_training()
            
        else:
            print("❌ 未知参数. 可用选项:")
            print("   --demo: 快速演示")
            print("   --test: 运行测试")
            print("   --train: 真实训练")
            print("   --train-advanced: 高级训练")
    else:
        # 默认运行真实训练
        print("🚀 启动MAP-Elites真实训练 (默认模式)")
        start_real_training()