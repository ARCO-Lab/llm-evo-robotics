import time
import argparse
from typing import List, Optional
import numpy as np

from map_elites_core import MAPElitesArchive, RobotMutator, Individual, RobotGenotype, RobotPhenotype
from training_adapter import MAPElitesTrainingAdapter


class MAPElitesEvolutionTrainer:
    """MAP-Elites进化训练器 - 支持可视化配置"""
    
    def __init__(self, base_args, num_initial_random: int = 20, 
                 training_steps_per_individual: int = 3000,
                 enable_rendering: bool = False,    # 🆕 是否启用渲染
                 silent_mode: bool = True):         # 🆕 是否静默模式
        
        # 初始化组件
        self.archive = MAPElitesArchive()
        self.mutator = RobotMutator()
        self.adapter = MAPElitesTrainingAdapter(
            base_args, 
            enable_rendering=enable_rendering,  # 🆕 传递渲染设置
            silent_mode=silent_mode            # 🆕 传递静默设置
        )
        
        self.num_initial_random = num_initial_random
        self.training_steps_per_individual = training_steps_per_individual
        
        print("🧬 MAP-Elites进化训练器已初始化")
        print(f"🎯 选择策略: 基于reward比例选择")
        print(f"🎨 可视化: {'启用' if enable_rendering else '禁用'}")
    
    def run_evolution(self, num_generations: int = 50, individuals_per_generation: int = 10):
        """运行MAP-Elites进化"""
        print(f"🚀 开始MAP-Elites进化")
        print(f"📊 参数: {num_generations}代, 每代{individuals_per_generation}个个体")
        
        # 第0代：随机初始化
        print(f"\n🎲 第0代: 随机初始化")
        self._initialize_random_population()  # 修复方法名
        
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
                self.archive.generation = generation  # 修复属性名：generate -> generation
                self.archive.save_archive()
        
        # 修复缩进：最终结果打印应该在循环外
        print(f"\n🎉 进化完成!")
        self._print_final_results()
    
    def _initialize_random_population(self):  # 修复方法名
        """初始化随机种群"""
        for i in range(self.num_initial_random):
            print(f"  初始化个体 {i+1}/{self.num_initial_random}")
            individual = self._create_random_individual(0)
            evaluated_individual = self.adapter.evaluate_individual(
                individual, self.training_steps_per_individual
            )
            self.archive.add_individual(evaluated_individual)
        
        stats = self.archive.get_statistics()
        print(f"📊 初始化完成: 存档大小={stats['size']}, 最佳适应度={stats['best_fitness']:.2f}")
    
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
        parent = self.archive.get_random_elite()  # 修复方法名：get_random_elites -> get_random_elite
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
        """打印代际统计信息"""
        stats = self.archive.get_statistics()
        
        # 🆕 添加详细的种群信息
        print(f"\n🧬 第{generation}代详细分析:")
        print(f"📊 基础统计:")
        print(f"   存档大小: {stats['size']}")
        print(f"   覆盖率: {stats['coverage']:.3f}")
        print(f"   最佳适应度: {stats['best_fitness']:.2f}")
        print(f"   平均适应度: {stats['avg_fitness']:.2f}")
        print(f"   改善率: {stats['improvement_rate']:.3f}")
        
        # 🆕 添加个体详情
        if self.archive.archive:
            individuals = list(self.archive.archive.values())
            print(f"🤖 形态多样性:")
            
            # 关节数统计
            joint_counts = {}
            for ind in individuals:
                joints = ind.genotype.num_links
                joint_counts[joints] = joint_counts.get(joints, 0) + 1
            
            for joints, count in sorted(joint_counts.items()):
                percentage = count / len(individuals) * 100
                print(f"   {joints}关节: {count}个 ({percentage:.1f}%)")
            
            # 前5名个体
            sorted_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
            print(f"🏆 前5名个体:")
            for i, ind in enumerate(sorted_individuals[:5]):
                print(f"   #{i+1}: 适应度={ind.fitness:.2f}, "
                    f"{ind.genotype.num_links}关节, "
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
        print(f"   最佳适应度: {stats['best_fitness']:.2f}")
        
        if best_individual:
            print(f"🥇 最佳个体:")
            print(f"   ID: {best_individual.individual_id}")
            print(f"   适应度: {best_individual.fitness:.2f}")
            print(f"   链节数: {best_individual.genotype.num_links}")
            print(f"   链节长度: {[f'{x:.1f}' for x in best_individual.genotype.link_lengths]}")
            print(f"   学习率: {best_individual.genotype.lr:.2e}")
            print(f"   Alpha: {best_individual.genotype.alpha:.3f}")


# 🧪 测试函数
def test_map_elites_trainer():
    """测试MAP-Elites训练器"""
    print("🧪 开始测试MAP-Elites训练器\n")
    
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
    
    # 2. 创建训练器
    print("\n📊 测试2: 创建MAP-Elites训练器")
    trainer = MAPElitesEvolutionTrainer(
        base_args=base_args,
        num_initial_random=3,  # 使用更少的个体进行快速测试
        training_steps_per_individual=100  # 使用更少的训练步数
    )
    print("✅ 训练器创建成功")
    
    # 3. 测试个体创建方法
    print("\n📊 测试3: 测试个体创建方法")
    
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
    print("\n📊 测试4: 测试统计信息")
    trainer._print_generation_stats(0)
    
    # 5. 测试小规模进化
    print("\n📊 测试5: 小规模进化测试")
    print("⚠️  这将运行一个小规模的进化过程...")
    
    try:
        trainer.run_evolution(
            num_generations=2,  # 只运行2代
            individuals_per_generation=2  # 每代只有2个个体
        )
        print("✅ 小规模进化测试成功完成")
    except Exception as e:
        print(f"❌ 进化测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. 测试最终结果
    print("\n📊 测试6: 测试最终结果打印")
    trainer._print_final_results()
    
    print("\n🎉 MAP-Elites训练器测试完成!")
    return trainer


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
    adapter = MAPElitesTrainingAdapter(base_args)
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
    print("🚀 开始MAP-Elites训练器完整测试\n")
    
    # 测试组件
    print("=" * 50)
    print("测试1: 组件测试")
    print("=" * 50)
    test_trainer_components()
    
    # 测试训练器
    print("\n" + "=" * 50)
    print("测试2: 训练器测试")
    print("=" * 50)
    trainer = test_map_elites_trainer()
    
    print("\n" + "=" * 50)
    print("🎉 所有测试完成!")
    print("=" * 50)


if __name__ == "__main__":
    # 可以选择运行完整测试或者简单演示
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        # 运行一个简单的演示
        print("🎬 运行MAP-Elites演示")
        base_args = argparse.Namespace()
        base_args.env_type = 'reacher2d'
        base_args.num_processes = 1
        base_args.seed = 42
        base_args.save_dir = './demo_results'
        
        trainer = MAPElitesEvolutionTrainer(
            base_args=base_args,
            num_initial_random=5,
            training_steps_per_individual=200
        )
        
        trainer.run_evolution(
            num_generations=3,
            individuals_per_generation=3
        )
    else:
        # 运行完整测试
        main()