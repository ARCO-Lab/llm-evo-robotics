#!/usr/bin/env python3
"""
分层测试MAP-Elites + enhanced_train.py集成系统
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import time
import numpy as np
import argparse
from typing import Dict, Any

# 导入系统组件
from map_elites_core import MAPElitesArchive, Individual, RobotGenotype, RobotPhenotype, RobotMutator
from training_adapter import MAPElitesTrainingAdapter
from map_elites_trainer import MAPElitesEvolutionTrainer

def test_level_1_components():
    """Level 1: 测试各个组件是否能正常工作"""
    print("🧪 Level 1: 组件级测试")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # 测试1: enhanced_train_interface导入
    total_tests += 1
    try:
        from enhanced_train_interface import MAPElitesTrainingInterface
        interface = MAPElitesTrainingInterface(silent_mode=True, enable_rendering=False)
        print("✅ Test 1.1: enhanced_train_interface导入成功")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1.1: enhanced_train_interface导入失败: {e}")
    
    # 测试2: 随机基因型生成
    total_tests += 1
    try:
        mutator = RobotMutator()
        genotype = mutator.random_genotype()
        assert 3 <= genotype.num_links <= 6
        assert len(genotype.link_lengths) == genotype.num_links
        assert 1e-6 <= genotype.lr <= 1e-3
        print(f"✅ Test 1.2: 随机基因型生成正常 (links={genotype.num_links}, lr={genotype.lr:.2e})")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1.2: 随机基因型生成失败: {e}")
    
    # 测试3: 存档基本功能
    total_tests += 1
    try:
        archive = MAPElitesArchive()
        test_individual = Individual(
            genotype=genotype,
            phenotype=RobotPhenotype(avg_reward=50.0),
            fitness=50.0
        )
        # 手动添加到存档 (跳过特征提取)
        archive.archive[(0, 0, 0, 0, 0)] = test_individual
        selected = archive.get_random_elite()
        assert selected is not None
        print("✅ Test 1.3: 存档基本功能正常")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1.3: 存档功能失败: {e}")
    
    print(f"\n📊 Level 1 结果: {tests_passed}/{total_tests} 通过")
    return tests_passed == total_tests

def test_level_2_single_individual():
    """Level 2: 测试单个个体的完整训练流程"""
    print("\n🧪 Level 2: 单个体完整流程测试")
    print("=" * 50)
    
    try:
        # 创建基础参数
        base_args = argparse.Namespace()
        base_args.seed = 42
        base_args.update_frequency = 1
        
        # 创建训练适配器
        adapter = MAPElitesTrainingAdapter(
            base_args=base_args,
            enable_rendering=False,  # 禁用渲染以提高测试速度
            silent_mode=True
        )
        
        # 创建随机个体
        mutator = RobotMutator()
        genotype = mutator.random_genotype()
        individual = Individual(
            genotype=genotype,
            phenotype=RobotPhenotype(),
            generation=0
        )
        
        print(f"🤖 测试个体: {genotype.num_links}关节, lr={genotype.lr:.2e}")
        
        # 训练个体 (使用较短的训练时间)
        start_time = time.time()
        evaluated_individual = adapter.evaluate_individual(individual, training_steps=1000)
        training_time = time.time() - start_time
        
        # 验证结果
        assert evaluated_individual.fitness != 0.0
        assert evaluated_individual.phenotype.avg_reward != 0.0
        
        print(f"✅ 个体训练成功:")
        print(f"   适应度: {evaluated_individual.fitness:.2f}")
        print(f"   平均奖励: {evaluated_individual.phenotype.avg_reward:.2f}")
        print(f"   成功率: {evaluated_individual.phenotype.success_rate:.2f}")
        print(f"   训练时间: {training_time:.1f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ Level 2 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_level_3_population_generation():
    """Level 3: 测试种群生成和选择机制"""
    print("\n🧪 Level 3: 种群生成和选择测试")
    print("=" * 50)
    
    try:
        # 创建基础参数
        base_args = argparse.Namespace()
        base_args.seed = 42
        base_args.update_frequency = 1
        
        # 创建训练器 (小规模配置)
        trainer = MAPElitesEvolutionTrainer(
            base_args=base_args,
            num_initial_random=3,  # 只生成3个初始个体
            training_steps_per_individual=1000,
            enable_rendering=False,
            silent_mode=True
        )
        
        print("🚀 生成初始种群...")
        initial_start = time.time()
        trainer._initialize_random_population()
        initial_time = time.time() - initial_start
        
        # 验证初始种群
        stats = trainer.archive.get_statistics()
        assert stats['size'] >= 1  # 至少有一个个体成功
        
        print(f"✅ 初始种群生成成功:")
        print(f"   存档大小: {stats['size']}")
        print(f"   最佳适应度: {stats['best_fitness']:.2f}")
        print(f"   平均适应度: {stats['avg_fitness']:.2f}")
        print(f"   生成时间: {initial_time:.1f}秒")
        
        # 测试选择机制
        print("\n🎯 测试选择机制...")
        selection_counts = {}
        for _ in range(10):
            selected = trainer.archive.get_random_elite()
            if selected:
                key = f"fitness_{selected.fitness:.1f}"
                selection_counts[key] = selection_counts.get(key, 0) + 1
        
        print(f"✅ 选择机制工作正常:")
        for key, count in selection_counts.items():
            print(f"   {key}: 被选择{count}次")
        
        # 测试变异机制
        print("\n🧬 测试变异机制...")
        parent = trainer.archive.get_random_elite()
        if parent:
            mutant = trainer._create_mutant_individual(1)
            assert mutant is not None
            assert mutant.generation == 1
            assert mutant.parent_id == parent.individual_id
            print(f"✅ 变异机制正常: 父代fitness={parent.fitness:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Level 3 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_level_4_mini_evolution():
    """Level 4: 测试完整的迷你进化过程"""
    print("\n🧪 Level 4: 迷你进化过程测试")
    print("=" * 50)
    
    try:
        # 创建基础参数
        base_args = argparse.Namespace()
        base_args.seed = 42
        base_args.update_frequency = 1
        
        # 创建训练器 (最小规模配置)
        trainer = MAPElitesEvolutionTrainer(
            base_args=base_args,
            num_initial_random=3,
            training_steps_per_individual=800,  # 更短的训练时间
            enable_rendering=False,
            silent_mode=True
        )
        
        print("🚀 运行迷你进化 (2代, 每代2个个体)...")
        evolution_start = time.time()
        
        # 记录进化前的状态
        trainer._initialize_random_population()
        initial_stats = trainer.archive.get_statistics()
        
        # 运行2代进化
        for generation in range(1, 3):
            print(f"\n🧬 第{generation}代:")
            
            # 生成新个体
            new_individuals = []
            for i in range(2):  # 每代只生成2个个体
                if np.random.random() < 0.5:  # 50-50随机vs变异
                    individual = trainer._create_random_individual(generation)
                    print(f"  个体{i+1}: 随机生成")
                else:
                    individual = trainer._create_mutant_individual(generation)
                    print(f"  个体{i+1}: 变异生成")
                
                if individual:
                    new_individuals.append(individual)
            
            # 评估新个体
            for i, individual in enumerate(new_individuals):
                print(f"  评估个体 {i+1}/{len(new_individuals)}...")
                evaluated_individual = trainer.adapter.evaluate_individual(
                    individual, trainer.training_steps_per_individual
                )
                trainer.archive.add_individual(evaluated_individual)
            
            # 输出代际统计
            trainer._print_generation_stats(generation)
        
        evolution_time = time.time() - evolution_start
        final_stats = trainer.archive.get_statistics()
        
        print(f"\n✅ 迷你进化完成:")
        print(f"   总时间: {evolution_time:.1f}秒")
        print(f"   最终存档大小: {final_stats['size']}")
        print(f"   适应度提升: {initial_stats['best_fitness']:.2f} → {final_stats['best_fitness']:.2f}")
        print(f"   覆盖率: {final_stats['coverage']:.3f}")
        
        # 验证是否有改进
        improvement = final_stats['best_fitness'] - initial_stats['best_fitness']
        if improvement > 0:
            print(f"🎉 检测到性能提升: +{improvement:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Level 4 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_level_5_stress_test():
    """Level 5: 压力测试 - 更长时间运行"""
    print("\n🧪 Level 5: 系统压力测试")
    print("=" * 50)
    
    try:
        # 创建基础参数
        base_args = argparse.Namespace()
        base_args.seed = 42
        base_args.update_frequency = 1
        
        # 创建训练器 (中等规模配置)
        trainer = MAPElitesEvolutionTrainer(
            base_args=base_args,
            num_initial_random=5,
            training_steps_per_individual=1500,
            enable_rendering=False,
            silent_mode=True
        )
        
        print("🚀 运行压力测试 (5代, 每代3个个体)...")
        stress_start = time.time()
        
        # 运行完整进化
        trainer.run_evolution(num_generations=5, individuals_per_generation=3)
        
        stress_time = time.time() - stress_start
        final_stats = trainer.archive.get_statistics()
        
        print(f"\n✅ 压力测试完成:")
        print(f"   总时间: {stress_time:.1f}秒")
        print(f"   最终存档大小: {final_stats['size']}")
        print(f"   最佳适应度: {final_stats['best_fitness']:.2f}")
        print(f"   平均适应度: {final_stats['avg_fitness']:.2f}")
        print(f"   覆盖率: {final_stats['coverage']:.3f}")
        
        # 性能基准
        individuals_trained = 5 + 5 * 3  # 初始5个 + 5代*3个/代
        avg_time_per_individual = stress_time / individuals_trained
        print(f"   平均每个体训练时间: {avg_time_per_individual:.1f}秒")
        
        if final_stats['best_fitness'] > -50:
            print("🎉 系统性能正常!")
            return True
        else:
            print("⚠️  系统运行但性能较低")
            return True
        
    except Exception as e:
        print(f"❌ Level 5 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行完整的分层测试"""
    print("🧪 MAP-Elites + enhanced_train.py 集成系统测试")
    print("=" * 60)
    
    # 记录测试结果
    test_results = []
    
    # Level 1: 组件测试
    result1 = test_level_1_components()
    test_results.append(("Level 1 - 组件测试", result1))
    
    if not result1:
        print("🚨 Level 1失败，停止后续测试")
        return
    
    # Level 2: 单个体测试
    result2 = test_level_2_single_individual()
    test_results.append(("Level 2 - 单个体流程", result2))
    
    if not result2:
        print("🚨 Level 2失败，建议检查enhanced_train_interface连接")
        return
    
    # Level 3: 种群测试
    result3 = test_level_3_population_generation()
    test_results.append(("Level 3 - 种群生成", result3))
    
    if not result3:
        print("🚨 Level 3失败，建议检查MAP-Elites核心逻辑")
        return
    
    # Level 4: 迷你进化
    result4 = test_level_4_mini_evolution()
    test_results.append(("Level 4 - 迷你进化", result4))
    
    # Level 5: 压力测试 (可选)
    print("\n❓ 是否运行Level 5压力测试? (可能需要10-15分钟) [y/N]: ", end="")
    try:
        choice = input().strip().lower()
        if choice in ['y', 'yes']:
            result5 = test_level_5_stress_test()
            test_results.append(("Level 5 - 压力测试", result5))
    except:
        print("跳过Level 5测试")
    
    # 输出最终报告
    print("\n" + "=" * 60)
    print("📋 测试总结报告:")
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    if passed_tests == total_tests:
        print(f"\n🎉 所有测试通过! ({passed_tests}/{total_tests})")
        print("💪 你的MAP-Elites + enhanced_train.py系统运行正常!")
    else:
        print(f"\n⚠️  部分测试失败 ({passed_tests}/{total_tests})")
        print("🔧 建议检查失败的测试模块")

if __name__ == "__main__":
    main()