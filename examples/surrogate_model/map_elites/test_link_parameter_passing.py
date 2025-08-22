#!/usr/bin/env python3
"""
验证link数量和长度从MAP-Elites到enhanced_train的传递
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import time
import numpy as np
import argparse
from typing import Dict, Any

# 导入必要组件
from map_elites_core import RobotGenotype, RobotMutator, Individual, RobotPhenotype
from training_adapter import MAPElitesTrainingAdapter
from enhanced_train_interface import MAPElitesTrainingInterface

def create_test_genotypes():
    """创建不同配置的测试基因型"""
    test_cases = [
        {
            "name": "最小配置",
            "num_links": 2,
            "link_lengths": [30.0, 40.0]
        },
        {
            "name": "标准配置", 
            "num_links": 4,
            "link_lengths": [80.0, 70.0, 60.0, 50.0]
        },
        {
            "name": "大型配置",
            "num_links": 6,
            "link_lengths": [100.0, 90.0, 80.0, 70.0, 60.0, 50.0]
        },
        {
            "name": "不规则配置",
            "num_links": 3,
            "link_lengths": [45.5, 120.0, 25.3]
        },
        {
            "name": "随机配置",
            "num_links": np.random.randint(3, 7),
            "link_lengths": [np.random.uniform(40, 100) for _ in range(np.random.randint(3, 7))]
        }
    ]
    
    genotypes = []
    for case in test_cases:
        # 同步 num_links 和 link_lengths 的长度
        if case["name"] == "随机配置":
            num_links = len(case["link_lengths"])
            case["num_links"] = num_links
        
        genotype = RobotGenotype(
            num_links=case["num_links"],
            link_lengths=case["link_lengths"].copy(),
            lr=1e-4,  # 固定其他参数以专注于link测试
            alpha=0.2,
            tau=0.005,
            gamma=0.99,
            batch_size=64,
            buffer_capacity=10000,
            warmup_steps=1000,
            target_entropy_factor=0.8
        )
        genotypes.append((case["name"], genotype))
    
    return genotypes

def test_training_adapter_parameter_conversion():
    """测试TrainingAdapter的参数转换"""
    print("🧪 测试1: TrainingAdapter参数转换")
    print("=" * 50)
    
    # 创建基础参数
    base_args = argparse.Namespace()
    base_args.seed = 42
    base_args.update_frequency = 1
    
    # 创建训练适配器
    adapter = MAPElitesTrainingAdapter(
        base_args=base_args,
        enable_rendering=False,
        silent_mode=True
    )
    
    # 测试不同的基因型
    test_genotypes = create_test_genotypes()
    
    for test_name, genotype in test_genotypes:
        print(f"\n🔬 测试案例: {test_name}")
        print(f"   输入基因型: {genotype.num_links}关节, 长度={[f'{x:.1f}' for x in genotype.link_lengths]}")
        
        try:
            # 调用参数转换方法
            training_args = adapter._genotype_to_training_args(genotype, training_steps=1000)
            
            # 验证转换结果
            assert hasattr(training_args, 'num_joints'), "缺少num_joints属性"
            assert hasattr(training_args, 'link_lengths'), "缺少link_lengths属性"
            assert training_args.num_joints == genotype.num_links, f"num_joints不匹配: {training_args.num_joints} != {genotype.num_links}"
            assert len(training_args.link_lengths) == genotype.num_links, f"link_lengths长度不匹配: {len(training_args.link_lengths)} != {genotype.num_links}"
            
            # 检查数值是否正确传递
            for i, (expected, actual) in enumerate(zip(genotype.link_lengths, training_args.link_lengths)):
                assert abs(expected - actual) < 1e-6, f"第{i}节长度不匹配: {actual} != {expected}"
            
            print(f"   ✅ 转换成功: {training_args.num_joints}关节, 长度={[f'{x:.1f}' for x in training_args.link_lengths]}")
            
            # 额外验证其他参数
            assert training_args.lr == genotype.lr, "学习率不匹配"
            assert training_args.alpha == genotype.alpha, "alpha不匹配"
            
            print(f"   ✅ 其他参数: lr={training_args.lr:.2e}, alpha={training_args.alpha:.3f}")
            
        except Exception as e:
            print(f"   ❌ 转换失败: {e}")
            return False
    
    print(f"\n✅ 所有参数转换测试通过!")
    return True

def test_enhanced_train_interface_parameter_passing():
    """测试enhanced_train_interface的参数传递"""
    print("\n🧪 测试2: enhanced_train_interface参数传递")
    print("=" * 50)
    
    # 创建训练接口
    interface = MAPElitesTrainingInterface(
        silent_mode=True,
        enable_rendering=False
    )
    
    # 测试不同的训练参数
    test_genotypes = create_test_genotypes()
    
    for test_name, genotype in test_genotypes[:3]:  # 只测试前3个以节省时间
        print(f"\n🔬 测试案例: {test_name}")
        
        try:
            # 创建训练参数
            training_args = argparse.Namespace()
            training_args.seed = 42
            training_args.num_joints = genotype.num_links
            training_args.link_lengths = genotype.link_lengths.copy()
            training_args.lr = genotype.lr
            training_args.alpha = genotype.alpha
            training_args.tau = genotype.tau
            training_args.gamma = genotype.gamma
            training_args.batch_size = genotype.batch_size
            training_args.buffer_capacity = genotype.buffer_capacity
            training_args.warmup_steps = genotype.warmup_steps
            training_args.target_entropy_factor = genotype.target_entropy_factor
            training_args.total_steps = 500  # 很短的训练时间用于测试
            training_args.update_frequency = 1
            training_args.save_dir = './test_link_passing'
            
            print(f"   输入参数: {training_args.num_joints}关节, 长度={[f'{x:.1f}' for x in training_args.link_lengths]}")
            
            # 调用训练接口
            start_time = time.time()
            result_metrics = interface.train_individual(training_args)
            training_time = time.time() - start_time
            
            # 验证返回结果
            assert isinstance(result_metrics, dict), "返回结果不是字典"
            assert 'avg_reward' in result_metrics, "缺少avg_reward指标"
            
            print(f"   ✅ 训练成功: avg_reward={result_metrics['avg_reward']:.2f}, 耗时={training_time:.1f}s")
            
            # 检查是否有错误信息表明参数传递问题
            if result_metrics['avg_reward'] == -100 and result_metrics['success_rate'] == 0:
                print(f"   ⚠️  警告: 可能的参数传递问题 (性能异常低)")
            
        except Exception as e:
            print(f"   ❌ 训练失败: {e}")
            return False
    
    print(f"\n✅ enhanced_train_interface参数传递测试通过!")
    return True

def test_full_pipeline_with_parameter_tracking():
    """测试完整流程并跟踪参数传递"""
    print("\n🧪 测试3: 完整流程参数跟踪")
    print("=" * 50)
    
    # 创建基础参数
    base_args = argparse.Namespace()
    base_args.seed = 42
    base_args.update_frequency = 1
    
    # 创建训练适配器
    adapter = MAPElitesTrainingAdapter(
        base_args=base_args,
        enable_rendering=False,
        silent_mode=True
    )
    
    # 创建测试个体
    test_genotype = RobotGenotype(
        num_links=2,
        link_lengths=[65.5, 45.2],
        lr=2.5e-4,
        alpha=0.15,
        tau=0.007,
        gamma=0.995,
        batch_size=32,
        buffer_capacity=5000,
        warmup_steps=500,
        target_entropy_factor=0.9
    )
    
    individual = Individual(
        genotype=test_genotype,
        phenotype=RobotPhenotype(),
        generation=0
    )
    
    print(f"🤖 测试个体:")
    print(f"   关节数: {test_genotype.num_links}")
    print(f"   长度: {[f'{x:.1f}' for x in test_genotype.link_lengths]}")
    print(f"   学习率: {test_genotype.lr:.2e}")
    print(f"   批大小: {test_genotype.batch_size}")
    
    # 开启详细跟踪
    original_method = adapter._genotype_to_training_args
    
    def tracking_genotype_to_training_args(genotype, training_steps):
        print(f"\n📊 参数转换跟踪:")
        print(f"   输入基因型: {genotype.num_links}关节, {[f'{x:.1f}' for x in genotype.link_lengths]}")
        
        result = original_method(genotype, training_steps)
        
        print(f"   输出参数: {result.num_joints}关节, {[f'{x:.1f}' for x in result.link_lengths]}")
        print(f"   其他参数: lr={result.lr:.2e}, batch_size={result.batch_size}")
        
        return result
    
    # 临时替换方法以进行跟踪
    adapter._genotype_to_training_args = tracking_genotype_to_training_args
    
    try:
        # 运行完整评估
        print(f"\n🚀 开始完整评估...")
        start_time = time.time()
        
        evaluated_individual = adapter.evaluate_individual(individual, training_steps=800)
        
        evaluation_time = time.time() - start_time
        
        print(f"\n✅ 完整流程成功:")
        print(f"   适应度: {evaluated_individual.fitness:.2f}")
        print(f"   成功率: {evaluated_individual.phenotype.success_rate:.2f}")
        print(f"   评估时间: {evaluation_time:.1f}s")
        
        # 验证个体的基因型没有被意外修改
        assert evaluated_individual.genotype.num_links == test_genotype.num_links, "关节数被意外修改"
        assert len(evaluated_individual.genotype.link_lengths) == len(test_genotype.link_lengths), "长度数组被意外修改"
        
        for i, (original, current) in enumerate(zip(test_genotype.link_lengths, evaluated_individual.genotype.link_lengths)):
            assert abs(original - current) < 1e-6, f"第{i}节长度被意外修改: {current} != {original}"
        
        print(f"✅ 基因型完整性验证通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 完整流程失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 恢复原始方法
        adapter._genotype_to_training_args = original_method

def test_random_genotype_generation():
    """测试随机基因型生成的合理性"""
    print("\n🧪 测试4: 随机基因型生成验证")
    print("=" * 50)
    
    mutator = RobotMutator()
    
    print("🎲 生成10个随机基因型:")
    for i in range(10):
        genotype = mutator.random_genotype()
        
        # 验证约束
        assert 3 <= genotype.num_links <= 6, f"关节数超出范围: {genotype.num_links}"
        assert len(genotype.link_lengths) == genotype.num_links, f"长度数组长度不匹配: {len(genotype.link_lengths)} != {genotype.num_links}"
        
        for j, length in enumerate(genotype.link_lengths):
            assert 40 <= length <= 100, f"第{j}节长度超出范围: {length}"
        
        print(f"   基因型{i+1}: {genotype.num_links}关节, 长度={[f'{x:.1f}' for x in genotype.link_lengths]}")
    
    print(f"✅ 随机基因型生成验证通过!")
    return True

def main():
    """运行完整的link参数传递验证"""
    print("🔗 Link参数传递验证测试")
    print("=" * 60)
    
    test_results = []
    
    # 测试1: 参数转换
    result1 = test_training_adapter_parameter_conversion()
    test_results.append(("参数转换测试", result1))
    
    if not result1:
        print("🚨 参数转换测试失败，停止后续测试")
        return
    
    # 测试2: enhanced_train_interface
    result2 = test_enhanced_train_interface_parameter_passing()
    test_results.append(("enhanced_train_interface测试", result2))
    
    # 测试3: 完整流程
    result3 = test_full_pipeline_with_parameter_tracking()
    test_results.append(("完整流程跟踪测试", result3))
    
    # 测试4: 随机生成
    result4 = test_random_genotype_generation()
    test_results.append(("随机基因型生成测试", result4))
    
    # 输出测试报告
    print("\n" + "=" * 60)
    print("📋 Link参数传递验证报告:")
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    if passed_tests == total_tests:
        print(f"\n🎉 所有测试通过! ({passed_tests}/{total_tests})")
        print("💪 Link数量和长度参数传递完全正常!")
    else:
        print(f"\n⚠️  部分测试失败 ({passed_tests}/{total_tests})")
        print("🔧 请检查失败的测试项")

if __name__ == "__main__":
    main()