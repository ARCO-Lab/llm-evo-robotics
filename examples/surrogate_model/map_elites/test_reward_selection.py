#!/usr/bin/env python3
"""
测试基于reward的比例选择
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from map_elites_core import MAPElitesArchive, Individual, RobotGenotype, RobotPhenotype

def test_reward_based_selection():
    """测试基于reward的选择是否工作"""
    print("🧪 测试基于reward的比例选择\n")
    
    # 创建存档
    archive = MAPElitesArchive()
    
    # 手动创建一些测试个体，设置不同的reward
    test_individuals = []
    
    # 个体1: 高reward
    ind1 = Individual(
        genotype=RobotGenotype(num_links=3, link_lengths=[50, 50, 50]),
        phenotype=RobotPhenotype(avg_reward=100.0),  # 高reward
        fitness=100.0,
        individual_id="high_reward"
    )
    
    # 个体2: 中等reward  
    ind2 = Individual(
        genotype=RobotGenotype(num_links=4, link_lengths=[40, 40, 40, 40]),
        phenotype=RobotPhenotype(avg_reward=50.0),   # 中等reward
        fitness=50.0,
        individual_id="medium_reward"
    )
    
    # 个体3: 低reward
    ind3 = Individual(
        genotype=RobotGenotype(num_links=2, link_lengths=[60, 60]),
        phenotype=RobotPhenotype(avg_reward=10.0),   # 低reward
        fitness=10.0,
        individual_id="low_reward"
    )
    
    # 添加到存档（需要手动设置坐标以避免特征提取）
    archive.archive[(0, 0, 0, 0, 0)] = ind1
    archive.archive[(1, 1, 1, 1, 1)] = ind2  
    archive.archive[(2, 2, 2, 2, 2)] = ind3
    
    print(f"📊 存档中有 {len(archive.archive)} 个个体:")
    for coords, ind in archive.archive.items():
        print(f"   位置 {coords}: ID={ind.individual_id}, reward={ind.fitness}")
    
    # 进行多次选择，统计结果
    print(f"\n🎯 进行100次选择，统计选择频率:")
    
    selection_counts = {
        "high_reward": 0,
        "medium_reward": 0, 
        "low_reward": 0
    }
    
    num_trials = 100
    for i in range(num_trials):
        selected = archive.get_random_elite()
        if selected:
            selection_counts[selected.individual_id] += 1
    
    print(f"\n📈 选择结果统计:")
    total_reward = 100 + 50 + 10  # 160
    for ind_id, count in selection_counts.items():
        actual_rate = count / num_trials
        if ind_id == "high_reward":
            expected_rate = (100 + 1) / (160 + 3)  # 调整后的概率
        elif ind_id == "medium_reward":
            expected_rate = (50 + 1) / (160 + 3)
        else:
            expected_rate = (10 + 1) / (160 + 3)
        
        print(f"   {ind_id}: {count}/{num_trials} = {actual_rate:.2%} "
              f"(预期约 {expected_rate:.2%})")
    
    # 验证是否符合预期
    high_rate = selection_counts["high_reward"] / num_trials
    low_rate = selection_counts["low_reward"] / num_trials
    
    if high_rate > low_rate:
        print(f"\n✅ 测试成功！高reward个体的选择频率 ({high_rate:.2%}) > 低reward个体 ({low_rate:.2%})")
        return True
    else:
        print(f"\n❌ 测试失败！选择频率不符合预期")
        return False

if __name__ == "__main__":
    success = test_reward_based_selection()
    if success:
        print("🎉 基于reward的比例选择正常工作！")
    else:
        print("🔧 需要检查实现...")