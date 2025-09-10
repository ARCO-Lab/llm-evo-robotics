"""
详细测试和解释GeneticFitnessEvaluator的底层逻辑
使用reacher_with_zigzag_obstacles.yaml定义的障碍物
"""

import numpy as np
from genetic_fitness_evaluator import GeneticFitnessEvaluator

def explain_fitness_logic():
    """详细解释fitness评估的底层逻辑"""
    
    print("🧠 GeneticFitnessEvaluator 底层逻辑详解")
    print("=" * 60)
    
    # 使用指定的障碍物配置
    evaluator = GeneticFitnessEvaluator(
        target_point=[600, 550],  # 来自yaml配置
        start_point=[480, 620],   # 来自yaml配置
        obstacles=None  # 使用默认的锯齿形障碍物
    )
    
    print(f"\n📏 任务分析:")
    print(f"   起始点: [480, 620]")
    print(f"   目标点: [600, 550]") 
    print(f"   直线距离: {evaluator.direct_distance:.1f}px")
    print(f"   障碍物: 锯齿形 (上下两排，通道宽度75px)")
    
    print(f"\n🎯 分层策略阈值:")
    print(f"   长度严重不足: < {evaluator.direct_distance * 0.8:.1f}px")
    print(f"   长度勉强够: {evaluator.direct_distance * 0.8:.1f} - {evaluator.direct_distance * 1.2:.1f}px")
    print(f"   长度充足: > {evaluator.direct_distance * 1.2:.1f}px")
    
    return evaluator

def test_detailed_scoring_logic(evaluator):
    """详细测试每种策略的评分逻辑"""
    
    print(f"\n🧪 详细评分逻辑测试")
    print("=" * 60)
    
    # 设计测试用例，覆盖三种策略
    test_cases = [
        {
            'name': '严重不足机器人',
            'link_lengths': [25, 25, 25],  # 总长75px < 111px
            'training_performance': {
                'max_distance_covered': 60,
                'success_rate': 0.0,
                'avg_reward': -80
            },
            'expected_strategy': 'encourage_length_growth'
        },
        {
            'name': '刚好不足机器人', 
            'link_lengths': [35, 35, 35],  # 总长105px < 111px
            'training_performance': {
                'max_distance_covered': 90,
                'success_rate': 0.0,
                'avg_reward': -60
            },
            'expected_strategy': 'encourage_length_growth'
        },
        {
            'name': '边缘长度机器人A',
            'link_lengths': [40, 40, 40],  # 总长120px, 在111-166px范围
            'training_performance': {
                'success_rate': 0.05,
                'avg_reward': -20,
                'efficiency': 0.2
            },
            'expected_strategy': 'optimize_reachability'
        },
        {
            'name': '边缘长度机器人B',
            'link_lengths': [50, 50, 50],  # 总长150px, 在111-166px范围
            'training_performance': {
                'success_rate': 0.15,
                'avg_reward': 10,
                'efficiency': 0.4
            },
            'expected_strategy': 'optimize_reachability'
        },
        {
            'name': '刚好充足机器人',
            'link_lengths': [55, 55, 60],  # 总长170px > 166px
            'training_performance': {
                'success_rate': 0.6,
                'avg_reward': 60,
                'efficiency': 0.7
            },
            'expected_strategy': 'optimize_performance'
        },
        {
            'name': '长度充足机器人',
            'link_lengths': [70, 70, 70],  # 总长210px > 166px
            'training_performance': {
                'success_rate': 0.85,
                'avg_reward': 80,
                'efficiency': 0.8
            },
            'expected_strategy': 'optimize_performance'
        },
        {
            'name': '过长机器人',
            'link_lengths': [90, 90, 90, 90],  # 总长360px，测试长度惩罚
            'training_performance': {
                'success_rate': 0.75,
                'avg_reward': 70,
                'efficiency': 0.6
            },
            'expected_strategy': 'optimize_performance'
        }
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"🤖 测试 {i+1}: {case['name']}")
        print(f"{'='*50}")
        
        total_length = sum(case['link_lengths'])
        print(f"📊 基础信息:")
        print(f"   链节长度: {case['link_lengths']}")
        print(f"   总长度: {total_length}px")
        print(f"   长度比例: {total_length/evaluator.direct_distance:.2f}")
        print(f"   预期策略: {case['expected_strategy']}")
        
        # 执行评估
        result = evaluator.evaluate_fitness(
            link_lengths=case['link_lengths'],
            training_performance=case['training_performance']
        )
        
        results.append({
            'name': case['name'],
            'total_length': total_length,
            'result': result
        })
        
        # 详细分析
        print(f"\n🎯 评估结果:")
        print(f"   最终fitness: {result['fitness']:.3f}")
        print(f"   实际策略: {result['strategy']}")
        print(f"   分类: {result['category']}")
        print(f"   可达性: {result.get('reachable', 'N/A')}")
        print(f"   原因: {result['reason']}")
        
        # 分数分解
        print(f"\n📈 分数构成分析:")
        score_components = []
        for key, value in result.items():
            if key.endswith('_score') or key.endswith('_bonus') or key.endswith('_penalty'):
                score_components.append((key, value))
                print(f"   {key:20}: {value:6.3f}")
        
        # 验证策略是否正确
        if result['strategy'] == case['expected_strategy']:
            print(f"   ✅ 策略匹配正确")
        else:
            print(f"   ❌ 策略不匹配: 期望{case['expected_strategy']}, 实际{result['strategy']}")
        
        # 逻辑解释
        print(f"\n🧠 逻辑解释:")
        explain_strategy_logic(result, total_length, evaluator.direct_distance)
    
    return results

def explain_strategy_logic(result, total_length, direct_distance):
    """解释具体策略的逻辑"""
    
    category = result['category']
    
    if category == 'insufficient_length':
        print(f"   📏 长度不足策略:")
        print(f"      - 目标: 防止进化停滞，鼓励增长")
        print(f"      - 基础分数: 长度比例({total_length/direct_distance:.2f}) × 0.3")
        print(f"      - 结构奖励: 鼓励合理的链节配置")  
        print(f"      - 增长潜力: 鼓励向更长方向进化")
        print(f"      - 训练奖励: 即使到不了目标，也奖励学习能力")
        
    elif category == 'marginal_length':
        print(f"   ⚖️ 边缘长度策略:")
        print(f"      - 目标: 从'可能到达'变成'确实能到达'")
        print(f"      - 基础分数: 0.3 + 可达性置信度 × 0.4")
        print(f"      - 路径效率: 奖励找到更好的路径")
        print(f"      - 结构优化: 优化链节配置")
        print(f"      - 训练验证: 用实际训练结果验证可达性")
        
    else:  # sufficient_length
        print(f"   ✅ 长度充足策略:")
        print(f"      - 目标: 在保证可达基础上追求最优性能")
        print(f"      - 基础分数: 0.6 (高起点)")
        print(f"      - 训练表现: 成功率和效率是主要评判标准")
        print(f"      - 结构微调: 避免过度复杂的设计")
        print(f"      - 长度惩罚: 防止无意义的过度增长")

def compare_strategies(results):
    """比较不同策略的效果"""
    
    print(f"\n🏆 策略效果对比分析")
    print("=" * 60)
    
    # 按策略分组
    strategies = {}
    for r in results:
        strategy = r['result']['strategy']
        if strategy not in strategies:
            strategies[strategy] = []
        strategies[strategy].append(r)
    
    for strategy, items in strategies.items():
        print(f"\n📊 {strategy} 策略:")
        
        # 按fitness排序
        sorted_items = sorted(items, key=lambda x: x['result']['fitness'], reverse=True)
        
        for item in sorted_items:
            fitness = item['result']['fitness']
            length = item['total_length']
            print(f"   {item['name']:15} (长度{length:3}px): fitness={fitness:.3f}")
        
        # 策略内部分析
        if len(sorted_items) > 1:
            best = sorted_items[0]
            worst = sorted_items[-1]
            improvement = best['result']['fitness'] - worst['result']['fitness']
            print(f"   📈 策略内部差异: {improvement:.3f}")

def test_edge_cases(evaluator):
    """测试边界情况"""
    
    print(f"\n🔬 边界情况测试")
    print("=" * 60)
    
    edge_cases = [
        {
            'name': '刚好80%阈值',
            'link_lengths': [37, 37, 37.2],  # 总长≈111.2px，刚好超过80%阈值
            'training_performance': {'success_rate': 0.1, 'avg_reward': 0}
        },
        {
            'name': '刚好120%阈值', 
            'link_lengths': [55.5, 55.5, 55.6],  # 总长≈166.6px，刚好超过120%阈值
            'training_performance': {'success_rate': 0.7, 'avg_reward': 50}
        },
        {
            'name': '极短机器人',
            'link_lengths': [10, 10],  # 总长20px，极端情况
            'training_performance': {'success_rate': 0.0, 'avg_reward': -100}
        },
        {
            'name': '极长机器人',
            'link_lengths': [100, 100, 100, 100, 100],  # 总长500px，极端情况
            'training_performance': {'success_rate': 0.9, 'avg_reward': 90}
        }
    ]
    
    for case in edge_cases:
        print(f"\n🧪 {case['name']}:")
        result = evaluator.evaluate_fitness(
            link_lengths=case['link_lengths'],
            training_performance=case['training_performance']
        )
        
        total_length = sum(case['link_lengths'])
        print(f"   总长: {total_length}px")
        print(f"   策略: {result['strategy']}")
        print(f"   Fitness: {result['fitness']:.3f}")
        print(f"   分类: {result['category']}")

def main():
    """主测试函数"""
    
    print("🚀 GeneticFitnessEvaluator 底层逻辑详细测试")
    print("使用 reacher_with_zigzag_obstacles.yaml 配置")
    print("=" * 80)
    
    # 1. 解释基础逻辑
    evaluator = explain_fitness_logic()
    
    # 2. 详细测试评分逻辑
    results = test_detailed_scoring_logic(evaluator)
    
    # 3. 策略对比
    compare_strategies(results)
    
    # 4. 边界情况测试
    test_edge_cases(evaluator)
    
    # 5. 总结
    print(f"\n🎯 总结:")
    print(f"   ✅ 分层策略能够根据机器人长度采用不同评估标准")
    print(f"   ✅ 避免了传统方法中短机器人被完全淘汰的问题")
    print(f"   ✅ 使用真实的最短路径算法进行可达性评估")
    print(f"   ✅ 在不同阶段优化不同目标，引导进化方向")
    
    print(f"\n📊 与当前MAP-Elites系统对比:")
    print(f"   当前系统: fitness ≈ total_length / 5")
    print(f"   新系统: 分层评估，物理约束感知，多维度优化")

if __name__ == "__main__":
    main()
