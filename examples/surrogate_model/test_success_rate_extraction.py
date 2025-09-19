#!/usr/bin/env python3
"""
测试成功率提取功能
"""

import re
from enhanced_multi_network_extractor import EnhancedMultiNetworkExtractor

def test_success_rate_extraction():
    """测试成功率提取功能"""
    print("🧪 测试成功率提取功能")
    
    # 创建提取器
    extractor = EnhancedMultiNetworkExtractor("test_success_rate")
    
    # 模拟训练输出
    test_lines = [
        "============================================================",
        "📊 PPO训练进度报告 [Step 1500]",
        "============================================================", 
        "🎯 当前Episode: 1/2",
        "📈 Episode内步数: 1500",
        "🏆 当前最佳距离: 382.4px",
        "📊 当前Episode最佳距离: 371.9px",
        "✅ 当前成功率: 85.5%",
        "🔄 连续成功次数: 3",
        "📋 已完成Episodes: 2",
        "🤖 PPO模型状态:",
        "   📈 学习率: 8.24e-06",
        "   🔄 更新次数: 7",
        "   💾 Buffer大小: 53",
        "============================================================",
        "",
        "🔥 PPO网络Loss更新 [Step 1511]:",
        "   📊 Actor Loss: 0.370621",
        "   📊 Critic Loss: 28.414595",
        "   📊 总Loss: 28.785216",
        "   🎭 Entropy: -0.486282",
        "   📈 学习率: 4.92e-07",
        "   🔄 更新次数: 8",
        "   💾 Buffer大小: 0",
        "   =================================================="
    ]
    
    print("📊 处理模拟训练输出...")
    
    for line in test_lines:
        extractor._process_line(line)
    
    # 保存数据
    extractor._save_all_data()
    
    print("\n📈 测试结果:")
    print(f"   PPO损失记录: {len(extractor.loss_data['ppo'])} 条")
    print(f"   性能指标记录: {len(extractor.loss_data['performance'])} 条")
    
    if extractor.loss_data['performance']:
        perf_data = extractor.loss_data['performance'][0]
        print(f"   成功率: {perf_data.get('success_rate', 'N/A')}%")
        print(f"   最佳距离: {perf_data.get('best_distance', 'N/A')}px")
        print(f"   连续成功: {perf_data.get('consecutive_success', 'N/A')}")
        print(f"   完成Episodes: {perf_data.get('completed_episodes', 'N/A')}")
    
    print("✅ 成功率提取测试完成")

if __name__ == "__main__":
    test_success_rate_extraction()

