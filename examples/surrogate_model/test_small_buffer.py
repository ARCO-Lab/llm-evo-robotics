#!/usr/bin/env python3
"""
测试小Buffer的SAC训练效果
"""

import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)

# 修改SAC model使用小buffer
def test_small_buffer_config():
    print("🔧 测试小Buffer配置的建议...")
    print("="*40)
    
    configs = [
        {"name": "当前配置", "capacity": 100000, "batch_size": 256},
        {"name": "小Buffer", "capacity": 10000, "batch_size": 256},
        {"name": "微Buffer", "capacity": 5000, "batch_size": 128},
        {"name": "极小Buffer", "capacity": 2000, "batch_size": 64},
    ]
    
    for config in configs:
        capacity = config["capacity"]
        batch_size = config["batch_size"]
        
        # 计算关键指标
        memory_mb = capacity * 0.004  # 估算内存使用
        turnover_steps = capacity  # buffer完全更新需要的步数
        freshness_1k = min(1.0, 1000 / capacity)  # 1000步内的新鲜度
        
        print(f"\n{config['name']}:")
        print(f"  Buffer容量: {capacity:,}")
        print(f"  Batch大小: {batch_size}")
        print(f"  内存使用: ~{memory_mb:.1f}MB")
        print(f"  Buffer周转: {turnover_steps:,}步")
        print(f"  1000步新鲜度: {freshness_1k:.1%}")
        
        if capacity <= 10000:
            print(f"  ✅ 优势: 更新鲜的经验，更快适应策略变化")
        if capacity >= 50000:
            print(f"  ⚠️ 风险: 包含过多陈旧经验")
    
    print(f"\n💡 推荐配置: 小Buffer (10000)")
    print(f"  理由: 平衡样本效率和经验新鲜度")

if __name__ == "__main__":
    test_small_buffer_config()
