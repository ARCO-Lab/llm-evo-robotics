#!/usr/bin/env python3
"""
测试MAP-Elites与enhanced_train_interface的连接
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from map_elites_trainer import MAPElitesEvolutionTrainer
import argparse

def test_enhanced_connection():
    """测试与enhanced_train的连接"""
    print("🧪 测试MAP-Elites与enhanced_train_interface的连接\n")
    
    # 创建基础参数
    base_args = argparse.Namespace()
    base_args.seed = 42
    base_args.update_frequency = 1
    
    # 🔧 测试不同配置
    configs = [
        {"name": "静默模式", "rendering": False, "silent": True},
        {"name": "可视化模式", "rendering": True, "silent": False},
    ]
    
    for config in configs:
        print(f"🧪 测试配置: {config['name']}")
        
        try:
            # 创建训练器
            trainer = MAPElitesEvolutionTrainer(
                base_args=base_args,
                num_initial_random=2,  # 只测试2个个体
                training_steps_per_individual=1000,  # 短时间训练
                enable_rendering=config['rendering'],
                silent_mode=config['silent']
            )
            
            print(f"✅ 训练器创建成功")
            
            # 测试单个个体训练
            print(f"🚀 开始单个体测试...")
            individual = trainer._create_random_individual(0)
            evaluated_individual = trainer.adapter.evaluate_individual(individual, 1000)
            
            print(f"✅ 个体训练成功: fitness={evaluated_individual.fitness:.2f}")
            
            if config['rendering']:
                print(f"   📝 注意: 如果能看到机械臂可视化，说明渲染正常")
            
        except Exception as e:
            print(f"❌ 配置 {config['name']} 测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"{'='*50}\n")

if __name__ == "__main__":
    test_enhanced_connection()