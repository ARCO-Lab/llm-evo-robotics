#!/usr/bin/env python3
"""
演示共享PPO模型恢复功能
"""

import os
import sys
import torch
import time
import argparse

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared_ppo_trainer import SharedPPOTrainer

def create_demo_model():
    """创建一个演示模型文件"""
    print("🎬 创建演示模型文件...")
    
    model_config = {
        'observation_dim': 14,
        'action_dim': 3,
        'hidden_dim': 256
    }
    
    training_config = {
        'lr': 2e-4,
        'buffer_size': 5000,
        'min_batch_size': 100,
        'model_path': './map_elites_shared_ppo_results/shared_ppo_model.pth',
        'update_interval': 10
    }
    
    trainer = SharedPPOTrainer(model_config, training_config)
    
    try:
        trainer.start_training()
        print("✅ 训练器启动成功")
        
        # 添加一些经验来触发模型保存
        for i in range(150):
            fake_experience = {
                'observation': torch.randn(model_config['observation_dim']).numpy(),
                'action': torch.randn(model_config['action_dim']).numpy(),
                'reward': float(torch.randn(1).item()),
                'next_observation': torch.randn(model_config['observation_dim']).numpy(),
                'done': False,
                'worker_id': 0,
                'robot_config': {'num_links': 3, 'link_lengths': [90.0, 90.0, 90.0]}
            }
            trainer.add_experience(fake_experience)
        
        print("⏳ 等待模型训练和保存...")
        time.sleep(8)
        
    finally:
        trainer.stop_training()
    
    # 检查模型是否保存成功
    if os.path.exists(training_config['model_path']):
        print(f"✅ 演示模型已创建: {training_config['model_path']}")
        return True
    else:
        print(f"❌ 演示模型创建失败")
        return False

def demo_resume_functionality():
    """演示恢复功能"""
    print("\n" + "="*60)
    print("🎬 共享PPO模型恢复功能演示")
    print("="*60)
    
    # 第1步：创建演示模型
    print("\n📝 第1步：创建初始模型")
    if not create_demo_model():
        print("❌ 无法创建演示模型，退出")
        return
    
    # 第2步：显示命令示例
    print("\n📝 第2步：命令使用示例")
    print("现在你可以使用以下命令：")
    print()
    print("🆕 开始新训练（会覆盖现有模型）:")
    print("   python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared")
    print()
    print("🔄 恢复训练（从现有模型继续）:")
    print("   python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared --resume")
    print()
    print("🎨 恢复训练 + 禁用可视化:")
    print("   python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared --resume --no-render")
    print()
    print("🔇 恢复训练 + 静默模式:")
    print("   python examples/surrogate_model/map_elites/map_elites_trainer.py --train-shared --resume --silent")
    
    # 第3步：检查模型信息
    print("\n📝 第3步：当前模型信息")
    model_path = './map_elites_shared_ppo_results/shared_ppo_model.pth'
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"📊 模型文件: {model_path}")
            print(f"📊 更新次数: {checkpoint.get('update_count', 0)}")
            print(f"📊 文件大小: {os.path.getsize(model_path)/1024:.1f} KB")
            print(f"📊 包含组件: {', '.join(checkpoint.keys())}")
        except Exception as e:
            print(f"⚠️ 无法读取模型信息: {e}")
    
    print("\n🎉 演示完成！你现在可以使用 --resume 参数来恢复训练了。")

if __name__ == "__main__":
    demo_resume_functionality()
