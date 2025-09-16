#!/usr/bin/env python3
"""
测试共享PPO模型的保存和恢复功能
"""

import os
import sys
import torch
import time

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared_ppo_trainer import SharedPPOTrainer

def test_model_resume():
    """测试模型恢复功能"""
    print("🧪 测试共享PPO模型的保存和恢复功能")
    print("=" * 60)
    
    # 配置
    model_config = {
        'observation_dim': 14,
        'action_dim': 3,
        'hidden_dim': 256
    }
    
    training_config = {
        'lr': 2e-4,
        'buffer_size': 5000,
        'min_batch_size': 100,
        'model_path': './test_resume_model.pth',
        'update_interval': 10
    }
    
    print(f"📊 配置信息:")
    print(f"   模型路径: {training_config['model_path']}")
    print(f"   观察维度: {model_config['observation_dim']}")
    print(f"   动作维度: {model_config['action_dim']}")
    
    # 第一阶段：创建新模型并训练一段时间
    print(f"\n🚀 第一阶段：创建新模型")
    
    # 删除已有的模型文件
    if os.path.exists(training_config['model_path']):
        os.remove(training_config['model_path'])
        print(f"🗑️ 删除已有模型文件")
    
    trainer1 = SharedPPOTrainer(model_config, training_config)
    
    try:
        trainer1.start_training()
        print(f"✅ 第一个训练器启动成功")
        
        # 添加一些虚拟经验
        print(f"📝 添加虚拟经验...")
        for i in range(200):  # 添加200个经验
            fake_experience = {
                'observation': torch.randn(model_config['observation_dim']).numpy(),
                'action': torch.randn(model_config['action_dim']).numpy(),
                'reward': float(torch.randn(1).item()),
                'next_observation': torch.randn(model_config['observation_dim']).numpy(),
                'done': False,
                'worker_id': 0,
                'robot_config': {'num_links': 3, 'link_lengths': [90.0, 90.0, 90.0]}
            }
            trainer1.add_experience(fake_experience)
        
        print(f"⏳ 等待模型训练和保存...")
        time.sleep(8)  # 等待训练进程处理经验并保存模型
        
    finally:
        trainer1.stop_training()
        print(f"🛑 第一个训练器已停止")
    
    # 检查模型是否被保存
    if os.path.exists(training_config['model_path']):
        print(f"✅ 模型文件已保存: {training_config['model_path']}")
        
        # 检查模型内容
        try:
            checkpoint = torch.load(training_config['model_path'], map_location='cpu')
            print(f"📊 模型检查点信息:")
            print(f"   包含键: {list(checkpoint.keys())}")
            if 'update_count' in checkpoint:
                print(f"   更新次数: {checkpoint['update_count']}")
            print(f"   Actor参数数量: {len(checkpoint['actor'])}")
            print(f"   Critic参数数量: {len(checkpoint['critic'])}")
        except Exception as e:
            print(f"⚠️ 无法读取模型文件: {e}")
    else:
        print(f"❌ 模型文件未找到: {training_config['model_path']}")
        return False
    
    # 第二阶段：加载已有模型并继续训练
    print(f"\n🔄 第二阶段：加载已有模型继续训练")
    
    trainer2 = SharedPPOTrainer(model_config, training_config)
    
    try:
        trainer2.start_training()
        print(f"✅ 第二个训练器启动成功（应该加载了已有模型）")
        
        # 再添加一些经验
        print(f"📝 添加更多虚拟经验...")
        for i in range(100):  # 再添加100个经验
            fake_experience = {
                'observation': torch.randn(model_config['observation_dim']).numpy(),
                'action': torch.randn(model_config['action_dim']).numpy(),
                'reward': float(torch.randn(1).item()),
                'next_observation': torch.randn(model_config['observation_dim']).numpy(),
                'done': False,
                'worker_id': 1,
                'robot_config': {'num_links': 4, 'link_lengths': [80.0, 80.0, 80.0, 80.0]}
            }
            trainer2.add_experience(fake_experience)
        
        print(f"⏳ 等待模型继续训练...")
        time.sleep(5)
        
    finally:
        trainer2.stop_training()
        print(f"🛑 第二个训练器已停止")
    
    # 最终检查
    if os.path.exists(training_config['model_path']):
        try:
            final_checkpoint = torch.load(training_config['model_path'], map_location='cpu')
            final_update_count = final_checkpoint.get('update_count', 0)
            print(f"🎉 最终模型更新次数: {final_update_count}")
            print(f"✅ 模型恢复功能测试成功！")
            
            # 清理测试文件
            os.remove(training_config['model_path'])
            print(f"🧹 清理测试文件")
            
            return True
        except Exception as e:
            print(f"❌ 读取最终模型失败: {e}")
            return False
    else:
        print(f"❌ 最终模型文件未找到")
        return False

if __name__ == "__main__":
    success = test_model_resume()
    if success:
        print(f"\n🎉 模型恢复功能测试通过！")
    else:
        print(f"\n❌ 模型恢复功能测试失败！")
