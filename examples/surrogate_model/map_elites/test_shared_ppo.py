#!/usr/bin/env python3
"""
测试共享PPO训练器的独立演示脚本
"""

import os
import sys
import time
import torch
import numpy as np

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_shared_ppo_basic():
    """测试基础共享PPO功能"""
    print("🚀 测试共享PPO训练器 - 基础功能")
    print("=" * 50)
    
    try:
        from shared_ppo_trainer import SharedPPOTrainer
        
        # 配置
        model_config = {
            'observation_dim': 10,
            'action_dim': 3,
            'hidden_dim': 128
        }
        
        training_config = {
            'lr': 1e-3,
            'buffer_size': 5000,
            'min_batch_size': 100,  # 较小的批次大小
            'model_path': './test_shared_ppo_model.pth',
            'update_interval': 20
        }
        
        # 创建训练器
        print("🤖 创建共享PPO训练器...")
        trainer = SharedPPOTrainer(model_config, training_config)
        
        # 启动训练
        print("🚀 启动训练进程...")
        trainer.start_training()
        
        # 模拟添加一些经验
        print("📊 添加模拟经验数据...")
        for i in range(150):  # 添加150个经验（超过min_batch_size）
            experience = {
                'observation': np.random.randn(10).astype(np.float32),
                'action': np.random.randn(3).astype(np.float32),
                'reward': float(np.random.randn()),  # 🔧 修复：转换为Python float
                'next_observation': np.random.randn(10).astype(np.float32),
                'done': i % 50 == 49,  # 每50步结束一个episode
                'step': i
            }
            trainer.add_experience(experience)
            
            if i % 50 == 0:
                print(f"   📈 已添加 {i+1} 个经验...")
        
        # 等待训练处理
        print("⏳ 等待训练处理经验...")
        time.sleep(5)
        
        # 检查模型是否保存
        if os.path.exists(training_config['model_path']):
            print("✅ 模型文件已创建")
            model_state = torch.load(training_config['model_path'])
            print(f"   📊 模型包含: {list(model_state.keys())}")
        else:
            print("⚠️ 模型文件未找到")
        
        # 停止训练
        print("🛑 停止训练...")
        trainer.stop_training()
        
        print("✅ 共享PPO基础功能测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_shared_ppo_multi_worker():
    """测试多工作器共享PPO（简化版）"""
    print("\n🚀 测试共享PPO训练器 - 多工作器（简化版）")
    print("=" * 50)
    
    try:
        from shared_ppo_trainer import SharedPPOTrainer
        
        # 配置
        model_config = {
            'observation_dim': 8,
            'action_dim': 2,
            'hidden_dim': 64
        }
        
        training_config = {
            'lr': 2e-3,
            'buffer_size': 3000,
            'min_batch_size': 50,
            'model_path': './test_multi_worker_ppo.pth',
            'update_interval': 10
        }
        
        # 创建训练器
        trainer = SharedPPOTrainer(model_config, training_config)
        trainer.start_training()
        
        # 🔧 简化：在主进程中模拟多个工作器添加经验
        print("👥 模拟3个工作器添加经验...")
        for worker_id in range(3):
            print(f"   🤖 工作器 {worker_id} 添加经验...")
            for step in range(60):  # 每个工作器60步
                experience = {
                    'observation': np.random.randn(8).astype(np.float32),
                    'action': np.random.randn(2).astype(np.float32),
                    'reward': float(np.random.randn() * 0.1),
                    'next_observation': np.random.randn(8).astype(np.float32),
                    'done': step % 20 == 19,
                    'worker_id': worker_id,
                    'step': step
                }
                trainer.add_experience(experience)
            
            print(f"   ✅ 工作器 {worker_id} 完成 (60个经验)")
        
        # 等待训练处理
        print("⏳ 等待训练处理所有经验...")
        time.sleep(3)
        
        # 停止训练
        trainer.stop_training()
        
        print("✅ 多工作器模拟测试完成")
        
    except Exception as e:
        print(f"❌ 多工作器测试失败: {e}")
        import traceback
        traceback.print_exc()

def cleanup_test_files():
    """清理测试文件"""
    test_files = [
        './test_shared_ppo_model.pth',
        './test_multi_worker_ppo.pth',
        './shared_ppo_demo.pth'
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🧹 清理文件: {file_path}")

if __name__ == "__main__":
    print("🧪 共享PPO训练器测试套件")
    print("=" * 60)
    
    # 设置多进程启动方法
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    try:
        # 运行测试
        test_shared_ppo_basic()
        time.sleep(1)
        test_shared_ppo_multi_worker()
        
        print("\n🎉 所有测试完成!")
        
    except KeyboardInterrupt:
        print("\n🛑 测试被中断")
    except Exception as e:
        print(f"\n❌ 测试套件失败: {e}")
    finally:
        # 清理
        print("\n🧹 清理测试文件...")
        cleanup_test_files()
        print("✅ 清理完成")
