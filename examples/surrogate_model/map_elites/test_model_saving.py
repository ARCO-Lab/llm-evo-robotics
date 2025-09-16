#!/usr/bin/env python3
"""
测试共享PPO模型保存功能
"""

import os
import sys
import time
import torch
import numpy as np
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.insert(0, current_dir)
from shared_ppo_trainer import SharedPPOTrainer

def test_model_saving():
    """测试模型保存功能"""
    print("🧪 测试共享PPO模型保存功能")
    print("=" * 50)
    
    # 创建测试目录
    test_dir = "./test_model_saving"
    os.makedirs(test_dir, exist_ok=True)
    
    # 配置
    model_config = {
        'observation_dim': 10,
        'action_dim': 3,
        'hidden_dim': 64
    }
    
    training_config = {
        'lr': 1e-3,
        'buffer_size': 1000,
        'min_batch_size': 50,  # 很小的批次大小，快速触发
        'model_path': f'{test_dir}/test_shared_ppo_model.pth',
        'update_interval': 10
    }
    
    print(f"📊 测试配置:")
    print(f"   模型保存路径: {training_config['model_path']}")
    print(f"   最小批次大小: {training_config['min_batch_size']}")
    print(f"   观察维度: {model_config['observation_dim']}")
    print(f"   动作维度: {model_config['action_dim']}")
    
    # 创建共享PPO训练器
    shared_trainer = SharedPPOTrainer(model_config, training_config)
    
    try:
        print("\n🚀 启动共享PPO训练...")
        shared_trainer.start_training()
        
        print("\n🎯 开始添加模拟经验...")
        
        # 添加模拟经验数据
        for i in range(200):  # 添加200个经验，应该触发4次更新（50*4=200）
            experience = {
                'observation': np.random.randn(model_config['observation_dim']).astype(np.float32),
                'action': np.random.randn(model_config['action_dim']).astype(np.float32),
                'reward': float(np.random.randn()),
                'next_observation': np.random.randn(model_config['observation_dim']).astype(np.float32),
                'done': bool(np.random.rand() > 0.9),  # 10%概率结束
                'worker_id': 0,
                'robot_config': {'num_links': 3, 'link_lengths': [50, 50, 50]}
            }
            
            shared_trainer.add_experience(experience)
            
            # 每50个经验检查一次
            if (i + 1) % 50 == 0:
                print(f"✅ 已添加 {i+1} 个经验")
                time.sleep(1)  # 给训练进程时间处理
        
        print("\n⏳ 等待训练进程处理所有经验...")
        time.sleep(10)  # 等待足够时间让训练进程处理
        
        # 检查模型文件是否存在
        model_path = training_config['model_path']
        if os.path.exists(model_path):
            print(f"✅ 模型文件已生成: {model_path}")
            
            # 检查文件大小
            file_size = os.path.getsize(model_path)
            print(f"📏 文件大小: {file_size} 字节 ({file_size/1024:.1f} KB)")
            
            # 尝试加载模型
            try:
                model_state = torch.load(model_path, map_location='cpu')
                print(f"🔍 模型内容:")
                for key in model_state.keys():
                    if isinstance(model_state[key], dict):
                        print(f"   - {key}: {len(model_state[key])} 个参数")
                    else:
                        print(f"   - {key}: {model_state[key]}")
                        
                print(f"✅ 模型加载成功！")
                
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
        else:
            print(f"❌ 模型文件未生成: {model_path}")
            
        # 检查备份文件
        backup_files = [f for f in os.listdir(test_dir) if f.startswith('test_shared_ppo_model_backup_')]
        if backup_files:
            print(f"📦 备份文件: {len(backup_files)} 个")
            for backup in backup_files:
                print(f"   - {backup}")
        else:
            print("📦 没有生成备份文件")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🛑 停止共享PPO训练...")
        shared_trainer.stop_training()
        
    print("\n🧹 清理测试文件...")
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"✅ 已删除测试目录: {test_dir}")
        
    print("\n🎉 测试完成！")

def test_directory_creation():
    """测试目录创建功能"""
    print("\n🧪 测试目录创建功能")
    print("-" * 30)
    
    # 测试深层目录创建
    deep_path = "./test_deep/level1/level2/level3/model.pth"
    deep_dir = os.path.dirname(deep_path)
    
    print(f"📁 测试路径: {deep_path}")
    print(f"📁 目录路径: {deep_dir}")
    
    # 模拟shared_ppo_trainer中的目录创建逻辑
    if deep_dir and not os.path.exists(deep_dir):
        os.makedirs(deep_dir, exist_ok=True)
        print(f"✅ 目录创建成功: {deep_dir}")
    else:
        print(f"ℹ️ 目录已存在: {deep_dir}")
    
    # 测试文件写入
    try:
        dummy_data = {'test': 'data', 'timestamp': datetime.now().isoformat()}
        torch.save(dummy_data, deep_path)
        print(f"✅ 文件写入成功: {deep_path}")
        
        # 验证文件
        loaded_data = torch.load(deep_path)
        print(f"✅ 文件读取成功: {loaded_data}")
        
    except Exception as e:
        print(f"❌ 文件操作失败: {e}")
    
    # 清理
    import shutil
    if os.path.exists("./test_deep"):
        shutil.rmtree("./test_deep")
        print("🧹 已清理测试文件")

if __name__ == "__main__":
    # 设置多进程启动方法
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    test_directory_creation()
    test_model_saving()
