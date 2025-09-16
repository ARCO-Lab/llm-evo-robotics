#!/usr/bin/env python3
"""
检查共享PPO模型存储和架构信息
"""

import os
import sys
import torch
import argparse
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def check_shared_model_info():
    """检查共享PPO模型的存储信息"""
    
    print("=" * 60)
    print("🤖 共享PPO模型架构信息")
    print("=" * 60)
    
    # 1. 架构说明
    print("\n📊 共享PPO训练架构:")
    print("   ┌─────────────────────────────────────────────┐")
    print("   │  主进程 (MAP-Elites Controller)            │")
    print("   │  ├── 共享PPO训练进程 (1个)                  │")
    print("   │  │   └── 更新单一共享模型                   │")
    print("   │  └── 工作进程 (4个)                        │")
    print("   │      ├── Worker 1: 收集经验 + 可视化        │")
    print("   │      ├── Worker 2: 收集经验 + 可视化        │")
    print("   │      ├── Worker 3: 收集经验 + 可视化        │")
    print("   │      └── Worker 4: 收集经验 + 可视化        │")
    print("   └─────────────────────────────────────────────┘")
    
    print("\n🔄 数据流:")
    print("   经验收集: Workers → 共享经验缓冲区 → PPO训练进程")
    print("   模型更新: PPO训练进程 → 共享模型文件 → Workers")
    print("   可视化:   每个Worker独立显示自己的机器人")
    
    # 2. 检查默认模型路径
    default_save_dir = "./map_elites_shared_results"
    model_paths = [
        f"{default_save_dir}/shared_ppo_model.pth",
        "./shared_ppo_model.pth",
        "./shared_ppo_demo.pth"
    ]
    
    print(f"\n💾 模型存储位置检查:")
    for model_path in model_paths:
        abs_path = os.path.abspath(model_path)
        exists = os.path.exists(model_path)
        size_info = ""
        
        if exists:
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
            size_info = f" ({size_mb:.2f}MB, 修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')})"
            
        status = "✅ 存在" if exists else "❌ 不存在"
        print(f"   {status} {abs_path}{size_info}")
    
    # 3. 检查具体的模型内容
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"\n🔍 检查模型: {model_path}")
            try:
                model_state = torch.load(model_path, map_location='cpu')
                print(f"   模型组件:")
                for key in model_state.keys():
                    if isinstance(model_state[key], dict):
                        print(f"     - {key}: {len(model_state[key])} 个参数")
                    else:
                        print(f"     - {key}: {model_state[key]}")
                
                if 'update_count' in model_state:
                    print(f"   🔄 模型更新次数: {model_state['update_count']}")
                    
            except Exception as e:
                print(f"   ⚠️ 加载模型失败: {e}")
    
    # 4. 显示配置信息
    print(f"\n⚙️ 共享PPO配置:")
    print(f"   缓冲区大小: 20,000 个经验")
    print(f"   最小批次大小: 500 个经验")
    print(f"   更新间隔: 每50个批次保存一次")
    print(f"   学习率: 2e-4")
    print(f"   并行工作进程: 4个")
    
    print(f"\n🎯 优势:")
    print(f"   • 多个机器人并行训练收集经验")
    print(f"   • 单一模型避免参数冲突")
    print(f"   • 每个机器人独立可视化")
    print(f"   • 经验共享提高训练效率")

def check_current_training_status():
    """检查当前训练状态"""
    print("\n" + "=" * 60)
    print("📈 当前训练状态")
    print("=" * 60)
    
    # 检查进程
    try:
        import psutil
        ppo_processes = []
        map_elites_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if 'shared_ppo_trainer' in cmdline:
                    ppo_processes.append(proc)
                elif 'map_elites_trainer' in cmdline:
                    map_elites_processes.append(proc)
            except:
                pass
        
        print(f"\n🔄 运行中的进程:")
        if map_elites_processes:
            for proc in map_elites_processes:
                print(f"   ✅ MAP-Elites主进程 (PID: {proc.info['pid']})")
        else:
            print(f"   ❌ 没有运行中的MAP-Elites进程")
            
        if ppo_processes:
            for proc in ppo_processes:
                print(f"   ✅ 共享PPO训练进程 (PID: {proc.info['pid']})")
        else:
            print(f"   ❌ 没有运行中的共享PPO进程")
            
    except ImportError:
        print("   ⚠️ 需要安装psutil来检查进程状态: pip install psutil")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查共享PPO模型信息")
    parser.add_argument("--status", action="store_true", help="检查当前训练状态")
    args = parser.parse_args()
    
    check_shared_model_info()
    
    if args.status:
        check_current_training_status()
    
    print("\n" + "=" * 60)
    print("✅ 检查完成")
    print("=" * 60)
