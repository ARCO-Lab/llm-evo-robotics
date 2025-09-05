#!/usr/bin/env python3
"""
截图版本的enhanced_train.py
专门用于截取前5个训练步骤的截图
"""

import sys
import os
import numpy as np
import time
import torch

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/env_config'))

from reacher2d_env import Reacher2DEnv
from env_wrapper import make_reacher2d_vec_envs

def enhanced_train_screenshot():
    """模拟enhanced_train.py的前5步并截图"""
    print("=" * 60)
    print("🖼️ Enhanced Train 前5步截图模式")
    print("=" * 60)
    
    # 环境参数 - 与enhanced_train.py相同
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': 'human',
        'config_path': "/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    }
    
    print("🚀 创建向量化环境...")
    
    # 创建向量化环境 - 单进程便于控制
    envs = make_reacher2d_vec_envs(
        env_params=env_params,
        seed=42,
        num_processes=1,
        gamma=0.99,
        log_dir=None,
        device=torch.device('cpu'),
        allow_early_resets=False
    )
    
    # 创建同步渲染环境 - 启用截图模式
    sync_env = Reacher2DEnv(**env_params)
    sync_env.screenshot_mode = True
    sync_env.screenshot_dir = 'screenshots/enhanced_train'
    
    print("✅ 环境创建完成")
    
    # 初始重置
    print("\n🔄 重置环境")
    current_obs = envs.reset()
    sync_env.reset()
    
    print(f"📊 初始观察形状: {current_obs.shape}")
    print(f"📐 初始角度 (从观察): {current_obs[0][0]:.4f} 弧度 = {np.degrees(current_obs[0][0].item()):.2f}°")
    
    # 渲染初始状态（step 0）
    print(f"\n📸 渲染初始状态 (Step 0)")
    sync_env.render()
    time.sleep(0.5)
    
    # 模拟前5个训练步骤
    for step in range(1, 6):
        print(f"\n📸 训练步骤 {step}")
        
        # 生成随机动作 - 模拟训练过程
        action_batch = torch.from_numpy(np.array([envs.action_space.sample() for _ in range(1)]))
        
        print(f"   动作: [{', '.join([f'{a:.3f}' for a in action_batch[0].numpy()])}]")
        
        # 执行动作
        next_obs, reward, done, infos = envs.step(action_batch)
        
        # 同步渲染环境
        sync_action = action_batch[0].cpu().numpy() if hasattr(action_batch, 'cpu') else action_batch[0]
        sync_obs, sync_reward, sync_done, sync_info = sync_env.step(sync_action)
        
        print(f"   奖励: {reward[0].item():.2f}")
        print(f"   结束: {done[0].item()}")
        
        if 'end_effector_pos' in sync_info:
            end_pos = sync_info['end_effector_pos']
            print(f"   末端位置: [{end_pos[0]:.1f}, {end_pos[1]:.1f}]")
        
        # 渲染并自动截图
        sync_env.render()
        time.sleep(0.5)
        
        # 更新观察
        current_obs = next_obs.clone()
        
        if done[0].item():
            print(f"   Episode在步骤{step}结束")
            break
    
    print(f"\n✅ 截图完成，保存在: {sync_env.screenshot_dir}")
    
    # 保持窗口打开一会儿
    print("🖼️ 保持窗口打开3秒...")
    time.sleep(3)
    
    # 清理
    envs.close()
    sync_env.close()
    print("🔒 环境已关闭")

if __name__ == "__main__":
    enhanced_train_screenshot()
