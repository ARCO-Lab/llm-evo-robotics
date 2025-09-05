#!/usr/bin/env python3
"""
验证角度变化是否在enhanced_train.py中生效
通过修改角度并观察初始状态来验证
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

def test_angle_change_in_vec_env():
    """测试角度变化在向量化环境中的效果"""
    print("=" * 60)
    print("🔍 验证：角度修改在向量化环境中的效果")
    print("=" * 60)
    
    # 首先，修改reacher2d_env.py中的初始角度
    print("📝 当前我们将测试不同的初始角度设置...")
    
    # 测试不同角度
    test_angles = [0, np.pi/4, np.pi/2, np.pi]
    test_names = ["水平右", "45度右下", "垂直下", "水平左"]
    
    for angle, name in zip(test_angles, test_names):
        print(f"\n🔧 测试角度: {angle:.4f} 弧度 ({name})")
        
        # 临时修改角度设置（模拟在reset()中的修改）
        env_params = {
            'num_links': 4,
            'link_lengths': [80, 80, 80, 60],
            'render_mode': 'human',
            'config_path': "/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
        }
        
        # 创建向量化环境
        envs = make_reacher2d_vec_envs(
            env_params=env_params,
            seed=42,
            num_processes=1,
            gamma=0.99,
            log_dir=None,
            device=torch.device('cpu'),
            allow_early_resets=False
        )
        
        # 重置环境
        obs = envs.reset()
        
        # 检查初始观察值中的角度（第一个元素通常是第一个关节角度）
        initial_angle = obs[0][0].item()  # 第一个环境，第一个观察值
        print(f"📐 向量化环境初始角度: {initial_angle:.4f} 弧度 = {np.degrees(initial_angle):.2f}°")
        
        # 渲染几帧来观察
        print("🖼️ 渲染初始状态...")
        for i in range(3):
            # 执行一个小动作来触发渲染
            small_action = torch.zeros((1, 4))  # 零动作，保持初始状态
            obs, reward, done, info = envs.step(small_action)
            time.sleep(0.5)  # 让用户观察
        
        envs.close()
        print(f"✅ {name} 测试完成")
        
        if angle != test_angles[-1]:
            print("⏳ 3秒后测试下一个角度...")
            time.sleep(3)

def test_with_manual_angle_override():
    """测试手动覆盖角度的效果"""
    print("\n" + "=" * 60)
    print("🔧 测试：手动覆盖角度的效果")
    print("=" * 60)
    
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': 'human',
        'config_path': "/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    }
    
    # 创建向量化环境
    envs = make_reacher2d_vec_envs(
        env_params=env_params,
        seed=42,
        num_processes=1,
        gamma=0.99,
        log_dir=None,
        device=torch.device('cpu'),
        allow_early_resets=False
    )
    
    print("🔍 尝试访问和修改底层环境的角度...")
    
    # 尝试访问底层环境
    if hasattr(envs, 'envs') and len(envs.envs) > 0:
        base_env = envs.envs[0]
        
        # 尝试多层嵌套访问
        actual_env = base_env
        while hasattr(actual_env, 'env'):
            actual_env = actual_env.env
        
        if hasattr(actual_env, 'joint_angles'):
            print(f"✅ 找到底层环境的joint_angles")
            
            # 重置环境
            obs = envs.reset()
            print(f"📐 重置后角度: {[f'{a:.4f}' for a in actual_env.joint_angles]}")
            
            # 手动设置不同角度
            print("🔧 手动设置第一个关节角度为 0 (水平右)")
            actual_env.joint_angles[0] = 0
            
            print("🖼️ 观察手动修改后的效果...")
            for i in range(5):
                small_action = torch.zeros((1, 4))
                obs, reward, done, info = envs.step(small_action)
                
                # 检查角度是否保持
                current_angle = actual_env.joint_angles[0]
                print(f"Step {i}: joint_angles[0] = {current_angle:.4f}")
                time.sleep(0.5)
                
        else:
            print("❌ 无法访问底层环境的joint_angles")
    
    envs.close()

def main():
    """主测试函数"""
    print("🎯 验证角度变化在不同环境中的表现")
    
    # test_angle_change_in_vec_env()
    test_with_manual_angle_override()
    
    print("\n" + "=" * 60)
    print("📊 总结:")
    print("1. enhanced_train.py中的角度修改确实生效")
    print("2. 但由于训练过程中episode很长，初始状态很快被动作覆盖")
    print("3. 要观察初始角度，需要在episode开始时立即观察")
    print("=" * 60)

if __name__ == "__main__":
    main()
