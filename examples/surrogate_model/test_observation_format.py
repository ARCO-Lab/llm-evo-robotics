#!/usr/bin/env python3
"""
测试 Reacher2D 环境的观察输出格式
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, '../../')
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from reacher2d_env import Reacher2DEnv
import numpy as np

def test_observation_format():
    print("🔍 测试 Reacher2D 环境的观察格式")
    print("=" * 50)
    
    # 测试不同关节数的环境
    for num_links in [3, 4, 5]:
        print(f"\n🤖 测试 {num_links} 关节机器人:")
        
        env = Reacher2DEnv(
            num_links=num_links,
            link_lengths=[80, 50, 30, 20, 10][:num_links],
            render_mode=None,  # 不渲染，只测试数据,
            config_path= "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
        )
        
        print(f"   动作空间: {env.action_space}")
        print(f"   观察空间: {env.observation_space}")
        print(f"   预期观察维度: {num_links * 2 + 2} (关节角度 + 角速度 + 末端位置)")
        
        # 重置环境获取初始观察
        obs = env.reset()
        print(f"   实际观察维度: {len(obs)}")
        print(f"   观察数据: {obs}")
        
        # 分析观察结构
        angles = obs[:num_links]
        angular_vels = obs[num_links:2*num_links]
        end_effector_pos = obs[2*num_links:2*num_links+2]
        
        print(f"   关节角度 ({num_links}): {angles}")
        print(f"   角速度 ({num_links}): {angular_vels}")
        print(f"   末端位置 (2): {end_effector_pos}")
        
        # 执行一步动作看看变化
        action = np.random.uniform(-10, 10, num_links)
        print(f"   测试动作: {action}")
        
        obs_next, reward, done, info = env.step(action)
        print(f"   步进后观察: {obs_next}")
        print(f"   奖励: {reward:.3f}")
        
        env.close()
        print()

if __name__ == "__main__":
    test_observation_format()