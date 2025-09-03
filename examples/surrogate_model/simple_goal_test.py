#!/usr/bin/env python3
"""
简单测试goal位置
"""

import sys
import os
import numpy as np
sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/envs')
from reacher2d_env import Reacher2DEnv

def test_goal_positions():
    print("🎯 测试goal位置加载")
    print("="*50)
    
    env = Reacher2DEnv(
        num_links=3,
        link_lengths=[60, 60, 60],
        render_mode=None,
        config_path='../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml'
    )
    
    obs = env.reset()
    
    print(f"✅ 配置文件加载:")
    print(f"   config goal: {env.config['goal']['position']}")
    print(f"   base_goal_pos: {env.base_goal_pos}")
    print(f"   实际goal_pos: {env.goal_pos}")
    
    print(f"\n📍 位置对比:")
    print(f"   锚点: {env.anchor_point}")
    print(f"   末端: {env._get_end_effector_position()}")
    print(f"   目标: {env.goal_pos}")
    
    # 计算距离
    anchor = np.array(env.anchor_point)
    goal = np.array(env.goal_pos)
    end_pos = np.array(env._get_end_effector_position())
    
    anchor_to_goal = np.linalg.norm(goal - anchor)
    end_to_goal = np.linalg.norm(goal - end_pos)
    
    print(f"\n📏 距离分析:")
    print(f"   锚点到目标: {anchor_to_goal:.1f}px")
    print(f"   末端到目标: {end_to_goal:.1f}px")
    print(f"   理论reach: 180px")
    print(f"   可达性: {'✅ 可达' if anchor_to_goal <= 180 else '❌ 不可达'}")
    
    # 现在让我们手动修改goal_pos并看看区别
    print(f"\n🧪 测试手动修改goal_pos:")
    
    original_goal = env.goal_pos.copy()
    test_goals = [
        [300, 550],  # 锚点
        [400, 550],  # 右侧100px
        [500, 550],  # 右侧200px
    ]
    
    for test_goal in test_goals:
        env.goal_pos = np.array(test_goal)
        distance = np.linalg.norm(np.array(test_goal) - anchor)
        print(f"   设置goal为{test_goal}: 距离锚点{distance:.1f}px")
        print(f"      渲染坐标: {env.goal_pos.astype(int)}")
    
    # 恢复原始goal
    env.goal_pos = original_goal
    print(f"\n🔄 恢复原始goal: {env.goal_pos}")

if __name__ == "__main__":
    test_goal_positions()
