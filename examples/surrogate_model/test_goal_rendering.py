#!/usr/bin/env python3
"""
测试goal渲染问题
"""

import sys
import os
sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/envs')
from reacher2d_env import Reacher2DEnv

def test_goal_rendering():
    print("🔍 测试goal渲染问题")
    print("="*50)
    
    env = Reacher2DEnv(
        num_links=3,
        link_lengths=[60, 60, 60],
        render_mode=None,  # 先不渲染，只检查数据
        config_path='../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml'
    )
    
    print(f"📋 配置文件内容:")
    print(f"   config['goal']: {env.config.get('goal', '未找到goal配置')}")
    print(f"   base_goal_pos: {env.base_goal_pos}")
    
    print(f"\n🔄 reset前:")
    print(f"   goal_pos: {getattr(env, 'goal_pos', '未设置')}")
    
    obs = env.reset()
    
    print(f"\n🔄 reset后:")
    print(f"   goal_pos: {env.goal_pos}")
    print(f"   goal_radius: {getattr(env, 'goal_radius', '未设置')}")
    
    # 测试渲染坐标
    end_pos = env._get_end_effector_position()
    print(f"\n🎯 位置信息:")
    print(f"   锚点: {env.anchor_point}")
    print(f"   末端执行器: {end_pos}")
    print(f"   目标位置: {env.goal_pos}")
    
    # 检查渲染时使用的坐标
    goal_render_pos = env.goal_pos.astype(int)
    end_render_pos = (int(end_pos[0]), int(end_pos[1]))
    
    print(f"\n🖼️ 渲染坐标:")
    print(f"   目标渲染位置: {goal_render_pos}")
    print(f"   末端渲染位置: {end_render_pos}")
    print(f"   两点距离: {((goal_render_pos[0]-end_render_pos[0])**2 + (goal_render_pos[1]-end_render_pos[1])**2)**0.5:.1f}px")
    
    # 测试不同的goal位置
    print(f"\n🧪 测试修改goal位置:")
    
    # 手动设置不同的goal
    test_goals = [
        [300, 550],  # 锚点位置
        [400, 550],  # 水平右侧
        [300, 400],  # 垂直上方
        [500, 600],  # 对角线
    ]
    
    for i, test_goal in enumerate(test_goals):
        env.goal_pos = test_goal
        goal_render = env.goal_pos.astype(int)
        print(f"   测试{i+1}: goal_pos={test_goal} -> 渲染位置={goal_render}")

if __name__ == "__main__":
    test_goal_rendering()
