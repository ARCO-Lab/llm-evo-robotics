#!/usr/bin/env python3
"""
测试可视化变化
"""

import sys
import os
sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/envs')
from reacher2d_env import Reacher2DEnv

def main():
    print("🎮 测试可视化变化")
    print("="*50)
    print("修改说明:")
    print("🟢 绿色大圆 = 目标点")  
    print("🔵 蓝色小圆 = 末端执行器")
    print("="*50)
    
    env = Reacher2DEnv(
        num_links=3,
        link_lengths=[60, 60, 60],
        render_mode="human",
        config_path='../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml'
    )
    
    obs = env.reset()
    
    print(f"✅ 位置信息:")
    print(f"   🎯 目标位置: {env.goal_pos}")
    print(f"   🤖 末端位置: {env._get_end_effector_position()}")
    print(f"   📏 距离: {((env.goal_pos[0] - env._get_end_effector_position()[0])**2 + (env.goal_pos[1] - env._get_end_effector_position()[1])**2)**0.5:.1f}px")
    
    print("\n🎮 渲染中...")
    print("应该看到:")
    print(f"   🟢 绿色大圆在 ({env.goal_pos[0]}, {env.goal_pos[1]})")
    print(f"   🔵 蓝色小圆在 ({env._get_end_effector_position()[0]:.0f}, {env._get_end_effector_position()[1]:.0f})")
    
    # 渲染几帧
    for i in range(5):
        env.render()
        print(f"渲染帧 {i+1}/5")
    
    print("\n现在修改goal位置测试...")
    
    # 修改goal位置到一个明显不同的地方
    new_goal = [200, 400]
    env.goal_pos = new_goal
    
    print(f"新目标位置: {new_goal}")
    
    for i in range(5):
        env.render()
        print(f"新位置渲染帧 {i+1}/5")

if __name__ == "__main__":
    main()
