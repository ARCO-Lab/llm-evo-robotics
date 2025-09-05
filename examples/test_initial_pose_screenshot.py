#!/usr/bin/env python3
"""
截图版本的初始姿态测试脚本
专门用于截取前5个步骤的截图
"""

import sys
import os
import numpy as np
import time

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from reacher2d_env import Reacher2DEnv

def test_initial_pose_with_screenshots():
    """测试机器人初始姿态并截图"""
    print("=" * 60)
    print("🖼️ 测试机器人初始姿态 - 截图模式")
    print("=" * 60)
    
    # 创建环境 - 启用截图模式
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 80, 80, 60],
        render_mode='human',
        config_path="/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='INFO'
    )
    
    # 启用截图模式
    env.screenshot_mode = True
    env.screenshot_dir = 'screenshots/test_initial_pose'
    
    print(f"📍 基座位置: {env.anchor_point}")
    print(f"🎯 目标位置: {env.goal_pos}")
    
    # 重置环境
    print(f"\n🔄 重置环境并截取前5步")
    obs = env.reset()
    
    print(f"📐 初始关节角度:")
    for i, angle in enumerate(env.joint_angles):
        degrees = np.degrees(angle)
        print(f"   关节{i}: {angle:.4f} 弧度 = {degrees:.2f}°")
    
    # 渲染并截图初始状态（step 0）
    env.render()
    time.sleep(0.5)
    
    # 执行前5个步骤
    for step in range(1, 6):
        print(f"\n📸 执行步骤 {step}")
        
        # 使用零动作来保持机器人相对静止，便于观察初始姿态
        action = np.zeros(4)  # 零动作
        
        obs, reward, done, info = env.step(action)
        
        print(f"   奖励: {reward:.2f}")
        print(f"   结束: {done}")
        if 'end_effector_pos' in info:
            end_pos = info['end_effector_pos']
            print(f"   末端位置: [{end_pos[0]:.1f}, {end_pos[1]:.1f}]")
        
        # 渲染并自动截图
        env.render()
        time.sleep(0.5)  # 等待截图保存
        
        if done:
            print(f"   Episode在步骤{step}结束")
            break
    
    print(f"\n✅ 截图完成，保存在: {env.screenshot_dir}")
    
    # 保持窗口打开一会儿让用户观察
    print("🖼️ 保持窗口打开3秒...")
    time.sleep(3)
    
    env.close()
    print("🔒 环境已关闭")

if __name__ == "__main__":
    test_initial_pose_with_screenshots()
