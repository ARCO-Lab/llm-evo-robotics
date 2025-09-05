#!/usr/bin/env python3
"""
自动截图版本的初始姿态测试脚本
与enhanced_train.py进行对比
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

def test_initial_pose_auto_screenshot():
    """测试机器人初始姿态并自动截图前5步"""
    print("=" * 60)
    print("🖼️ 测试机器人初始姿态 - 自动截图前5步")
    print("=" * 60)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 80, 80, 60],
        render_mode='human',
        config_path="/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='INFO'
    )
    
    print(f"📍 基座位置: {env.anchor_point}")
    print(f"🎯 目标位置: {env.goal_pos}")
    
    # 重置环境
    print(f"\n🔄 重置环境")
    obs = env.reset()
    
    print(f"📐 初始关节角度:")
    for i, angle in enumerate(env.joint_angles):
        degrees = np.degrees(angle)
        print(f"   关节{i}: {angle:.4f} 弧度 = {degrees:.2f}°")
    
    # 临时修改截图目录以区分
    original_render = env.render
    def custom_render(mode='human'):
        result = original_render(mode)
        # 自定义截图逻辑
        if env.step_count <= 5:
            import os
            screenshot_dir = 'screenshots/test_initial_pose_auto'
            os.makedirs(screenshot_dir, exist_ok=True)
            filename = f'{screenshot_dir}/step_{env.step_count:02d}.png'
            
            import pygame
            pygame.image.save(env.screen, filename)
            print(f"🖼️ [Step {env.step_count}] 自动保存截图: {filename}")
            
            # 显示详细信息
            end_pos = env._get_end_effector_position()
            print(f"    📍 末端位置: [{end_pos[0]:.1f}, {end_pos[1]:.1f}]")
            print(f"    📐 关节角度: [{', '.join([f'{a:.3f}' for a in env.joint_angles])}]")
            print(f"    🎯 目标位置: [{env.goal_pos[0]:.1f}, {env.goal_pos[1]:.1f}]")
        return result
    
    env.render = custom_render
    
    # 渲染初始状态（step 0）
    print(f"\n📸 渲染初始状态")
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
        
        # 渲染并自动截图
        env.render()
        time.sleep(0.5)
        
        if done:
            print(f"   Episode在步骤{step}结束")
            break
    
    print(f"\n✅ 截图完成，保存在: screenshots/test_initial_pose_auto")
    
    # 保持窗口打开一会儿
    print("🖼️ 保持窗口打开3秒...")
    time.sleep(3)
    
    env.close()
    print("🔒 环境已关闭")

if __name__ == "__main__":
    test_initial_pose_auto_screenshot()
