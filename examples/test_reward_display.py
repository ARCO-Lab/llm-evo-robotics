#!/usr/bin/env python3
"""
测试实时奖励和末端执行器位置显示
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

def test_reward_display():
    """测试奖励和位置的实时显示"""
    print("=" * 60)
    print("🎯 测试实时奖励和末端执行器位置显示")
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
    print(f"🎮 动作空间: {env.action_space}")
    print("\n🎯 显示说明:")
    print("  - 左侧: 基本信息 (步数、距离、末端位置、目标位置)")
    print("  - 右侧: 奖励信息 (总奖励及各组成部分)")
    print("  - 颜色含义: 绿色=正奖励, 红色=惩罚, 灰色=零")
    print("  - 末端执行器: 红色圆圈 (外圈红色, 内圈白色, 中心红点)")
    print("  - 目标: 绿色圆圈")
    print("\n🎮 控制:")
    print("  - WASD: 控制前两个关节")
    print("  - 方向键: 控制后两个关节")
    print("  - ESC/Q: 退出")
    print("  - R: 重置环境")
    
    # 重置环境
    obs = env.reset()
    
    print(f"\n🎮 开始交互式测试...")
    print(f"   初始奖励: {env.current_reward:.3f}")
    print(f"   奖励组成: {env.reward_components}")
    
    try:
        step_count = 0
        episode_reward = 0
        
        while True:
            # 渲染环境
            env.render()
            
            # 检查键盘输入
            import pygame
            action = np.zeros(env.action_space.shape[0])
            
            keys = pygame.key.get_pressed()
            
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n✅ 用户关闭窗口")
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        print("\n✅ 用户按ESC退出")
                        return
                    elif event.key == pygame.K_r:
                        print("\n🔄 重置环境...")
                        obs = env.reset()
                        step_count = 0
                        episode_reward = 0
                        print(f"   新目标位置: {env.goal_pos}")
                        continue
            
            # WASD控制前两个关节
            if keys[pygame.K_w]:
                action[0] += 50  # 第一个关节逆时针
            if keys[pygame.K_s]:
                action[0] -= 50  # 第一个关节顺时针
            if keys[pygame.K_a]:
                action[1] += 50  # 第二个关节逆时针
            if keys[pygame.K_d]:
                action[1] -= 50  # 第二个关节顺时针
            
            # 方向键控制后两个关节
            if keys[pygame.K_UP]:
                action[2] += 50  # 第三个关节
            if keys[pygame.K_DOWN]:
                action[2] -= 50
            if keys[pygame.K_LEFT]:
                action[3] += 50  # 第四个关节
            if keys[pygame.K_RIGHT]:
                action[3] -= 50
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            step_count += 1
            episode_reward += reward
            
            # 每10步输出一次详细信息
            if step_count % 10 == 0:
                end_pos = env._get_end_effector_position()
                distance = np.linalg.norm(end_pos - env.goal_pos)
                print(f"\n📊 Step {step_count}:")
                print(f"   末端位置: [{end_pos[0]:.1f}, {end_pos[1]:.1f}]")
                print(f"   距离目标: {distance:.1f} pixels")
                print(f"   当前奖励: {reward:.3f}")
                print(f"   累计奖励: {episode_reward:.3f}")
                print(f"   奖励组成: Distance={env.reward_components['distance_reward']:.3f}, "
                      f"Reach={env.reward_components['reach_reward']:.3f}, "
                      f"Collision={env.reward_components['collision_penalty']:.3f}, "
                      f"Control={env.reward_components['control_penalty']:.3f}")
                
                # 检查是否到达目标
                if distance < 35.0:
                    print(f"🎉 接近目标! 距离: {distance:.1f} < 35.0")
                if distance < 20.0:
                    print(f"🏆 到达目标! 获得到达奖励: +10.0")
            
            # 检查是否完成
            if done:
                print(f"\n🏁 Episode 完成!")
                print(f"   总步数: {step_count}")
                print(f"   总奖励: {episode_reward:.3f}")
                print(f"   平均奖励: {episode_reward/step_count:.3f}")
                
                print(f"\n⏳ 3秒后自动重置...")
                time.sleep(3)
                obs = env.reset()
                step_count = 0
                episode_reward = 0
            
            # 控制帧率
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n✅ 用户中断测试")
    
    finally:
        env.close()
        print("🔒 环境已关闭")

if __name__ == "__main__":
    test_reward_display()
