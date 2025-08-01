#!/usr/bin/env python3
"""
随机动作演示程序 - 展示Reacher2D机器人的运动效果
"""

from reacher2d_env import Reacher2DEnv
import numpy as np
import time
import pygame

def demo_random_actions():
    print("🤖 启动Reacher2D随机动作演示...")
    
    # 创建环境
    env = Reacher2DEnv(num_links=5, link_lengths=[80, 50, 30, 20, 10], render_mode="human")
    
    obs, info = env.reset()
    print("✅ 环境初始化完成")
    print("🎮 控制说明：按ESC或关闭窗口退出")
    print("🎯 红色圆点是目标位置")
    print()
    
    running = True
    step_count = 0
    demo_mode = 0  # 0: 随机, 1: 波浪, 2: 协调摆动, 3: 追踪目标
    mode_duration = 300  # 每个模式持续的步数
    
    try:
        while running:
            # 处理pygame事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # 空格键切换模式
                        demo_mode = (demo_mode + 1) % 4
                        print(f"切换到模式 {demo_mode}")
            
            # 根据当前模式生成动作
            if demo_mode == 0:  # 完全随机模式
                actions = np.random.uniform(-20, 20, size=env.num_links)  # 增加随机动作幅度
                if step_count % 60 == 0:
                    print("🎲 模式0: 完全随机动作")
                    
            elif demo_mode == 1:  # 波浪模式
                t = step_count * 0.05
                actions = np.array([
                    15.0 * np.sin(t + 0),      # 增加幅度
                    12.0 * np.sin(t + 0.5),
                    8.0 * np.sin(t + 1.0),
                    5.0 * np.sin(t + 1.5),
                    3.0 * np.sin(t + 2.0)
                ])
                if step_count % 60 == 0:
                    print("🌊 模式1: 波浪式运动")
                    
            elif demo_mode == 2:  # 协调摆动模式
                t = step_count * 0.03
                actions = np.array([
                    20.0 * np.sin(t),          # 增加幅度
                    15.0 * np.cos(t * 1.2),
                    10.0 * np.sin(t * 0.8),
                    6.0 * np.cos(t * 1.5),
                    3.0 * np.sin(t * 2.0)
                ])
                if step_count % 60 == 0:
                    print("🔄 模式2: 协调摆动")
                    
            elif demo_mode == 3:  # 尝试追踪目标模式
                # 简单的"追踪"逻辑（不是真正的控制算法）
                end_pos = env._get_end_effector_position()
                goal_pos = env.goal_pos
                
                # 计算误差
                error_x = goal_pos[0] - end_pos[0]
                error_y = goal_pos[1] - end_pos[1]
                
                # 增加控制增益
                base_action = 0.1 * error_x  # 增加比例增益
                actions = np.array([
                    base_action + 5.0 * np.sin(step_count * 0.02),   # 增加幅度
                    0.8 * base_action + 4.0 * np.cos(step_count * 0.03),
                    0.6 * base_action + 3.0 * np.sin(step_count * 0.025),
                    0.4 * base_action + 2.0 * np.cos(step_count * 0.035),
                    0.2 * base_action + 1.0 * np.sin(step_count * 0.04)
                ])
                actions = np.clip(actions, -25, 25)  # 增加限制范围
                
                if step_count % 60 == 0:
                    print(f"🎯 模式3: 追踪目标 (误差: x={error_x:.1f}, y={error_y:.1f})")
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(actions)
            env.render()
            
            # 打印状态信息
            if step_count % 120 == 0:  # 每2秒打印一次
                end_pos = env._get_end_effector_position()
                print(f"步数: {step_count}, 末端位置: ({end_pos[0]:.1f}, {end_pos[1]:.1f}), 奖励: {reward:.3f}")
            
            step_count += 1
            
            # 自动切换模式
            if step_count % mode_duration == 0:
                demo_mode = (demo_mode + 1) % 4
                print(f"\n🔄 自动切换到模式 {demo_mode}")
            
            time.sleep(0.016)  # 60fps
            
    except KeyboardInterrupt:
        print("\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        env.close()
        print("👋 演示结束，感谢观看！")

if __name__ == "__main__":
    demo_random_actions() 