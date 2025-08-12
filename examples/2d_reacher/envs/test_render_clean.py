#!/usr/bin/env python3
"""
渲染模式对比测试脚本
按键说明：
- 空格键：切换渲染模式
- ESC/Q：退出
"""

from reacher2d_env import Reacher2DEnv
import pygame
import numpy as np

def main():
    # 创建环境
    env = Reacher2DEnv(
        num_links=5,
        link_lengths=[80, 50, 30, 20, 50],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml"
    )
    
    print("🎯 渲染模式对比测试")
    print("按键说明：")
    print("- 空格键：切换渲染模式")
    print("- ESC/Q：退出")
    print("- 当前模式：调试模式（显示所有约束）")
    print("\n🔍 观察要点：")
    print("- 调试模式：每个关节位置显示多个约束符号")
    print("- 清洁模式：每个关节只显示一个连接点")
    
    obs = env.reset()
    clean_mode = False  # 开始时使用调试模式
    running = True
    step = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # 切换渲染模式
                    clean_mode = not clean_mode
                    if clean_mode:
                        print("✅ 切换到清洁模式（只显示机器人结构）")
                        print("   现在每个关节只显示一个连接点")
                    else:
                        print("🔧 切换到调试模式（显示所有约束）")
                        print("   现在显示PivotJoint、Motor、RotaryLimit等约束")
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
        
        # 使用小幅度动作让机器人慢慢移动
        actions = np.array([0.2, 0.1, -0.1, 0.15, -0.05])  # 缓慢运动
        obs, reward, done, info = env.step(actions)
        
        # 使用选定的渲染模式
        env.render(clean_mode=clean_mode)
        
        step += 1
        
        # 每50步显示一次状态
        if step % 50 == 0:
            mode_str = "清洁模式" if clean_mode else "调试模式"
            print(f"步骤 {step} - 当前渲染模式: {mode_str}")
    
    env.close()
    print("测试完成")

if __name__ == "__main__":
    main() 