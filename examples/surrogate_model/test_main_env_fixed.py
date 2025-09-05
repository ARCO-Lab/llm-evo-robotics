#!/usr/bin/env python3
"""
测试修复后的主环境文件
验证基座关节穿透问题是否已解决
"""

import sys
import os
import numpy as np
import pygame
import time

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from envs.reacher2d_env import Reacher2DEnv

def test_main_env_fixed():
    """测试修复后的主环境"""
    print("🛠️ 测试修复后的主环境")
    print("=" * 50)
    
    # 创建修复后的主环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='INFO'  # 使用INFO级别查看碰撞日志
    )
    
    env.reset()
    
    print(f"\n🎮 开始测试:")
    print("  自动执行D+W组合动作")
    print("  期望: 基座关节能够正确与障碍物碰撞")
    print("  Q: 退出")
    
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    running = True
    step_count = 0
    
    while running and step_count < 3000:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        # 自动执行强力测试动作 - 专门让基座关节接触障碍物
        actions = np.array([100, -80, 0, 0])  # 基座关节大力转动
        
        # 执行step
        obs, reward, done, info = env.step(actions)
        
        # 渲染
        env.render()
        
        # 显示统计信息
        base_pos = env.bodies[0].position
        base_angle = env.bodies[0].angle
        
        # 检查碰撞统计
        base_collision_count = getattr(env, 'base_collision_count', 0)
        collision_count = getattr(env, 'collision_count', 0)
        
        info_texts = [
            f"步数: {step_count}",
            f"修复版主环境测试",
            "",
            "🤖 基座关节状态:",
            f"位置: ({base_pos[0]:.0f}, {base_pos[1]:.0f})",
            f"角度: {np.degrees(base_angle):.0f}°",
            "",
            "🚨 碰撞统计:",
            f"基座专用碰撞: {base_collision_count}",
            f"总碰撞: {collision_count}",
            "",
            f"🔍 修复状态:",
            f"{'✅ 成功!' if base_collision_count > 0 else '❌ 仍有问题'}",
            "",
            "Q: 退出"
        ]
        
        # 显示信息
        info_surface = pygame.Surface((300, len(info_texts) * 22 + 20))
        info_surface.set_alpha(180)
        info_surface.fill((50, 50, 50))
        env.screen.blit(info_surface, (10, 10))
        
        for i, text in enumerate(info_texts):
            if text:
                color = (255, 255, 255)
                if "基座关节状态" in text:
                    color = (100, 200, 255)
                elif "碰撞统计" in text:
                    color = (255, 200, 100)
                elif f"基座专用碰撞: {base_collision_count}" in text and base_collision_count > 0:
                    color = (100, 255, 100)  # 绿色表示成功
                elif "✅ 成功!" in text:
                    color = (100, 255, 100)
                elif "❌ 仍有问题" in text:
                    color = (255, 100, 100)
                
                surface = font.render(text, True, color)
                env.screen.blit(surface, (20, 20 + i * 22))
        
        pygame.display.flip()
        
        step_count += 1
        
        # 每200步输出统计
        if step_count % 200 == 0:
            print(f"\n📊 步数{step_count}统计:")
            print(f"   基座专用碰撞: {base_collision_count}")
            print(f"   总碰撞: {collision_count}")
            
            if base_collision_count > 0:
                print("✅ 主环境修复成功! 基座关节可以正确碰撞!")
                # 注释掉提前退出，继续完整的3000步测试
                # break
        
        if done:
            env.reset()
        
        clock.tick(60)
    
    # 最终结果
    base_collision_count = getattr(env, 'base_collision_count', 0)
    collision_count = getattr(env, 'collision_count', 0)
    
    print(f"\n🎯 最终测试结果:")
    print("=" * 40)
    print(f"测试步数: {step_count}")
    print(f"基座专用碰撞: {base_collision_count}")
    print(f"总碰撞: {collision_count}")
    
    if base_collision_count > 0:
        print(f"\n🎉 主环境修复成功!")
        print("   基座关节现在可以正确与障碍物碰撞")
        print("   可以正常用于训练和测试")
    else:
        print(f"\n😔 主环境修复失败")
        print("   需要检查修复是否正确应用")
    
    env.close()

if __name__ == "__main__":
    test_main_env_fixed()
