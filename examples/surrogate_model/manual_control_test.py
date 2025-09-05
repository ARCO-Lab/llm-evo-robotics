#!/usr/bin/env python3
"""
手动控制测试修复后的基座关节
用键盘控制机器人，验证基座关节碰撞修复效果
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

from envs.reacher2d_env_fixed import Reacher2DEnv

def manual_control_test():
    """手动控制测试"""
    print("🎮 手动控制测试修复后的基座关节")
    print("=" * 50)
    
    # 创建修复后的主环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='INFO'  # 显示碰撞日志
    )
    
    env.reset()
    
    print(f"\n🎮 手动控制说明:")
    print("  WASD: 控制各个关节")
    print("  W: 基座关节顺时针 ⟲")
    print("  S: 基座关节逆时针 ⟳ - 重点测试这个！")
    print("  A/D: 其他关节")
    print("  R: 重置机器人")
    print("  Q/ESC: 退出")
    print("  期望: 基座关节能够正确与障碍物碰撞")
    
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    running = True
    step_count = 0
    last_collision_count = 0
    
    # 控制状态
    keys_pressed = {
        'w': False,
        's': False, 
        'a': False,
        'd': False
    }
    
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    print("🔄 机器人已重置")
                elif event.key == pygame.K_w:
                    keys_pressed['w'] = True
                elif event.key == pygame.K_s:
                    keys_pressed['s'] = True
                elif event.key == pygame.K_a:
                    keys_pressed['a'] = True
                elif event.key == pygame.K_d:
                    keys_pressed['d'] = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    keys_pressed['w'] = False
                elif event.key == pygame.K_s:
                    keys_pressed['s'] = False
                elif event.key == pygame.K_a:
                    keys_pressed['a'] = False
                elif event.key == pygame.K_d:
                    keys_pressed['d'] = False
        
        # 根据按键生成动作
        actions = np.array([0.0, 0.0, 0.0, 0.0])
        
        if keys_pressed['w']:
            actions[0] = 50   # 🚀 增加基座关节力度：从20提升到50
        elif keys_pressed['s']:
            actions[0] = -50  # 🚀 增加基座关节力度：从-20提升到-50
            
        if keys_pressed['a']:
            actions[1] = -30  # 🚀 增加第二关节力度：从-15提升到-30
            actions[2] = -20  # 🚀 增加第三关节力度：从-10提升到-20
        elif keys_pressed['d']:
            actions[1] = 30   # 🚀 增加第二关节力度：从15提升到30
            actions[2] = 20   # 🚀 增加第三关节力度：从10提升到20
        
        # 执行step
        obs, reward, done, info = env.step(actions)
        
        # 渲染
        env.render()
        
        # 获取碰撞统计
        base_collision_count = getattr(env, 'base_collision_count', 0)
        collision_count = getattr(env, 'collision_count', 0)
        
        # 检测新碰撞
        if base_collision_count > last_collision_count:
            print(f"🎯 基座关节碰撞! 总计: {base_collision_count}")
            last_collision_count = base_collision_count
        
        # 获取基座关节状态
        base_pos = env.bodies[0].position
        base_angle = env.bodies[0].angle
        base_vel = env.bodies[0].velocity
        base_angular_vel = env.bodies[0].angular_velocity
        
        # 显示实时信息
        info_texts = [
            f"步数: {step_count}",
            f"手动控制模式",
            "",
            "🎮 控制:",
            "W: 基座关节顺时针 ⟲",
            "S: 基座关节逆时针 ⟳ ⭐",
            "A/D: 其他关节",
            "R: 重置 | Q: 退出",
            "",
            "🤖 基座关节状态:",
            f"位置: ({base_pos[0]:.0f}, {base_pos[1]:.0f})",
            f"角度: {np.degrees(base_angle):.0f}°",
            f"速度: {np.linalg.norm(base_vel):.1f}",
            f"角速度: {np.degrees(base_angular_vel):.1f}°/s",
            "",
            "🚨 碰撞统计:",
            f"基座专用碰撞: {base_collision_count}",
            f"其他碰撞: {collision_count}",
            "",
            f"🔍 修复状态:",
            f"{'✅ 正常工作!' if base_collision_count > 0 else '⏳ 尝试撞击障碍物'}",
            "",
            "💡 提示: 用S键让基座关节",
            "逆时针撞击附近的障碍物!"
        ]
        
        # 显示信息面板
        info_surface = pygame.Surface((350, len(info_texts) * 22 + 20))
        info_surface.set_alpha(180)
        info_surface.fill((50, 50, 50))
        env.screen.blit(info_surface, (10, 10))
        
        for i, text in enumerate(info_texts):
            if text:
                color = (255, 255, 255)
                if "控制" in text:
                    color = (100, 200, 255)
                elif "基座关节状态" in text:
                    color = (255, 200, 100)
                elif "碰撞统计" in text:
                    color = (255, 150, 150)
                elif f"基座专用碰撞: {base_collision_count}" in text and base_collision_count > 0:
                    color = (100, 255, 100)  # 绿色表示成功
                elif "✅ 正常工作!" in text:
                    color = (100, 255, 100)
                elif "⏳ 尝试撞击障碍物" in text:
                    color = (255, 255, 100)
                elif "S: 基座关节逆时针 ⟳ ⭐" in text:
                    color = (255, 100, 255)  # 高亮重要控制
                elif "提示" in text or "撞击" in text:
                    color = (100, 255, 255)
                
                surface = font.render(text, True, color)
                env.screen.blit(surface, (20, 20 + i * 22))
        
        pygame.display.flip()
        
        step_count += 1
        
        # 每500步输出统计
        if step_count % 500 == 0:
            print(f"\n📊 步数{step_count}统计:")
            print(f"   基座专用碰撞: {base_collision_count}")
            print(f"   其他碰撞: {collision_count}")
            print(f"   基座位置: ({base_pos[0]:.0f}, {base_pos[1]:.0f})")
        
        if done:
            env.reset()
        
        clock.tick(60)
    
    # 最终结果
    base_collision_count = getattr(env, 'base_collision_count', 0)
    collision_count = getattr(env, 'collision_count', 0)
    
    print(f"\n🎯 手动控制测试结果:")
    print("=" * 40)
    print(f"测试步数: {step_count}")
    print(f"基座专用碰撞: {base_collision_count}")
    print(f"其他碰撞: {collision_count}")
    
    if base_collision_count > 0:
        print(f"\n🎉 手动测试确认修复成功!")
        print("   基座关节可以正确与障碍物碰撞")
        print("   虚拟固定基座Body方案有效")
    else:
        print(f"\n🤔 未检测到基座关节碰撞")
        print("   请尝试用W/S键让基座关节撞击障碍物")
    
    env.close()

if __name__ == "__main__":
    manual_control_test()
