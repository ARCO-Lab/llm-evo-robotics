#!/usr/bin/env python3
"""
测试基座固定效果
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

def test_base_fixed():
    """测试基座是否正确固定"""
    print("🔧 测试基座固定效果")
    print("=" * 40)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='WARNING'
    )
    
    env.reset()
    
    # 检查基座关节位置
    base_body = env.bodies[0]
    anchor_point = env.anchor_point
    
    print(f"🎯 基座锚点位置: {anchor_point}")
    print(f"🤖 基座关节初始位置: {base_body.position}")
    print(f"📏 距离锚点距离: {np.linalg.norm(np.array(base_body.position) - np.array(anchor_point)):.2f} px")
    
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    running = True
    step_count = 0
    max_distance = 0
    
    # 记录基座位置变化
    base_positions = []
    
    print("\n🚀 开始测试...")
    print("   应用随机大力测试基座稳定性")
    
    while running and step_count < 300:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
        
        # 应用随机大力测试基座稳定性
        if step_count < 200:
            # 前200步应用随机大力
            random_actions = np.random.uniform(-100, 100, 4)
        else:
            # 后100步应用零力，看基座是否回到原位
            random_actions = np.zeros(4)
        
        # 执行step
        obs, reward, done, info = env.step(random_actions)
        
        # 记录基座位置
        current_pos = base_body.position
        base_positions.append(current_pos)
        distance_from_anchor = np.linalg.norm(np.array(current_pos) - np.array(anchor_point))
        max_distance = max(max_distance, distance_from_anchor)
        
        # 渲染
        env.render()
        
        # 显示实时信息
        info_texts = [
            f"🔧 基座固定测试",
            f"步数: {step_count}/300",
            "",
            f"🎯 锚点位置: ({anchor_point[0]:.1f}, {anchor_point[1]:.1f})",
            f"🤖 基座位置: ({current_pos[0]:.1f}, {current_pos[1]:.1f})",
            f"📏 当前距离: {distance_from_anchor:.2f} px",
            f"📏 最大距离: {max_distance:.2f} px",
            "",
            f"🎮 当前动作: [{random_actions[0]:.1f}, {random_actions[1]:.1f}, {random_actions[2]:.1f}, {random_actions[3]:.1f}]",
            "",
            "🧪 测试阶段:",
            f"   {'随机大力测试 (0-200步)' if step_count < 200 else '零力恢复测试 (200-300步)'}",
            "",
            "✅ 预期结果:",
            "   基座应该始终保持在锚点附近",
            "   最大偏移应该很小 (< 5px)",
            "",
            "Q: 退出测试"
        ]
        
        # 显示信息面板
        info_surface = pygame.Surface((400, len(info_texts) * 22 + 20))
        info_surface.set_alpha(180)
        info_surface.fill((50, 50, 50))
        env.screen.blit(info_surface, (10, 10))
        
        for i, text in enumerate(info_texts):
            if text:
                color = (255, 255, 255)
                if "基座固定测试" in text:
                    color = (100, 200, 255)
                elif "当前距离" in text:
                    if distance_from_anchor < 2:
                        color = (100, 255, 100)
                    elif distance_from_anchor < 5:
                        color = (255, 255, 100)
                    else:
                        color = (255, 100, 100)
                elif "最大距离" in text:
                    if max_distance < 2:
                        color = (100, 255, 100)
                    elif max_distance < 5:
                        color = (255, 255, 100)
                    else:
                        color = (255, 100, 100)
                elif "随机大力测试" in text:
                    color = (255, 200, 100)
                elif "零力恢复测试" in text:
                    color = (100, 255, 200)
                
                surface = font.render(text, True, color)
                env.screen.blit(surface, (20, 20 + i * 22))
        
        pygame.display.flip()
        
        step_count += 1
        clock.tick(60)
    
    # 最终测试结果
    final_distance = np.linalg.norm(np.array(base_body.position) - np.array(anchor_point))
    
    print(f"\n🎯 基座固定测试结果:")
    print(f"   总测试步数: {step_count}")
    print(f"   最大偏移距离: {max_distance:.2f} px")
    print(f"   最终距离: {final_distance:.2f} px")
    
    # 判断测试结果
    if max_distance < 5 and final_distance < 2:
        print(f"\n🎉 测试通过! 基座固定效果良好")
        print(f"   ✅ 最大偏移 < 5px")
        print(f"   ✅ 最终距离 < 2px")
        print(f"   ✅ 基座稳定固定在锚点")
    elif max_distance < 10:
        print(f"\n⚠️ 测试部分通过")
        print(f"   ⚠️ 最大偏移: {max_distance:.2f}px (可接受但不理想)")
        print(f"   💡 基座基本固定，但可能需要调整参数")
    else:
        print(f"\n❌ 测试失败")
        print(f"   ❌ 最大偏移过大: {max_distance:.2f}px")
        print(f"   🔧 需要检查基座锚点约束参数")
    
    env.close()

if __name__ == "__main__":
    test_base_fixed()

