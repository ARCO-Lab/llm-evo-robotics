#!/usr/bin/env python3
"""
调试渲染问题 - 检查物理位置和渲染位置的对应关系
"""

import sys
import os
import numpy as np
import pygame
import time
import pymunk
import pymunk.pygame_util

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from envs.reacher2d_env import Reacher2DEnv

def debug_rendering():
    """调试渲染问题"""
    print("🎨 调试渲染坐标系统")
    print("=" * 50)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='WARNING'
    )
    
    env.reset()
    
    print(f"🔍 物理空间检查:")
    print(f"   基座锚点: {env.anchor_point}")
    print(f"   屏幕尺寸: {env.screen.get_size()}")
    
    # 检查每个body的物理位置
    print(f"\n🤖 机器人body物理位置:")
    for i, body in enumerate(env.bodies):
        print(f"   Link{i}: 物理位置 = {body.position}")
    
    # 创建一个简单的测试：手动渲染基座关节
    pygame.init()
    screen = pygame.display.set_mode((1200, 1200))
    pygame.display.set_caption("渲染调试")
    clock = pygame.time.Clock()
    
    running = True
    frame_count = 0
    
    while running and frame_count < 100:  # 运行100帧
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 清空屏幕
        screen.fill((255, 255, 255))
        
        # 🎯 手动绘制基座锚点（红色十字）
        anchor_x, anchor_y = int(env.anchor_point[0]), int(env.anchor_point[1])
        pygame.draw.line(screen, (255, 0, 0), (anchor_x-20, anchor_y), (anchor_x+20, anchor_y), 3)
        pygame.draw.line(screen, (255, 0, 0), (anchor_x, anchor_y-20), (anchor_x, anchor_y+20), 3)
        
        # 🤖 手动绘制基座关节位置（蓝色圆圈）
        base_body = env.bodies[0]
        base_x, base_y = int(base_body.position[0]), int(base_body.position[1])
        pygame.draw.circle(screen, (0, 0, 255), (base_x, base_y), 15, 3)
        
        # 🔍 显示坐标信息
        font = pygame.font.Font(None, 36)
        anchor_text = f"Anchor: ({anchor_x}, {anchor_y})"
        base_text = f"Base: ({base_x}, {base_y})"
        
        anchor_surface = font.render(anchor_text, True, (255, 0, 0))
        base_surface = font.render(base_text, True, (0, 0, 255))
        
        screen.blit(anchor_surface, (50, 50))
        screen.blit(base_surface, (50, 100))
        
        # 🔧 使用PyMunk的debug_draw渲染整个物理世界
        draw_options = pymunk.pygame_util.DrawOptions(screen)
        env.space.debug_draw(draw_options)
        
        # 🎯 绘制目标点
        goal_pos = env.goal_pos.astype(int)
        pygame.draw.circle(screen, (0, 255, 0), goal_pos, 15)
        
        pygame.display.flip()
        clock.tick(10)  # 慢速播放便于观察
        
        frame_count += 1
        
        # 每10帧打印一次位置信息
        if frame_count % 10 == 0:
            print(f"帧 {frame_count}: 基座物理位置 = {base_body.position}")
            
            # 检查是否有位置变化
            if abs(base_body.position[0] - env.anchor_point[0]) > 0.1 or abs(base_body.position[1] - env.anchor_point[1]) > 0.1:
                print(f"⚠️  警告：基座位置偏离锚点！")
    
    pygame.quit()
    env.close()
    
    print(f"\n✅ 渲染调试完成")
    print(f"   如果你看到红色十字和蓝色圆圈重合在(450,620)位置，说明物理和渲染是一致的")
    print(f"   如果蓝色圆圈在屏幕下方，说明存在渲染问题")

if __name__ == "__main__":
    debug_rendering()

