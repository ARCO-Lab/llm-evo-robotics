#!/usr/bin/env python3
"""
修复版碰撞检测调试脚本 - 使用PyMunk 7.1.0正确API
"""

import pymunk
import pymunk.pygame_util
import pygame
import numpy as np
import math
import sys
import os

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")
sys.path.append(base_dir)

def debug_collision_detection_fixed():
    print("🔧 开始碰撞检测修复调试...")
    print(f"🔍 PyMunk版本: {pymunk.version}")
    
    # 创建物理空间
    space = pymunk.Space()
    space.gravity = (0.0, 981.0)
    
    # 初始化渲染
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("修复版碰撞检测调试 - 按ESC退出")
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    # 🎯 创建障碍物和机器人
    OBSTACLE_COLLISION_TYPE = 100
    ROBOT_COLLISION_TYPE = 1
    
    print(f"📍 障碍物碰撞类型: {OBSTACLE_COLLISION_TYPE}")
    print(f"📍 机器人碰撞类型: {ROBOT_COLLISION_TYPE}")
    
    # 创建障碍物 (中央竖直线)
    obstacle_shape = pymunk.Segment(space.static_body, (400, 200), (400, 400), 8.0)
    obstacle_shape.collision_type = OBSTACLE_COLLISION_TYPE
    obstacle_shape.friction = 1.0
    obstacle_shape.color = (255, 0, 0, 255)  # 红色
    space.add(obstacle_shape)
    print(f"✅ 创建障碍物: collision_type = {obstacle_shape.collision_type}")
    
    # 创建机器人 (圆形)
    robot_mass = 10
    robot_radius = 20
    robot_moment = pymunk.moment_for_circle(robot_mass, 0, robot_radius)
    robot_body = pymunk.Body(robot_mass, robot_moment)
    robot_body.position = 300, 300  # 障碍物左侧
    
    robot_shape = pymunk.Circle(robot_body, robot_radius)
    robot_shape.collision_type = ROBOT_COLLISION_TYPE
    robot_shape.friction = 1.0
    robot_shape.color = (0, 255, 0, 255)  # 绿色
    space.add(robot_body, robot_shape)
    print(f"✅ 创建机器人: collision_type = {robot_shape.collision_type}")
    
    # 🎯 使用PyMunk 7.1.0的正确API设置碰撞检测
    collision_count = 0
    
    def collision_callback(space, arbiter):
        nonlocal collision_count
        collision_count += 1
        shape_a, shape_b = arbiter.shapes
        print(f"🚨 检测到碰撞! 计数: {collision_count}")
        print(f"   碰撞对象: {shape_a.collision_type} vs {shape_b.collision_type}")
        return True  # 返回True允许碰撞继续
    
    print("\n🧪 使用PyMunk 7.1.0 on_collision API...")
    try:
        # 使用on_collision方法 - 这是PyMunk 7.x的正确方式
        space.on_collision = collision_callback
        print("✅ on_collision 设置成功")
    except Exception as e:
        print(f"❌ on_collision 设置失败: {e}")
    
    # 🎯 强制机器人移动撞击障碍物
    print("\n🚀 开始强制碰撞测试...")
    running = True
    step_count = 0
    move_right = True
    
    while running and step_count < 300:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # 强制机器人左右移动穿过障碍物
        if move_right:
            robot_body.velocity = (150, 0)  # 向右移动
            if robot_body.position.x > 500:
                move_right = False
        else:
            robot_body.velocity = (-150, 0)  # 向左移动
            if robot_body.position.x < 200:
                move_right = True
        
        # 物理步进
        space.step(1/60.0)
        
        # 渲染
        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)
        
        # 显示信息
        font = pygame.font.Font(None, 36)
        text1 = font.render(f"碰撞计数: {collision_count}", True, (0, 0, 0))
        text2 = font.render(f"机器人位置: {robot_body.position.x:.1f}", True, (0, 0, 0))
        text3 = font.render("ESC退出", True, (0, 0, 0))
        screen.blit(text1, (10, 10))
        screen.blit(text2, (10, 50))
        screen.blit(text3, (10, 90))
        
        pygame.display.flip()
        clock.tick(60)
        
        step_count += 1
        
        # 每20步检查一次
        if step_count % 20 == 0:
            print(f"步骤 {step_count}: 位置={robot_body.position.x:.1f}, 碰撞={collision_count}")
    
    print(f"\n📊 测试结果:")
    print(f"   总步数: {step_count}")
    print(f"   总碰撞数: {collision_count}")
    print(f"   pymunk版本: {pymunk.version}")
    
    if collision_count > 0:
        print("🎉 碰撞检测修复成功!")
        print("✅ 现在可以正确检测碰撞了!")
    else:
        print("❌ 碰撞检测仍然失效!")
        print("🔍 需要进一步调查...")
    
    pygame.quit()

if __name__ == "__main__":
    debug_collision_detection_fixed() 