#!/usr/bin/env python3
"""
碰撞检测深度调试脚本
强制机器人与障碍物发生碰撞，诊断回调函数问题
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

def debug_collision_detection():
    print("🔧 开始碰撞检测深度调试...")
    
    # 创建物理空间
    space = pymunk.Space()
    space.gravity = (0.0, 981.0)
    
    # 初始化渲染
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("碰撞检测调试 - 按ESC退出")
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    # 🎯 1. 创建简单的障碍物
    OBSTACLE_COLLISION_TYPE = 100
    ROBOT_COLLISION_TYPE = 1
    
    print(f"📍 障碍物碰撞类型: {OBSTACLE_COLLISION_TYPE}")
    print(f"📍 机器人碰撞类型: {ROBOT_COLLISION_TYPE}")
    
    # 创建一个简单的障碍物 (中央竖直线)
    obstacle_shape = pymunk.Segment(space.static_body, (400, 200), (400, 400), 5.0)
    obstacle_shape.collision_type = OBSTACLE_COLLISION_TYPE
    obstacle_shape.friction = 1.0
    obstacle_shape.color = (255, 0, 0, 255)  # 红色
    space.add(obstacle_shape)
    print(f"✅ 创建障碍物: collision_type = {obstacle_shape.collision_type}")
    
    # 🎯 2. 创建简单的机器人 (单个圆形)
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
    
    # 🎯 3. 设置碰撞检测回调 - 多种方法测试
    collision_count = 0
    
    def collision_handler_method1(arbiter, space, data):
        nonlocal collision_count
        collision_count += 1
        print(f"🚨 方法1检测到碰撞! 计数: {collision_count}")
        print(f"   碰撞对象: {arbiter.shapes[0].collision_type} vs {arbiter.shapes[1].collision_type}")
        return True
    
    # 测试方法1: add_collision_handler
    print("\n🧪 测试方法1: add_collision_handler")
    try:
        if hasattr(space, 'add_collision_handler'):
            handler1 = space.add_collision_handler(ROBOT_COLLISION_TYPE, OBSTACLE_COLLISION_TYPE)
            handler1.pre_solve = collision_handler_method1
            print("✅ add_collision_handler 设置成功")
        else:
            print("❌ space没有add_collision_handler方法")
    except Exception as e:
        print(f"❌ add_collision_handler 失败: {e}")
    
    # 测试方法2: add_wildcard_collision_handler
    print("\n🧪 测试方法2: add_wildcard_collision_handler")
    def wildcard_handler(arbiter, space, data):
        nonlocal collision_count
        collision_count += 1
        print(f"🚨 通配符检测到碰撞! 计数: {collision_count}")
        return True
    
    try:
        if hasattr(space, 'add_wildcard_collision_handler'):
            handler2 = space.add_wildcard_collision_handler(ROBOT_COLLISION_TYPE)
            handler2.pre_solve = wildcard_handler
            print("✅ add_wildcard_collision_handler 设置成功")
        else:
            print("❌ space没有add_wildcard_collision_handler方法")
    except Exception as e:
        print(f"❌ add_wildcard_collision_handler 失败: {e}")
    
    # 🎯 4. 强制机器人移动撞击障碍物
    print("\n🚀 开始强制碰撞测试...")
    running = True
    step_count = 0
    move_right = True
    
    while running and step_count < 500:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # 强制机器人左右移动穿过障碍物
        if move_right:
            robot_body.velocity = (200, 0)  # 向右移动
            if robot_body.position.x > 500:
                move_right = False
        else:
            robot_body.velocity = (-200, 0)  # 向左移动
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
        
        # 每10步检查一次
        if step_count % 10 == 0:
            print(f"步骤 {step_count}: 位置={robot_body.position.x:.1f}, 碰撞={collision_count}")
    
    print(f"\n📊 测试结果:")
    print(f"   总步数: {step_count}")
    print(f"   总碰撞数: {collision_count}")
    print(f"   pymunk版本: {pymunk.version}")
    
    if collision_count > 0:
        print("✅ 碰撞检测工作正常!")
    else:
        print("❌ 碰撞检测完全失效!")
        print("🔍 可能原因:")
        print("   1. pymunk版本不兼容")
        print("   2. collision_type设置错误")
        print("   3. 回调函数注册失败")
        print("   4. 物理步进问题")
    
    pygame.quit()

if __name__ == "__main__":
    debug_collision_detection() 