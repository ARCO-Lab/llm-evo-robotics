#!/usr/bin/env python3
"""
调试碰撞检测问题的简单测试脚本
"""

import sys
import os
import numpy as np
import pygame

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from envs.reacher2d_env import Reacher2DEnv

def debug_collision_setup():
    """调试碰撞检测设置"""
    print("🔍 调试碰撞检测设置...")
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='DEBUG'  # 使用DEBUG级别查看更多信息
    )
    
    env.reset()
    
    # 检查环境设置
    print(f"✅ 环境创建成功")
    print(f"   Links数量: {env.num_links}")
    print(f"   Bodies数量: {len(env.bodies)}")
    print(f"   Shapes数量: {len([shape for body in env.bodies for shape in body.shapes])}")
    
    # 检查collision_type设置
    print(f"\n🔍 检查collision_type设置:")
    for i, body in enumerate(env.bodies):
        for j, shape in enumerate(body.shapes):
            print(f"   Body{i} Shape{j}: collision_type = {shape.collision_type}")
    
    # 检查PyMunk版本和API
    import pymunk
    print(f"\n🔍 PyMunk信息:")
    print(f"   版本: {pymunk.version}")
    print(f"   Space方法: {[attr for attr in dir(env.space) if 'collision' in attr.lower()]}")
    
    # 手动设置简单的碰撞检测
    collision_count = 0
    
    def simple_collision_handler(arbiter, space, data):
        nonlocal collision_count
        collision_count += 1
        shape_a, shape_b = arbiter.shapes
        print(f"🚨 检测到碰撞! 计数: {collision_count}")
        print(f"   碰撞对象: collision_type {shape_a.collision_type} vs {shape_b.collision_type}")
        return True
    
    # 尝试设置碰撞检测
    print(f"\n🔧 尝试设置碰撞检测...")
    
    # 方法1: 使用add_collision_handler
    try:
        if hasattr(env.space, 'add_collision_handler'):
            # 为所有Link对设置碰撞检测
            for i in range(env.num_links):
                for j in range(i + 1, env.num_links):
                    handler = env.space.add_collision_handler(i + 1, j + 1)
                    handler.begin = simple_collision_handler
                    print(f"   ✅ 设置Link{i}-Link{j}碰撞检测 (types: {i+1} vs {j+1})")
        else:
            print("   ❌ add_collision_handler方法不存在")
    except Exception as e:
        print(f"   ❌ add_collision_handler失败: {e}")
    
    # 方法2: 使用on_collision
    try:
        if hasattr(env.space, 'on_collision'):
            print("   🔧 尝试on_collision方法...")
            # 只测试一对
            env.space.on_collision(
                collision_type_a=1,
                collision_type_b=2,
                begin=simple_collision_handler
            )
            print("   ✅ on_collision设置成功 (Link0 vs Link1)")
        else:
            print("   ❌ on_collision方法不存在")
    except Exception as e:
        print(f"   ❌ on_collision失败: {e}")
    
    # 简单测试
    print(f"\n🎮 开始简单测试 (按空格键退出)...")
    
    pygame.init()
    clock = pygame.time.Clock()
    
    running = True
    step_count = 0
    
    while running and step_count < 1000:
        # 处理事件
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = False
        
        # 施加大力让Link折叠
        actions = np.array([50, -50, 50, -50])  # 强制折叠
        
        obs, reward, done, info = env.step(actions)
        env.render()
        
        step_count += 1
        
        # 每100步输出统计
        if step_count % 100 == 0:
            print(f"   步数: {step_count}, 碰撞计数: {collision_count}")
        
        if done:
            env.reset()
        
        clock.tick(60)
    
    print(f"\n📊 测试结果:")
    print(f"   总步数: {step_count}")
    print(f"   总碰撞数: {collision_count}")
    print(f"   碰撞率: {collision_count/step_count*100:.2f}%")
    
    if collision_count == 0:
        print("❌ 没有检测到任何碰撞，碰撞检测系统可能有问题")
    else:
        print("✅ 碰撞检测系统工作正常")
    
    env.close()

if __name__ == "__main__":
    debug_collision_setup()

