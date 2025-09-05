#!/usr/bin/env python3
"""
专门测试基座关节穿透障碍物的问题
- 检查基座关节（Link0）的collision_type设置
- 验证基座关节与障碍物的碰撞检测
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

def test_base_joint_penetration():
    """测试基座关节穿透障碍物的问题"""
    print("🔍 测试基座关节穿透障碍物问题")
    print("=" * 50)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='DEBUG'
    )
    
    env.reset()
    
    # 详细检查collision_type设置
    print("\n🔍 检查碰撞类型设置:")
    print("=" * 30)
    
    # 检查机器人Link的collision_type
    print("🤖 机器人Link碰撞类型:")
    for i, body in enumerate(env.bodies):
        for j, shape in enumerate(body.shapes):
            print(f"   Link{i} Shape{j}: collision_type = {shape.collision_type}")
    
    # 检查障碍物的collision_type
    print("\n🚧 障碍物碰撞类型:")
    for i, obstacle in enumerate(env.obstacles):
        print(f"   Obstacle{i}: collision_type = {obstacle.collision_type}")
    
    # 检查碰撞处理器设置
    print(f"\n🔧 碰撞处理器检查:")
    OBSTACLE_COLLISION_TYPE = 100
    
    # 创建碰撞统计
    collision_stats = {
        'link0_obstacle': 0,
        'link1_obstacle': 0,
        'link2_obstacle': 0,
        'link3_obstacle': 0,
        'total_obstacle': 0
    }
    
    # 手动设置基座关节与障碍物的碰撞检测
    def base_joint_collision_handler(arbiter, space, data):
        collision_stats['link0_obstacle'] += 1
        collision_stats['total_obstacle'] += 1
        shape_a, shape_b = arbiter.shapes
        print(f"🚨 基座关节碰撞障碍物! collision_type: {shape_a.collision_type} vs {shape_b.collision_type}")
        return True
    
    def other_link_collision_handler(link_id):
        def handler(arbiter, space, data):
            collision_stats[f'link{link_id}_obstacle'] += 1
            collision_stats['total_obstacle'] += 1
            shape_a, shape_b = arbiter.shapes
            print(f"🚨 Link{link_id}碰撞障碍物! collision_type: {shape_a.collision_type} vs {shape_b.collision_type}")
            return True
        return handler
    
    # 为每个Link设置与障碍物的碰撞检测
    try:
        # 基座关节 (collision_type = 1)
        env.space.on_collision(
            collision_type_a=1,  # Link0
            collision_type_b=OBSTACLE_COLLISION_TYPE,
            begin=base_joint_collision_handler
        )
        print("✅ 设置基座关节-障碍物碰撞检测")
        
        # 其他Link
        for i in range(1, env.num_links):
            env.space.on_collision(
                collision_type_a=i + 1,  # Link1,2,3...
                collision_type_b=OBSTACLE_COLLISION_TYPE,
                begin=other_link_collision_handler(i)
            )
            print(f"✅ 设置Link{i}-障碍物碰撞检测")
            
    except Exception as e:
        print(f"❌ 设置碰撞检测失败: {e}")
    
    print(f"\n🎮 开始测试:")
    print("  W/S: 控制基座关节 (Link0)")
    print("  A/D: 控制第二关节 (Link1)")  
    print("  1/2: 控制第三关节 (Link2)")
    print("  3/4: 控制第四关节 (Link3)")
    print("  Space: 自动测试基座关节穿透")
    print("  Q: 退出")
    
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    running = True
    step_count = 0
    auto_test = False
    penetration_test_counter = 0
    
    while running and step_count < 2000:
        # 处理事件
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    auto_test = not auto_test
                    print(f"🔄 {'启用' if auto_test else '禁用'}基座关节穿透测试")
        
        # 生成动作
        if auto_test:
            # 专门测试基座关节的大幅度旋转
            actions = np.array([100, 0, 0, 0])  # 只控制基座关节
            penetration_test_counter += 1
            if penetration_test_counter > 100:  # 每100步改变方向
                actions[0] = -100
                if penetration_test_counter > 200:
                    penetration_test_counter = 0
        else:
            # 手动控制
            actions = np.zeros(4)
            if keys[pygame.K_w]:
                actions[0] = 80  # 基座关节正向
            if keys[pygame.K_s]:
                actions[0] = -80  # 基座关节反向
            if keys[pygame.K_a]:
                actions[1] = 60  # 第二关节
            if keys[pygame.K_d]:
                actions[1] = -60
            if keys[pygame.K_1]:
                actions[2] = 40
            if keys[pygame.K_2]:
                actions[2] = -40
            if keys[pygame.K_3]:
                actions[3] = 40
            if keys[pygame.K_4]:
                actions[3] = -40
        
        # 执行step
        obs, reward, done, info = env.step(actions)
        
        # 检查基座关节位置是否穿透障碍物
        base_link_pos = env.bodies[0].position
        base_link_angle = env.bodies[0].angle
        
        # 计算基座关节末端位置
        link_length = env.link_lengths[0]
        end_x = base_link_pos[0] + link_length * np.cos(base_link_angle)
        end_y = base_link_pos[1] + link_length * np.sin(base_link_angle)
        
        # 检查是否在障碍物区域内（简单的几何检测）
        in_obstacle_zone = False
        for obstacle in env.obstacles:
            # 这里简化检测，实际应该用更精确的几何算法
            # 假设障碍物是垂直线段，检查x坐标是否在障碍物附近
            if hasattr(obstacle, 'a') and hasattr(obstacle, 'b'):
                obs_x = obstacle.a[0]  # 障碍物x坐标
                if abs(base_link_pos[0] - obs_x) < 20 or abs(end_x - obs_x) < 20:
                    in_obstacle_zone = True
                    break
        
        # 渲染
        env.render()
        
        # 显示调试信息
        info_texts = [
            f"步数: {step_count}",
            f"模式: {'自动基座测试' if auto_test else '手动控制'}",
            "",
            "🤖 基座关节状态:",
            f"位置: ({base_link_pos[0]:.1f}, {base_link_pos[1]:.1f})",
            f"角度: {np.degrees(base_link_angle):.1f}°",
            f"末端: ({end_x:.1f}, {end_y:.1f})",
            f"在障碍物区域: {'是' if in_obstacle_zone else '否'}",
            "",
            "🚨 碰撞统计:",
            f"基座-障碍物: {collision_stats['link0_obstacle']}",
            f"Link1-障碍物: {collision_stats['link1_obstacle']}",
            f"Link2-障碍物: {collision_stats['link2_obstacle']}",
            f"Link3-障碍物: {collision_stats['link3_obstacle']}",
            f"总计: {collision_stats['total_obstacle']}",
            "",
            "🎮 控制说明:",
            "W/S: 基座关节",
            "A/D: 第二关节",
            "1-4: 其他关节",
            "Space: 自动测试",
            "Q: 退出"
        ]
        
        # 创建信息背景
        info_surface = pygame.Surface((320, len(info_texts) * 22 + 20))
        info_surface.set_alpha(180)
        info_surface.fill((50, 50, 50))
        env.screen.blit(info_surface, (10, 10))
        
        # 显示信息
        for i, text in enumerate(info_texts):
            if text:
                color = (255, 255, 255)
                if "基座关节状态" in text:
                    color = (100, 200, 255)
                elif "在障碍物区域: 是" in text:
                    color = (255, 100, 100)  # 红色警告
                elif "碰撞统计" in text:
                    color = (255, 200, 100)
                elif any(f"{k}: " in text and collision_stats[k] > 0 for k in collision_stats.keys() if k != 'total_obstacle'):
                    color = (100, 255, 100)  # 绿色表示有碰撞检测
                elif "总计:" in text and collision_stats['total_obstacle'] > 0:
                    color = (100, 255, 100)
                
                surface = font.render(text, True, color)
                env.screen.blit(surface, (20, 20 + i * 22))
        
        # 在基座关节位置画一个特殊标记
        pygame.draw.circle(env.screen, (255, 255, 0), 
                         (int(base_link_pos[0]), int(base_link_pos[1])), 5)
        pygame.draw.circle(env.screen, (255, 0, 255), 
                         (int(end_x), int(end_y)), 3)
        
        pygame.display.flip()
        
        step_count += 1
        
        # 每500步输出统计
        if step_count % 500 == 0:
            print(f"\n📊 步数{step_count}统计:")
            print(f"   基座-障碍物碰撞: {collision_stats['link0_obstacle']}")
            print(f"   其他Link-障碍物碰撞: {sum(collision_stats[k] for k in collision_stats.keys() if k.startswith('link') and k != 'link0_obstacle')}")
            print(f"   总障碍物碰撞: {collision_stats['total_obstacle']}")
            
            if in_obstacle_zone and collision_stats['link0_obstacle'] == 0:
                print("🚨 警告: 基座关节在障碍物区域但没有碰撞检测!")
        
        if done:
            env.reset()
        
        clock.tick(60)
    
    # 最终分析
    print(f"\n🎯 最终分析结果:")
    print("=" * 40)
    print(f"总测试步数: {step_count}")
    print(f"基座关节-障碍物碰撞: {collision_stats['link0_obstacle']}")
    print(f"其他Link-障碍物碰撞: {sum(collision_stats[k] for k in collision_stats.keys() if k.startswith('link') and k != 'link0_obstacle')}")
    print(f"总障碍物碰撞: {collision_stats['total_obstacle']}")
    
    if collision_stats['link0_obstacle'] == 0:
        print("\n❌ 问题确认: 基座关节没有与障碍物产生碰撞检测!")
        print("   可能原因:")
        print("   1. 基座关节的collision_type设置有问题")
        print("   2. 碰撞处理器没有正确设置")
        print("   3. 基座关节的物理形状有问题")
    else:
        print("\n✅ 基座关节碰撞检测正常工作")
    
    if collision_stats['total_obstacle'] == 0:
        print("\n❌ 严重问题: 没有任何Link与障碍物产生碰撞!")
        print("   需要检查整个碰撞检测系统")
    
    env.close()

if __name__ == "__main__":
    test_base_joint_penetration()

