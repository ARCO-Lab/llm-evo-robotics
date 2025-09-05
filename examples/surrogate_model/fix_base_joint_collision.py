#!/usr/bin/env python3
"""
修复基座关节穿透障碍物的问题
专门针对基座关节（Link0）与障碍物的碰撞检测进行调试和修复
"""

import sys
import os
import numpy as np
import pygame
import time
import pymunk

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from envs.reacher2d_env import Reacher2DEnv

def diagnose_base_joint_collision():
    """诊断基座关节碰撞问题"""
    print("🔍 诊断基座关节碰撞问题")
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
    
    # 🔍 1. 检查基座关节的物理属性
    print("\n🔍 基座关节物理属性检查:")
    print("=" * 30)
    
    base_body = env.bodies[0]  # 基座关节
    base_shapes = base_body.shapes
    
    print(f"基座关节Body信息:")
    print(f"  位置: {base_body.position}")
    print(f"  角度: {np.degrees(base_body.angle):.1f}°")
    print(f"  质量: {base_body.mass}")
    print(f"  转动惯量: {base_body.moment}")
    print(f"  形状数量: {len(base_shapes)}")
    
    for i, shape in enumerate(base_shapes):
        print(f"  形状{i}:")
        print(f"    类型: {type(shape).__name__}")
        print(f"    collision_type: {shape.collision_type}")
        print(f"    摩擦力: {shape.friction}")
        print(f"    collision_slop: {getattr(shape, 'collision_slop', 'N/A')}")
        
        if isinstance(shape, pymunk.Segment):
            print(f"    线段端点: {shape.a} -> {shape.b}")
            print(f"    线段半径: {shape.radius}")
            
            # 计算世界坐标中的线段位置
            world_a = base_body.local_to_world(shape.a)
            world_b = base_body.local_to_world(shape.b)
            print(f"    世界坐标端点: {world_a} -> {world_b}")
    
    # 🔍 2. 检查障碍物属性
    print(f"\n🔍 障碍物属性检查:")
    print("=" * 30)
    
    for i, obstacle in enumerate(env.obstacles):
        print(f"障碍物{i}:")
        print(f"  类型: {type(obstacle).__name__}")
        print(f"  collision_type: {obstacle.collision_type}")
        print(f"  摩擦力: {obstacle.friction}")
        
        if isinstance(obstacle, pymunk.Segment):
            print(f"  线段端点: {obstacle.a} -> {obstacle.b}")
            print(f"  线段半径: {obstacle.radius}")
    
    # 🔍 3. 手动检查几何碰撞
    print(f"\n🔍 手动几何碰撞检查:")
    print("=" * 30)
    
    def check_segment_collision(seg1_a, seg1_b, seg1_r, seg2_a, seg2_b, seg2_r):
        """简单的线段碰撞检测"""
        # 计算两线段的最短距离
        # 这里使用简化版本，实际PyMunk使用更复杂的算法
        
        # 检查端点到线段的距离
        def point_to_segment_distance(px, py, x1, y1, x2, y2):
            A = px - x1
            B = py - y1
            C = x2 - x1
            D = y2 - y1
            
            dot = A * C + B * D
            len_sq = C * C + D * D
            
            if len_sq == 0:
                return np.sqrt(A * A + B * B)
            
            param = dot / len_sq
            
            if param < 0:
                xx, yy = x1, y1
            elif param > 1:
                xx, yy = x2, y2
            else:
                xx = x1 + param * C
                yy = y1 + param * D
            
            dx = px - xx
            dy = py - yy
            return np.sqrt(dx * dx + dy * dy)
        
        # 检查各种距离
        dist1 = point_to_segment_distance(seg1_a[0], seg1_a[1], seg2_a[0], seg2_a[1], seg2_b[0], seg2_b[1])
        dist2 = point_to_segment_distance(seg1_b[0], seg1_b[1], seg2_a[0], seg2_a[1], seg2_b[0], seg2_b[1])
        dist3 = point_to_segment_distance(seg2_a[0], seg2_a[1], seg1_a[0], seg1_a[1], seg1_b[0], seg1_b[1])
        dist4 = point_to_segment_distance(seg2_b[0], seg2_b[1], seg1_a[0], seg1_a[1], seg1_b[0], seg1_b[1])
        
        min_distance = min(dist1, dist2, dist3, dist4)
        collision_threshold = seg1_r + seg2_r
        
        return min_distance, collision_threshold, min_distance <= collision_threshold
    
    # 获取基座关节的世界坐标
    base_shapes_list = list(base_body.shapes)
    if len(base_shapes_list) > 0:
        base_shape = base_shapes_list[0]
        if isinstance(base_shape, pymunk.Segment):
            base_world_a = base_body.local_to_world(base_shape.a)
            base_world_b = base_body.local_to_world(base_shape.b)
            base_radius = base_shape.radius
            
            print(f"基座关节世界坐标: {base_world_a} -> {base_world_b}, 半径: {base_radius}")
            
            # 检查与每个障碍物的碰撞
            for i, obstacle in enumerate(env.obstacles):
                if isinstance(obstacle, pymunk.Segment):
                    obs_a = obstacle.a
                    obs_b = obstacle.b
                    obs_radius = obstacle.radius
                    
                    min_dist, threshold, is_collision = check_segment_collision(
                        base_world_a, base_world_b, base_radius,
                        obs_a, obs_b, obs_radius
                    )
                    
                    print(f"  vs 障碍物{i}: 最短距离={min_dist:.2f}, 阈值={threshold:.2f}, 碰撞={is_collision}")
    
    return env

def test_base_joint_collision_fix():
    """测试基座关节碰撞修复"""
    print("\n🛠️ 测试基座关节碰撞修复")
    print("=" * 50)
    
    # 诊断问题
    env = diagnose_base_joint_collision()
    
    # 创建增强的碰撞统计
    collision_stats = {
        'base_obstacle': 0,
        'other_obstacle': 0,
        'total': 0
    }
    
    # 🔧 重新设置基座关节的碰撞检测
    print(f"\n🔧 重新设置基座关节碰撞检测:")
    
    def enhanced_base_collision_handler(arbiter, space, data):
        collision_stats['base_obstacle'] += 1
        collision_stats['total'] += 1
        shape_a, shape_b = arbiter.shapes
        
        print(f"🚨 [ENHANCED] 基座关节碰撞障碍物!")
        print(f"   collision_type: {shape_a.collision_type} vs {shape_b.collision_type}")
        print(f"   碰撞点数: {len(arbiter.contact_point_set.points)}")
        
        if len(arbiter.contact_point_set.points) > 0:
            for i, point in enumerate(arbiter.contact_point_set.points):
                print(f"   碰撞点{i}: 位置={point.point_a}, 深度={point.distance}")
        
        # 设置碰撞参数
        arbiter.restitution = 0.1  # 低弹性
        arbiter.friction = 0.9     # 高摩擦
        
        return True  # 允许碰撞处理
    
    def enhanced_other_collision_handler(arbiter, space, data):
        collision_stats['other_obstacle'] += 1
        collision_stats['total'] += 1
        shape_a, shape_b = arbiter.shapes
        
        print(f"🚨 [ENHANCED] 其他Link碰撞障碍物!")
        print(f"   collision_type: {shape_a.collision_type} vs {shape_b.collision_type}")
        
        return True
    
    # 清除现有的碰撞处理器并重新设置
    OBSTACLE_COLLISION_TYPE = 100
    BASE_COLLISION_TYPE = 1
    
    try:
        # 专门为基座关节设置增强碰撞检测
        env.space.on_collision(
            collision_type_a=BASE_COLLISION_TYPE,
            collision_type_b=OBSTACLE_COLLISION_TYPE,
            begin=enhanced_base_collision_handler,
            pre_solve=enhanced_base_collision_handler,
            post_solve=enhanced_base_collision_handler,
            separate=lambda arbiter, space, data: print("🔄 基座关节与障碍物分离")
        )
        print("✅ 设置增强基座关节碰撞检测")
        
        # 为其他Link设置碰撞检测
        for i in range(1, env.num_links):
            env.space.on_collision(
                collision_type_a=i + 1,
                collision_type_b=OBSTACLE_COLLISION_TYPE,
                begin=enhanced_other_collision_handler
            )
            print(f"✅ 设置增强Link{i+1}碰撞检测")
            
    except Exception as e:
        print(f"❌ 设置增强碰撞检测失败: {e}")
    
    # 🔧 额外：检查并修复基座关节的物理属性
    base_body = env.bodies[0]
    base_shapes_list = list(base_body.shapes)
    base_shape = base_shapes_list[0] if len(base_shapes_list) > 0 else None
    
    print(f"\n🔧 修复基座关节物理属性:")
    if base_shape:
        print(f"修复前 - collision_slop: {getattr(base_shape, 'collision_slop', 'N/A')}")
        
        # 确保collision_slop设置正确
        if hasattr(base_shape, 'collision_slop'):
            base_shape.collision_slop = 0.1  # 设置较小的碰撞容差
            print(f"修复后 - collision_slop: {base_shape.collision_slop}")
        
        # 确保friction设置正确
        base_shape.friction = 1.0
        print(f"修复后 - friction: {base_shape.friction}")
    else:
        print("❌ 无法获取基座关节形状")
    
    # 🎮 开始测试
    print(f"\n🎮 开始碰撞测试:")
    print("  D + W: 让基座关节接触障碍物")
    print("  其他控制: W/S/A/D/1-4")
    print("  Q: 退出")
    
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    running = True
    step_count = 0
    
    while running and step_count < 1000:
        # 处理事件
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        # 生成动作 - 专门测试D+W组合
        actions = np.zeros(4)
        if keys[pygame.K_d] and keys[pygame.K_w]:
            # 同时按D+W：让基座关节接触障碍物
            actions[0] = 100  # 基座关节大力转动
            actions[1] = -80  # 第二关节配合
            print(f"🎯 执行D+W组合动作 - 基座关节接触测试")
        else:
            # 正常控制
            if keys[pygame.K_w]:
                actions[0] = 80
            if keys[pygame.K_s]:
                actions[0] = -80
            if keys[pygame.K_a]:
                actions[1] = 60
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
        
        # 渲染
        env.render()
        
        # 显示增强调试信息
        base_pos = env.bodies[0].position
        base_angle = env.bodies[0].angle
        
        info_texts = [
            f"步数: {step_count}",
            "",
            "🤖 基座关节状态:",
            f"位置: ({base_pos[0]:.1f}, {base_pos[1]:.1f})",
            f"角度: {np.degrees(base_angle):.1f}°",
            "",
            "🚨 碰撞统计:",
            f"基座-障碍物: {collision_stats['base_obstacle']}",
            f"其他-障碍物: {collision_stats['other_obstacle']}",
            f"总计: {collision_stats['total']}",
            "",
            "🎮 测试说明:",
            "D+W: 基座关节接触测试",
            "单独按键: 正常控制",
            "Q: 退出",
            "",
            f"🔍 期望: 基座关节应该碰撞障碍物",
            f"实际: {'✅ 正常' if collision_stats['base_obstacle'] > 0 else '❌ 穿透'}"
        ]
        
        # 显示信息
        info_surface = pygame.Surface((350, len(info_texts) * 22 + 20))
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
                elif "基座-障碍物:" in text and collision_stats['base_obstacle'] > 0:
                    color = (100, 255, 100)  # 绿色表示有碰撞
                elif "❌ 穿透" in text:
                    color = (255, 100, 100)  # 红色警告
                elif "✅ 正常" in text:
                    color = (100, 255, 100)  # 绿色正常
                
                surface = font.render(text, True, color)
                env.screen.blit(surface, (20, 20 + i * 22))
        
        pygame.display.flip()
        
        step_count += 1
        
        # 每100步输出统计
        if step_count % 100 == 0:
            print(f"\n📊 步数{step_count}统计:")
            print(f"   基座-障碍物碰撞: {collision_stats['base_obstacle']}")
            print(f"   其他-障碍物碰撞: {collision_stats['other_obstacle']}")
            print(f"   总碰撞: {collision_stats['total']}")
            
            if collision_stats['base_obstacle'] == 0:
                print("🚨 警告: 基座关节仍然没有碰撞检测!")
        
        if done:
            env.reset()
        
        clock.tick(60)
    
    # 最终分析
    print(f"\n🎯 最终测试结果:")
    print("=" * 40)
    print(f"测试步数: {step_count}")
    print(f"基座关节-障碍物碰撞: {collision_stats['base_obstacle']}")
    print(f"其他Link-障碍物碰撞: {collision_stats['other_obstacle']}")
    print(f"总碰撞: {collision_stats['total']}")
    
    if collision_stats['base_obstacle'] == 0:
        print(f"\n❌ 基座关节碰撞修复失败!")
        print("   可能需要更深层的PyMunk调试")
    else:
        print(f"\n✅ 基座关节碰撞修复成功!")
        print("   基座关节现在可以正确与障碍物碰撞")
    
    env.close()

if __name__ == "__main__":
    test_base_joint_collision_fix()
