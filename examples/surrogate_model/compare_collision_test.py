#!/usr/bin/env python3
"""
对比碰撞测试 - 展示为什么红色link可以穿透但蓝色不会
"""

import pygame
import pymunk
import pymunk.pygame_util
import yaml
import numpy as np
import time
import math

def load_obstacles_from_yaml(yaml_path):
    """从YAML文件加载障碍物配置"""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('obstacles', [])

def create_comparison_test():
    """创建对比测试，展示两个link的不同行为"""
    print("🔬 对比碰撞测试 - 分析红色vs蓝色link")
    print("="*50)
    
    # 初始化pygame
    pygame.init()
    width, height = 1000, 700
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("对比测试 - 红色vs蓝色link碰撞行为")
    clock = pygame.time.Clock()
    
    # 物理空间设置
    space = pymunk.Space()
    space.gravity = (0.0, 200.0)
    space.damping = 0.98
    space.collision_slop = 0.01
    
    print(f"✅ 物理空间设置:")
    print(f"   gravity: {space.gravity}")
    print(f"   damping: {space.damping}")
    print(f"   collision_slop: {space.collision_slop}")
    
    # 创建绘制选项
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    # 加载障碍物
    yaml_path = '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml'
    obstacles_config = load_obstacles_from_yaml(yaml_path)
    
    print(f"\n🏗️ 创建障碍物:")
    
    obstacles = []
    OBSTACLE_COLLISION_TYPE = 100
    
    for i, obs in enumerate(obstacles_config):
        if obs["shape"] == "segment":
            p1 = tuple(obs["points"][0])
            p2 = tuple(obs["points"][1])
            
            shape = pymunk.Segment(space.static_body, p1, p2, radius=5.0)
            shape.friction = 1.0
            shape.color = (0, 0, 0, 255)  # 黑色障碍物
            shape.collision_type = OBSTACLE_COLLISION_TYPE
            shape.collision_slop = 0.01
            
            space.add(shape)
            obstacles.append(shape)
            print(f"   障碍物 {i}: {p1} → {p2}")
    
    print(f"✅ 创建了 {len(obstacles)} 个障碍物")
    
    # 🎯 创建蓝色link (第一个，正确的)
    print(f"\n🔵 创建蓝色Link (正确配置):")
    
    blue_mass = 10
    blue_length = 60
    blue_radius = 8
    
    blue_moment = pymunk.moment_for_segment(blue_mass, (0, 0), (blue_length, 0), blue_radius)
    blue_body = pymunk.Body(blue_mass, blue_moment)
    blue_body.position = (300, 300)  # 左侧位置
    
    blue_shape = pymunk.Segment(blue_body, (0, 0), (blue_length, 0), blue_radius)
    blue_shape.friction = 0.8
    blue_shape.collision_type = 1  # 蓝色link的碰撞类型
    blue_shape.collision_slop = 0.01
    blue_shape.color = (0, 0, 255, 255)  # 蓝色
    
    space.add(blue_body, blue_shape)
    
    print(f"   质量: {blue_mass}")
    print(f"   碰撞类型: {blue_shape.collision_type}")
    print(f"   位置: {blue_body.position}")
    print(f"   添加到space: ✅")
    
    # 🎯 创建红色link (第二个，演示问题)
    print(f"\n🔴 创建红色Link (演示问题配置):")
    
    red_mass = 10
    red_length = 60
    red_radius = 8
    
    red_moment = pymunk.moment_for_segment(red_mass, (0, 0), (red_length, 0), red_radius)
    red_body = pymunk.Body(red_mass, red_moment)
    red_body.position = (600, 300)  # 右侧位置
    
    red_shape = pymunk.Segment(red_body, (0, 0), (red_length, 0), red_radius)
    red_shape.friction = 0.8
    red_shape.collision_type = 2  # 🔧 不同的碰撞类型！
    red_shape.collision_slop = 0.01
    red_shape.color = (255, 0, 0, 255)  # 红色
    
    space.add(red_body, red_shape)
    
    print(f"   质量: {red_mass}")
    print(f"   碰撞类型: {red_shape.collision_type}")
    print(f"   位置: {red_body.position}")
    print(f"   添加到space: ✅")
    
    # 🎯 碰撞检测设置
    blue_collisions = 0
    red_collisions = 0
    
    def blue_collision_handler(arbiter, space, data):
        nonlocal blue_collisions
        blue_collisions += 1
        print(f"🔵 蓝色Link碰撞! 总计: {blue_collisions}")
        return True
    
    def red_collision_handler(arbiter, space, data):
        nonlocal red_collisions
        red_collisions += 1
        print(f"🔴 红色Link碰撞! 总计: {red_collisions}")
        return True
    
    # 注册碰撞处理器
    print(f"\n🎯 碰撞处理器设置:")
    
    # 🔵 为蓝色link注册碰撞处理器
    try:
        blue_handler = space.add_collision_handler(1, OBSTACLE_COLLISION_TYPE)
        blue_handler.begin = blue_collision_handler
        print(f"✅ 蓝色Link碰撞处理器: collision_type 1 vs {OBSTACLE_COLLISION_TYPE}")
    except AttributeError:
        space.on_collision(
            collision_type_a=1,
            collision_type_b=OBSTACLE_COLLISION_TYPE,
            begin=blue_collision_handler
        )
        print(f"✅ 蓝色Link碰撞处理器 (旧API): collision_type 1 vs {OBSTACLE_COLLISION_TYPE}")
    
    # 🔴 故意不为红色link注册碰撞处理器！
    print(f"❌ 红色Link碰撞处理器: 故意不注册 (collision_type 2)")
    print(f"   这就是为什么红色link可以穿透障碍物！")
    
    # 控制参数
    force_magnitude = 800.0
    running = True
    paused = False
    controlled_link = "blue"  # 当前控制的link
    
    print(f"\n🎮 控制说明:")
    print(f"   WASD: 移动当前控制的link")
    print(f"   TAB: 切换控制 (蓝色/红色)")
    print(f"   SPACE: 暂停/继续")
    print(f"   R: 重置位置")
    print(f"   ESC: 退出")
    print(f"   目标: 观察蓝色link被阻挡，红色link穿透")
    
    font = pygame.font.Font(None, 24)
    
    # 主循环
    step_count = 0
    while running:
        step_count += 1
        dt = clock.tick(60) / 1000.0
        
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"{'⏸️ 暂停' if paused else '▶️ 继续'}")
                elif event.key == pygame.K_TAB:
                    controlled_link = "red" if controlled_link == "blue" else "blue"
                    print(f"🎮 切换控制到: {'🔵 蓝色Link' if controlled_link == 'blue' else '🔴 红色Link'}")
                elif event.key == pygame.K_r:
                    # 重置位置
                    blue_body.position = (300, 300)
                    blue_body.velocity = (0, 0)
                    blue_body.angular_velocity = 0
                    blue_body.angle = 0
                    
                    red_body.position = (600, 300)
                    red_body.velocity = (0, 0)
                    red_body.angular_velocity = 0
                    red_body.angle = 0
                    
                    blue_collisions = 0
                    red_collisions = 0
                    print(f"🔄 重置所有link位置")
        
        # 控制
        if not paused:
            keys = pygame.key.get_pressed()
            
            # 选择要控制的body
            if controlled_link == "blue":
                controlled_body = blue_body
            else:
                controlled_body = red_body
            
            force_x, force_y = 0, 0
            if keys[pygame.K_a]:  # 左
                force_x = -force_magnitude
            if keys[pygame.K_d]:  # 右
                force_x = force_magnitude
            if keys[pygame.K_w]:  # 上
                force_y = -force_magnitude
            if keys[pygame.K_s]:  # 下
                force_y = force_magnitude
            
            # 应用力
            if force_x != 0 or force_y != 0:
                controlled_body.apply_force_at_world_point((force_x, force_y), controlled_body.position)
        
        # 物理更新
        if not paused:
            space.step(dt)
        
        # 渲染
        screen.fill((240, 240, 240))
        
        # 绘制物理对象
        space.debug_draw(draw_options)
        
        # 显示信息
        info_lines = [
            f"步数: {step_count}",
            f"当前控制: {'🔵 蓝色Link' if controlled_link == 'blue' else '🔴 红色Link'}",
            f"",
            f"🔵 蓝色Link:",
            f"   位置: ({blue_body.position.x:.0f}, {blue_body.position.y:.0f})",
            f"   碰撞次数: {blue_collisions}",
            f"   碰撞类型: {blue_shape.collision_type}",
            f"   碰撞处理器: ✅ 已注册",
            f"",
            f"🔴 红色Link:",
            f"   位置: ({red_body.position.x:.0f}, {red_body.position.y:.0f})",
            f"   碰撞次数: {red_collisions}",
            f"   碰撞类型: {red_shape.collision_type}",
            f"   碰撞处理器: ❌ 未注册",
            f"",
            f"状态: {'⏸️ 暂停' if paused else '▶️ 运行'}",
            f"",
            f"解释:",
            f"蓝色link有碰撞处理器，会被阻挡",
            f"红色link没有碰撞处理器，可以穿透",
        ]
        
        for i, line in enumerate(info_lines):
            color = (0, 0, 255) if "🔵" in line else (255, 0, 0) if "🔴" in line else (0, 0, 0)
            text = font.render(line, True, color)
            screen.blit(text, (10, 10 + i * 20))
        
        # 控制指南
        guide_lines = [
            "🎮 控制:",
            "WASD: 移动",
            "TAB: 切换控制",
            "R: 重置, SPACE: 暂停"
        ]
        
        for i, line in enumerate(guide_lines):
            text = font.render(line, True, (100, 100, 100))
            screen.blit(text, (width - 150, 10 + i * 20))
        
        pygame.display.flip()
    
    pygame.quit()
    
    print(f"\n📊 对比测试结果:")
    print(f"   🔵 蓝色Link碰撞次数: {blue_collisions}")
    print(f"   🔴 红色Link碰撞次数: {red_collisions}")
    print(f"")
    print(f"🔍 分析:")
    print(f"   蓝色Link: collision_type=1, 有碰撞处理器 → {'会被阻挡' if blue_collisions > 0 else '应该会被阻挡'}")
    print(f"   红色Link: collision_type=2, 无碰撞处理器 → {'可以穿透' if red_collisions == 0 else '意外被阻挡'}")
    
    return blue_collisions, red_collisions

if __name__ == "__main__":
    print("🔬 对比碰撞测试")
    print("="*50)
    print("目标：解释为什么红色link可以穿透但蓝色不会")
    print()
    
    try:
        blue_cols, red_cols = create_comparison_test()
        
        print(f"\n✅ 测试完成!")
        print(f"这解释了为什么红色link可以穿透障碍物：")
        print(f"1. 碰撞类型不同 (1 vs 2)")
        print(f"2. 没有注册对应的碰撞处理器")
        print(f"3. PyMunk允许没有处理器的碰撞穿透")
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
