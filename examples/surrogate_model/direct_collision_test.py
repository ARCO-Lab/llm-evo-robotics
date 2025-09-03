#!/usr/bin/env python3
"""
直接碰撞测试 - 将Link直接放置在障碍物附近测试碰撞
"""

import pygame
import pymunk
import pymunk.pygame_util
import yaml
import math

def create_direct_collision_test():
    """创建直接碰撞测试"""
    print("🔧 直接碰撞测试")
    print("="*40)
    print("🎯 将Link直接放置在障碍物附近")
    
    # 初始化pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("直接碰撞测试 - 按数字键移动到不同障碍物")
    clock = pygame.time.Clock()
    
    # 物理空间设置
    space = pymunk.Space()
    space.gravity = (0.0, 100.0)  # 轻微重力
    space.damping = 0.98
    space.collision_slop = 0.01
    
    # 创建绘制选项
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    # 🎯 手动创建几个简单的障碍物进行测试
    obstacles = []
    OBSTACLE_COLLISION_TYPE = 100
    
    # 障碍物位置 - 容易到达的位置
    obstacle_configs = [
        {"start": (300, 300), "end": (400, 300), "name": "水平线1"},
        {"start": (500, 200), "end": (500, 300), "name": "竖直线1"},
        {"start": (200, 400), "end": (300, 500), "name": "斜线1"},
        {"start": (600, 350), "end": (700, 400), "name": "斜线2"},
    ]
    
    print(f"✅ 创建简单障碍物:")
    for i, config in enumerate(obstacle_configs):
        start, end, name = config["start"], config["end"], config["name"]
        
        shape = pymunk.Segment(space.static_body, start, end, radius=8.0)  # 增大半径便于观察
        shape.friction = 1.0
        shape.color = (255, 0, 0, 255)  # 红色
        shape.collision_type = OBSTACLE_COLLISION_TYPE
        shape.collision_slop = 0.01
        
        space.add(shape)
        obstacles.append(shape)
        print(f"   障碍物 {i+1}: {name}, {start} → {end}")
    
    # 🎯 创建Link
    link_mass = 20
    link_length = 60
    link_radius = 8
    
    moment = pymunk.moment_for_segment(link_mass, (0, 0), (link_length, 0), link_radius)
    test_body = pymunk.Body(link_mass, moment)
    test_body.position = (250, 250)  # 初始位置
    
    test_shape = pymunk.Segment(test_body, (0, 0), (link_length, 0), link_radius)
    test_shape.friction = 0.8
    test_shape.collision_type = 1
    test_shape.collision_slop = 0.01
    test_shape.color = (0, 255, 0, 255)  # 绿色
    
    space.add(test_body, test_shape)
    
    print(f"✅ 创建Link: 质量={link_mass}, 长度={link_length}")
    
    # 🎯 碰撞检测
    collision_count = 0
    
    def collision_handler(arbiter, space, data):
        nonlocal collision_count
        collision_count += 1
        print(f"🚨 碰撞! 总计: {collision_count}")
        print(f"   Link位置: {test_body.position}")
        return True
    
    # 注册碰撞处理器
    try:
        handler = space.add_collision_handler(1, OBSTACLE_COLLISION_TYPE)
        handler.begin = collision_handler
    except AttributeError:
        space.on_collision(
            collision_type_a=1,
            collision_type_b=OBSTACLE_COLLISION_TYPE,
            begin=collision_handler
        )
    
    print(f"✅ 碰撞处理器已设置")
    
    # 🎯 预设位置 - 直接放到障碍物附近
    test_positions = [
        (280, 290, "接近水平线1"),
        (490, 250, "接近竖直线1"), 
        (240, 430, "接近斜线1"),
        (620, 370, "接近斜线2"),
        (100, 100, "远离所有障碍物"),
    ]
    
    current_pos_index = 0
    running = True
    paused = False
    force_magnitude = 800.0
    
    print(f"\n🎮 控制说明:")
    print(f"   1-5: 跳转到预设位置")
    print(f"   WASD: 移动Link")
    print(f"   SPACE: 暂停/继续")
    print(f"   R: 重置到当前预设位置")
    print(f"   ESC: 退出")
    
    # 立即移动到第一个测试位置
    pos_x, pos_y, desc = test_positions[current_pos_index]
    test_body.position = (pos_x, pos_y)
    test_body.velocity = (0, 0)
    print(f"🎯 移动到位置 {current_pos_index+1}: {desc} ({pos_x}, {pos_y})")
    
    font = pygame.font.Font(None, 24)
    
    # 主循环
    while running:
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
                elif event.key == pygame.K_r:
                    # 重置到当前位置
                    pos_x, pos_y, desc = test_positions[current_pos_index]
                    test_body.position = (pos_x, pos_y)
                    test_body.velocity = (0, 0)
                    test_body.angular_velocity = 0
                    test_body.angle = 0
                    print(f"🔄 重置到位置: {desc}")
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                    # 跳转到预设位置
                    pos_index = event.key - pygame.K_1
                    if pos_index < len(test_positions):
                        current_pos_index = pos_index
                        pos_x, pos_y, desc = test_positions[current_pos_index]
                        test_body.position = (pos_x, pos_y)
                        test_body.velocity = (0, 0)
                        test_body.angular_velocity = 0
                        test_body.angle = 0
                        print(f"🎯 跳转到位置 {pos_index+1}: {desc} ({pos_x}, {pos_y})")
        
        # 控制
        if not paused:
            keys = pygame.key.get_pressed()
            
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
                test_body.apply_force_at_world_point((force_x, force_y), test_body.position)
        
        # 物理更新
        if not paused:
            space.step(dt)
        
        # 渲染
        screen.fill((240, 240, 240))
        
        # 绘制物理对象
        space.debug_draw(draw_options)
        
        # 显示信息
        pos_x, pos_y, desc = test_positions[current_pos_index]
        info_lines = [
            f"当前位置 {current_pos_index+1}: {desc}",
            f"Link位置: ({test_body.position.x:.0f}, {test_body.position.y:.0f})",
            f"Link速度: ({test_body.velocity.x:.0f}, {test_body.velocity.y:.0f})",
            f"碰撞次数: {collision_count}",
            f"状态: {'⏸️ 暂停' if paused else '▶️ 运行'}",
            "",
            f"测试位置:",
            f"1: 水平线1附近 (280, 290)",
            f"2: 竖直线1附近 (490, 250)",
            f"3: 斜线1附近 (240, 430)",
            f"4: 斜线2附近 (620, 370)",
            f"5: 远离障碍物 (100, 100)",
        ]
        
        for i, line in enumerate(info_lines):
            color = (255, 0, 0) if "碰撞" in line and collision_count > 0 else (0, 0, 0)
            text = font.render(line, True, color)
            screen.blit(text, (10, 10 + i * 22))
        
        # 碰撞状态
        if collision_count > 0:
            collision_text = font.render(f"🚨 已检测到碰撞!", True, (255, 0, 0))
            screen.blit(collision_text, (10, height - 40))
        
        pygame.display.flip()
    
    pygame.quit()
    
    print(f"\n📊 直接碰撞测试结果:")
    print(f"   总碰撞次数: {collision_count}")
    print(f"   碰撞检测: {'✅ 正常工作' if collision_count > 0 else '❌ 需要检查'}")
    
    return collision_count > 0

if __name__ == "__main__":
    print("🔧 直接碰撞测试")
    print("="*50)
    print("测试方法：将Link直接放置在障碍物附近")
    print()
    
    try:
        success = create_direct_collision_test()
        
        if success:
            print("✅ 碰撞检测正常工作")
        else:
            print("❌ 碰撞检测可能有问题")
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()