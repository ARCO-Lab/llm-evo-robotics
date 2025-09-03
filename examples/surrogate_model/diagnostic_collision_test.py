#!/usr/bin/env python3
"""
诊断碰撞穿透问题 - 复制实际环境设置
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

def create_diagnostic_test():
    """创建诊断测试，完全复制reacher2d_env的设置"""
    print("🔬 诊断碰撞穿透问题")
    print("="*50)
    print("📋 复制实际环境的所有物理参数...")
    
    # 初始化pygame
    pygame.init()
    width, height = 1200, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("诊断测试 - 与实际环境相同的设置")
    clock = pygame.time.Clock()
    
    # 🎯 完全复制reacher2d_env.py的space设置
    space = pymunk.Space()
    space.gravity = (0.0, 100.0)  # 与环境完全一致
    space.damping = 0.999         # 与环境完全一致
    space.collision_slop = 0.01   # 与环境完全一致
    space.collision_bias = (1-0.1) ** 60  # 与环境完全一致
    space.sleep_time_threshold = 0.5      # 与环境完全一致
    
    print(f"✅ PyMunk空间设置 (与reacher2d_env.py一致):")
    print(f"   gravity: {space.gravity}")
    print(f"   damping: {space.damping}")
    print(f"   collision_slop: {space.collision_slop}")
    print(f"   collision_bias: {space.collision_bias}")
    print(f"   sleep_time_threshold: {space.sleep_time_threshold}")
    
    # 创建绘制选项
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    # 加载障碍物
    yaml_path = '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml'
    obstacles_config = load_obstacles_from_yaml(yaml_path)
    
    print(f"\n🏗️ 创建障碍物 (与reacher2d_env.py一致):")
    print(f"   障碍物数量: {len(obstacles_config)}")
    
    # 🎯 完全复制_create_obstacle()的逻辑
    obstacles = []
    OBSTACLE_COLLISION_TYPE = 100  # 与环境一致
    
    for i, obs in enumerate(obstacles_config):
        if obs["shape"] == "segment":
            p1 = tuple(obs["points"][0])
            p2 = tuple(obs["points"][1])
            
            # 与reacher2d_env.py完全一致的障碍物创建
            shape = pymunk.Segment(space.static_body, p1, p2, radius=5.0)  # 半径与环境一致
            shape.friction = 1.0                                          # 摩擦力与环境一致
            shape.color = (0,0,0,255)                                     # 颜色与环境一致
            shape.density = 1000                                          # 密度与环境一致
            shape.collision_type = OBSTACLE_COLLISION_TYPE                # 碰撞类型与环境一致
            shape.collision_slop = 0.01                                   # collision_slop与环境一致
            
            space.add(shape)
            obstacles.append(shape)
            print(f"   障碍物 {i}: {p1} → {p2}, collision_type={shape.collision_type}")
    
    print(f"✅ 创建了 {len(obstacles)} 个障碍物")
    
    # 🎯 创建类似robot link的测试对象
    print(f"\n🤖 创建测试Link (模拟robot link):")
    
    # 参数与reacher2d_env._create_robot()一致
    density = 1  # 与环境一致
    link_length = 60  # 与环境一致
    link_radius = 8   # 与环境一致 (shape半径)
    mass = density * link_length * 10  # 质量计算与环境一致
    
    moment = pymunk.moment_for_segment(mass, (0, 0), (link_length, 0), link_radius)
    test_body = pymunk.Body(mass, moment)
    test_body.position = (500, 400)  # 起始位置
    
    # 创建shape - 与环境完全一致
    test_shape = pymunk.Segment(test_body, (0, 0), (link_length, 0), link_radius)
    test_shape.friction = 0.8  # 与环境一致
    test_shape.collision_type = 1  # 与环境中robot link一致
    test_shape.collision_slop = 0.01  # 与环境一致
    test_shape.color = (0, 0, 255, 255)  # 蓝色便于识别
    
    space.add(test_body, test_shape)
    
    print(f"   Link参数 (与reacher2d_env一致):")
    print(f"   mass: {mass}")
    print(f"   length: {link_length}")
    print(f"   radius: {link_radius}")
    print(f"   friction: {test_shape.friction}")
    print(f"   collision_type: {test_shape.collision_type}")
    print(f"   collision_slop: {test_shape.collision_slop}")
    
    # 🎯 先移除蓝色link，只保留红色link和正确的碰撞处理器
    space.remove(test_body, test_shape)
    print(f"🗑️ 移除蓝色Link，只保留红色Link")
    
    # 碰撞检测变量
    collision_count = 0
    last_collision_time = 0
    penetration_detected = False
    
    def collision_handler(arbiter, space, data):
        nonlocal collision_count, last_collision_time
        current_time = time.time()
        if current_time - last_collision_time > 0.05:  # 减少重复输出
            collision_count += 1
            last_collision_time = current_time
            print(f"🚨 红色Link碰撞检测! 总计: {collision_count}")
        return True  # 允许物理碰撞响应
    
    # 🎯 穿透检测函数
    def check_penetration():
        """检测是否发生穿透"""
        nonlocal penetration_detected
        
        # 获取test_shape的线段端点
        body_pos = test_body.position
        angle = test_body.angle
        
        # 计算link两端的世界坐标
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        start_world = (body_pos.x, body_pos.y)
        end_world = (
            body_pos.x + link_length * cos_a,
            body_pos.y + link_length * sin_a
        )
        
        # 检查与每个障碍物的穿透
        for i, obstacle in enumerate(obstacles):
            # 障碍物是Segment，获取其端点
            if hasattr(obstacle, 'a') and hasattr(obstacle, 'b'):
                obs_start = obstacle.a
                obs_end = obstacle.b
                
                # 简单的线段相交检测
                if line_segments_intersect(start_world, end_world, obs_start, obs_end):
                    if not penetration_detected:
                        print(f"⚠️ 检测到穿透! Link与障碍物{i}相交")
                        print(f"   Link: {start_world} → {end_world}")
                        print(f"   障碍物: {obs_start} → {obs_end}")
                        penetration_detected = True
                    return True
        
        return False
    
    def line_segments_intersect(p1, p2, p3, p4):
        """检测两条线段是否相交"""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)
    
    # 控制参数
    force_magnitude = 500.0
    running = True
    paused = False
    dt = 1/120.0  # 与环境一致的时间步长
    
    print(f"\n🎮 控制说明:")
    print(f"   WASD: 对Link施加力")
    print(f"   SPACE: 暂停/继续")
    print(f"   R: 重置位置")
    print(f"   ESC: 退出")
    print(f"   目标: 测试Link是否能穿透障碍物")
    
    font = pygame.font.Font(None, 24)
    
    # 主循环
       # 创建可控制的link
    link_mass = 5  # 减少质量便于控制
    link_length = 60
    link_radius = 10
    
    # Link的初始位置
    start_pos = (400, 300)  # 移到屏幕中央易于控制
    
    moment = pymunk.moment_for_segment(link_mass, (0, 0), (link_length, 0), link_radius)
    link_body = pymunk.Body(link_mass, moment)
    link_body.position = start_pos
    
    link_shape = pymunk.Segment(link_body, (0, 0), (link_length, 0), link_radius)
    link_shape.friction = 0.5
    link_shape.collision_type = 1
    link_shape.collision_slop = 0.01
    link_shape.color = pygame.Color("red")
    
    space.add(link_body, link_shape)
    
    # 🎯 为红色link注册碰撞处理器
    print(f"✅ 红色Link创建完成:")
    print(f"   collision_type: {link_shape.collision_type}")
    print(f"   位置: {link_body.position}")
    
    # 注册红色link的碰撞处理器
    try:
        handler = space.add_collision_handler(1, OBSTACLE_COLLISION_TYPE)
        handler.begin = collision_handler
        print(f"✅ 红色Link碰撞处理器已设置: Link(1) vs Obstacle({OBSTACLE_COLLISION_TYPE})")
    except AttributeError:
        space.on_collision(
            collision_type_a=1,
            collision_type_b=OBSTACLE_COLLISION_TYPE,
            begin=collision_handler
        )
        print(f"✅ 红色Link碰撞处理器已设置 (旧API): Link(1) vs Obstacle({OBSTACLE_COLLISION_TYPE})")
    
    # 控制参数
    move_force = 2000.0  # 🔧 大幅增加力度，确保能有效移动
    max_velocity = 400.0  # 最大速度限制
    running = True
    paused = False
    
    # 按键状态追踪
    keys_pressed = set()
    
    print(f"\n🎮 改进的控制说明:")
    print(f"   1. 点击窗口获得焦点")
    print(f"   2. WASD: 移动link (现在应该更明显)")
    print(f"   3. QE: 旋转link")
    print(f"   4. SPACE: 暂停/继续")
    print(f"   5. R: 重置位置")
    print(f"   6. ESC: 退出")
    
    font = pygame.font.Font(None, 24)
    
    # 主循环
    while running:
        dt = clock.tick(60) / 1000.0
        
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                keys_pressed.add(event.key)
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"{'⏸️ 暂停' if paused else '▶️ 继续'}")
                elif event.key == pygame.K_r:
                    link_body.position = start_pos
                    link_body.velocity = (0, 0)
                    link_body.angular_velocity = 0
                    link_body.angle = 0
                    print(f"🔄 重置位置")
                # 调试按键
                print(f"🔧 按键按下: {pygame.key.name(event.key)}")
            elif event.type == pygame.KEYUP:
                keys_pressed.discard(event.key)
        
        # 物理控制
        if not paused:
            # 获取当前按键状态
            keys = pygame.key.get_pressed()
            
            # 🔧 修复：使用力而不是直接设置速度，这样PyMunk可以正确处理碰撞
            force_x, force_y = 0, 0
            force_applied = False
            
            if keys[pygame.K_a]:  # 左
                force_x = -move_force
                force_applied = True
            if keys[pygame.K_d]:  # 右
                force_x = move_force
                force_applied = True
            if keys[pygame.K_w]:  # 上
                force_y = -move_force
                force_applied = True
            if keys[pygame.K_s]:  # 下
                force_y = move_force
                force_applied = True
            
            # 🔧 使用施加力而不是直接设置速度
            if force_applied:
                link_body.apply_force_at_world_point((force_x, force_y), link_body.position)
                # 调试输出
                if pygame.K_a in keys_pressed or pygame.K_d in keys_pressed or pygame.K_w in keys_pressed or pygame.K_s in keys_pressed:
                    print(f"🎮 控制生效: 施加力=({force_x:.0f}, {force_y:.0f})")
            
            # 🔧 速度限制（但不覆盖碰撞响应）
            vel = link_body.velocity
            speed = math.sqrt(vel.x**2 + vel.y**2)
            if speed > max_velocity:
                scale = max_velocity / speed
                link_body.velocity = (vel.x * scale, vel.y * scale)
            
            # 旋转控制
            if keys[pygame.K_q]:
                link_body.angular_velocity = -3.0
            elif keys[pygame.K_e]:
                link_body.angular_velocity = 3.0
            else:
                link_body.angular_velocity *= 0.9
        
        # 物理更新
        if not paused:
            space.step(dt)
        
        # 渲染
        screen.fill((220, 220, 220))  # 浅灰色背景
        
        # 绘制物理对象
        space.debug_draw(draw_options)
        
        # 显示详细信息
        info_lines = [
            f"碰撞次数: {collision_count}",
            f"Link位置: ({link_body.position.x:.0f}, {link_body.position.y:.0f})",
            f"Link速度: ({link_body.velocity.x:.0f}, {link_body.velocity.y:.0f})",
            f"Link角度: {math.degrees(link_body.angle):.1f}°",
            f"状态: {'⏸️ 暂停' if paused else '▶️ 运行'}",
            "",
            f"控制指南:",
            f"WASD: 移动 (速度控制)",
            f"QE: 旋转",
            f"R: 重置, SPACE: 暂停, ESC: 退出",
            "",
            f"当前按键: {[pygame.key.name(k) for k in keys_pressed]}"
        ]
        
        for i, line in enumerate(info_lines):
            color = (255, 0, 0) if "碰撞" in line and collision_count > 0 else (0, 0, 0)
            text = font.render(line, True, color)
            screen.blit(text, (10, 10 + i * 22))
        
        pygame.display.flip()
        clock.tick(120)  # 与环境一致的120 FPS
    
    pygame.quit()
    
    print(f"\n📊 诊断结果:")
    print(f"   总碰撞次数: {collision_count}")
    print(f"   发现穿透: {'是' if penetration_detected else '否'}")
    print(f"   碰撞检测: {'✅ 正常工作' if collision_count > 0 else '❌ 可能有问题'}")
    
    if collision_count == 0:
        print(f"\n🔍 可能的问题:")
        print(f"   1. collision_type设置不正确")
        print(f"   2. 物理对象没有正确添加到space")
        print(f"   3. 碰撞处理器没有正确注册")
        print(f"   4. 力的大小不足以产生有效碰撞")
    
    return collision_count > 0, penetration_detected

if __name__ == "__main__":
    try:
        has_collisions, has_penetration = create_diagnostic_test()
        
        if has_collisions and not has_penetration:
            print("✅ 结论: 碰撞检测正常，无穿透问题")
        elif has_collisions and has_penetration:
            print("⚠️ 结论: 碰撞检测工作，但仍有穿透")
        else:
            print("❌ 结论: 碰撞检测可能存在问题")
            
    except KeyboardInterrupt:
        print("\n⏹️ 诊断被用户中断")
    except Exception as e:
        print(f"\n❌ 诊断出错: {e}")
        import traceback
        traceback.print_exc()
