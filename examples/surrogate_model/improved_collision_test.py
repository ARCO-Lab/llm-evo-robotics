#!/usr/bin/env python3
"""
改进的可视化碰撞测试 - 修复控制问题
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

def create_improved_collision_test():
    """创建改进的碰撞测试环境"""
    print("🔧 创建改进的碰撞测试环境")
    print("="*40)
     
    # 初始化pygame
    pygame.init()
    width, height = 1000, 700
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("改进的碰撞测试 - 点击窗口获得焦点，然后使用WASD")
    clock = pygame.time.Clock()
    
    # 创建pymunk空间
    space = pymunk.Space()
    space.gravity = (0, 100)  # 减少重力影响
    space.damping = 0.95  # 增加阻尼
    space.collision_slop = 0.01
    
    # 创建绘制选项
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    # 加载障碍物
    yaml_path = '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml'
    obstacles_config = load_obstacles_from_yaml(yaml_path)
    
    print(f"✅ 加载了 {len(obstacles_config)} 个障碍物")
    
    # 创建障碍物
    obstacles = []
    OBSTACLE_COLLISION_TYPE = 100
    
    for i, obs in enumerate(obstacles_config):
        if obs["shape"] == "segment":
            p1 = tuple(obs["points"][0])
            p2 = tuple(obs["points"][1])
            
            shape = pymunk.Segment(space.static_body, p1, p2, radius=8.0)  # 增大半径便于看见
            shape.friction = 1.0
            shape.collision_type = OBSTACLE_COLLISION_TYPE
            shape.collision_slop = 0.01
            shape.color = pygame.Color("red")
            
            space.add(shape)
            obstacles.append(shape)
    
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
    link_shape.color = pygame.Color("blue")
    
    space.add(link_body, link_shape)
    
    # 碰撞处理器
    collision_count = 0
    last_collision_time = 0
    
    def collision_handler(arbiter, space, data):
        nonlocal collision_count, last_collision_time
        current_time = time.time()
        if current_time - last_collision_time > 0.1:  # 防止重复计数
            collision_count += 1
            last_collision_time = current_time
            print(f"🚨 碰撞检测! 总计: {collision_count}")
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
    
    # 控制参数
    move_force = 500.0  # 增大力度
    max_velocity = 300.0  # 最大速度限制
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
            
            # 移动控制 - 直接设置速度而不是施加力
            vel_x, vel_y = 0, 0
            force_applied = False
            
            if keys[pygame.K_a]:  # 左
                vel_x = -max_velocity
                force_applied = True
            if keys[pygame.K_d]:  # 右
                vel_x = max_velocity
                force_applied = True
            if keys[pygame.K_w]:  # 上
                vel_y = -max_velocity
                force_applied = True
            if keys[pygame.K_s]:  # 下
                vel_y = max_velocity
                force_applied = True
            
            # 应用速度
            if force_applied:
                link_body.velocity = (vel_x, vel_y)
                # 调试输出
                if pygame.K_a in keys_pressed or pygame.K_d in keys_pressed or pygame.K_w in keys_pressed or pygame.K_s in keys_pressed:
                    print(f"🎮 控制生效: 速度=({vel_x:.0f}, {vel_y:.0f})")
            else:
                # 阻尼
                link_body.velocity = (link_body.velocity[0] * 0.9, link_body.velocity[1] * 0.9)
            
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
        
        # 绘制边界
        pygame.draw.rect(screen, (100, 100, 100), (0, 0, width, height), 3)
        
        pygame.display.flip()
    
    pygame.quit()
    
    print(f"\n📊 最终测试结果:")
    print(f"   总碰撞次数: {collision_count}")
    print(f"   碰撞检测: {'✅ 正常工作' if collision_count > 0 else '❌ 未检测到碰撞'}")
    
    return collision_count > 0

if __name__ == "__main__":
    print("🔬 改进的可视化碰撞测试")
    print("="*50)
    
    try:
        success = create_improved_collision_test()
        
        if success:
            print("✅ 碰撞检测和控制都正常工作")
        else:
            print("⚠️ 测试完成，但未检测到碰撞")
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
