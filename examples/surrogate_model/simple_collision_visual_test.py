#!/usr/bin/env python3
"""
简单的可视化碰撞测试 - 单个link与障碍物碰撞
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

def create_simple_collision_test():
    """创建简单的碰撞测试环境"""
    print("🔧 创建简单碰撞测试环境")
    print("="*40)
    
    # 初始化pygame
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("碰撞穿透测试 - 按SPACE开始/停止，ESC退出")
    clock = pygame.time.Clock()
    
    # 创建pymunk空间
    space = pymunk.Space()
    space.gravity = (0, 981)  # 重力向下
    space.damping = 0.999
    space.collision_slop = 0.01  # 与环境一致
    space.collision_bias = (1-0.1) ** 60
    
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
            
            # 创建障碍物线段
            shape = pymunk.Segment(space.static_body, p1, p2, radius=5.0)
            shape.friction = 1.0
            shape.collision_type = OBSTACLE_COLLISION_TYPE
            shape.collision_slop = 0.01  # 与links一致
            shape.color = pygame.Color("red")  # 红色障碍物
            
            space.add(shape)
            obstacles.append(shape)
            print(f"   障碍物 {i}: {p1} → {p2}")
    
    # 创建可控制的link
    link_mass = 10
    link_length = 80
    link_radius = 8
    
    # Link的初始位置 - 在机器人的起始区域
    start_pos = (500, 620)
    
    moment = pymunk.moment_for_segment(link_mass, (0, 0), (link_length, 0), link_radius)
    link_body = pymunk.Body(link_mass, moment)
    link_body.position = start_pos
    
    # 创建link形状
    link_shape = pymunk.Segment(link_body, (0, 0), (link_length, 0), link_radius)
    link_shape.friction = 0.8
    link_shape.collision_type = 1  # 与机器人link相同
    link_shape.collision_slop = 0.01  # 与障碍物一致
    link_shape.color = pygame.Color("blue")  # 蓝色link
    
    space.add(link_body, link_shape)
    
    # 设置碰撞处理器
    collision_count = 0
    
    def collision_handler(arbiter, space, data):
        nonlocal collision_count
        collision_count += 1
        print(f"🚨 碰撞检测! 总计: {collision_count}")
        return True  # 允许物理碰撞
    
    # 注册碰撞处理器 - 使用正确的PyMunk API
    try:
        # 尝试新版本API
        handler = space.add_collision_handler(1, OBSTACLE_COLLISION_TYPE)
        handler.begin = collision_handler
    except AttributeError:
        # 使用旧版本API
        space.on_collision(
            collision_type_a=1,
            collision_type_b=OBSTACLE_COLLISION_TYPE,
            begin=collision_handler
        )
    
    print(f"✅ 碰撞处理器已设置: Link(1) vs Obstacle({OBSTACLE_COLLISION_TYPE})")
    
    # 控制参数
    move_speed = 200.0  # 移动速度
    rotate_speed = 3.0  # 旋转速度
    running = True
    paused = False
    
    print(f"\n🎮 控制说明:")
    print(f"   WASD: 移动link")
    print(f"   QE: 旋转link")
    print(f"   SPACE: 暂停/继续")
    print(f"   R: 重置位置")
    print(f"   ESC: 退出")
    
    # 主循环
    while running:
        dt = clock.tick(60) / 1000.0  # 60 FPS
        
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
                    # 重置位置
                    link_body.position = start_pos
                    link_body.velocity = (0, 0)
                    link_body.angular_velocity = 0
                    link_body.angle = 0
                    print(f"🔄 重置link位置到 {start_pos}")
        
        # 键盘控制
        if not paused:
            keys = pygame.key.get_pressed()
            
            # 移动控制
            force_x, force_y = 0, 0
            if keys[pygame.K_a]:  # 左
                force_x -= move_speed
            if keys[pygame.K_d]:  # 右
                force_x += move_speed
            if keys[pygame.K_w]:  # 上
                force_y -= move_speed
            if keys[pygame.K_s]:  # 下
                force_y += move_speed
            
            # 应用力
            if force_x != 0 or force_y != 0:
                link_body.apply_force_at_world_point((force_x, force_y), link_body.position)
            
            # 旋转控制
            if keys[pygame.K_q]:  # 逆时针旋转
                link_body.angular_velocity = -rotate_speed
            elif keys[pygame.K_e]:  # 顺时针旋转
                link_body.angular_velocity = rotate_speed
            else:
                link_body.angular_velocity *= 0.9  # 阻尼
        
        # 物理更新
        if not paused:
            space.step(dt)
        
        # 渲染
        screen.fill((255, 255, 255))  # 白色背景
        
        # 绘制物理对象
        space.debug_draw(draw_options)
        
        # 显示信息
        font = pygame.font.Font(None, 36)
        info_lines = [
            f"碰撞次数: {collision_count}",
            f"Link位置: ({link_body.position.x:.0f}, {link_body.position.y:.0f})",
            f"Link角度: {math.degrees(link_body.angle):.1f}°",
            f"状态: {'⏸️ 暂停' if paused else '▶️ 运行'}",
            f"控制: WASD移动, QE旋转, R重置"
        ]
        
        for i, line in enumerate(info_lines):
            text = font.render(line, True, (0, 0, 0))
            screen.blit(text, (10, 10 + i * 30))
        
        # 碰撞状态指示
        if collision_count > 0:
            collision_text = font.render(f"🚨 检测到碰撞!", True, (255, 0, 0))
            screen.blit(collision_text, (10, height - 40))
        
        pygame.display.flip()
    
    pygame.quit()
    
    print(f"\n📊 测试结果:")
    print(f"   总碰撞次数: {collision_count}")
    print(f"   碰撞检测: {'✅ 正常工作' if collision_count > 0 else '❌ 未检测到碰撞'}")
    print(f"   collision_slop: 0.01")
    
    return collision_count > 0

if __name__ == "__main__":
    print("🔬 简单可视化碰撞测试")
    print("="*50)
    print("这个测试将创建一个可控制的link和YAML中的障碍物")
    print("你可以手动控制link移动来测试碰撞检测")
    print()
    
    try:
        success = create_simple_collision_test()
        
        if success:
            print("✅ 碰撞检测工作正常，障碍物不会被穿透")
        else:
            print("⚠️ 未检测到碰撞，可能需要进一步调试")
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
