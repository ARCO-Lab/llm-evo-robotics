#!/usr/bin/env python3
"""
修复版碰撞测试 - 增强WASD控制和调试
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

def create_fixed_collision_test():
    """创建修复版碰撞测试"""
    print("🔧 修复版碰撞测试")
    print("="*40)
    print("🎯 重点：确保WASD控制有效")
    
    # 初始化pygame
    pygame.init()
    width, height = 1000, 700
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("修复版碰撞测试 - 点击窗口获得焦点，然后使用WASD")
    clock = pygame.time.Clock()
    
    # 🎯 调整物理参数以便更好的控制
    space = pymunk.Space()
    space.gravity = (0.0, 200.0)    # 🔧 减少重力，便于控制
    space.damping = 0.95            # 🔧 增加阻尼，便于精确控制
    space.collision_slop = 0.01
    space.collision_bias = (1-0.1) ** 60
    
    print(f"✅ 修改后的物理参数:")
    print(f"   gravity: {space.gravity} (减少重力)")
    print(f"   damping: {space.damping} (增加阻尼)")
    print(f"   collision_slop: {space.collision_slop}")
    
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
            
            shape = pymunk.Segment(space.static_body, p1, p2, radius=5.0)
            shape.friction = 1.0
            shape.color = (255, 0, 0, 255)  # 红色障碍物
            shape.collision_type = OBSTACLE_COLLISION_TYPE
            shape.collision_slop = 0.01
            
            space.add(shape)
            obstacles.append(shape)
            print(f"   障碍物 {i}: {p1} → {p2}")
    
    # 🎯 创建更容易控制的Link
    link_mass = 50     # 🔧 减少质量，便于控制
    link_length = 80   # 🔧 增加长度，便于观察
    link_radius = 10   # 🔧 增加半径，便于观察
    
    moment = pymunk.moment_for_segment(link_mass, (0, 0), (link_length, 0), link_radius)
    test_body = pymunk.Body(link_mass, moment)
    test_body.position = (400, 300)  # 左上角起始位置
    
    test_shape = pymunk.Segment(test_body, (0, 0), (link_length, 0), link_radius)
    test_shape.friction = 0.8
    test_shape.collision_type = 1
    test_shape.collision_slop = 0.01
    test_shape.color = (0, 255, 0, 255)  # 绿色Link便于观察
    
    space.add(test_body, test_shape)
    
    print(f"✅ 创建测试Link:")
    print(f"   质量: {link_mass} (减轻便于控制)")
    print(f"   长度: {link_length}")
    print(f"   半径: {link_radius}")
    print(f"   起始位置: {test_body.position}")
    
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
    
    # 🎯 增强控制参数
    base_force = 1000.0     # 🔧 增大基础力
    boost_multiplier = 3.0  # 🔧 冲刺倍数
    max_velocity = 400.0    # 🔧 最大速度限制
    
    running = True
    paused = False
    dt = 1/60.0  # 标准60 FPS
    
    # 控制状态追踪
    keys_pressed = set()
    force_applied = False
    
    print(f"\n🎮 增强控制说明:")
    print(f"   1. 点击窗口获得焦点")
    print(f"   2. WASD: 移动Link (力: {base_force})")
    print(f"   3. SHIFT+WASD: 冲刺移动 (力: {base_force * boost_multiplier})")
    print(f"   4. QE: 旋转Link")
    print(f"   5. SPACE: 暂停/继续")
    print(f"   6. R: 重置位置")
    print(f"   7. ESC: 退出")
    
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 18)
    
    # 主循环
    step_count = 0
    while running:
        step_count += 1
        
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
                    test_body.position = (400, 300)
                    test_body.velocity = (0, 0)
                    test_body.angular_velocity = 0
                    test_body.angle = 0
                    collision_count = 0
                    print(f"🔄 重置位置")
                # 调试：显示按键
                print(f"🔧 按下: {pygame.key.name(event.key)}")
            elif event.type == pygame.KEYUP:
                keys_pressed.discard(event.key)
        
        # 控制
        if not paused:
            keys = pygame.key.get_pressed()
            
            # 检查是否按住SHIFT进行冲刺
            boost = boost_multiplier if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.0
            current_force = base_force * boost
            
            # 移动控制
            force_x, force_y = 0, 0
            force_applied = False
            
            if keys[pygame.K_a]:  # 左
                force_x = -current_force
                force_applied = True
            if keys[pygame.K_d]:  # 右
                force_x = current_force
                force_applied = True
            if keys[pygame.K_w]:  # 上
                force_y = -current_force
                force_applied = True
            if keys[pygame.K_s]:  # 下
                force_y = current_force
                force_applied = True
            
            # 应用力
            if force_applied:
                test_body.apply_force_at_world_point((force_x, force_y), test_body.position)
                # 调试输出
                if step_count % 30 == 0:  # 每半秒输出一次
                    print(f"🎮 施加力: ({force_x:.0f}, {force_y:.0f}), 冲刺: {boost > 1}")
            
            # 速度限制
            vel = test_body.velocity
            speed = math.sqrt(vel.x**2 + vel.y**2)
            if speed > max_velocity:
                scale = max_velocity / speed
                test_body.velocity = (vel.x * scale, vel.y * scale)
            
            # 旋转控制
            if keys[pygame.K_q]:  # 逆时针
                test_body.angular_velocity = -3.0
            elif keys[pygame.K_e]:  # 顺时针
                test_body.angular_velocity = 3.0
            else:
                test_body.angular_velocity *= 0.9  # 旋转阻尼
        
        # 物理更新
        if not paused:
            space.step(dt)
        
        # 渲染
        screen.fill((240, 240, 240))  # 浅灰色背景
        
        # 绘制物理对象
        space.debug_draw(draw_options)
        
        # 显示详细信息
        info_lines = [
            f"步数: {step_count}",
            f"碰撞次数: {collision_count}",
            f"Link位置: ({test_body.position.x:.0f}, {test_body.position.y:.0f})",
            f"Link速度: ({test_body.velocity.x:.0f}, {test_body.velocity.y:.0f})",
            f"速度大小: {math.sqrt(test_body.velocity.x**2 + test_body.velocity.y**2):.0f}",
            f"Link角度: {math.degrees(test_body.angle):.1f}°",
            f"状态: {'⏸️ 暂停' if paused else '▶️ 运行'}",
            f"施加力: {'✅ 是' if force_applied else '❌ 否'}",
            "",
            f"控制参数:",
            f"基础力: {base_force}",
            f"冲刺倍数: {boost_multiplier}",
            f"当前力: {base_force * (boost_multiplier if any(k in keys_pressed for k in [pygame.K_LSHIFT, pygame.K_RSHIFT]) else 1.0):.0f}",
        ]
        
        for i, line in enumerate(info_lines):
            color = (255, 0, 0) if "碰撞" in line and collision_count > 0 else (0, 0, 0)
            text = font.render(line, True, color)
            screen.blit(text, (10, 10 + i * 22))
        
        # 显示当前按键
        pressed_keys = [pygame.key.name(k) for k in keys_pressed if k in [
            pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d, 
            pygame.K_q, pygame.K_e, pygame.K_LSHIFT, pygame.K_RSHIFT
        ]]
        
        if pressed_keys:
            keys_text = small_font.render(f"当前按键: {', '.join(pressed_keys)}", True, (0, 100, 0))
            screen.blit(keys_text, (10, height - 80))
        
        # 控制指南
        guide_lines = [
            "🎮 控制指南:",
            "WASD: 基础移动",
            "SHIFT+WASD: 冲刺移动", 
            "QE: 旋转",
            "R: 重置, SPACE: 暂停"
        ]
        
        for i, line in enumerate(guide_lines):
            text = small_font.render(line, True, (100, 100, 100))
            screen.blit(text, (width - 200, 10 + i * 18))
        
        # 碰撞状态指示
        if collision_count > 0:
            collision_text = font.render(f"🚨 检测到 {collision_count} 次碰撞!", True, (255, 0, 0))
            screen.blit(collision_text, (10, height - 40))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    
    print(f"\n📊 测试结果:")
    print(f"   总物理步数: {step_count}")
    print(f"   总碰撞次数: {collision_count}")
    print(f"   碰撞检测: {'✅ 正常工作' if collision_count > 0 else '❌ 可能有问题'}")
    
    return collision_count > 0

if __name__ == "__main__":
    print("🔧 修复版可视化碰撞测试")
    print("="*50)
    print("重点解决WASD控制问题")
    print()
    
    try:
        success = create_fixed_collision_test()
        
        if success:
            print("✅ 碰撞检测和控制都正常工作")
        else:
            print("⚠️ 测试完成，检查控制是否有效")
            
    except KeyboardInterrupt:
        print("\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
