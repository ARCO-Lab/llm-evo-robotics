#!/usr/bin/env python3
"""
最简单的碰撞检测测试 - 验证PyMunk 7.1.0 on_collision API
"""

import pymunk
import pygame

def test_minimal_collision():
    print("🔧 最简单的碰撞检测测试...")
    
    # 创建空间
    space = pymunk.Space()
    
    # 初始化pygame (简单版本)
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("最简碰撞测试")
    clock = pygame.time.Clock()
    
    # 创建两个会碰撞的球
    # 球1 - 动态
    body1 = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 10))
    body1.position = 200, 100
    body1.velocity = 0, 100  # 向下运动
    shape1 = pymunk.Circle(body1, 10)
    shape1.collision_type = 1
    space.add(body1, shape1)
    
    # 球2 - 静态
    shape2 = pymunk.Circle(space.static_body, 15, (200, 300))
    shape2.collision_type = 2
    space.add(shape2)
    
    collision_count = 0
    
    # 设置碰撞回调
    def collision_callback(arbiter, space, data):
        nonlocal collision_count
        collision_count += 1
        print(f"🚨 碰撞检测成功! 计数: {collision_count}")
        print(f"   碰撞对象: {arbiter.shapes[0].collision_type} vs {arbiter.shapes[1].collision_type}")
        return True
    
    # 使用on_collision API
    try:
        space.on_collision(
            collision_type_a=1, 
            collision_type_b=2, 
            pre_solve=collision_callback
        )
        print("✅ on_collision 设置成功")
    except Exception as e:
        print(f"❌ on_collision 设置失败: {e}")
        return
    
    # 运行物理模拟
    running = True
    steps = 0
    
    while running and steps < 100:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 物理步进
        space.step(1/60.0)
        steps += 1
        
        # 简单渲染
        screen.fill((255, 255, 255))
        
        # 绘制球1
        pos1 = int(body1.position.x), int(body1.position.y)
        pygame.draw.circle(screen, (255, 0, 0), pos1, 10)
        
        # 绘制球2
        pos2 = int(shape2.body.position.x + shape2.offset.x), int(shape2.body.position.y + shape2.offset.y)
        pygame.draw.circle(screen, (0, 0, 255), pos2, 15)
        
        # 显示信息
        font = pygame.font.Font(None, 36)
        text = font.render(f"碰撞: {collision_count}", True, (0, 0, 0))
        screen.blit(text, (10, 10))
        
        pygame.display.flip()
        clock.tick(60)
        
        if steps % 10 == 0:
            print(f"步骤 {steps}: 球1位置={body1.position.y:.1f}, 碰撞={collision_count}")
    
    print(f"\n📊 测试结果:")
    print(f"   总步数: {steps}")
    print(f"   总碰撞数: {collision_count}")
    
    if collision_count > 0:
        print("🎉 碰撞检测API工作正常!")
    else:
        print("❌ 碰撞检测API仍然失效!")
    
    pygame.quit()

if __name__ == "__main__":
    test_minimal_collision() 