#!/usr/bin/env python3
"""
简单的键盘测试 - 只测试WASD响应
"""

import pygame
import sys

def test_keyboard():
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("键盘测试 - 按WASD看反应")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    position = [200, 150]
    last_key = "无"
    
    print("🎮 键盘测试开始")
    print("点击窗口，然后按WASD键")
    print("ESC退出")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_w:
                    position[1] -= 10
                    last_key = "W (上)"
                    print("🔼 W键按下 - 向上移动")
                elif event.key == pygame.K_s:
                    position[1] += 10
                    last_key = "S (下)"
                    print("🔽 S键按下 - 向下移动")
                elif event.key == pygame.K_a:
                    position[0] -= 10
                    last_key = "A (左)"
                    print("◀️ A键按下 - 向左移动")
                elif event.key == pygame.K_d:
                    position[0] += 10
                    last_key = "D (右)"
                    print("▶️ D键按下 - 向右移动")
        
        # 渲染
        screen.fill((255, 255, 255))
        
        # 绘制移动的圆点
        pygame.draw.circle(screen, (0, 0, 255), position, 20)
        
        # 显示信息
        text1 = font.render(f"最后按键: {last_key}", True, (0, 0, 0))
        text2 = font.render(f"位置: ({position[0]}, {position[1]})", True, (0, 0, 0))
        text3 = font.render("按WASD移动蓝点", True, (0, 0, 0))
        
        screen.blit(text1, (10, 10))
        screen.blit(text2, (10, 50))
        screen.blit(text3, (10, 90))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    print("键盘测试完成")

if __name__ == "__main__":
    test_keyboard()
