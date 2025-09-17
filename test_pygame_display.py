#!/usr/bin/env python3
"""
测试pygame显示是否正常工作
"""
import os
import sys
import pygame
import time
import math

print("🎮 开始pygame显示测试...")

# 设置显示环境
print("🔧 设置显示环境...")
os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'  # 强制窗口位置
os.environ['SDL_VIDEO_CENTERED'] = '1'  # 居中显示

# 检查DISPLAY环境变量
if 'DISPLAY' in os.environ:
    print(f"✅ DISPLAY环境变量: {os.environ['DISPLAY']}")
else:
    print("⚠️ 未设置DISPLAY环境变量（可能是SSH连接问题）")

try:
    # 初始化pygame
    print("🔧 初始化pygame...")
    pygame.init()
    
    # 检查可用显示驱动
    drivers = pygame.display.get_driver()
    print(f"🖥️ 当前显示驱动: {drivers}")
    
    # 创建窗口
    print("🪟 创建测试窗口...")
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("🧪 Pygame显示测试 - 如果看到这个窗口请按SPACE")
    
    # 创建时钟
    clock = pygame.time.Clock()
    
    print("✅ pygame窗口创建成功")
    print("🎯 如果您能看到窗口，请按SPACE键确认")
    print("⏳ 测试将运行10秒...")
    
    # 测试循环
    start_time = time.time()
    frame_count = 0
    user_confirmed = False
    
    while time.time() - start_time < 10:  # 运行10秒
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("🚪 用户关闭窗口")
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("✅ 用户确认看到窗口！")
                    user_confirmed = True
                elif event.key == pygame.K_ESCAPE:
                    print("⏹️ 用户按ESC退出")
                    pygame.quit()
                    sys.exit(0)
        
        # 绘制测试内容
        # 背景色渐变
        bg_color = int(128 + 127 * abs(math.sin(time.time() * 2)))
        screen.fill((bg_color, 50, 50))
        
        # 绘制移动的圆
        circle_x = int(400 + 200 * math.sin(time.time() * 3))
        circle_y = int(300 + 100 * math.cos(time.time() * 2))
        pygame.draw.circle(screen, (255, 255, 0), (circle_x, circle_y), 50)
        
        # 绘制文字
        font = pygame.font.Font(None, 36)
        texts = [
            "Pygame Display Test",
            f"Frame: {frame_count}",
            f"Time: {time.time() - start_time:.1f}s",
            "Press SPACE if you see this",
            "Press ESC to exit"
        ]
        
        for i, text in enumerate(texts):
            color = (255, 255, 255) if i != 3 else (255, 255, 0)
            text_surface = font.render(text, True, color)
            screen.blit(text_surface, (50, 50 + i * 40))
        
        # 更新显示
        pygame.display.flip()
        clock.tick(60)
        frame_count += 1
    
    # 测试结果
    if user_confirmed:
        print("🎉 pygame显示测试成功！")
        print("✅ 窗口可以正常显示和交互")
    else:
        print("⚠️ 测试完成，但未收到用户确认")
        print("💡 可能的原因：")
        print("   1. 窗口被其他窗口遮挡")
        print("   2. 窗口在其他显示器上")
        print("   3. SSH连接未启用X11转发")
        print("   4. 显示驱动问题")
    
    pygame.quit()
    
except Exception as e:
    print(f"❌ pygame测试失败: {e}")
    import traceback
    traceback.print_exc()
    
    # 尝试获取更多错误信息
    print("\n🔍 诊断信息:")
    try:
        import pygame
        print(f"pygame版本: {pygame.version.ver}")
        print(f"SDL版本: {pygame.version.SDL}")
    except:
        print("无法获取pygame版本信息")
    
    # 检查显示相关环境变量
    display_vars = ['DISPLAY', 'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE']
    for var in display_vars:
        value = os.environ.get(var, '未设置')
        print(f"{var}: {value}")
