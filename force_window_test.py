#!/usr/bin/env python3
"""
强制显示pygame窗口的测试
"""
import os
import sys
import time

# 设置pygame显示
os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'  # 强制窗口位置
os.environ['SDL_VIDEO_CENTERED'] = '1'  # 居中显示

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..', '2d_reacher', 'envs'))

try:
    import pygame
    from reacher2d_env import Reacher2DEnv
    
    print("🎮 强制窗口显示测试...")
    print("🔧 设置pygame显示环境...")
    
    # 初始化pygame
    pygame.init()
    pygame.display.init()
    
    # 创建一个简单的测试窗口
    print("🪟 创建测试窗口...")
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("🤖 Reacher2D 渲染测试 - 如果看到这个窗口请按SPACE键")
    
    # 填充背景
    screen.fill((50, 50, 50))  # 深灰色背景
    
    # 添加文字
    font = pygame.font.Font(None, 36)
    text1 = font.render("Reacher2D Rendering Test", True, (255, 255, 255))
    text2 = font.render("Press SPACE if you can see this window", True, (255, 255, 0))
    text3 = font.render("Press ESC to exit", True, (255, 100, 100))
    
    screen.blit(text1, (200, 250))
    screen.blit(text2, (150, 300))
    screen.blit(text3, (300, 350))
    
    pygame.display.flip()
    
    print("🎯 如果您能看到窗口，请按SPACE键确认")
    print("⏳ 等待用户输入... (10秒后自动继续)")
    
    # 等待用户输入或超时
    start_time = time.time()
    user_confirmed = False
    
    while time.time() - start_time < 10:  # 10秒超时
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("🚪 用户关闭窗口")
                pygame.quit()
                exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("✅ 用户确认看到窗口！")
                    user_confirmed = True
                    break
                elif event.key == pygame.K_ESCAPE:
                    print("⏹️ 用户按ESC退出")
                    pygame.quit()
                    exit(0)
        
        if user_confirmed:
            break
            
        time.sleep(0.1)
    
    if user_confirmed:
        print("🎉 窗口显示正常！开始reacher2d测试...")
        
        # 现在测试reacher2d
        env = Reacher2DEnv(
            num_links=3,
            link_lengths=[60.0, 40.0, 30.0],
            render_mode='human'
        )
        
        obs, info = env.reset()
        
        for i in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            # 检查事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    env.close()
                    pygame.quit()
                    exit(0)
            
            time.sleep(0.1)
            
            if terminated or truncated:
                obs, info = env.reset()
        
        env.close()
        print("✅ Reacher2D渲染测试完成")
        
    else:
        print("⚠️ 10秒内未收到确认，可能窗口不可见")
        print("💡 建议：")
        print("   1. 检查任务栏是否有pygame窗口")
        print("   2. 尝试Alt+Tab切换窗口")
        print("   3. 如果是SSH连接，确保启用了X11转发")
    
    pygame.quit()
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()


