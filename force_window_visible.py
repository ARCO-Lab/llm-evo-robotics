#!/usr/bin/env python3
"""
强制显示pygame窗口到前台
"""
import os
import sys
import pygame
import time

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'examples', '2d_reacher', 'envs'))

# 设置窗口环境变量
os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'  # 固定位置
os.environ['SDL_VIDEO_CENTERED'] = '1'          # 居中

print("🚨 强制显示reacher2d窗口测试")

try:
    from reacher2d_env import Reacher2DEnv
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=3,
        link_lengths=[60.0, 40.0, 30.0],
        render_mode='human'
    )
    
    # 强制设置窗口属性
    pygame.display.set_caption("🔥🔥🔥 REACHER2D - 请看这里！🔥🔥🔥")
    
    # 尝试强制窗口到前台（Linux特有）
    try:
        import subprocess
        # 获取pygame窗口ID并尝试置顶
        subprocess.run(['wmctrl', '-a', 'REACHER2D'], capture_output=True, timeout=1)
    except:
        pass
    
    print("🔥 创建了一个**非常明显**的窗口")
    print("🔍 窗口标题: '🔥🔥🔥 REACHER2D - 请看这里！🔥🔥🔥'")
    print("📍 窗口应该在屏幕左上角位置(100,100)")
    
    # 重置环境
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    print("🎨 开始超明显的渲染测试...")
    
    for i in range(20):  # 只运行20步
        # 随机动作
        action = env.action_space.sample()
        
        # 执行步骤
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        
        # 渲染前清除事件队列，确保窗口响应
        pygame.event.pump()
        
        # 渲染
        env.render()
        
        # 强制刷新显示
        pygame.display.update()
        
        # 检查事件
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                print("🚪 用户关闭窗口")
                env.close()
                exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("🎉🎉🎉 成功！用户看到了reacher2d窗口！")
                    env.close()
                    exit(0)
                elif event.key == pygame.K_ESCAPE:
                    print("⏹️ 用户按ESC退出")
                    env.close()
                    exit(0)
        
        print(f"🔥 步骤 {i+1}/20 - 寻找火焰窗口！")
        time.sleep(0.2)  # 5 FPS，给足够时间观察
        
        if terminated or truncated:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
    
    print("⚠️ 测试完成")
    print("💭 如果您仍然没看到窗口，可能的原因：")
    print("   1. 使用远程桌面/SSH，需要X11转发")
    print("   2. 窗口管理器限制")
    print("   3. 多显示器配置问题")
    print("   4. 虚拟环境显示问题")
    
    # 保持窗口打开一段时间
    print("🔍 窗口将保持打开5秒钟...")
    for i in range(50):
        pygame.event.pump()
        env.render()
        pygame.display.update()
        time.sleep(0.1)
    
    env.close()
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()


