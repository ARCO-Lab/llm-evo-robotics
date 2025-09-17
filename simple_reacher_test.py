#!/usr/bin/env python3
"""
最简单的reacher2d渲染测试
"""
import os
import sys
import pygame
import time

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'examples', '2d_reacher', 'envs'))

print("🤖 最简单的reacher2d渲染测试...")

try:
    from reacher2d_env import Reacher2DEnv
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=3,
        link_lengths=[60.0, 40.0, 30.0],
        render_mode='human'
    )
    
    print("✅ 环境创建成功")
    print(f"📐 窗口大小: {env.screen.get_size()}")
    
    # 设置窗口标题更明显
    pygame.display.set_caption("🚨 REACHER2D 测试 - 如果看到请按SPACE 🚨")
    
    # 重置环境
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    print("🎨 开始渲染测试...")
    print("🔍 寻找标题为 '🚨 REACHER2D 测试' 的窗口")
    print("⚠️ 如果看到窗口，请按SPACE键确认")
    
    for i in range(50):  # 只运行50步
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
        
        # 渲染
        env.render()
        
        # 检查事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("🚪 用户关闭窗口")
                env.close()
                exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("🎉 用户确认看到reacher2d渲染！")
                    print("✅ 渲染测试成功！")
                    env.close()
                    exit(0)
                elif event.key == pygame.K_ESCAPE:
                    print("⏹️ 用户按ESC退出")
                    env.close()
                    exit(0)
        
        print(f"📊 步骤 {i+1}/50")
        time.sleep(0.1)  # 10 FPS，更容易观察
        
        if terminated or truncated:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
    
    print("⚠️ 测试完成，但未收到用户确认")
    print("💡 窗口可能被隐藏或在其他位置")
    env.close()
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

