#!/usr/bin/env python3
"""
专门测试reacher2d环境的渲染
"""
import os
import sys
import pygame
import time

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'examples', '2d_reacher', 'envs'))

print("🤖 开始reacher2d渲染测试...")

try:
    from reacher2d_env import Reacher2DEnv
    
    print("✅ 成功导入reacher2d环境")
    
    # 创建环境 - 强制启用渲染
    print("🔧 创建reacher2d环境...")
    env = Reacher2DEnv(
        num_links=3,
        link_lengths=[60.0, 40.0, 30.0],
        render_mode='human'  # 强制启用human模式
    )
    
    print("✅ reacher2d环境创建成功")
    print(f"🔍 render_mode: {env.render_mode}")
    print(f"🔍 screen对象: {hasattr(env, 'screen')}")
    
    # 重置环境
    print("🔄 重置环境...")
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}
    
    print("✅ 环境重置完成")
    print(f"🔍 重置后screen对象: {hasattr(env, 'screen')}")
    
    if hasattr(env, 'screen') and env.screen:
        print("✅ pygame screen已创建")
        
        # 检查窗口标题
        caption = pygame.display.get_caption()[0]
        print(f"🪟 窗口标题: {caption}")
        
        # 检查窗口大小
        size = env.screen.get_size()
        print(f"📐 窗口大小: {size}")
        
    else:
        print("❌ pygame screen未创建")
    
    print("🎬 开始渲染测试循环...")
    print("⚠️ 如果看到窗口，请按ESC退出")
    
    for step in range(100):
        # 随机动作
        action = env.action_space.sample()
        
        # 执行步骤
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 强制调用渲染
        print(f"🎨 第{step}步：调用render()...")
        env.render()
        
        # 检查pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("🚪 用户关闭窗口")
                env.close()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("⏹️ 用户按ESC退出")
                    env.close()
                    sys.exit(0)
                elif event.key == pygame.K_SPACE:
                    print("✅ 用户确认看到reacher2d渲染！")
        
        # 打印一些调试信息
        if step % 10 == 0:
            end_pos = info.get('end_effector_pos', [0, 0])
            goal_pos = info.get('goal_pos', [0, 0])
            distance = info.get('distance_to_target', 0)
            print(f"📊 步骤{step}: 末端={end_pos}, 目标={goal_pos}, 距离={distance:.1f}")
        
        # 控制帧率
        time.sleep(0.05)  # 20 FPS
        
        if terminated or truncated:
            print("🔄 Episode结束，重置...")
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
    
    print("✅ reacher2d渲染测试完成")
    env.close()
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请检查reacher2d_env.py路径")
except Exception as e:
    print(f"❌ 测试错误: {e}")
    import traceback
    traceback.print_exc()
    
    # 尝试清理
    try:
        if 'env' in locals():
            env.close()
    except:
        pass
