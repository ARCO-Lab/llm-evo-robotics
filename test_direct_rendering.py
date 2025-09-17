#!/usr/bin/env python3
"""
直接测试reacher2d环境的渲染功能
"""
import os
import sys
import time

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', '..', '2d_reacher', 'envs'))

try:
    from reacher2d_env import Reacher2DEnv
    import pygame
    import numpy as np
    
    print("🎮 开始直接渲染测试...")
    
    # 创建环境
    env_params = {
        'num_links': 4,
        'link_lengths': [79.7, 86.6, 91.6, 60.3],
        'render_mode': 'human',  # 强制人类模式
        'config_path': None
    }
    
    print(f"🤖 创建机器人: {env_params['num_links']}关节")
    env = Reacher2DEnv(**env_params)
    
    print("🔄 重置环境...")
    obs, info = env.reset()
    
    print("🎨 开始渲染循环...")
    print("⚠️ 如果看到窗口，请按ESC键退出")
    
    for step in range(100):  # 运行100步
        # 随机动作
        action = env.action_space.sample()
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 渲染
        env.render()
        
        # 检查pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("🚪 用户关闭窗口")
                env.close()
                exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("⏹️ 用户按ESC退出")
                    env.close()
                    exit(0)
        
        # 打印进度
        if step % 10 == 0:
            end_pos = info.get('end_effector_pos', [0, 0])
            goal_pos = info.get('goal_pos', [0, 0])
            distance = info.get('distance_to_target', 0)
            print(f"步骤 {step}: 末端位置={end_pos}, 目标={goal_pos}, 距离={distance:.1f}")
        
        # 控制帧率
        time.sleep(0.05)  # 20 FPS
        
        if terminated or truncated:
            print("🏁 Episode结束，重置...")
            obs, info = env.reset()
    
    print("✅ 渲染测试完成")
    env.close()
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请检查环境路径和依赖")
except Exception as e:
    print(f"❌ 运行错误: {e}")
    import traceback
    traceback.print_exc()


