#!/usr/bin/env python3
"""
测试渲染器问题诊断脚本
"""

import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)

print("🔍 开始诊断渲染器问题...")

try:
    # 1. 测试基础导入
    print("1️⃣ 测试基础导入...")
    sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
    from reacher2d_env import Reacher2DEnv
    print("   ✅ Reacher2DEnv 导入成功")
    
    from async_renderer import AsyncRenderer, StateExtractor
    print("   ✅ AsyncRenderer 导入成功")
    
    # 2. 测试环境创建
    print("2️⃣ 测试环境创建...")
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': 'human',
        'config_path': '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml',
        'debug_level': 'SILENT'
    }
    
    env = Reacher2DEnv(**env_params)
    print("   ✅ 环境创建成功")
    
    # 3. 测试异步渲染器创建
    print("3️⃣ 测试异步渲染器创建...")
    async_renderer = AsyncRenderer(env_params)
    print("   ✅ 异步渲染器创建成功")
    
    # 4. 测试启动渲染器
    print("4️⃣ 测试启动渲染器...")
    async_renderer.start()
    print(f"   ✅ 异步渲染器启动成功 (PID: {async_renderer.render_process.pid})")
    
    # 5. 测试环境reset
    print("5️⃣ 测试环境reset...")
    obs = env.reset()
    print("   ✅ 环境reset成功")
    
    # 6. 测试step
    print("6️⃣ 测试环境step...")
    import numpy as np
    action = np.array([0.1, 0.1, 0.1, 0.1])
    obs, reward, done, info = env.step(action)
    print(f"   ✅ 环境step成功，reward: {reward:.3f}")
    
    # 7. 清理
    print("7️⃣ 清理资源...")
    env.close()
    async_renderer.stop()
    print("   ✅ 清理完成")
    
    print("\n🎉 所有测试通过！渲染器应该正常工作")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
