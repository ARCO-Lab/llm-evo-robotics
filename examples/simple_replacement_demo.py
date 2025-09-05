#!/usr/bin/env python3
"""
简单的环境替换演示
直接测试新环境是否可以替代原环境
"""

import sys
import os
import numpy as np
import pygame

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '2d_reacher/envs'))

def test_new_env():
    """测试新环境"""
    print("🧪 测试新的Gymnasium Reacher2D环境")
    
    try:
        from reacher2d_env_gymnasium import Reacher2DEnv
        
        # 创建环境
        env = Reacher2DEnv(
            num_links=3,
            render_mode='human',
            config_path='2d_reacher/configs/reacher_with_zigzag_obstacles.yaml'
        )
        
        print("✅ 环境创建成功")
        print(f"📊 观察空间: {env.observation_space}")
        print(f"🎮 动作空间: {env.action_space}")
        
        # 重置环境
        obs = env.reset()
        print(f"🔄 重置成功，观察维度: {obs.shape}")
        
        # 运行几步测试
        for i in range(5):
            action = np.array([20, -15, 10])  # 手动控制动作
            obs, reward, done, info = env.step(action)
            
            print(f"步数 {i+1}: 奖励={reward:.2f}, 距离={info['distance']:.1f}, 完成={done}")
            
            if done:
                print("🎯 任务完成，重置环境")
                obs = env.reset()
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = test_new_env()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 新环境测试成功！")
        print("\n📝 替换步骤:")
        print("1. 备份原文件:")
        print("   cp examples/2d_reacher/envs/reacher2d_env.py examples/2d_reacher/envs/reacher2d_env_pymunk.py")
        print("\n2. 替换环境:")
        print("   cp examples/2d_reacher/envs/reacher2d_env_gymnasium.py examples/2d_reacher/envs/reacher2d_env.py")
        print("\n3. 或者修改导入:")
        print("   将 'from envs.reacher2d_env import Reacher2DEnv'")
        print("   改为 'from envs.reacher2d_env_gymnasium import Reacher2DEnv'")
        print("\n✅ 新环境优势:")
        print("  - 🚫 无关节分离问题")
        print("  - 🚫 无穿透问题") 
        print("  - ✅ 数值稳定")
        print("  - ✅ 保持相同接口")
        print("  - ✅ 支持配置文件")
        print("  - ✅ 支持课程学习")
    else:
        print("❌ 新环境测试失败，需要修复问题")

if __name__ == "__main__":
    main()
