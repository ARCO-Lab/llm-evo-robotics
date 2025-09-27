#!/usr/bin/env python3
"""
强制显示MuJoCo窗口测试
"""

import gymnasium as gym
import time
import numpy as np
import os

def force_mujoco_window_test():
    """强制显示MuJoCo窗口"""
    print("🎮 强制MuJoCo窗口显示测试")
    print("💡 窗口将保持30秒，请仔细查看")
    
    # 设置环境变量确保渲染
    os.environ['MUJOCO_GL'] = 'glfw'  # 强制使用GLFW
    
    try:
        # 创建环境
        print("🌍 创建Reacher环境...")
        env = gym.make('Reacher-v5', render_mode='human')
        
        print("✅ 环境创建成功")
        print("🎯 开始长时间渲染测试...")
        print("   窗口标题应该是: MuJoCo")
        print("   如果没有看到窗口，请检查任务栏或其他桌面")
        
        # 重置环境
        obs, info = env.reset()
        
        # 运行30秒
        start_time = time.time()
        step = 0
        
        while time.time() - start_time < 30:
            step += 1
            
            # 慢速随机动作
            action = env.action_space.sample() * 0.5  # 减小动作幅度
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 10 == 0:
                elapsed = time.time() - start_time
                remaining = 30 - elapsed
                print(f"   运行中... {elapsed:.1f}s / 30s (剩余 {remaining:.1f}s)")
                print(f"   Step {step}: 奖励={reward:.3f}")
            
            # 慢一点让您看清楚
            time.sleep(0.1)
            
            if terminated or truncated:
                obs, info = env.reset()
                print("   环境重置")
        
        print("✅ 长时间渲染测试完成")
        env.close()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_with_trained_model():
    """使用训练好的模型测试"""
    print("\n🤖 使用训练好的模型测试")
    
    try:
        from stable_baselines3 import SAC
        from real_multi_joint_reacher import RealMultiJointWrapper
        
        # 设置渲染
        os.environ['MUJOCO_GL'] = 'glfw'
        
        print("🌍 创建多关节环境...")
        env = RealMultiJointWrapper(
            num_joints=2,
            link_lengths=[0.1, 0.1],
            render_mode='human'
        )
        
        print("📂 加载训练好的模型...")
        model = SAC.load("models/multi_joint_2j_sac", env=env)
        
        print("✅ 模型加载成功")
        print("🎯 开始智能控制演示 (60秒)...")
        print("   机器人将尝试到达目标位置")
        
        obs, info = env.reset()
        start_time = time.time()
        step = 0
        episode = 1
        
        while time.time() - start_time < 60:
            step += 1
            
            # 使用训练好的策略
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            distance = info.get('distance_to_target', float('inf'))
            
            if step % 20 == 0:
                elapsed = time.time() - start_time
                remaining = 60 - elapsed
                print(f"   Episode {episode}, Step {step}: 距离={distance:.3f}, 奖励={reward:.3f}")
                print(f"   运行时间: {elapsed:.1f}s / 60s (剩余 {remaining:.1f}s)")
            
            # 稍微快一点
            time.sleep(0.05)
            
            if terminated or truncated:
                episode += 1
                obs, info = env.reset()
                if step % 20 != 0:  # 避免重复打印
                    print(f"   Episode {episode-1} 结束，开始 Episode {episode}")
        
        print("✅ 智能控制演示完成")
        env.close()
        
    except Exception as e:
        print(f"❌ 智能控制测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🌟 强制MuJoCo窗口显示测试")
    print("💡 请仔细观察屏幕，窗口可能在任务栏或其他位置")
    print("=" * 60)
    
    # 测试1: 标准环境长时间显示
    force_mujoco_window_test()
    
    print("\n" + "=" * 60)
    
    # 测试2: 训练模型演示
    test_with_trained_model()
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成！")
    print("💡 如果仍然没有看到窗口，可能是以下原因：")
    print("   1. 窗口被其他程序遮挡")
    print("   2. 窗口在其他虚拟桌面")
    print("   3. MuJoCo使用了离屏渲染")
    print("   4. 图形驱动问题")

if __name__ == "__main__":
    main()


