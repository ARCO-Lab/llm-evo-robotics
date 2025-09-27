#!/usr/bin/env python3
"""
修复3关节渲染问题
强制使用真正的窗口渲染
"""

import os
import time
import numpy as np
import gymnasium as gym
from natural_3joint_reacher import Natural3JointReacherEnv

def force_window_rendering():
    """强制窗口渲染"""
    print("🎮 强制窗口渲染测试")
    
    # 设置环境变量强制使用GLFW窗口
    os.environ['MUJOCO_GL'] = 'glfw'
    os.environ['MUJOCO_RENDERER'] = 'glfw'
    
    print("✅ 设置渲染环境变量:")
    print(f"   MUJOCO_GL = {os.environ.get('MUJOCO_GL')}")
    print(f"   MUJOCO_RENDERER = {os.environ.get('MUJOCO_RENDERER')}")
    
    # 创建环境
    env = Natural3JointReacherEnv(render_mode='human')
    
    print("✅ 环境创建完成")
    
    # 手动初始化viewer
    try:
        # 强制创建viewer
        if hasattr(env, '_initialize_simulation'):
            env._initialize_simulation()
        
        # 尝试手动渲染
        obs, info = env.reset()
        
        print("🎯 开始强制渲染测试 (10步，每步2秒)")
        
        for step in range(10):
            print(f"\n📍 Step {step + 1}/10")
            
            # 大幅度动作让运动更明显
            if step < 3:
                action = [1.0, 0.0, 0.0]  # 只动第1关节
            elif step < 6:
                action = [0.0, 1.0, 0.0]  # 只动第2关节
            else:
                action = [0.0, 0.0, 1.0]  # 只动第3关节
            
            print(f"   动作: {action}")
            
            # 执行动作
            start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - start_time
            
            # 强制渲染
            render_start = time.time()
            try:
                render_result = env.render()
                render_time = time.time() - render_start
                print(f"   渲染结果: {type(render_result)}")
                print(f"   渲染时间: {render_time*1000:.1f}ms")
            except Exception as e:
                render_time = time.time() - render_start
                print(f"   渲染失败: {e}")
                print(f"   渲染时间: {render_time*1000:.1f}ms")
            
            # 显示状态
            joint_angles = np.arctan2(obs[3:6], obs[0:3])
            print(f"   关节角度: {np.degrees(joint_angles):.1f}度")
            print(f"   末端位置: {obs[9:11]:.3f}")
            print(f"   距离目标: {info['distance_to_target']:.3f}")
            print(f"   步骤时间: {step_time*1000:.1f}ms")
            
            # 长时间暂停让您观察
            print("   (暂停2秒让您观察窗口...)")
            time.sleep(2.0)
            
            if terminated or truncated:
                obs, info = env.reset()
                print("   环境重置")
        
        print("✅ 强制渲染测试完成")
        
    except Exception as e:
        print(f"❌ 强制渲染失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()

def test_standard_reacher_comparison():
    """对比标准Reacher的渲染"""
    print("\n🔍 对比标准Reacher渲染")
    
    # 设置渲染环境
    os.environ['MUJOCO_GL'] = 'glfw'
    
    try:
        # 创建标准Reacher
        print("📊 创建标准Reacher-v5...")
        standard_env = gym.make('Reacher-v5', render_mode='human')
        
        print("✅ 标准Reacher创建完成")
        
        obs, info = standard_env.reset()
        
        print("🎯 标准Reacher渲染测试 (5步)")
        
        for step in range(5):
            action = standard_env.action_space.sample()
            
            start_time = time.time()
            obs, reward, terminated, truncated, info = standard_env.step(action)
            step_time = time.time() - start_time
            
            print(f"   Step {step+1}: 时间={step_time*1000:.1f}ms, 奖励={reward:.3f}")
            
            time.sleep(1.0)
            
            if terminated or truncated:
                obs, info = standard_env.reset()
        
        standard_env.close()
        print("✅ 标准Reacher测试完成")
        
    except Exception as e:
        print(f"❌ 标准Reacher测试失败: {e}")

def main():
    """主函数"""
    print("🌟 修复3关节渲染问题")
    print("💡 强制使用真正的窗口渲染")
    print()
    
    try:
        # 1. 强制窗口渲染
        force_window_rendering()
        
        # 2. 对比标准Reacher
        test_standard_reacher_comparison()
        
        print(f"\n🎉 渲染修复测试完成！")
        print(f"💡 如果仍然没有看到窗口，可能需要检查:")
        print(f"   1. X11转发 (如果是SSH)")
        print(f"   2. 图形驱动")
        print(f"   3. OpenGL支持")
        
    except Exception as e:
        print(f"\n❌ 渲染修复失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


