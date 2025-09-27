#!/usr/bin/env python3
"""
测试简化3关节环境的FPS和渲染性能
对比标准Reacher
"""

import time
import numpy as np
from simple_3joint_reacher import Simple3JointReacherEnv
import gymnasium as gym

def test_3joint_fps():
    """测试3关节环境的FPS"""
    print("📊 测试简化3关节环境FPS")
    
    env = Simple3JointReacherEnv(render_mode='human')
    obs, info = env.reset()
    
    print("🎯 测量100步的FPS...")
    
    num_steps = 100
    start_time = time.time()
    
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 每20步报告一次
        if step % 20 == 0 and step > 0:
            elapsed = time.time() - start_time
            current_fps = step / elapsed
            print(f"   Step {step}: FPS = {current_fps:.1f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    total_time = time.time() - start_time
    final_fps = num_steps / total_time
    
    print(f"\n📈 3关节环境FPS结果:")
    print(f"   总步数: {num_steps}")
    print(f"   总时间: {total_time:.2f}秒")
    print(f"   平均FPS: {final_fps:.1f}")
    print(f"   每步时间: {total_time/num_steps*1000:.1f}ms")
    
    env.close()
    return final_fps

def test_standard_reacher_fps():
    """测试标准Reacher的FPS作为对比"""
    print("\n📊 测试标准Reacher FPS (对比)")
    
    env = gym.make('Reacher-v5', render_mode='human')
    obs, info = env.reset()
    
    print("🎯 测量100步的FPS...")
    
    num_steps = 100
    start_time = time.time()
    
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 每20步报告一次
        if step % 20 == 0 and step > 0:
            elapsed = time.time() - start_time
            current_fps = step / elapsed
            print(f"   Step {step}: FPS = {current_fps:.1f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    total_time = time.time() - start_time
    final_fps = num_steps / total_time
    
    print(f"\n📈 标准Reacher FPS结果:")
    print(f"   总步数: {num_steps}")
    print(f"   总时间: {total_time:.2f}秒")
    print(f"   平均FPS: {final_fps:.1f}")
    print(f"   每步时间: {total_time/num_steps*1000:.1f}ms")
    
    env.close()
    return final_fps

def test_joint_movement():
    """测试关节运动是否明显"""
    print("\n🔧 测试3关节运动")
    
    env = Simple3JointReacherEnv(render_mode='human')
    obs, info = env.reset()
    
    print("🎯 大幅度动作测试 (每步2秒)")
    
    # 测试大幅度动作
    test_actions = [
        [1.0, 0.0, 0.0],   # 只动第1关节
        [0.0, 1.0, 0.0],   # 只动第2关节
        [0.0, 0.0, 1.0],   # 只动第3关节
        [1.0, 1.0, 0.0],   # 前两关节
        [0.0, 1.0, 1.0],   # 后两关节
        [1.0, 1.0, 1.0],   # 所有关节
        [-1.0, -1.0, -1.0] # 反向所有关节
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n🔧 动作 {i+1}: {action}")
        
        # 记录前状态
        prev_obs = obs.copy()
        prev_angles = np.arctan2(prev_obs[3:6], prev_obs[0:3])
        
        # 执行动作
        start_time = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - start_time
        
        # 计算变化
        new_angles = np.arctan2(obs[3:6], obs[0:3])
        angle_changes = new_angles - prev_angles
        
        print(f"   执行时间: {step_time*1000:.1f}ms")
        print(f"   角度变化: {np.degrees(angle_changes):.1f}度")
        print(f"   关节速度: {obs[6:9]:.3f}")
        print(f"   末端位置: {obs[9:11]:.3f}")
        print(f"   距离目标: {info['distance_to_target']:.3f}")
        
        # 暂停观察
        print("   (暂停2秒观察渲染...)")
        time.sleep(2.0)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

def main():
    """主函数"""
    print("🌟 简化3关节环境性能测试")
    print("💡 对比标准Reacher，检查FPS和渲染问题是否解决")
    print()
    
    try:
        # 1. 测试3关节FPS
        fps_3joint = test_3joint_fps()
        
        # 2. 测试标准Reacher FPS
        fps_standard = test_standard_reacher_fps()
        
        # 3. 对比结果
        print(f"\n🔍 FPS对比结果:")
        print(f"   简化3关节: {fps_3joint:.1f} FPS")
        print(f"   标准Reacher: {fps_standard:.1f} FPS")
        print(f"   差异: {abs(fps_3joint - fps_standard):.1f} FPS")
        
        if abs(fps_3joint - fps_standard) < 20:
            print("✅ FPS差异在正常范围内")
        else:
            print("⚠️ FPS差异较大，可能仍有问题")
        
        # 4. 测试关节运动
        print("\n" + "="*50)
        print("准备测试关节运动...")
        print("如果FPS正常，按Enter继续关节运动测试")
        print("如果仍有问题，按Ctrl+C退出")
        print("="*50)
        input("按Enter继续...")
        
        test_joint_movement()
        
        print(f"\n🎉 所有测试完成！")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


