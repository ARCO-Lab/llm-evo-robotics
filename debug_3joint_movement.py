#!/usr/bin/env python3
"""
调试3关节运动和渲染问题
检查关节是否真的在动，以及FPS为什么这么高
"""

import time
import numpy as np
from natural_3joint_reacher import Natural3JointReacherEnv

def debug_joint_movement():
    """调试关节运动"""
    print("🔍 调试3关节运动")
    
    # 创建环境
    env = Natural3JointReacherEnv(render_mode='human')
    
    print("✅ 环境创建完成")
    print(f"   观察空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    
    # 重置环境
    obs, info = env.reset()
    print(f"\n📊 初始观察:")
    print(f"   观察维度: {obs.shape}")
    print(f"   cos角度: {obs[0:3]}")  # cos0, cos1, cos2
    print(f"   sin角度: {obs[3:6]}")  # sin0, sin1, sin2
    print(f"   关节速度: {obs[6:9]}")  # vel0, vel1, vel2
    print(f"   末端位置: {obs[9:11]}")  # ee_x, ee_y
    print(f"   目标位置: {obs[11:13]}")  # target_x, target_y
    
    print(f"\n🎯 测试关节运动 (慢速，每步暂停1秒)")
    
    # 测试每个关节单独运动
    test_actions = [
        [1.0, 0.0, 0.0],   # 只动第1个关节
        [0.0, 1.0, 0.0],   # 只动第2个关节  
        [0.0, 0.0, 1.0],   # 只动第3个关节
        [-1.0, 0.0, 0.0],  # 反向第1个关节
        [0.0, -1.0, 0.0],  # 反向第2个关节
        [0.0, 0.0, -1.0],  # 反向第3个关节
        [1.0, 1.0, 1.0],   # 所有关节正向
        [-1.0, -1.0, -1.0] # 所有关节反向
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n🔧 测试动作 {i+1}: {action}")
        
        # 记录执行前的状态
        prev_obs = obs.copy()
        prev_joint_angles = np.arctan2(prev_obs[3:6], prev_obs[0:3])  # 从cos,sin计算角度
        
        # 执行动作
        start_time = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - start_time
        
        # 记录执行后的状态
        new_joint_angles = np.arctan2(obs[3:6], obs[0:3])
        angle_changes = new_joint_angles - prev_joint_angles
        
        print(f"   执行时间: {step_time*1000:.1f}ms")
        print(f"   关节角度变化: {np.degrees(angle_changes)}度")
        print(f"   关节速度: {obs[6:9]}")
        print(f"   末端位置变化: {obs[9:11] - prev_obs[9:11]}")
        print(f"   奖励: {reward:.3f}")
        print(f"   距离: {info['distance_to_target']:.3f}")
        
        # 暂停让您观察
        print("   (暂停1秒让您观察渲染...)")
        time.sleep(1.0)
        
        if terminated or truncated:
            obs, info = env.reset()
            print("   环境已重置")
    
    env.close()
    print("✅ 关节运动测试完成")

def debug_fps_issue():
    """调试FPS问题"""
    print("\n🔍 调试FPS问题")
    
    # 创建环境
    env = Natural3JointReacherEnv(render_mode='human')
    obs, info = env.reset()
    
    print("🎯 手动测量真实FPS")
    
    num_steps = 100
    start_time = time.time()
    
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 20 == 0:
            elapsed = time.time() - start_time
            current_fps = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"   Step {step}: 当前FPS = {current_fps:.1f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    total_time = time.time() - start_time
    actual_fps = num_steps / total_time
    
    print(f"\n📊 FPS测量结果:")
    print(f"   总步数: {num_steps}")
    print(f"   总时间: {total_time:.2f}秒")
    print(f"   实际FPS: {actual_fps:.1f}")
    print(f"   每步平均时间: {total_time/num_steps*1000:.1f}ms")
    
    env.close()

def debug_rendering_backend():
    """调试渲染后端"""
    print("\n🔍 调试渲染后端")
    
    import mujoco
    
    # 检查MuJoCo版本和渲染器
    print(f"📊 MuJoCo信息:")
    print(f"   MuJoCo版本: {mujoco.__version__}")
    
    # 创建环境并检查渲染器
    env = Natural3JointReacherEnv(render_mode='human')
    
    print(f"   渲染模式: {env.render_mode}")
    print(f"   MuJoCo模型: {hasattr(env, 'model')}")
    print(f"   MuJoCo数据: {hasattr(env, 'data')}")
    
    if hasattr(env, 'viewer') and env.viewer is not None:
        print(f"   Viewer类型: {type(env.viewer)}")
    else:
        print(f"   Viewer: None")
    
    # 尝试手动渲染一帧
    print(f"\n🎮 手动渲染测试:")
    obs, info = env.reset()
    
    render_start = time.time()
    try:
        # 尝试获取渲染图像
        rgb_array = env.render()
        render_time = time.time() - render_start
        
        if rgb_array is not None:
            print(f"   渲染成功: {rgb_array.shape if hasattr(rgb_array, 'shape') else type(rgb_array)}")
            print(f"   渲染时间: {render_time*1000:.1f}ms")
        else:
            print(f"   渲染返回None")
            print(f"   渲染时间: {render_time*1000:.1f}ms")
            
    except Exception as e:
        render_time = time.time() - render_start
        print(f"   渲染失败: {e}")
        print(f"   渲染时间: {render_time*1000:.1f}ms")
    
    env.close()

def main():
    """主函数"""
    print("🌟 3关节运动和渲染调试")
    print("💡 检查关节是否真的在动，以及FPS异常问题")
    print()
    
    try:
        # 1. 调试关节运动
        debug_joint_movement()
        
        # 2. 调试FPS问题
        debug_fps_issue()
        
        # 3. 调试渲染后端
        debug_rendering_backend()
        
        print(f"\n🎉 调试完成！")
        
    except Exception as e:
        print(f"\n❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
