#!/usr/bin/env python3
"""
最终疯狂旋转修复测试
"""

import sys
import os
import numpy as np
import torch
import time

# 添加路径
sys.path.append('examples/surrogate_model/sac')
sys.path.append('examples/2d_reacher/envs')

from sb3_sac_adapter import SB3SACFactory
from reacher_env_factory import create_reacher_env

def final_rotation_test():
    print("🏁 最终疯狂旋转修复测试")
    print("🔧 最激进修复: max_torque=0.01, 强制重置")
    print("🎯 目标: 彻底解决电风扇旋转问题")
    print("👁️ 请仔细观察 Reacher 的行为")
    
    # 创建环境
    env = create_reacher_env(version='mujoco', render_mode='human')
    
    # 创建 SAC
    sac = SB3SACFactory.create_reacher_sac(
        action_dim=2,
        buffer_capacity=500,
        batch_size=32,
        lr=1e-3,
        device='cpu'
    )
    sac.set_env(env)
    
    # 重置环境
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
    
    print("✅ 环境和 SAC 初始化完成")
    print(f"🎮 动作空间: {env.action_space}")
    print("=" * 60)
    
    # 统计信息
    velocity_history = []
    max_velocity_seen = 0
    episode_count = 0
    forced_resets = 0
    
    for step in range(200):  # 运行200步观察
        # 获取动作
        action = sac.get_action(obs, deterministic=False)
        
        # 记录重置前的状态
        prev_obs = obs.copy()
        
        # 执行动作
        step_result = env.step(action.cpu().numpy())
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        
        # 检查是否发生了强制重置
        if np.linalg.norm(obs - prev_obs) > 1.0:  # 观察值大幅变化可能表示重置
            forced_resets += 1
        
        # 获取关节速度
        joint_velocities = obs[2:4] if len(obs) > 4 else [0, 0]
        velocity_magnitude = np.linalg.norm(joint_velocities)
        velocity_history.append(velocity_magnitude)
        max_velocity_seen = max(max_velocity_seen, velocity_magnitude)
        
        # 获取距离信息
        distance = obs[8] if len(obs) > 8 else 999
        
        # 每40步打印一次状态
        if step % 40 == 0:
            recent_velocities = velocity_history[-20:] if len(velocity_history) >= 20 else velocity_history
            avg_recent_velocity = np.mean(recent_velocities)
            
            print(f"📊 Step {step:3d}: 动作=[{action[0]:.3f}, {action[1]:.3f}]")
            print(f"         速度: {velocity_magnitude:.2f} rad/s (平均: {avg_recent_velocity:.2f})")
            print(f"         距离: {distance:.1f}px, 强制重置: {forced_resets}次")
            
            # 速度状态判断
            if velocity_magnitude < 0.5:
                print(f"         🎉 速度极低 - 完全解决旋转问题!")
            elif velocity_magnitude < 2.0:
                print(f"         ✅ 速度正常 - 旋转问题基本解决")
            elif velocity_magnitude < 5.0:
                print(f"         ⚠️ 速度较高但可控")
            else:
                print(f"         🚨 速度过高 - 仍有旋转问题")
        
        # Episode 结束处理
        if terminated or truncated:
            episode_count += 1
            print(f"🔄 Episode {episode_count} 结束! 距离: {distance:.1f}px")
            
            # 重置
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
        
        # 短暂暂停，便于观察
        time.sleep(0.08)  # 80ms 暂停，便于人眼观察
    
    # 最终统计
    print("\n" + "=" * 60)
    print("🏆 最终修复测试结果:")
    print("=" * 60)
    
    avg_velocity = np.mean(velocity_history) if velocity_history else 0
    print(f"⚡ 最大关节速度: {max_velocity_seen:.2f} rad/s")
    print(f"📊 平均关节速度: {avg_velocity:.2f} rad/s")
    print(f"🔄 完成的Episodes: {episode_count}")
    print(f"🚨 强制重置次数: {forced_resets}")
    print(f"🎮 最终扭矩设置: {getattr(env, 'max_torque', 0.01):.3f}")
    
    # 修复效果判断
    print("\n" + "=" * 60)
    print("🔍 最终修复效果评估:")
    
    if max_velocity_seen < 1.0:
        print("🎉 完美! 疯狂旋转问题彻底解决!")
        print("✅ Reacher 动作已完全可控")
        success = True
    elif max_velocity_seen < 3.0:
        print("👍 优秀! 旋转速度大幅降低")
        print("✅ 基本解决了电风扇问题")
        success = True
    elif max_velocity_seen < 10.0:
        print("⚠️ 有改善，但仍需进一步优化")
        success = False
    else:
        print("❌ 修复效果有限，需要重新考虑方案")
        success = False
    
    # 人工观察确认
    print("\n" + "=" * 60)
    print("👁️ 请您最终确认:")
    print("1. Reacher 是否不再像电风扇疯狂旋转？")
    print("2. 您能否清楚看到机械臂的每个动作？")
    print("3. 机械臂是否表现出合理的控制行为？")
    
    if success:
        print("\n🎊 恭喜! 疯狂旋转问题已解决，可以开始正常的 SAC 训练了!")
    else:
        print("\n😔 仍需进一步调试，可能需要考虑其他解决方案")
    
    print("\n✅ 最终修复测试完成!")
    env.close()
    
    return success

if __name__ == "__main__":
    success = final_rotation_test()
    if success:
        print("\n🚀 准备开始正常的 SAC 训练...")
    else:
        print("\n🔧 需要继续调试旋转问题...")


