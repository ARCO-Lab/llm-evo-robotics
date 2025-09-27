#!/usr/bin/env python3
"""
紧急疯狂旋转修复测试
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

def emergency_rotation_test():
    print("🚨 紧急疯狂旋转修复测试")
    print("🔧 修复: 动作缩放逻辑错误")
    print("🎯 新参数: max_torque=0.1, 直接使用 custom_action")
    print("👁️ 关键观察: Reacher 是否不再像电风扇旋转")
    
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
    
    for step in range(150):  # 运行150步观察
        # 获取动作
        action = sac.get_action(obs, deterministic=False)
        
        # 执行动作
        step_result = env.step(action.cpu().numpy())
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        
        # 获取关节速度
        joint_velocities = obs[2:4] if len(obs) > 4 else [0, 0]
        velocity_magnitude = np.linalg.norm(joint_velocities)
        velocity_history.append(velocity_magnitude)
        max_velocity_seen = max(max_velocity_seen, velocity_magnitude)
        
        # 获取距离信息
        distance = obs[8] if len(obs) > 8 else 999
        
        # 每30步打印一次状态
        if step % 30 == 0:
            recent_velocities = velocity_history[-10:] if len(velocity_history) >= 10 else velocity_history
            avg_recent_velocity = np.mean(recent_velocities)
            
            print(f"📊 Step {step:3d}: 动作=[{action[0]:.3f}, {action[1]:.3f}]")
            print(f"         速度: {velocity_magnitude:.2f} rad/s (平均: {avg_recent_velocity:.2f})")
            print(f"         距离: {distance:.1f}px")
            
            # 速度状态判断
            if velocity_magnitude < 1.0:
                print(f"         ✅ 速度正常 - 不再疯狂旋转!")
            elif velocity_magnitude < 5.0:
                print(f"         ⚠️ 速度较高但可控")
            else:
                print(f"         🚨 速度过高 - 仍在疯狂旋转!")
        
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
        time.sleep(0.1)  # 100ms 暂停，便于人眼观察
    
    # 最终统计
    print("\n" + "=" * 60)
    print("🏆 紧急修复测试结果:")
    print("=" * 60)
    
    avg_velocity = np.mean(velocity_history) if velocity_history else 0
    print(f"⚡ 最大关节速度: {max_velocity_seen:.2f} rad/s")
    print(f"📊 平均关节速度: {avg_velocity:.2f} rad/s")
    print(f"🔄 完成的Episodes: {episode_count}")
    print(f"🎮 最终扭矩设置: {getattr(env, 'max_torque', 0.1):.2f}")
    
    # 修复效果判断
    print("\n" + "=" * 60)
    print("🔍 修复效果评估:")
    
    if max_velocity_seen < 2.0:
        print("🎉 优秀! 疯狂旋转问题已完全解决")
        print("✅ Reacher 动作已变得可控")
    elif max_velocity_seen < 10.0:
        print("👍 良好! 旋转速度大幅降低")
        print("⚠️ 可能需要进一步微调")
    else:
        print("❌ 修复失败! 仍然存在疯狂旋转")
        print("🔧 需要更激进的修复措施")
    
    # 人工观察确认
    print("\n" + "=" * 60)
    print("👁️ 请您确认观察结果:")
    print("1. Reacher 是否不再像电风扇一样疯狂旋转？")
    print("2. 机械臂的动作是否变得平缓可控？")
    print("3. 您是否能清楚看到机械臂的运动轨迹？")
    
    print("\n✅ 紧急修复测试完成!")
    env.close()

if __name__ == "__main__":
    emergency_rotation_test()


