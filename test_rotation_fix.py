#!/usr/bin/env python3
"""
测试疯狂旋转修复效果
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

def test_rotation_fix():
    print("🔧 测试疯狂旋转修复效果")
    print("🎯 新参数: max_torque=0.5, velocity_threshold=2.0")
    print("👁️ 观察: Reacher 是否还像电风扇一样旋转")
    
    # 创建环境
    env = create_reacher_env(version='mujoco', render_mode='human')
    
    # 创建 SAC
    sac = SB3SACFactory.create_reacher_sac(
        action_dim=2,
        buffer_capacity=1000,
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
    print("=" * 60)
    
    # 统计信息
    rotation_detections = 0
    torque_adjustments = 0
    max_velocity_seen = 0
    episode_count = 0
    
    for step in range(300):  # 运行300步观察
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
        max_velocity_seen = max(max_velocity_seen, velocity_magnitude)
        
        # 获取距离信息
        distance = obs[8] if len(obs) > 8 else 999
        
        # 检查是否检测到疯狂旋转
        if hasattr(env, 'rotation_monitor'):
            if env.rotation_monitor['crazy_rotation_count'] > rotation_detections:
                rotation_detections = env.rotation_monitor['crazy_rotation_count']
                print(f"🚨 Step {step}: 检测到疯狂旋转! 计数: {rotation_detections}")
        
        # 检查是否调整了扭矩
        current_torque = getattr(env, 'max_torque', 0.5)
        if step == 0:
            initial_torque = current_torque
        elif current_torque < initial_torque:
            torque_adjustments += 1
            print(f"🔧 Step {step}: 扭矩自动调整为 {current_torque:.2f}")
            initial_torque = current_torque
        
        # 每50步打印一次状态
        if step % 50 == 0:
            print(f"📊 Step {step:3d}: 动作=[{action[0]:.2f}, {action[1]:.2f}], 速度={velocity_magnitude:.2f}, 距离={distance:.1f}px")
            
            # 人工观察提示
            if velocity_magnitude > 1.5:
                print(f"   ⚠️ 关节速度较高: {velocity_magnitude:.2f} rad/s")
            else:
                print(f"   ✅ 关节速度正常: {velocity_magnitude:.2f} rad/s")
        
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
        time.sleep(0.05)  # 50ms 暂停，便于人眼观察
    
    # 最终统计
    print("\n" + "=" * 60)
    print("🔍 疯狂旋转修复测试结果:")
    print("=" * 60)
    print(f"🌪️ 疯狂旋转检测次数: {rotation_detections}")
    print(f"🔧 扭矩自动调整次数: {torque_adjustments}")
    print(f"⚡ 最大关节速度: {max_velocity_seen:.2f} rad/s")
    print(f"🔄 完成的Episodes: {episode_count}")
    print(f"🎮 最终扭矩设置: {getattr(env, 'max_torque', 0.5):.2f}")
    
    # 人工观察结果询问
    print("\n" + "=" * 60)
    print("👁️ 请您观察并回答:")
    print("1. Reacher 是否还像电风扇一样疯狂旋转？")
    print("2. 旋转速度是否明显降低了？")
    print("3. 机械臂的动作是否更加可控？")
    
    if max_velocity_seen < 2.0:
        print("✅ 技术指标: 关节速度已控制在合理范围内")
    else:
        print("⚠️ 技术指标: 关节速度仍然较高，可能需要进一步调整")
    
    print("\n✅ 测试完成!")
    env.close()

if __name__ == "__main__":
    test_rotation_fix()


