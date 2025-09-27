#!/usr/bin/env python3
"""
诊断 SAC 无法到达目标的具体问题
"""

import sys
import os
import numpy as np
import torch
import time
from collections import deque

# 添加路径
sys.path.append('examples/surrogate_model/sac')
sys.path.append('examples/2d_reacher/envs')

from sb3_sac_adapter import SB3SACFactory
from reacher_env_factory import create_reacher_env

def diagnose_sac_issues():
    print("🔍 诊断 SAC 无法到达目标的问题")
    print("🎯 重点观察:")
    print("   1. 两个关节的动作范围和变化")
    print("   2. 末端执行器的实际移动距离")
    print("   3. 奖励函数的各个组成部分")
    print("   4. 是否存在物理限制")
    print("=" * 70)
    
    # 创建环境
    env = create_reacher_env(version='mujoco', render_mode='human')
    
    # 创建 SAC
    sac = SB3SACFactory.create_reacher_sac(
        action_dim=2,
        buffer_capacity=3000,
        batch_size=64,
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
    print("=" * 70)
    
    # 诊断统计
    joint_actions = {'joint1': [], 'joint2': []}
    end_effector_positions = []
    distances = []
    rewards_breakdown = []
    
    print("🎯 开始诊断测试...")
    start_time = time.time()
    
    for step in range(500):  # 运行500步进行诊断
        # 获取动作
        action = sac.get_action(obs, deterministic=False)
        joint_actions['joint1'].append(action[0].item())
        joint_actions['joint2'].append(action[1].item())
        
        # 执行动作
        step_result = env.step(action.cpu().numpy())
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        
        # 记录末端执行器位置和距离
        end_pos = obs[4:6] if len(obs) > 6 else [0, 0]
        goal_pos = obs[6:8] if len(obs) > 8 else [0, 0]
        distance = obs[8] if len(obs) > 8 else 999
        
        end_effector_positions.append(end_pos.copy())
        distances.append(distance)
        
        # 记录奖励分解（如果环境提供）
        if hasattr(env, 'reward_components'):
            rewards_breakdown.append(env.reward_components.copy())
        
        # 每100步进行详细分析
        if (step + 1) % 100 == 0:
            print(f"\n📊 诊断报告 - Step {step + 1}/500")
            print("=" * 50)
            
            # 分析关节动作
            recent_joint1 = joint_actions['joint1'][-100:]
            recent_joint2 = joint_actions['joint2'][-100:]
            
            joint1_range = [min(recent_joint1), max(recent_joint1)]
            joint2_range = [min(recent_joint2), max(recent_joint2)]
            joint1_std = np.std(recent_joint1)
            joint2_std = np.std(recent_joint2)
            
            print(f"🎮 关节动作分析:")
            print(f"   关节1: 范围 [{joint1_range[0]:.4f}, {joint1_range[1]:.4f}], 标准差 {joint1_std:.4f}")
            print(f"   关节2: 范围 [{joint2_range[0]:.4f}, {joint2_range[1]:.4f}], 标准差 {joint2_std:.4f}")
            
            if joint1_std < 0.001:
                print("   ⚠️ 关节1 动作变化很小")
            if joint2_std < 0.001:
                print("   ⚠️ 关节2 动作变化很小")
            
            # 分析末端执行器移动
            recent_positions = end_effector_positions[-100:]
            if len(recent_positions) > 1:
                position_changes = []
                for i in range(1, len(recent_positions)):
                    change = np.linalg.norm(np.array(recent_positions[i]) - np.array(recent_positions[i-1]))
                    position_changes.append(change)
                
                avg_movement = np.mean(position_changes)
                max_movement = max(position_changes)
                
                print(f"📍 末端执行器移动分析:")
                print(f"   平均每步移动: {avg_movement:.4f}")
                print(f"   最大单步移动: {max_movement:.4f}")
                print(f"   当前位置: [{end_pos[0]:.2f}, {end_pos[1]:.2f}]")
                print(f"   目标位置: [{goal_pos[0]:.2f}, {goal_pos[1]:.2f}]")
                
                if avg_movement < 0.001:
                    print("   ⚠️ 末端执行器几乎不移动")
            
            # 分析距离变化
            recent_distances = distances[-100:]
            distance_improvement = recent_distances[0] - recent_distances[-1]
            min_distance = min(recent_distances)
            
            print(f"📏 距离分析:")
            print(f"   当前距离: {distance:.2f}px")
            print(f"   最近100步最小距离: {min_distance:.2f}px")
            print(f"   距离改善: {distance_improvement:.2f}px")
            
            if abs(distance_improvement) < 1.0:
                print("   ⚠️ 距离没有明显改善")
            
            # 分析奖励分解
            if len(rewards_breakdown) > 0:
                recent_rewards = rewards_breakdown[-10:]
                if len(recent_rewards) > 0:
                    avg_components = {}
                    for key in recent_rewards[0].keys():
                        avg_components[key] = np.mean([r[key] for r in recent_rewards])
                    
                    print(f"🎁 奖励分解分析:")
                    for key, value in avg_components.items():
                        print(f"   {key}: {value:.3f}")
            
            print("=" * 50)
        
        # Episode 结束处理
        if terminated or truncated:
            print(f"🔄 Episode 结束，重置环境")
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
    
    # 最终诊断报告
    print("\n" + "=" * 70)
    print("🏥 最终诊断报告")
    print("=" * 70)
    
    # 关节动作总体分析
    joint1_overall_std = np.std(joint_actions['joint1'])
    joint2_overall_std = np.std(joint_actions['joint2'])
    joint1_range = [min(joint_actions['joint1']), max(joint_actions['joint1'])]
    joint2_range = [min(joint_actions['joint2']), max(joint_actions['joint2'])]
    
    print(f"🎮 关节动作总体分析:")
    print(f"   关节1: 范围 [{joint1_range[0]:.4f}, {joint1_range[1]:.4f}], 标准差 {joint1_overall_std:.4f}")
    print(f"   关节2: 范围 [{joint2_range[0]:.4f}, {joint2_range[1]:.4f}], 标准差 {joint2_overall_std:.4f}")
    
    # 距离分析
    min_distance_achieved = min(distances)
    avg_distance = np.mean(distances)
    distance_std = np.std(distances)
    
    print(f"📏 距离总体分析:")
    print(f"   最小距离: {min_distance_achieved:.2f}px")
    print(f"   平均距离: {avg_distance:.2f}px")
    print(f"   距离标准差: {distance_std:.2f}px")
    
    # 问题诊断
    print(f"\n🔍 问题诊断:")
    issues = []
    
    if joint1_overall_std < 0.002:
        issues.append("关节1 探索不足")
    if joint2_overall_std < 0.002:
        issues.append("关节2 探索不足")
    if min_distance_achieved > 50:
        issues.append("从未接近目标")
    if distance_std < 5:
        issues.append("距离变化很小，可能陷入局部最优")
    
    if len(issues) == 0:
        print("   ✅ 未发现明显问题，可能需要更长时间训练")
    else:
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. ❌ {issue}")
    
    # 建议
    print(f"\n💡 改进建议:")
    if joint2_overall_std < 0.002:
        print("   1. 增加第二个关节的探索噪声")
    if min_distance_achieved > 30:
        print("   2. 检查扭矩是否足够大，能否产生有效运动")
    if distance_std < 5:
        print("   3. 增加熵系数，鼓励更多探索")
        print("   4. 调整奖励函数，提供更好的学习信号")
    
    print(f"\n✅ 诊断完成!")
    env.close()
    
    return {
        'min_distance': min_distance_achieved,
        'avg_distance': avg_distance,
        'joint1_std': joint1_overall_std,
        'joint2_std': joint2_overall_std,
        'issues': issues
    }

if __name__ == "__main__":
    results = diagnose_sac_issues()
    print(f"\n📋 诊断结果总结:")
    print(f"   最小距离: {results['min_distance']:.2f}px")
    print(f"   平均距离: {results['avg_distance']:.2f}px")
    print(f"   关节1探索: {results['joint1_std']:.4f}")
    print(f"   关节2探索: {results['joint2_std']:.4f}")
    print(f"   发现问题: {len(results['issues'])}个")


