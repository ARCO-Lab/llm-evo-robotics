#!/usr/bin/env python3
"""
测试简化配置的可达性
"""

import numpy as np
import sys
import os

sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/envs')
from reacher2d_env import Reacher2DEnv

def test_easy_config():
    """测试简化配置"""
    print("🎯 测试简化配置的可达性")
    print("="*50)
    
    # 使用3关节，每个80px = 240px总reach
    env = Reacher2DEnv(
        num_links=3,
        link_lengths=[80, 80, 80],
        render_mode=None,
        config_path="../2d_reacher/configs/reacher_easy.yaml"
    )
    
    obs = env.reset()
    initial_pos = env._get_end_effector_position()
    goal_pos = env.goal_pos
    initial_distance = np.linalg.norm(np.array(initial_pos) - goal_pos)
    max_reach = sum([80, 80, 80])
    
    print(f"📍 初始位置: ({initial_pos[0]:.1f}, {initial_pos[1]:.1f})")
    print(f"🎯 目标位置: ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})")
    print(f"📏 初始距离: {initial_distance:.1f}")
    print(f"🦾 理论最大reach: {max_reach}")
    print(f"📊 可达性: {'✅ 可达' if initial_distance <= max_reach else '❌ 不可达'}")
    
    if initial_distance <= max_reach:
        # 尝试通过简单策略接近目标
        print("\n🎮 测试简单策略能否接近目标...")
        
        best_distance = initial_distance
        success_steps = 0
        
        for attempt in range(20):
            env.reset()
            
            for step in range(100):
                current_pos = env._get_end_effector_position()
                direction = np.array(goal_pos) - np.array(current_pos)
                distance = np.linalg.norm(direction)
                
                if distance < 35:  # 成功
                    success_steps += 1
                    print(f"🎉 成功！第{attempt+1}次尝试，第{step+1}步达到目标")
                    break
                
                # 简单启发式：朝目标方向移动
                if direction[0] > 0:  # 目标在右边
                    action = np.array([40, 20, 10])
                else:
                    action = np.array([-40, -20, -10])
                
                if direction[1] < 0:  # 目标在下方，减小第一个关节角度
                    action[0] *= -1
                
                next_obs, reward, done, info = env.step(action)
                current_pos = env._get_end_effector_position()
                current_distance = np.linalg.norm(np.array(current_pos) - goal_pos)
                best_distance = min(best_distance, current_distance)
                
                if done:
                    break
        
        print(f"📈 最佳接近距离: {best_distance:.1f}")
        print(f"📊 距离改善: {initial_distance - best_distance:.1f}")
        print(f"🎯 成功次数: {success_steps}/20")
        
        if success_steps > 0:
            print("✅ 目标确实可达！策略可以学习！")
            return True
        elif best_distance < initial_distance * 0.6:
            print("⚠️  接近成功，但需要更好的策略")
            return True
        else:
            print("❌ 仍然无法有效接近")
            return False
    else:
        print("❌ 目标超出理论可达范围")
        return False

def demonstrate_reward_learning():
    """演示奖励函数是否能提供学习信号"""
    print("\n🧠 测试奖励函数学习信号")
    print("="*50)
    
    env = Reacher2DEnv(
        num_links=3,
        link_lengths=[80, 80, 80],
        render_mode=None,
        config_path="../2d_reacher/configs/reacher_easy.yaml"
    )
    
    # 收集不同动作的奖励
    rewards_good = []  # 朝目标的动作
    rewards_bad = []   # 远离目标的动作
    
    for test_round in range(10):
        obs = env.reset()
        current_pos = env._get_end_effector_position()
        goal_pos = env.goal_pos
        direction = np.array(goal_pos) - np.array(current_pos)
        
        # 测试朝向目标的动作
        if direction[0] > 0:
            good_action = np.array([30, 15, 5])
        else:
            good_action = np.array([-30, -15, -5])
        
        next_obs, reward_good, done, info = env.step(good_action)
        rewards_good.append(reward_good)
        
        # 重置并测试远离目标的动作
        env.reset()
        bad_action = -good_action  # 反方向
        next_obs, reward_bad, done, info = env.step(bad_action)
        rewards_bad.append(reward_bad)
    
    avg_good = np.mean(rewards_good)
    avg_bad = np.mean(rewards_bad)
    
    print(f"📊 朝向目标动作平均奖励: {avg_good:.3f}")
    print(f"📊 远离目标动作平均奖励: {avg_bad:.3f}")
    print(f"📊 奖励差异: {avg_good - avg_bad:.3f}")
    
    if avg_good > avg_bad + 0.1:  # 明显差异
        print("✅ 奖励函数提供明确学习信号")
        return True
    else:
        print("❌ 奖励信号不够明确")
        return False

if __name__ == "__main__":
    success1 = test_easy_config()
    success2 = demonstrate_reward_learning()
    
    print("\n" + "="*60)
    print("📋 简化配置测试结果")
    print("="*60)
    print(f"可达性测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"奖励信号测试: {'✅ 通过' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print("🎉 简化配置有效！可以开始SAC训练测试")
        print("💡 建议：使用这个配置重新训练你的模型")
    else:
        print("🚨 需要进一步调整")
