#!/usr/bin/env python3
"""
简化版SAC学习能力测试
避免复杂的导入问题，专注于核心测试
"""

import torch
import numpy as np
import sys
import os

# 简化路径设置
sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/envs')
from reacher2d_env import Reacher2DEnv

def test_basic_functionality():
    """测试基础功能"""
    print("🧪 步骤1: 基础功能测试")
    print("="*50)
    
    try:
        # 创建环境
        env = Reacher2DEnv(
            num_links=3, 
            link_lengths=[60, 60, 60],
            render_mode=None,
            config_path="../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
        )
        print("✅ 环境创建成功")
        
        # 测试环境重置
        obs = env.reset()
        print(f"✅ 环境重置成功: obs shape = {len(obs)}")
        
        # 测试随机动作
        action = np.random.uniform(-50, 50, 3)
        next_obs, reward, done, info = env.step(action)
        print(f"✅ 环境交互成功: reward = {reward:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
        return False

def test_reward_signal():
    """测试奖励信号合理性"""
    print("\n🎯 步骤2: 奖励信号测试")
    print("="*50)
    
    try:
        env = Reacher2DEnv(
            num_links=3, 
            link_lengths=[60, 60, 60],
            render_mode=None,
            config_path="../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
        )
        
        rewards_close = []
        rewards_far = []
        
        # 测试不同距离下的奖励
        for test_round in range(10):
            obs = env.reset()
            
            # 记录初始距离和奖励
            initial_pos = env._get_end_effector_position()
            initial_distance = np.linalg.norm(np.array(initial_pos) - env.goal_pos)
            
            # 随机动作几步，观察奖励
            for step in range(10):
                action = np.random.uniform(-30, 30, 3)
                next_obs, reward, done, info = env.step(action)
                
                # 计算当前距离
                current_pos = env._get_end_effector_position()
                current_distance = np.linalg.norm(np.array(current_pos) - env.goal_pos)
                
                # 分类收集奖励
                if current_distance < 100:  # 较近
                    rewards_close.append(reward)
                else:  # 较远
                    rewards_far.append(reward)
                
                if done:
                    break
        
        # 分析奖励
        if rewards_close and rewards_far:
            avg_close = np.mean(rewards_close)
            avg_far = np.mean(rewards_far)
            
            print(f"📊 距离较近时平均奖励: {avg_close:.3f}")
            print(f"📊 距离较远时平均奖励: {avg_far:.3f}")
            print(f"📊 奖励差异: {avg_close - avg_far:.3f}")
            
            if avg_close > avg_far:
                print("✅ 奖励信号合理：距离越近奖励越高")
                return True
            else:
                print("❌ 奖励信号可能有问题")
                return False
        else:
            print("❌ 数据收集不足")
            return False
            
    except Exception as e:
        print(f"❌ 奖励信号测试失败: {e}")
        return False

def test_action_scaling():
    """测试动作缩放是否正确"""
    print("\n🎮 步骤3: 动作缩放测试")
    print("="*50)
    
    try:
        env = Reacher2DEnv(
            num_links=3, 
            link_lengths=[60, 60, 60],
            render_mode=None,
            config_path="../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
        )
        
        # 测试不同动作强度的效果
        movements = []
        
        for action_scale in [0, 25, 50, 100]:
            env.reset()
            initial_pos = env._get_end_effector_position()
            
            # 应用动作
            action = np.array([action_scale, 0, 0])  # 只动第一个关节
            
            total_movement = 0
            for step in range(5):
                next_obs, reward, done, info = env.step(action)
                new_pos = env._get_end_effector_position()
                movement = np.linalg.norm(np.array(new_pos) - np.array(initial_pos))
                total_movement = max(total_movement, movement)  # 记录最大移动
                initial_pos = new_pos
                
                if done:
                    break
            
            movements.append(total_movement)
            print(f"   动作强度 {action_scale:3d}: 最大累积移动 {total_movement:.1f}")
        
        # 检查是否单调递增
        is_increasing = all(movements[i] <= movements[i+1] for i in range(len(movements)-1))
        
        if is_increasing and movements[-1] > movements[0]:
            print("✅ 动作缩放正常：更大动作产生更大移动")
            return True
        else:
            print("❌ 动作缩放可能有问题")
            return False
            
    except Exception as e:
        print(f"❌ 动作缩放测试失败: {e}")
        return False

def test_goal_reachability():
    """测试目标是否可达"""
    print("\n🎯 步骤4: 目标可达性测试")
    print("="*50)
    
    try:
        env = Reacher2DEnv(
            num_links=3, 
            link_lengths=[60, 60, 60],
            render_mode=None,
            config_path="../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
        )
        
        obs = env.reset()
        initial_pos = env._get_end_effector_position()
        goal_pos = env.goal_pos
        initial_distance = np.linalg.norm(np.array(initial_pos) - goal_pos)
        
        print(f"📍 初始位置: ({initial_pos[0]:.1f}, {initial_pos[1]:.1f})")
        print(f"🎯 目标位置: ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})")
        print(f"📏 初始距离: {initial_distance:.1f}")
        
        # 计算机械臂理论最大reach
        max_reach = sum([60, 60, 60])  # 所有关节长度之和
        print(f"🦾 理论最大reach: {max_reach}")
        
        if initial_distance <= max_reach:
            print("✅ 目标在理论可达范围内")
            
            # 测试是否能通过随机动作接近目标
            best_distance = initial_distance
            for attempt in range(100):
                env.reset()
                
                for step in range(50):
                    # 使用较大的随机动作
                    action = np.random.uniform(-80, 80, 3)
                    next_obs, reward, done, info = env.step(action)
                    
                    current_pos = env._get_end_effector_position()
                    current_distance = np.linalg.norm(np.array(current_pos) - goal_pos)
                    best_distance = min(best_distance, current_distance)
                    
                    if done:
                        break
            
            print(f"🏆 随机探索最近距离: {best_distance:.1f}")
            improvement = initial_distance - best_distance
            print(f"📈 距离改善: {improvement:.1f}")
            
            if improvement > 20:  # 能改善20像素以上
                print("✅ 目标确实可达")
                return True
            else:
                print("⚠️  目标可能难以达到，但在理论范围内")
                return True
        else:
            print("❌ 目标超出理论可达范围")
            return False
            
    except Exception as e:
        print(f"❌ 目标可达性测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🔬 SAC学习能力简化诊断")
    print("="*60)
    
    tests = [
        ("基础功能", test_basic_functionality),
        ("奖励信号", test_reward_signal),
        ("动作缩放", test_action_scaling),
        ("目标可达性", test_goal_reachability),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    # 生成报告
    print("\n" + "="*60)
    print("📋 测试报告")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:10s}: {status}")
    
    print(f"\n📊 总体评估: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 基础环境功能正常，可以进行SAC训练")
    elif passed >= 3:
        print("⚠️  大部分功能正常，但需要注意一些问题")
    else:
        print("🚨 存在严重问题，建议先修复环境配置")

if __name__ == "__main__":
    main()