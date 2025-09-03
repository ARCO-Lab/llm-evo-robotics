#!/usr/bin/env python3
"""
修复机器人可达性问题的测试脚本
"""

import numpy as np
import sys
import os

sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/envs')
from reacher2d_env import Reacher2DEnv

def test_reachability_fix():
    """测试修复方案"""
    print("🔧 测试可达性修复方案")
    print("="*50)
    
    # 方案1: 增加关节长度
    print("\n📏 方案1: 增加关节长度")
    env1 = Reacher2DEnv(
        num_links=3, 
        link_lengths=[100, 100, 100],  # 增加到300px总reach
        render_mode=None,
        config_path="../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    )
    
    obs1 = env1.reset()
    pos1 = env1._get_end_effector_position()
    goal1 = env1.goal_pos
    distance1 = np.linalg.norm(np.array(pos1) - goal1)
    max_reach1 = sum([100, 100, 100])
    
    print(f"   初始位置: ({pos1[0]:.1f}, {pos1[1]:.1f})")
    print(f"   目标位置: ({goal1[0]:.1f}, {goal1[1]:.1f})")
    print(f"   距离: {distance1:.1f}, 最大reach: {max_reach1}")
    print(f"   可达性: {'✅ 可达' if distance1 <= max_reach1 else '❌ 不可达'}")
    
    # 方案2: 增加关节数量
    print("\n🔗 方案2: 增加关节数量")
    env2 = Reacher2DEnv(
        num_links=5, 
        link_lengths=[60, 60, 60, 60, 60],  # 5个关节，300px总reach
        render_mode=None,
        config_path="../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    )
    
    obs2 = env2.reset()
    pos2 = env2._get_end_effector_position()
    goal2 = env2.goal_pos
    distance2 = np.linalg.norm(np.array(pos2) - goal2)
    max_reach2 = sum([60, 60, 60, 60, 60])
    
    print(f"   初始位置: ({pos2[0]:.1f}, {pos2[1]:.1f})")
    print(f"   目标位置: ({goal2[0]:.1f}, {goal2[1]:.1f})")
    print(f"   距离: {distance2:.1f}, 最大reach: {max_reach2}")
    print(f"   可达性: {'✅ 可达' if distance2 <= max_reach2 else '❌ 不可达'}")
    
    # 测试实际达到能力
    print("\n🎯 实际测试：尝试接近目标")
    
    best_distance = distance2
    for attempt in range(50):
        env2.reset()
        
        for step in range(30):
            # 朝目标方向的启发式动作
            current_pos = env2._get_end_effector_position()
            direction = np.array(goal2) - np.array(current_pos)
            
            # 简单的启发式：第一个关节向目标旋转
            if direction[0] > 0:  # 目标在右边
                action = np.array([50, 10, 10, 10, 10])
            else:  # 目标在左边
                action = np.array([-50, -10, -10, -10, -10])
            
            next_obs, reward, done, info = env2.step(action)
            new_pos = env2._get_end_effector_position()
            new_distance = np.linalg.norm(np.array(new_pos) - goal2)
            best_distance = min(best_distance, new_distance)
            
            if done:
                break
    
    print(f"   最佳接近距离: {best_distance:.1f}")
    print(f"   改善幅度: {distance2 - best_distance:.1f}")
    
    if best_distance < distance2 * 0.8:  # 至少改善20%
        print("✅ 修复成功：机器人可以显著接近目标")
        return True
    else:
        print("❌ 修复失败：仍然无法有效接近目标")
        return False

def create_custom_config():
    """创建可达性友好的配置文件"""
    print("\n📝 创建可达性友好的配置")
    
    config_content = """
start:
  position: [300, 550]

goal:
  position: [450, 580]  # 更近的目标
  radius: 35

obstacles:
  - shape: segment
    points: [[200, 500], [250, 500]]
  - shape: segment  
    points: [[500, 600], [550, 600]]
"""
    
    config_path = "../2d_reacher/configs/reacher_reachable.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ 创建配置文件: {config_path}")
    
    # 测试新配置
    env = Reacher2DEnv(
        num_links=3,
        link_lengths=[80, 80, 80],  # 总reach = 240px
        render_mode=None,
        config_path=config_path
    )
    
    obs = env.reset()
    pos = env._get_end_effector_position()
    goal = env.goal_pos
    distance = np.linalg.norm(np.array(pos) - goal)
    max_reach = sum([80, 80, 80])
    
    print(f"📊 新配置测试:")
    print(f"   距离: {distance:.1f}, 最大reach: {max_reach}")
    print(f"   可达性: {'✅ 可达' if distance <= max_reach else '❌ 不可达'}")
    
    return distance <= max_reach

if __name__ == "__main__":
    success1 = test_reachability_fix()
    success2 = create_custom_config()
    
    print("\n" + "="*60)
    print("📋 修复结果总结")
    print("="*60)
    print(f"方案1 (增加关节): {'✅ 成功' if success1 else '❌ 失败'}")
    print(f"方案2 (新配置): {'✅ 成功' if success2 else '❌ 失败'}")
    
    if success1 or success2:
        print("🎉 可达性问题已解决！现在可以重新训练SAC")
        print("💡 建议：使用5关节配置或自定义可达目标")
    else:
        print("🚨 需要进一步调整配置")
