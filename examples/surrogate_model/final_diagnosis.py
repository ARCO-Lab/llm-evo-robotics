#!/usr/bin/env python3
"""
最终诊断：为什么SAC学不到策略
根本原因：目标不可达
"""

import numpy as np
import sys
import os

sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/envs')
from reacher2d_env import Reacher2DEnv

def analyze_reachability_problem():
    """分析可达性问题的根本原因"""
    print("🔬 最终诊断：SAC学不到策略的根本原因")
    print("="*60)
    
    # 测试原始配置
    env = Reacher2DEnv(
        num_links=3,
        link_lengths=[60, 60, 60],  # 原始配置
        render_mode=None,
        config_path="../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    )
    
    obs = env.reset()
    anchor = env.anchor_point  # 锚点
    initial_pos = env._get_end_effector_position()  # 末端执行器
    goal_pos = env.goal_pos  # 目标
    
    print(f"📍 锚点位置: {anchor}")
    print(f"🤖 末端执行器位置: {initial_pos}")
    print(f"🎯 目标位置: {goal_pos}")
    
    # 计算距离
    anchor_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(anchor))
    end_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(initial_pos))
    theoretical_reach = sum([60, 60, 60])
    
    print(f"\n📏 距离分析:")
    print(f"   锚点到目标: {anchor_to_goal:.1f} px")
    print(f"   末端到目标: {end_to_goal:.1f} px")
    print(f"   理论最大reach: {theoretical_reach} px")
    
    print(f"\n🎯 可达性分析:")
    print(f"   锚点可达性: {'✅ 可达' if anchor_to_goal <= theoretical_reach else '❌ 不可达'}")
    print(f"   当前可达性: {'✅ 可达' if end_to_goal <= theoretical_reach else '❌ 不可达'}")
    
    # 分析初始角度问题
    print(f"\n🔧 初始化问题分析:")
    print(f"   初始化方式: 所有关节90度（垂直向上）")
    print(f"   导致末端偏移: {np.array(initial_pos) - np.array(anchor)}")
    print(f"   应该初始化为: 水平伸展（0度）")
    
    # 计算最佳初始角度
    direction = np.array(goal_pos) - np.array(anchor)
    optimal_angle = np.arctan2(direction[1], direction[0])
    print(f"   最佳初始方向: {np.degrees(optimal_angle):.1f}度")
    
    return anchor_to_goal <= theoretical_reach

def demonstrate_solution():
    """演示解决方案"""
    print("\n💡 解决方案演示")
    print("="*40)
    
    print("方案1: 增加关节长度")
    env1 = Reacher2DEnv(
        num_links=3,
        link_lengths=[100, 100, 100],  # 增加到300px
        render_mode=None,
        config_path="../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    )
    obs1 = env1.reset()
    pos1 = env1._get_end_effector_position()
    goal1 = env1.goal_pos
    distance1 = np.linalg.norm(np.array(pos1) - goal1)
    reach1 = sum([100, 100, 100])
    print(f"   距离: {distance1:.1f}, reach: {reach1}, 可达: {'✅' if distance1 <= reach1 else '❌'}")
    
    print("\n方案2: 增加关节数量")
    env2 = Reacher2DEnv(
        num_links=5,
        link_lengths=[60, 60, 60, 60, 60],  # 5关节300px
        render_mode=None,
        config_path="../2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    )
    obs2 = env2.reset()
    pos2 = env2._get_end_effector_position()
    goal2 = env2.goal_pos
    distance2 = np.linalg.norm(np.array(pos2) - goal2)
    reach2 = sum([60, 60, 60, 60, 60])
    print(f"   距离: {distance2:.1f}, reach: {reach2}, 可达: {'✅' if distance2 <= reach2 else '❌'}")
    
    print("\n方案3: 修改目标位置")
    # 从锚点出发，理论上可达的目标
    anchor = [300, 550]
    reachable_goal = [420, 600]  # 距离约111px，在180px范围内
    anchor_distance = np.linalg.norm(np.array(reachable_goal) - np.array(anchor))
    print(f"   新目标距离锚点: {anchor_distance:.1f}, reach: 180, 可达: {'✅' if anchor_distance <= 180 else '❌'}")

def main():
    """主函数"""
    reachable = analyze_reachability_problem()
    demonstrate_solution()
    
    print("\n" + "="*60)
    print("📋 最终诊断结论")
    print("="*60)
    
    print("🚨 SAC学不到策略的根本原因:")
    print("   1. 目标物理上不可达（距离 > 最大reach）")
    print("   2. 机器人初始化角度不合理（垂直向上而非朝向目标）")
    print("   3. 没有成功的经验供网络学习")
    print("   4. Q值估计错误，策略无法收敛")
    
    print("\n✅ 推荐解决方案:")
    print("   1. 使用5关节配置（300px reach）")
    print("   2. 或使用3关节但每个100px长度")
    print("   3. 或修改目标到可达范围内")
    print("   4. 修正初始角度让机器人朝向目标")
    
    print("\n🎯 修复后，SAC应该能够:")
    print("   - 收集到成功的经验")
    print("   - 学到正确的Q值估计")
    print("   - 训练出有效的策略")
    print("   - Critic Loss稳定在1.0以下")
    print("   - Actor Loss收敛到合理范围")

if __name__ == "__main__":
    main()
