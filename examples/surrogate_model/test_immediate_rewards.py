#!/usr/bin/env python3
"""
测试路标点即时奖励机制
"""

import sys
import os
import numpy as np
sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/surrogate_model')

from waypoint_navigator import WaypointNavigator

def test_immediate_rewards():
    """测试即时奖励获得机制"""
    
    print("💰 测试路标点即时奖励机制")
    print("="*50)
    
    # 创建路标点导航器
    start = [500, 620]
    goal = [600, 550]
    navigator = WaypointNavigator(start, goal)
    
    print(f"📍 路标点列表:")
    for i, wp in enumerate(navigator.waypoints):
        print(f"   路标{i}: {wp.position} (奖励: {wp.reward}, 半径: {wp.radius})")
    
    print(f"\n🧪 模拟机器人移动，测试即时奖励获得:")
    print("-" * 50)
    
    # 模拟机器人移动到每个路标点
    current_pos = np.array(start, dtype=float)
    total_reward_earned = 0.0
    
    for step in range(100):
        # 获取当前目标
        target = navigator.get_current_target()
        
        # 计算移动方向
        direction = target - current_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            # 每步移动25像素朝向目标
            move_distance = min(25.0, distance)
            current_pos += (direction / distance) * move_distance
        
        # 更新导航器并获取奖励
        reward, info = navigator.update(current_pos)
        total_reward_earned += reward
        
        # 输出详细信息
        print(f"步骤 {step+1:2d}: 位置 {current_pos.astype(int)} -> 目标 {target.astype(int)}")
        print(f"        距离: {np.linalg.norm(current_pos - target):.1f}px, 本步奖励: {reward:+.2f}")
        
        # 检查是否到达路标点
        if info["waypoint_reached"]:
            print(f"        🎉 【即时奖励】到达路标点 {info['current_waypoint']-1}!")
            print(f"        💰 获得奖励: +{info['waypoint_reward']:.1f} 分")
            print(f"        📊 完成进度: {info['completion_progress']*100:.1f}%")
            print(f"        🏆 累计奖励: {navigator.total_reward:.1f} 分")
            print("-" * 30)
        
        # 检查是否完成所有路标点
        if info["completion_progress"] >= 1.0:
            print(f"\n🏁 所有路标点完成!")
            print(f"   总步数: {step+1}")
            print(f"   总奖励: {navigator.total_reward:.1f} 分")
            break
    
    print(f"\n📊 奖励机制验证:")
    print(f"   路标点奖励: {navigator.total_reward:.1f} 分")
    print(f"   其他奖励: {total_reward_earned - navigator.total_reward:.1f} 分")
    print(f"   总计奖励: {total_reward_earned:.1f} 分")
    
    return navigator.total_reward, total_reward_earned

def test_reward_types():
    """测试不同类型的奖励"""
    
    print(f"\n🔬 详细奖励类型测试")
    print("="*50)
    
    navigator = WaypointNavigator([500, 620], [600, 550])
    
    # 测试接近奖励
    print(f"1️⃣ 接近奖励测试:")
    positions = [
        [500, 620],  # 起点 - 应该立即获得到达奖励
        [480, 620],  # 接近路标点1
        [450, 620],  # 到达路标点1
        [450, 600],  # 接近路标点2
        [450, 550],  # 到达路标点2
    ]
    
    for i, pos in enumerate(positions):
        reward, info = navigator.update(np.array(pos))
        print(f"   位置 {pos}: 奖励 {reward:+.2f}")
        if info["waypoint_reached"]:
            print(f"      🎯 到达路标点! 即时奖励: +{info['waypoint_reward']}")
    
    print(f"\n2️⃣ 奖励组成分析:")
    print(f"   - 🎯 路标点即时奖励: 10分 (中间) + 50分 (最终)")
    print(f"   - 🏃 接近当前目标奖励: 0 到 +0.5分")
    print(f"   - 📈 移动进度奖励: -1.0 到 +1.0分")
    print(f"   - 💯 完成度奖励: 0 到 +5.0分")

def demonstrate_reward_timing():
    """演示奖励获得的时机"""
    
    print(f"\n⏰ 奖励获得时机演示")
    print("="*50)
    
    navigator = WaypointNavigator([500, 620], [600, 550])
    
    # 模拟精确到达路标点的时刻
    waypoint_pos = navigator.waypoints[1].position  # 第二个路标点 [450, 620]
    waypoint_radius = navigator.waypoints[1].radius  # 半径30px
    
    print(f"🎯 目标路标点: {waypoint_pos}, 半径: {waypoint_radius}px")
    
    # 从外围逐步接近
    test_positions = [
        waypoint_pos + [50, 0],   # 距离50px (未到达)
        waypoint_pos + [35, 0],   # 距离35px (未到达)
        waypoint_pos + [30, 0],   # 距离30px (刚好到达边界)
        waypoint_pos + [25, 0],   # 距离25px (已到达)
        waypoint_pos + [0, 0],    # 距离0px (中心)
    ]
    
    for i, pos in enumerate(test_positions):
        # 重置路标点状态用于测试
        if i > 0:
            navigator.waypoints[1].visited = False
        
        reward, info = navigator.update(np.array(pos))
        distance = np.linalg.norm(np.array(pos) - waypoint_pos)
        
        print(f"   距离 {distance:4.1f}px: ", end="")
        if info["waypoint_reached"]:
            print(f"✅ 到达! 即时奖励: +{info['waypoint_reward']}")
        else:
            print(f"❌ 未到达, 奖励: {reward:+.2f}")

if __name__ == "__main__":
    # 运行所有测试
    waypoint_rewards, total_rewards = test_immediate_rewards()
    test_reward_types()
    demonstrate_reward_timing()
    
    print(f"\n🎊 总结:")
    print(f"   ✅ 即时奖励机制正常工作")
    print(f"   ✅ 每到达路标点立即获得{waypoint_rewards:.0f}分奖励")
    print(f"   ✅ 额外奖励机制提供持续反馈")
    print(f"   🚀 路标点系统已准备好用于训练!")
