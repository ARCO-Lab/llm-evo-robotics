#!/usr/bin/env python3
"""
测试路标点可视化
"""

import sys
import os
import time
import numpy as np
sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/envs')
sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/surrogate_model')

from reacher2d_env import Reacher2DEnv
from waypoint_navigator import WaypointNavigator

def test_waypoint_visualization():
    """测试路标点可视化效果"""
    
    print("🎮 测试路标点可视化")
    print("="*50)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=3,
        link_lengths=[60, 60, 60],
        render_mode="human",  # 开启可视化
        config_path='/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml'
    )
    
    # 添加路标点系统
    start_pos = env.anchor_point
    goal_pos = env.goal_pos
    
    # 创建路标点导航器
    waypoint_navigator = WaypointNavigator(start_pos, goal_pos)
    env.waypoint_navigator = waypoint_navigator
    
    print(f"✅ 环境创建成功，已添加路标点系统")
    print(f"   锚点: {start_pos}")
    print(f"   目标: {goal_pos}")
    print(f"   路标点数: {len(waypoint_navigator.waypoints)}")
    
    # 重置环境
    obs = env.reset()
    
    print(f"\n🎯 路标点可视化说明:")
    print(f"   🟡 黄色闪烁圆圈 = 当前目标路标点")
    print(f"   🔵 蓝色圆圈 = 未访问路标点") 
    print(f"   🟢 绿色圆圈 = 已访问路标点")
    print(f"   🟨 黄色虚线 = 当前路径段")
    print(f"   🟩 绿色实线 = 已完成路径段")
    print(f"   ⚪ 灰色虚线 = 未来路径段")
    print(f"   📊 左上角面板 = 导航进度信息")
    
    # 模拟机器人朝向路标点移动
    for step in range(200):
        # 获取当前位置
        current_pos = np.array(env._get_end_effector_position())
        
        # 更新路标点导航器
        waypoint_reward, waypoint_info = waypoint_navigator.update(current_pos)
        
        # 生成朝向当前路标点的动作
        target = waypoint_navigator.get_current_target()
        
        # 计算需要的关节角度（简化版逆运动学）
        action = env.action_space.sample() * 0.1  # 小幅随机动作
        
        # 如果接近目标，减小动作幅度
        distance_to_target = np.linalg.norm(current_pos - target)
        if distance_to_target < 50:
            action *= 0.5
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        # 渲染
        env.render()
        
        # 控制帧率
        time.sleep(0.05)
        
        # 输出进度信息
        if step % 20 == 0:
            progress = waypoint_navigator.get_progress_info()
            print(f"步骤 {step}: 进度 {progress['progress_percentage']:.1f}%, "
                  f"当前路标 {waypoint_navigator.current_waypoint_idx}, "
                  f"距离目标 {distance_to_target:.1f}px")
        
        # 如果完成所有路标点
        if waypoint_info.get('completion_progress', 0) >= 1.0:
            print(f"🏆 所有路标点完成! 总步数: {step}")
            break
        
        # 如果环境结束
        if done:
            print(f"🔄 环境结束，重置中...")
            obs = env.reset()
            waypoint_navigator.reset()
    
    print(f"\n⏱️ 演示结束，等待5秒后关闭...")
    for i in range(5):
        env.render()
        time.sleep(1)
        print(f"   {5-i}秒后关闭...")
    
    env.close()
    print(f"✅ 可视化测试完成")

if __name__ == "__main__":
    test_waypoint_visualization()
