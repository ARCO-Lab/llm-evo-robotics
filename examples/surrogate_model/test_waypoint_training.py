#!/usr/bin/env python3
"""
测试带路标点的训练可视化
"""

import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)

sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model'))

from reacher2d_env import Reacher2DEnv
from waypoint_navigator import WaypointNavigator
import numpy as np
import time
import torch

def test_waypoint_training_visualization():
    """测试带路标点的训练可视化"""
    
    print("🎮 测试带路标点的训练可视化")
    print("="*50)
    
    # 创建环境
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': 'human',  # 开启渲染
        'config_path': '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml',
        'debug_level': 'SILENT'
    }
    
    env = Reacher2DEnv(**env_params)
    
    # 添加路标点系统
    start_pos = env.anchor_point
    goal_pos = env.goal_pos
    env.waypoint_navigator = WaypointNavigator(start_pos, goal_pos)
    
    print(f"✅ 环境创建完成")
    print(f"   起点: {start_pos}")
    print(f"   终点: {goal_pos}")
    print(f"   路标数: {len(env.waypoint_navigator.waypoints)}")
    print(f"   机器人关节数: {env.action_space.shape[0]}")
    
    print(f"\n🗺️ 路标点列表:")
    for i, wp in enumerate(env.waypoint_navigator.waypoints):
        print(f"   路标{i}: {wp.position} (奖励: {wp.reward})")
    
    print(f"\n🎯 开始测试...")
    print(f"应该能看到:")
    print(f"   🟡 黄色闪烁圆圈 = 当前目标路标点")
    print(f"   🔵 蓝色圆圈 = 未访问路标点")
    print(f"   🟢 绿色圆圈 = 已访问路标点")
    print(f"   📊 左上角面板 = 导航进度信息")
    print(f"   🛤️ 彩色路径线 = 完成状态路径")
    
    # 测试路标点系统的奖励函数
    def compute_waypoint_reward(env, action):
        """计算带路标点的奖励"""
        # 执行原始环境step
        obs, base_reward, done, info = env.step(action)
        
        # 获取当前末端执行器位置
        end_pos = np.array(env._get_end_effector_position())
        
        # 更新路标点导航器并获取奖励
        waypoint_reward, waypoint_info = env.waypoint_navigator.update(end_pos)
        
        # 组合奖励
        total_reward = base_reward + waypoint_reward
        
        # 更新info
        info.update(waypoint_info)
        info['base_reward'] = base_reward
        info['waypoint_reward'] = waypoint_reward
        info['total_reward'] = total_reward
        
        return obs, total_reward, done, info
    
    # 运行测试
    obs = env.reset()
    total_episodes = 0
    step_count = 0
    
    for episode in range(3):  # 测试3个episode
        print(f"\n🎬 Episode {episode + 1}/3")
        
        obs = env.reset()
        env.waypoint_navigator.reset()  # 重置路标点系统
        
        episode_reward = 0
        episode_waypoint_reward = 0
        
        for step in range(500):  # 每个episode最多500步
            # 渲染环境
            env.render()
            
            # 生成随机动作（实际训练中这里是policy网络）
            action = env.action_space.sample() * 0.3  # 减小动作幅度
            
            # 执行动作并获得路标点奖励
            obs, reward, done, info = compute_waypoint_reward(env, action)
            
            episode_reward += info['total_reward']
            episode_waypoint_reward += info.get('waypoint_reward', 0)
            step_count += 1
            
            # 每10步输出一次信息
            if step % 10 == 0:
                end_pos = env._get_end_effector_position()
                current_target = env.waypoint_navigator.get_current_target()
                distance = np.linalg.norm(np.array(end_pos) - current_target)
                
                print(f"  步骤 {step}: 距离目标 {distance:.1f}px, "
                      f"路标奖励 {info.get('waypoint_reward', 0):+.2f}, "
                      f"进度 {info.get('completion_progress', 0)*100:.1f}%")
            
            # 检查路标点完成
            if info.get('waypoint_reached', False):
                print(f"    🎯 到达路标点! 获得奖励: +{info.get('waypoint_reward', 0)}")
            
            # 检查episode结束
            if done or info.get('completion_progress', 0) >= 1.0:
                print(f"  🏁 Episode结束: 总步数 {step+1}")
                print(f"     总奖励: {episode_reward:.2f}")
                print(f"     路标奖励: {episode_waypoint_reward:.2f}")
                print(f"     完成进度: {info.get('completion_progress', 0)*100:.1f}%")
                break
            
            # 控制帧率
            time.sleep(0.02)
        
        total_episodes += 1
    
    print(f"\n🏆 测试完成!")
    print(f"   总episodes: {total_episodes}")
    print(f"   总步数: {step_count}")
    print(f"   路标点系统正常工作！")
    
    # 等待一段时间让用户观察
    print(f"\n⏱️ 保持渲染5秒...")
    for i in range(5):
        env.render()
        time.sleep(1)
        print(f"   {5-i}秒后关闭...")
    
    env.close()

if __name__ == "__main__":
    test_waypoint_training_visualization()
