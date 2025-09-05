#!/usr/bin/env python3
"""
测试环境初始化状态的脚本
检查Reacher2D环境的各种初始化参数和状态
"""

import sys
import os
import numpy as np

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from reacher2d_env import Reacher2DEnv

def test_env_initialization():
    """测试环境初始化"""
    print("=" * 60)
    print("🧪 测试Reacher2D环境初始化状态")
    print("=" * 60)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[60, 60, 60, 60],
        render_mode="human",
        debug_level='INFO'
    )
    
    print(f"\n📊 环境基本信息:")
    print(f"   链接数量: {env.num_links}")
    print(f"   链接长度: {env.link_lengths}")
    print(f"   时间步长: {env.dt}")
    print(f"   最大扭矩: {env.max_torque}")
    print(f"   锚点位置: {env.anchor_point}")
    print(f"   目标位置: {env.goal_pos}")
    
    print(f"\n🎯 动作和观察空间:")
    print(f"   动作空间: {env.action_space}")
    print(f"   观察空间: {env.observation_space}")
    
    print(f"\n🚧 障碍物信息:")
    if env.obstacles:
        for i, obstacle in enumerate(env.obstacles):
            print(f"   障碍物{i+1}: {obstacle}")
    else:
        print("   无障碍物")
    
    # 重置环境并获取初始状态
    print(f"\n🔄 重置环境...")
    initial_obs = env.reset()
    
    print(f"\n📐 初始状态:")
    print(f"   关节角度: {env.joint_angles}")
    print(f"   关节角速度: {env.joint_velocities}")
    print(f"   步数: {env.step_count}")
    print(f"   碰撞计数: {env.collision_count}")
    print(f"   基座碰撞计数: {env.base_collision_count}")
    
    # 计算初始位置
    link_positions = env._calculate_link_positions()
    end_effector_pos = env._get_end_effector_position()
    
    print(f"\n📍 初始位置:")
    for i, pos in enumerate(link_positions):
        if i == 0:
            print(f"   基座位置: [{pos[0]:.1f}, {pos[1]:.1f}]")
        else:
            print(f"   关节{i}位置: [{pos[0]:.1f}, {pos[1]:.1f}]")
    
    print(f"   末端执行器: [{end_effector_pos[0]:.1f}, {end_effector_pos[1]:.1f}]")
    
    # 计算到目标的距离
    distance_to_goal = np.linalg.norm(end_effector_pos - env.goal_pos)
    print(f"   到目标距离: {distance_to_goal:.1f} 像素")
    
    print(f"\n📊 初始观察值:")
    print(f"   观察向量长度: {len(initial_obs)}")
    print(f"   观察值: {initial_obs}")
    
    # 检查碰撞状态
    collision = env._check_collision()
    print(f"\n💥 碰撞检查:")
    print(f"   初始碰撞状态: {collision}")
    
    # 计算初始奖励
    initial_reward = env._compute_reward()
    print(f"\n🎁 初始奖励: {initial_reward:.3f}")
    
    # 检查兼容性对象
    print(f"\n🔧 兼容性检查:")
    print(f"   Bodies数量: {len(env.bodies)}")
    print(f"   Space对象: {type(env.space)}")
    
    for i, body in enumerate(env.bodies):
        print(f"   Body{i} 位置: {body.position}, 角度: {body.angle:.3f}")
    
    # 测试几步动作
    print(f"\n🎮 测试几步随机动作...")
    for step in range(3):
        action = np.random.uniform(-10, 10, env.num_links)  # 小的随机动作
        obs, reward, done, info = env.step(action)
        
        print(f"   步骤{step+1}:")
        print(f"     动作: {action}")
        print(f"     奖励: {reward:.3f}")
        print(f"     完成: {done}")
        print(f"     末端位置: [{info['end_effector_pos'][0]:.1f}, {info['end_effector_pos'][1]:.1f}]")
        print(f"     距离: {info['distance']:.1f}")
        print(f"     碰撞: {info['collision_count']}")
        
        # 渲染一帧
        env.render()
        
        if done:
            print("     ✅ 任务完成！")
            break
    
    print(f"\n✅ 环境初始化测试完成！")
    print("=" * 60)
    
    # 保持窗口打开一会儿
    import time
    print("🖼️ 窗口将在5秒后关闭...")
    time.sleep(5)
    
    env.close()

if __name__ == "__main__":
    test_env_initialization()
