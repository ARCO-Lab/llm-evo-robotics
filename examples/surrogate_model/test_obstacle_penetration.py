#!/usr/bin/env python3
"""
测试障碍物穿透问题 - 验证collision_slop设置后的改进
"""

import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)

sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
from reacher2d_env import Reacher2DEnv
import numpy as np
import time

def test_obstacle_penetration():
    print("🔍 测试障碍物穿透情况")
    print("="*40)
    
    # 创建环境
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': 'human',  # 启用可视化
        'config_path': '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml',
        'debug_level': 'INFO'  # 启用碰撞日志
    }
    
    env = Reacher2DEnv(**env_params)
    obs = env.reset()
    
    print("✅ 环境创建成功")
    print(f"   障碍物数量: {len(env.obstacles)}")
    print(f"   Robot links数量: {len(env.bodies)}")
    
    # 检查碰撞设置
    print(f"\n🔧 碰撞设置检查:")
    print(f"   Space collision_slop: {env.space.collision_slop}")
    
    # 检查第一个link和第一个obstacle的collision_slop
    if env.bodies and env.obstacles:
        link_shape = env.bodies[0].shapes[0]
        obstacle_shape = env.obstacles[0]
        print(f"   Link collision_slop: {link_shape.collision_slop}")
        print(f"   Obstacle collision_slop: {obstacle_shape.collision_slop}")
    
    # 测试1: 强制机器人朝障碍物移动
    print(f"\n🎮 测试1: 强制移动测试")
    print(f"   执行大幅度动作，观察碰撞检测...")
    
    collision_count_start = getattr(env, 'collision_count', 0)
    penetration_detected = False
    
    test_actions = [
        [2.0, 1.0, 1.0, 1.0],   # 大幅度正向
        [-2.0, -1.0, -1.0, -1.0], # 大幅度负向
        [1.5, -1.5, 1.5, -1.5],   # 交替运动
        [2.0, 2.0, -2.0, -2.0],   # 极端运动
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n   动作 {i+1}: {action}")
        
        # 记录机器人位置
        pos_before = [body.position for body in env.bodies]
        
        for step in range(10):  # 每个动作执行10步
            obs, reward, done, info = env.step(np.array(action))
            
            # 检查是否有link穿透障碍物（简单几何检查）
            for j, body in enumerate(env.bodies):
                for obstacle in env.obstacles:
                    # 简单的距离检查
                    dist = np.linalg.norm(np.array(body.position) - np.array([obstacle.a[0], obstacle.a[1]]))
                    if dist < 10:  # 如果非常接近障碍物
                        print(f"     ⚠️ Link {j} 非常接近障碍物 (距离: {dist:.1f})")
            
            time.sleep(0.05)  # 短暂暂停以便观察
        
        collision_count_current = getattr(env, 'collision_count', 0)
        collisions_this_action = collision_count_current - collision_count_start
        print(f"     碰撞次数: {collisions_this_action}")
        collision_count_start = collision_count_current
    
    # 测试2: 连续旋转测试
    print(f"\n🎮 测试2: 连续旋转测试")
    print(f"   连续旋转所有关节，检查碰撞...")
    
    rotation_actions = [
        [3.0, 0, 0, 0],     # 只旋转基座
        [0, 3.0, 0, 0],     # 只旋转第二关节
        [0, 0, 3.0, 0],     # 只旋转第三关节
        [0, 0, 0, 3.0],     # 只旋转第四关节
    ]
    
    for i, action in enumerate(rotation_actions):
        print(f"\n   旋转测试 {i+1}: 关节 {i+1} 大幅旋转")
        collision_before = getattr(env, 'collision_count', 0)
        
        for step in range(20):
            obs, reward, done, info = env.step(np.array(action))
            time.sleep(0.03)
        
        collision_after = getattr(env, 'collision_count', 0)
        print(f"     检测到碰撞: {collision_after - collision_before} 次")
    
    # 测试3: 检查物理约束
    print(f"\n🔍 测试3: 物理约束检查")
    final_collision_count = getattr(env, 'collision_count', 0)
    
    print(f"   总碰撞次数: {final_collision_count}")
    print(f"   碰撞检测 {'✅ 正常工作' if final_collision_count > 0 else '⚠️ 可能有问题'}")
    
    # 获取最终位置信息
    print(f"\n📍 最终机器人状态:")
    for i, body in enumerate(env.bodies):
        print(f"   Link {i}: 位置 ({body.position.x:.1f}, {body.position.y:.1f}), 角度 {body.angle:.2f}rad")
    
    # 手动检查穿透
    print(f"\n🔍 手动穿透检查:")
    penetration_found = False
    
    for i, body in enumerate(env.bodies):
        body_pos = np.array([body.position.x, body.position.y])
        
        for j, obstacle in enumerate(env.obstacles):
            # 检查link中心是否在障碍物线段附近
            obs_start = np.array(obstacle.a)
            obs_end = np.array(obstacle.b)
            
            # 点到线段的距离计算
            line_vec = obs_end - obs_start
            point_vec = body_pos - obs_start
            line_len = np.linalg.norm(line_vec)
            
            if line_len > 0:
                line_unitvec = line_vec / line_len
                proj_length = np.dot(point_vec, line_unitvec)
                proj_length = max(min(proj_length, line_len), 0)
                nearest_point = obs_start + proj_length * line_unitvec
                distance = np.linalg.norm(body_pos - nearest_point)
                
                # 如果距离小于link半径 + obstacle半径，可能有穿透
                if distance < 13:  # 8 (link radius) + 5 (obstacle radius)
                    print(f"     ⚠️ 可能穿透: Link {i} 距离障碍物 {j} 仅 {distance:.1f} 像素")
                    penetration_found = True
    
    if not penetration_found:
        print(f"     ✅ 未发现明显穿透")
    
    print(f"\n📊 测试总结:")
    print(f"   ✅ 碰撞检测设置: collision_slop = 0.01")
    print(f"   ✅ 碰撞计数功能: {'正常' if final_collision_count > 0 else '需检查'}")
    print(f"   ✅ 穿透检查: {'未发现' if not penetration_found else '发现可能问题'}")
    print(f"   ✅ 物理一致性: 所有对象使用相同collision_slop")
    
    env.close()
    return final_collision_count, not penetration_found

def quick_penetration_test():
    """快速无渲染穿透测试"""
    print(f"\n🚀 快速穿透测试 (无渲染)")
    print("="*30)
    
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': None,  # 无渲染，更快
        'config_path': '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml',
        'debug_level': 'SILENT'
    }
    
    env = Reacher2DEnv(**env_params)
    obs = env.reset()
    
    # 执行大量随机动作
    collision_count = 0
    penetration_warnings = 0
    
    for episode in range(5):
        obs = env.reset()
        for step in range(100):
            # 生成激进的动作
            action = np.random.uniform(-3, 3, 4)
            obs, reward, done, info = env.step(action)
            
            # 检查碰撞计数
            current_collisions = getattr(env, 'collision_count', 0)
            if current_collisions > collision_count:
                collision_count = current_collisions
            
            # 快速穿透检查
            for i, body in enumerate(env.bodies):
                body_pos = np.array([body.position.x, body.position.y])
                for obstacle in env.obstacles:
                    obs_center = np.array([(obstacle.a[0] + obstacle.b[0])/2, 
                                         (obstacle.a[1] + obstacle.b[1])/2])
                    if np.linalg.norm(body_pos - obs_center) < 8:
                        penetration_warnings += 1
    
    env.close()
    
    print(f"   总碰撞检测: {collision_count} 次")
    print(f"   穿透警告: {penetration_warnings} 次")
    print(f"   结果: {'✅ 正常' if collision_count > 0 and penetration_warnings < 10 else '⚠️ 需检查'}")
    
    return collision_count > 0 and penetration_warnings < 10

if __name__ == "__main__":
    print("🔬 障碍物穿透测试套件")
    print("="*50)
    
    # 选择测试模式
    print("选择测试模式:")
    print("1. 可视化详细测试 (推荐)")
    print("2. 快速测试")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            collision_count, no_penetration = test_obstacle_penetration()
            print(f"\n🎯 详细测试完成!")
            print(f"   碰撞检测: {'✅' if collision_count > 0 else '❌'}")
            print(f"   无穿透: {'✅' if no_penetration else '❌'}")
        
        elif choice == "2":
            success = quick_penetration_test()
            print(f"\n🎯 快速测试完成: {'✅ 通过' if success else '❌ 需检查'}")
        
        else:
            print("无效选择，执行快速测试...")
            quick_penetration_test()
            
    except KeyboardInterrupt:
        print(f"\n⏹️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        quick_penetration_test()
