#!/usr/bin/env python3
"""
精确的Joint连接检测 - 检查Joint的实际连接状态
"""

import sys
import os
import numpy as np
import pygame
import time
import math

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from envs.reacher2d_env import Reacher2DEnv

def precise_joint_detection():
    """精确检测Joint连接状态"""
    print("🔍 精确Joint连接检测")
    print("=" * 50)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 60, 50, 40],
        render_mode="human",
        config_path="configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='WARNING'
    )
    
    env.reset()
    
    print(f"🔍 Joint配置检查:")
    for i, joint in enumerate(env.joints):
        print(f"   Joint {i}: max_force = {joint.max_force}")
        print(f"   Joint {i}: type = {type(joint).__name__}")
        # PyMunk Joint的body属性访问方式
        try:
            bodies = joint.bodies
            print(f"   Joint {i}: bodies = {bodies}")
        except:
            print(f"   Joint {i}: 无法访问bodies属性")
    
    print(f"\n🔍 Link配置:")
    for i, body in enumerate(env.bodies):
        print(f"   Link {i}: mass = {body.mass}, position = {body.position}")
    
    # 记录理论Joint连接点
    theoretical_connections = []
    for i in range(len(env.bodies)):
        if i == 0:
            # 基座关节连接点
            theoretical_connections.append({
                'joint_id': i,
                'link_a': 'static_body',
                'link_b': i,
                'expected_distance': 0.0,  # 基座应该固定在锚点
                'connection_point': env.anchor_point
            })
        else:
            # 其他关节连接点
            prev_link_length = env.link_lengths[i-1]
            theoretical_connections.append({
                'joint_id': i,
                'link_a': i-1,
                'link_b': i,
                'expected_distance': prev_link_length,
                'connection_point': None  # 动态计算
            })
    
    print(f"\n🎮 开始精确检测测试...")
    
    pygame.init()
    clock = pygame.time.Clock()
    
    running = True
    step_count = 0
    joint_disconnection_count = 0
    max_disconnection_distance = 0.0
    
    while running and step_count < 500:
        # 处理pygame事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # 获取按键状态进行手动控制
        keys = pygame.key.get_pressed()
        action = [0.0, 0.0, 0.0, 0.0]
        
        force = 5.0
        if keys[pygame.K_w]:
            action[0] = -force
        elif keys[pygame.K_s]:
            action[0] = force
        if keys[pygame.K_a]:
            action[1] = -force
        elif keys[pygame.K_d]:
            action[1] = force
        if keys[pygame.K_q]:
            action[2] = -force
        elif keys[pygame.K_e]:
            action[2] = force
        if keys[pygame.K_z]:
            action[3] = -force
        elif keys[pygame.K_c]:
            action[3] = force
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        # 🔍 精确检测Joint连接状态
        current_disconnections = []
        
        for connection in theoretical_connections:
            joint_id = connection['joint_id']
            
            if joint_id == 0:
                # 基座关节：检查Link0是否在锚点附近
                base_body = env.bodies[0]
                anchor_point = env.anchor_point
                distance = math.sqrt((base_body.position.x - anchor_point[0])**2 + 
                                   (base_body.position.y - anchor_point[1])**2)
                
                if distance > 5.0:  # 基座偏离锚点超过5像素
                    current_disconnections.append({
                        'joint_id': joint_id,
                        'type': 'base_disconnection',
                        'distance': distance,
                        'expected': 0.0,
                        'deviation': distance
                    })
                    print(f"❌ 基座关节断开! 距离锚点: {distance:.2f}px")
            
            else:
                # 其他关节：检查相邻Link间的距离
                link_a_id = connection['link_a']
                link_b_id = connection['link_b']
                expected_distance = connection['expected_distance']
                
                if link_a_id < len(env.bodies) and link_b_id < len(env.bodies):
                    body_a = env.bodies[link_a_id]
                    body_b = env.bodies[link_b_id]
                    
                    # 计算Link间的实际距离
                    actual_distance = math.sqrt((body_a.position.x - body_b.position.x)**2 + 
                                              (body_a.position.y - body_b.position.y)**2)
                    
                    # 检查是否偏离预期距离过多
                    deviation = abs(actual_distance - expected_distance)
                    deviation_percentage = deviation / expected_distance * 100
                    
                    if deviation > expected_distance * 0.5:  # 偏差超过50%
                        current_disconnections.append({
                            'joint_id': joint_id,
                            'type': 'link_disconnection',
                            'distance': actual_distance,
                            'expected': expected_distance,
                            'deviation': deviation,
                            'deviation_percentage': deviation_percentage
                        })
                        print(f"❌ Joint {joint_id} 断开! Link{link_a_id}-Link{link_b_id}")
                        print(f"   实际距离: {actual_distance:.2f}px")
                        print(f"   预期距离: {expected_distance:.2f}px") 
                        print(f"   偏差: {deviation:.2f}px ({deviation_percentage:.1f}%)")
        
        # 统计断开情况
        if current_disconnections:
            joint_disconnection_count += len(current_disconnections)
            for disconnection in current_disconnections:
                max_disconnection_distance = max(max_disconnection_distance, disconnection['deviation'])
        
        # 渲染
        env.render()
        
        # 每100步报告状态
        if step_count % 100 == 0:
            base_pos = env.bodies[0].position
            end_effector = env._get_end_effector_position()
            print(f"\n📊 步数{step_count}:")
            print(f"   基座位置: ({base_pos.x:.1f}, {base_pos.y:.1f})")
            print(f"   末端位置: [{end_effector[0]:.1f}, {end_effector[1]:.1f}]")
            print(f"   Joint断开次数: {joint_disconnection_count}")
            print(f"   最大断开距离: {max_disconnection_distance:.2f}px")
            
            # 详细检查当前所有Joint状态
            print(f"   当前Joint状态:")
            for i in range(len(env.bodies)):
                if i == 0:
                    base_body = env.bodies[0]
                    distance = math.sqrt((base_body.position.x - env.anchor_point[0])**2 + 
                                       (base_body.position.y - env.anchor_point[1])**2)
                    print(f"     基座-锚点距离: {distance:.2f}px")
                else:
                    if i-1 < len(env.bodies):
                        body_a = env.bodies[i-1]
                        body_b = env.bodies[i]
                        distance = math.sqrt((body_a.position.x - body_b.position.x)**2 + 
                                           (body_a.position.y - body_b.position.y)**2)
                        expected = env.link_lengths[i-1]
                        print(f"     Link{i-1}-Link{i}距离: {distance:.2f}px (预期: {expected}px)")
        
        clock.tick(60)
    
    print(f"\n📊 最终精确检测结果:")
    print(f"   总步数: {step_count}")
    print(f"   Joint断开总次数: {joint_disconnection_count}")
    print(f"   最大断开距离: {max_disconnection_distance:.2f}px")
    print(f"   Joint连接状态: {'❌ 不稳定' if joint_disconnection_count > 0 else '✅ 稳定'}")
    
    env.close()
    pygame.quit()

if __name__ == "__main__":
    precise_joint_detection()
