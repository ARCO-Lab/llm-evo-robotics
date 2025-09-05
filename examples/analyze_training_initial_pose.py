#!/usr/bin/env python3
"""
分析enhanced_train.py中的初始姿态
完全模拟训练脚本的环境创建过程
"""

import sys
import os
import numpy as np
import time

# 添加路径 - 完全模拟enhanced_train.py的路径设置
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/gnn_encoder'))
sys.path.insert(0, os.path.join(base_dir, 'examples/rl/train'))  
sys.path.insert(0, os.path.join(base_dir, 'examples/rl/common'))
sys.path.insert(0, os.path.join(base_dir, 'examples/rl/environments'))
sys.path.append(os.path.join(base_dir, 'examples/rl'))

from reacher2d_env import Reacher2DEnv

def analyze_training_initial_pose():
    """分析训练中的初始姿态"""
    print("=" * 70)
    print("🔍 分析enhanced_train.py中的机器人初始姿态")
    print("=" * 70)
    
    # 完全模拟enhanced_train.py的环境参数
    num_links = 4
    link_lengths = [80, 80, 80, 60]
    
    env_params = {
        'num_links': num_links,
        'link_lengths': link_lengths,
        'render_mode': 'human',
        'config_path': "/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    }
    
    print(f"📊 环境参数:")
    print(f"   链接数量: {env_params['num_links']}")
    print(f"   链接长度: {env_params['link_lengths']}")
    print(f"   配置文件: {env_params['config_path']}")
    
    # 创建环境 - 模拟训练脚本中的sync_env创建
    print(f"\n🤖 创建渲染环境 (sync_env)...")
    render_env_params = env_params.copy()
    render_env_params['render_mode'] = 'human'
    sync_env = Reacher2DEnv(**render_env_params)
    
    print(f"📍 基座位置: {sync_env.anchor_point}")
    print(f"🎯 基础目标位置: {sync_env.base_goal_pos}")
    
    # 多次重置观察
    for reset_count in range(5):
        print(f"\n{'='*50}")
        print(f"🔄 重置 #{reset_count + 1}")
        print(f"{'='*50}")
        
        # 重置环境
        obs = sync_env.reset()
        
        print(f"📐 关节角度 (弧度/度):")
        total_angle = 0
        for i, angle in enumerate(sync_env.joint_angles):
            degrees = np.degrees(angle)
            total_angle += angle
            print(f"   关节{i}: {angle:+7.4f} 弧度 = {degrees:+7.2f}° (累积: {np.degrees(total_angle):+7.2f}°)")
        
        # 计算位置
        positions = sync_env._calculate_link_positions()
        
        print(f"\n📍 Link位置和方向:")
        for i, pos in enumerate(positions):
            if i == 0:
                print(f"   基座: [{pos[0]:7.1f}, {pos[1]:7.1f}]")
            else:
                prev_pos = positions[i-1]
                dx = pos[0] - prev_pos[0]
                dy = pos[1] - prev_pos[1]
                
                # 计算方向角度
                link_angle = np.arctan2(dy, dx)
                link_degrees = np.degrees(link_angle)
                
                # 判断主要方向
                if abs(dx) > abs(dy):
                    main_dir = "→右" if dx > 0 else "←左"
                    dominant = "水平"
                else:
                    main_dir = "↓下" if dy > 0 else "↑上"  
                    dominant = "垂直"
                
                print(f"   Link{i}: [{pos[0]:7.1f}, {pos[1]:7.1f}] → dx={dx:+6.1f}, dy={dy:+6.1f}")
                print(f"           方向角度: {link_degrees:+7.2f}° ({dominant}为主, {main_dir})")
        
        # 特别分析第一个Link
        if len(positions) > 1:
            base_pos = positions[0]
            first_link_end = positions[1]
            dx = first_link_end[0] - base_pos[0]
            dy = first_link_end[1] - base_pos[1]
            
            print(f"\n🔍 第一个Link详细分析:")
            print(f"   基座 → Link1: [{base_pos[0]:.1f}, {base_pos[1]:.1f}] → [{first_link_end[0]:.1f}, {first_link_end[1]:.1f}]")
            print(f"   位移向量: dx={dx:+7.2f}, dy={dy:+7.2f}")
            print(f"   长度: {np.sqrt(dx*dx + dy*dy):.2f} (应该是 {sync_env.link_lengths[0]})")
            
            # 判断主要方向
            if abs(dx) > abs(dy):
                if abs(dx) > abs(dy) * 2:  # 明显水平
                    direction_desc = "🚨 明显水平向右" if dx > 0 else "🚨 明显水平向左"
                else:
                    direction_desc = "🔶 偏水平向右" if dx > 0 else "🔶 偏水平向左"
            else:
                if abs(dy) > abs(dx) * 2:  # 明显垂直
                    direction_desc = "✅ 明显垂直向下" if dy > 0 else "✅ 明显垂直向上"
                else:
                    direction_desc = "🔶 偏垂直向下" if dy > 0 else "🔶 偏垂直向上"
            
            print(f"   方向判断: {direction_desc}")
            
            # 角度分析
            link_angle = np.arctan2(dy, dx)
            link_degrees = np.degrees(link_angle)
            print(f"   实际角度: {link_degrees:+7.2f}° (0°=右, 90°=下, 180°=左, 270°=上)")
            
            # 与期望角度比较
            expected_angle = 90  # 期望垂直向下
            angle_diff = abs(link_degrees - expected_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            print(f"   与垂直向下的偏差: {angle_diff:.2f}°")
            
            if angle_diff < 10:
                print(f"   ✅ 基本垂直向下")
            elif angle_diff < 30:
                print(f"   ⚠️ 轻微偏离垂直")
            else:
                print(f"   🚨 明显偏离垂直!")
        
        # 渲染观察
        sync_env.render()
        
        if reset_count < 4:
            print(f"\n⏳ 2秒后下一次重置...")
            time.sleep(2)
    
    print(f"\n{'='*70}")
    print(f"🎯 总结:")
    print(f"   如果你看到的是'水平向右'，可能的原因：")
    print(f"   1. 随机扰动使角度偏离了90°")
    print(f"   2. 课程学习改变了初始设置")
    print(f"   3. 视觉上的误判")
    print(f"   4. 不同的环境实例有不同的行为")
    print(f"{'='*70}")
    
    try:
        print(f"\n🖼️ 保持渲染窗口打开，仔细观察机器人姿态...")
        print(f"   请特别注意第一个Link(蓝色线段)的方向")
        print(f"   按Ctrl+C结束观察")
        
        while True:
            sync_env.render()
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print(f"\n✅ 观察结束")
    
    finally:
        sync_env.close()
        print(f"🔒 环境已关闭")

if __name__ == "__main__":
    analyze_training_initial_pose()
