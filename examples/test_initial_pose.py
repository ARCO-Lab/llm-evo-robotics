#!/usr/bin/env python3
"""
测试机器人初始姿态的脚本
专门用于观察机器人初始化时的方向
"""

import sys
import os
import numpy as np
import time

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from reacher2d_env import Reacher2DEnv

def test_initial_pose():
    """测试机器人初始姿态"""
    print("=" * 60)
    print("🤖 测试机器人初始姿态")
    print("=" * 60)
    
    # 创建环境 - 使用与enhanced_train.py相同的配置
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 80, 80, 60],  # 与enhanced_train.py相同
        render_mode='human',
        config_path="/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='INFO'
    )
    
    print(f"📍 基座位置: {env.anchor_point}")
    print(f"🎯 目标位置: {env.goal_pos}")
    
    # 重置环境多次观察
    for reset_count in range(3):
        print(f"\n🔄 重置 #{reset_count + 1}")
        
        # 重置环境
        obs = env.reset()
        
        print(f"📐 关节角度:")
        for i, angle in enumerate(env.joint_angles):
            degrees = np.degrees(angle)
            print(f"   关节{i}: {angle:.4f} 弧度 = {degrees:.2f}°")
        
        # 计算并显示位置
        positions = env._calculate_link_positions()
        print(f"📍 Link位置:")
        for i, pos in enumerate(positions):
            if i == 0:
                print(f"   基座: [{pos[0]:.1f}, {pos[1]:.1f}]")
            else:
                prev_pos = positions[i-1]
                direction_x = pos[0] - prev_pos[0]
                direction_y = pos[1] - prev_pos[1]
                
                # 判断主要方向
                if abs(direction_x) > abs(direction_y):
                    main_dir = "→右" if direction_x > 0 else "←左"
                else:
                    main_dir = "↓下" if direction_y > 0 else "↑上"
                
                print(f"   Link{i}: [{pos[0]:.1f}, {pos[1]:.1f}] (相对方向: {main_dir})")
        
        # 渲染环境
        env.render()
        
        # 等待用户观察
        print(f"🖼️ 请观察渲染窗口中机器人的姿态...")
        print(f"   - 基座在 [{env.anchor_point[0]}, {env.anchor_point[1]}]")
        print(f"   - 第一个Link终点在 [{positions[1][0]:.1f}, {positions[1][1]:.1f}]")
        
        if reset_count < 2:
            print("⏳ 3秒后进行下一次重置...")
            time.sleep(3)
        else:
            print("🔍 请仔细观察最后一次的初始姿态...")
            print("   按Ctrl+C结束观察")
    
    try:
        # 保持窗口打开让用户观察
        print("\n🖼️ 保持渲染窗口打开，按Ctrl+C结束...")
        while True:
            env.render()
            time.sleep(0.1)  # 保持窗口刷新
            
    except KeyboardInterrupt:
        print("\n✅ 观察结束")
    
    finally:
        env.close()
        print("🔒 环境已关闭")

if __name__ == "__main__":
    test_initial_pose()
