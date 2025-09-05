#!/usr/bin/env python3
"""
测试角度修复是否生效
"""

import sys
import os
import numpy as np

# 添加路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from reacher2d_env import Reacher2DEnv

def test_angle_fix():
    """测试角度修复"""
    print("=" * 60)
    print("🔧 测试角度修复是否生效")
    print("=" * 60)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=4,
        link_lengths=[80, 80, 80, 60],
        render_mode='human',
        config_path="/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml",
        debug_level='SILENT'
    )
    
    print(f"📍 基座位置: {env.anchor_point}")
    
    # 测试3次重置
    for i in range(3):
        print(f"\n🔄 重置 #{i+1}")
        
        env.reset()
        
        # 显示基座角度
        base_angle_rad = env.joint_angles[0]
        base_angle_deg = np.degrees(base_angle_rad)
        
        print(f"📐 基座角度: {base_angle_rad:.4f} 弧度 = {base_angle_deg:.2f}°")
        
        # 计算第一个Link的终点
        positions = env._calculate_link_positions()
        base_pos = positions[0]
        first_link_end = positions[1]
        
        dx = first_link_end[0] - base_pos[0]
        dy = first_link_end[1] - base_pos[1]
        
        print(f"📍 基座: [{base_pos[0]:.1f}, {base_pos[1]:.1f}]")
        print(f"📍 Link1终点: [{first_link_end[0]:.1f}, {first_link_end[1]:.1f}]")
        print(f"📏 位移: dx={dx:+7.2f}, dy={dy:+7.2f}")
        
        # 判断方向
        if abs(dx) > abs(dy):
            if abs(dx) > abs(dy) * 2:
                direction = "🚨 明显水平向右" if dx > 0 else "🚨 明显水平向左"
            else:
                direction = "🔶 偏水平向右" if dx > 0 else "🔶 偏水平向左"
        else:
            if abs(dy) > abs(dx) * 2:
                direction = "✅ 明显垂直向下" if dy > 0 else "✅ 明显垂直向上"
            else:
                direction = "🔶 偏垂直向下" if dy > 0 else "🔶 偏垂直向上"
        
        print(f"🧭 方向: {direction}")
        
        # 渲染
        env.render()
        
        if i < 2:
            import time
            time.sleep(2)
    
    print(f"\n🖼️ 请观察渲染窗口，现在机器人应该是垂直向下的")
    print(f"按Ctrl+C结束...")
    
    try:
        while True:
            env.render()
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n✅ 测试完成")
    
    env.close()

if __name__ == "__main__":
    test_angle_fix()

