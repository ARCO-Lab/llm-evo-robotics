#!/usr/bin/env python3
"""
可视化测试goal渲染
"""

import sys
import os
import pygame
import time
sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/envs')
from reacher2d_env import Reacher2DEnv

def test_visual_goal():
    print("🎮 可视化测试goal位置变化")
    print("="*50)
    
    # 测试不同的goal位置
    test_positions = [
        [120, 600],  # 当前配置
        [300, 550],  # 锚点位置
        [400, 550],  # 右侧
        [300, 400],  # 上方
        [500, 650],  # 远处
    ]
    
    for i, goal_pos in enumerate(test_positions):
        print(f"\n📍 测试 {i+1}: goal位置 {goal_pos}")
        
        # 先修改配置文件
        config_content = f"""obstacles:
  # 锯齿形障碍物 - 移动到机器人可达范围内
  - shape: segment
    points: [[500, 500], [550, 550]]
  - shape: segment
    points: [[550, 550], [600, 500]]

goal:
  position: {goal_pos}  # 测试位置
  radius: 10

start:
  position: [300, 550]  # 保持起始位置不变
  angle: 0
"""
        
        # 写入配置文件
        with open("../2d_reacher/configs/test_goal.yaml", "w") as f:
            f.write(config_content)
        
        # 创建环境
        env = Reacher2DEnv(
            num_links=3,
            link_lengths=[60, 60, 60],
            render_mode="human",  # 开启渲染
            config_path="../2d_reacher/configs/test_goal.yaml"
        )
        
        obs = env.reset()
        
        print(f"   配置goal: {env.config['goal']['position']}")
        print(f"   实际goal_pos: {env.goal_pos}")
        print(f"   末端位置: {env._get_end_effector_position()}")
        
        # 渲染几帧
        for frame in range(10):
            env.render()
            pygame.display.flip()
            time.sleep(0.1)
        
        # 清理
        pygame.quit()
        
        # 等待用户确认
        input(f"   看到goal在 {goal_pos} 了吗？按回车继续...")

if __name__ == "__main__":
    test_visual_goal()
