#!/usr/bin/env python3
"""
将路标点系统集成到Reacher2D环境中
"""

import sys
import os
sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/envs')
sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/surrogate_model')

from reacher2d_env import Reacher2DEnv
from waypoint_navigator import WaypointNavigator
import numpy as np

def add_waypoint_system_to_env():
    """为环境添加路标点系统"""
    
    print("🔧 集成路标点系统到Reacher2D环境")
    print("="*50)
    
    # 读取当前环境配置
    config_path = '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml'
    
    # 创建修改后的奖励函数
    waypoint_reward_code = '''
    def _compute_reward_with_waypoints(self):
        """带路标点的奖励函数"""
        end_effector_pos = np.array(self._get_end_effector_position())
        
        # === 1. 路标点导航奖励 ===
        waypoint_reward, waypoint_info = self.waypoint_navigator.update(end_effector_pos)
        
        # === 2. 基础距离奖励（到当前目标的距离）===
        current_target = self.waypoint_navigator.get_current_target()
        distance_to_target = np.linalg.norm(end_effector_pos - current_target)
        
        # 根据当前目标计算距离奖励
        max_distance = 200.0
        distance_reward = -distance_to_target / max_distance * 0.5  # 较小的距离惩罚
        
        # === 3. 进度奖励 ===
        if not hasattr(self, 'prev_distance_to_target'):
            self.prev_distance_to_target = distance_to_target
        
        progress = self.prev_distance_to_target - distance_to_target
        progress_reward = np.clip(progress * 3.0, -1.0, 1.0)  # 加大进度奖励
        self.prev_distance_to_target = distance_to_target
        
        # === 4. 完成度奖励 ===
        completion_progress = waypoint_info.get('completion_progress', 0.0)
        completion_bonus = completion_progress * 5.0  # 根据完成度给予奖励
        
        # === 5. 碰撞惩罚（保持原有） ===
        collision_penalty = self._get_collision_penalty()
        
        # === 6. 总奖励计算 ===
        total_reward = (
            waypoint_reward +       # [0, 50] 路标点奖励
            distance_reward +       # [-1, 0] 距离惩罚
            progress_reward +       # [-1, 1] 进度奖励
            completion_bonus +      # [0, 5] 完成度奖励
            collision_penalty       # [-2, 0] 碰撞惩罚
        )
        
        # === 7. 调试信息 ===
        if hasattr(self, 'step_counter') and self.step_counter % 100 == 0:
            print(f"💰 [waypoint_reward] Step {self.step_counter}:")
            print(f"   路标奖励: {waypoint_reward:.2f}")
            print(f"   距离奖励: {distance_reward:.2f} (距离: {distance_to_target:.1f})")
            print(f"   进度奖励: {progress_reward:.2f}")
            print(f"   完成奖励: {completion_bonus:.2f}")
            print(f"   总奖励: {total_reward:.2f}")
            print(f"   当前目标: {current_target}")
            print(f"   完成进度: {completion_progress*100:.1f}%")
        
        return total_reward
    '''
    
    # 初始化代码
    init_waypoint_code = '''
    def _init_waypoint_system(self):
        """初始化路标点系统"""
        # 获取起点和终点
        start_pos = self.anchor_point  # 锚点作为起点
        goal_pos = self.goal_pos
        
        # 创建路标点导航器
        self.waypoint_navigator = WaypointNavigator(start_pos, goal_pos)
        
        print(f"🗺️ 路标点系统已初始化")
        print(f"   起点: {start_pos}")
        print(f"   终点: {goal_pos}")
    '''
    
    # 重置代码
    reset_waypoint_code = '''
    def _reset_waypoint_system(self):
        """重置路标点系统"""
        if hasattr(self, 'waypoint_navigator'):
            self.waypoint_navigator.reset()
        else:
            self._init_waypoint_system()
    '''
    
    print("📝 生成的集成代码:")
    print("="*30)
    print("1. 初始化函数:")
    print(init_waypoint_code)
    print("\n2. 重置函数:")
    print(reset_waypoint_code)
    print("\n3. 奖励函数 (部分):")
    print(waypoint_reward_code[:500] + "...")
    
    return waypoint_reward_code, init_waypoint_code, reset_waypoint_code

def create_waypoint_integration_patch():
    """创建环境集成补丁"""
    
    print("\n🔧 创建环境集成补丁")
    print("="*30)
    
    patch_content = '''
# === 路标点系统集成补丁 ===
# 将以下代码添加到 reacher2d_env.py 中

import sys
sys.path.append('/home/xli149/Documents/repos/test_robo/examples/surrogate_model')
from waypoint_navigator import WaypointNavigator

# 在 __init__ 方法中添加:
def _setup_waypoint_system(self):
    """设置路标点系统"""
    self.use_waypoints = True  # 启用路标点
    self._init_waypoint_system()

# 在 reset 方法中添加:
def _reset_waypoints_on_reset(self):
    """在环境重置时重置路标点"""
    if hasattr(self, 'waypoint_navigator'):
        self.waypoint_navigator.reset()

# 替换 _compute_reward 方法:
def _compute_reward_with_waypoints(self):
    """使用路标点的奖励函数"""
    if not hasattr(self, 'waypoint_navigator'):
        # fallback to original reward
        return self._compute_reward_original()
    
    end_effector_pos = np.array(self._get_end_effector_position())
    
    # 路标点导航奖励
    waypoint_reward, waypoint_info = self.waypoint_navigator.update(end_effector_pos)
    
    # 当前目标（动态切换的路标点）
    current_target = self.waypoint_navigator.get_current_target()
    distance_to_target = np.linalg.norm(end_effector_pos - current_target)
    
    # 距离奖励（针对当前路标点）
    distance_reward = -distance_to_target / 200.0 * 0.5
    
    # 进度奖励
    if not hasattr(self, 'prev_waypoint_distance'):
        self.prev_waypoint_distance = distance_to_target
    
    progress = self.prev_waypoint_distance - distance_to_target  
    progress_reward = np.clip(progress * 3.0, -1.0, 1.0)
    self.prev_waypoint_distance = distance_to_target
    
    # 碰撞惩罚
    collision_penalty = self._get_collision_penalty()
    
    total_reward = waypoint_reward + distance_reward + progress_reward + collision_penalty
    
    return total_reward
'''
    
    with open("examples/surrogate_model/waypoint_integration_patch.py", "w") as f:
        f.write(patch_content)
    
    print("✅ 集成补丁已保存到: waypoint_integration_patch.py")

def test_integration():
    """测试集成效果"""
    
    print("\n🧪 测试路标点集成效果")
    print("="*30)
    
    # 创建环境
    env = Reacher2DEnv(
        num_links=3,
        link_lengths=[60, 60, 60],
        render_mode=None,
        config_path='/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml'
    )
    
    # 手动添加路标点系统
    start_pos = env.anchor_point
    goal_pos = env.goal_pos
    
    navigator = WaypointNavigator(start_pos, goal_pos)
    env.waypoint_navigator = navigator
    
    print(f"✅ 测试环境创建成功")
    print(f"   锚点: {start_pos}")
    print(f"   目标: {goal_pos}")
    print(f"   路标点数: {len(navigator.waypoints)}")
    
    # 测试几步
    obs = env.reset()
    
    for step in range(10):
        # 随机动作
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # 获取当前位置
        end_pos = env._get_end_effector_position()
        
        # 更新路标点系统
        waypoint_reward, waypoint_info = navigator.update(np.array(end_pos))
        
        if step % 3 == 0:
            print(f"步骤 {step}: 位置 {np.array(end_pos).astype(int)}, "
                  f"路标奖励 {waypoint_reward:.2f}, 进度 {waypoint_info['completion_progress']*100:.1f}%")
    
    print("✅ 集成测试完成")

if __name__ == "__main__":
    # 生成集成代码
    reward_code, init_code, reset_code = add_waypoint_system_to_env()
    
    # 创建补丁文件
    create_waypoint_integration_patch()
    
    # 测试集成
    test_integration()
    
    print("\n🎯 下一步：要将路标点系统正式集成到环境中吗？")
    print("   这将修改 reacher2d_env.py 文件")
