
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
