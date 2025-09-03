#!/usr/bin/env python3
"""
临时禁用路标点系统，恢复基础奖励函数
"""

import sys
import os
sys.path.insert(0, '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/envs')

def disable_waypoint_system():
    """临时禁用路标点系统"""
    
    print("🔄 禁用路标点系统，恢复基础奖励")
    print("="*40)
    
    # 读取环境文件
    env_file = '/home/xli149/Documents/repos/test_robo/examples/2d_reacher/envs/reacher2d_env.py'
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    # 修改奖励选择逻辑
    old_logic = '''        # 🗺️ 如果有路标点系统，使用路标点奖励
        if hasattr(self, 'waypoint_navigator') and self.waypoint_navigator is not None:
            return self._compute_reward_with_waypoints()
        else:
            return self._compute_reward_basic()'''
    
    new_logic = '''        # 🗺️ 临时禁用路标点系统，强制使用基础奖励
        # if hasattr(self, 'waypoint_navigator') and self.waypoint_navigator is not None:
        #     return self._compute_reward_with_waypoints()
        # else:
        return self._compute_reward_basic()'''
    
    if old_logic in content:
        content = content.replace(old_logic, new_logic)
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ 已禁用路标点系统")
        print("   现在使用基础奖励函数")
        print("   建议重新训练以测试稳定性")
    else:
        print("❌ 未找到目标代码段")

if __name__ == "__main__":
    disable_waypoint_system()
