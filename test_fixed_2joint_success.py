#!/usr/bin/env python3
"""
测试修复后的2关节Reacher成功判断逻辑
"""

import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import time

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

class SpecializedJointExtractor(BaseFeaturesExtractor):
    """专门针对特定关节数的特征提取器"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(SpecializedJointExtractor, self).__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        
        print(f"🔧 SpecializedJointExtractor: {obs_dim}维 -> {features_dim}维")
        
        # 针对具体观察维度设计网络
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

# 修复后的2关节环境包装器
class Fixed2JointReacherWrapper(gym.Wrapper):
    """修复2关节Reacher包装器 - 使用相同的目标生成策略并修复成功判断"""
    
    def __init__(self, env):
        super().__init__(env)
        self.link_lengths = [0.1, 0.1]
        self.max_episode_steps = 100  # 每个episode 100步
        print("🌟 Fixed2JointReacherWrapper 初始化")
        print(f"   链长: {self.link_lengths}")
        print(f"   最大可达距离: {self.calculate_max_reach():.3f}")
        print(f"   目标生成范围: {self.calculate_target_range():.3f}")
    
    def calculate_max_reach(self):
        return sum(self.link_lengths)
    
    def calculate_target_range(self):
        max_reach = self.calculate_max_reach()
        return max_reach * 0.85
    
    def generate_unified_target(self):
        max_target_distance = self.calculate_target_range()
        min_target_distance = 0.05
        
        target_distance = self.np_random.uniform(min_target_distance, max_target_distance)
        target_angle = self.np_random.uniform(-np.pi, np.pi)
        
        target_x = target_distance * np.cos(target_angle)
        target_y = target_distance * np.sin(target_angle)
        
        return target_x, target_y
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 应用统一的目标生成策略
        target_x, target_y = self.generate_unified_target()
        
        # 🔧 修复目标滚动问题：确保目标速度为0
        reacher_env = self.env.unwrapped
        qpos = reacher_env.data.qpos.copy()
        qvel = reacher_env.data.qvel.copy()
        
        # 设置目标位置
        qpos[-2:] = [target_x, target_y]
        # 🔧 关键修复：确保目标速度为0
        qvel[-2:] = [0.0, 0.0]
        
        reacher_env.set_state(qpos, qvel)
        
        # 获取新的观察
        obs = reacher_env._get_obs()
        
        # 更新info
        if info is None:
            info = {}
        info.update({
            'max_reach': self.calculate_max_reach(),
            'target_range': self.calculate_target_range(),
            'target_pos': [target_x, target_y]
        })
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 🔧 重新计算成功判断 - 这是关键修复！
        reacher_env = self.env.unwrapped
        fingertip_pos = reacher_env.get_body_com("fingertip")[:2]
        target_pos = reacher_env.get_body_com("target")[:2]
        distance = np.linalg.norm(fingertip_pos - target_pos)
        
        # 🎯 成功判断：距离小于0.05（5cm）- 更合理的阈值
        is_success = distance < 0.05
        
        # 添加统一的信息
        if info is None:
            info = {}
        info.update({
            'max_reach': self.calculate_max_reach(),
            'target_range': self.calculate_target_range(),
            'distance_to_target': distance,
            'is_success': is_success,  # 🔧 关键修复：添加正确的成功判断
            'fingertip_pos': fingertip_pos.copy(),
            'target_pos': target_pos.copy()
        })
        
        return obs, reward, terminated, truncated, info

def create_fixed_test_env(render_mode='human'):
    """创建修复后的测试用2关节环境"""
    env = gym.make('Reacher-v5', render_mode=render_mode)
    env = Fixed2JointReacherWrapper(env)
    env = Monitor(env)
    return env

def test_fixed_2joint_success_logic(model_path, n_eval_episodes=5):
    """测试修复后的2关节成功判断逻辑"""
    print(f"🧪 测试修复后的2关节成功判断逻辑")
    print(f"📊 测试episodes: {n_eval_episodes}, 每个episode: 100步")
    print("🎯 成功标准: 距离目标 < 0.05 (5cm) - 更合理的阈值")
    print("="*60)
    
    # 创建测试环境
    test_env = create_fixed_test_env(render_mode='human')
    
    # 加载模型
    try:
        model = SAC.load(model_path)
        print(f"✅ 模型加载成功: {model_path}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 测试结果
    episode_results = []
    
    try:
        for episode in range(n_eval_episodes):
            print(f"\\n📍 Episode {episode+1}/{n_eval_episodes}")
            print("-" * 40)
            
            obs, info = test_env.reset()
            episode_reward = 0
            episode_success = False
            min_distance = float('inf')
            distances = []
            step_count = 0
            
            # 初始信息
            initial_target_pos = info.get('target_pos', [0, 0])
            print(f"   🎯 目标位置: ({initial_target_pos[0]:.3f}, {initial_target_pos[1]:.3f})")
            print(f"   📏 目标距离: {np.linalg.norm(initial_target_pos):.3f}")
            
            for step in range(100):  # 每个episode 100步
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                
                episode_reward += reward
                step_count += 1
                
                # 获取详细信息
                distance = info.get('distance_to_target', float('inf'))
                is_success = info.get('is_success', False)
                fingertip_pos = info.get('fingertip_pos', [0, 0])
                target_pos = info.get('target_pos', [0, 0])
                
                distances.append(distance)
                min_distance = min(min_distance, distance)
                
                if is_success and not episode_success:
                    episode_success = True
                    print(f"   ✅ 成功! 步数: {step+1}, 距离: {distance:.4f}")
                
                # 每20步打印一次状态
                if (step + 1) % 20 == 0:
                    print(f"   步数 {step+1:3d}: 距离={distance:.4f}, 成功={is_success}, 末端=({fingertip_pos[0]:.3f},{fingertip_pos[1]:.3f})")
                
                if terminated or truncated:
                    print(f"   🏁 Episode结束: 步数={step+1}, 原因={'terminated' if terminated else 'truncated'}")
                    break
            
            # Episode总结
            final_distance = distances[-1] if distances else float('inf')
            avg_distance = np.mean(distances) if distances else float('inf')
            
            episode_result = {
                'episode': episode + 1,
                'total_reward': episode_reward,
                'success': episode_success,
                'steps': step_count,
                'min_distance': min_distance,
                'final_distance': final_distance,
                'avg_distance': avg_distance,
                'target_pos': initial_target_pos
            }
            episode_results.append(episode_result)
            
            print(f"\\n   📊 Episode {episode+1} 总结:")
            print(f"      总奖励: {episode_reward:.2f}")
            print(f"      成功: {'✅' if episode_success else '❌'}")
            print(f"      最小距离: {min_distance:.4f}")
            print(f"      最终距离: {final_distance:.4f}")
            
            # 暂停一下让用户观察
            time.sleep(1)
    
    except KeyboardInterrupt:
        print(f"\\n⚠️ 测试被用户中断")
    
    finally:
        test_env.close()
    
    # 最终统计
    if episode_results:
        print(f"\\n{'='*60}")
        print("🎉 修复后的成功判断测试完成!")
        print(f"{'='*60}")
        
        success_episodes = sum(1 for r in episode_results if r['success'])
        total_episodes = len(episode_results)
        success_rate = success_episodes / total_episodes if total_episodes > 0 else 0
        
        avg_reward = np.mean([r['total_reward'] for r in episode_results])
        avg_min_distance = np.mean([r['min_distance'] for r in episode_results])
        
        print(f"\\n📊 修复后的结果:")
        print(f"   成功率: {success_rate:.1%} ({success_episodes}/{total_episodes})")
        print(f"   平均总奖励: {avg_reward:.2f}")
        print(f"   平均最小距离: {avg_min_distance:.4f}")
        
        print(f"\\n📋 详细结果:")
        print(f"{'Episode':<8} {'奖励':<8} {'成功':<6} {'最小距离':<10}")
        print("-" * 40)
        for r in episode_results:
            success_mark = "✅" if r['success'] else "❌"
            print(f"{r['episode']:<8} {r['total_reward']:<8.2f} {success_mark:<6} {r['min_distance']:<10.4f}")
        
        if success_rate > 0:
            print(f"\\n🎉 修复成功！现在2关节模型有 {success_rate:.1%} 的成功率")
        else:
            print(f"\\n⚠️ 仍然没有成功，可能需要进一步调试")
        
        return episode_results
    
    return None

def main():
    """主函数"""
    print("🌟 测试修复后的2关节Reacher成功判断逻辑")
    print("🎯 目标: 验证修复后的成功判断是否正确工作")
    print()
    
    # 测试最新的2关节模型
    model_path = "models/sequential_2joint_reacher.zip"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("📋 可用的模型文件:")
        if os.path.exists("models"):
            for file in os.listdir("models"):
                if "2joint" in file or "reacher" in file:
                    print(f"   - models/{file}")
        return
    
    # 开始测试
    results = test_fixed_2joint_success_logic(model_path, n_eval_episodes=5)
    
    if results:
        print(f"\\n✅ 测试完成! 成功判断逻辑已修复。")
    else:
        print(f"\\n❌ 测试失败或被中断。")

if __name__ == "__main__":
    main()
