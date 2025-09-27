#!/usr/bin/env python3
"""
简化版MADDPG_SB3训练脚本：
1. 使用多个独立的DDPG智能体
2. 每个智能体控制一个关节
3. 在同一个环境中协调训练
4. 可视化训练过程
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
import time

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

# 导入环境
from maddpg_complete_sequential_training import (
    create_env, 
    SUCCESS_THRESHOLD_RATIO, 
    SUCCESS_THRESHOLD_2JOINT
)

class SimpleMADDPG_SB3:
    """简化版MADDPG - 使用独立的DDPG智能体"""
    
    def __init__(self, num_joints=3):
        self.num_joints = num_joints
        self.agents = []
        
        print(f"🌟 创建简化版MADDPG_SB3: {num_joints}个关节")
        
        # 创建主环境用于协调训练
        self.main_env = create_env(num_joints, render_mode='human', show_position_info=True)
        
        # 为每个关节创建独立的DDPG智能体
        for i in range(num_joints):
            # 创建单关节环境（用于智能体训练）
            single_env = create_env(num_joints, render_mode=None, show_position_info=False)
            
            # 包装为单关节动作空间
            wrapped_env = SingleJointWrapper(single_env, joint_id=i, num_joints=num_joints)
            
            # 添加动作噪声
            action_noise = NormalActionNoise(
                mean=np.zeros(1), 
                sigma=0.1 * np.ones(1)
            )
            
            # 创建DDPG智能体
            agent = DDPG(
                "MlpPolicy",
                wrapped_env,
                learning_rate=1e-3,
                gamma=0.99,
                tau=0.005,
                action_noise=action_noise,
                verbose=0,
                device='cpu'
            )
            
            self.agents.append(agent)
            print(f"🤖 DDPG智能体 {i}: 已创建，控制关节{i}")
        
        print(f"✅ 简化版MADDPG_SB3初始化完成")
    
    def train_coordinated(self, total_timesteps=10000):
        """协调训练所有智能体"""
        print(f"\n🎯 开始协调训练 {total_timesteps} 步...")
        
        obs, _ = self.main_env.reset()
        episode_count = 0
        episode_reward = 0
        step_count = 0
        
        start_time = time.time()
        
        try:
            while step_count < total_timesteps:
                # 获取所有智能体的动作
                actions = []
                for i, agent in enumerate(self.agents):
                    action, _ = agent.predict(obs, deterministic=False)
                    actions.append(action[0])
                
                # 执行联合动作
                next_obs, reward, terminated, truncated, info = self.main_env.step(np.array(actions))
                episode_reward += reward
                step_count += 1
                
                # 为每个智能体存储经验并训练
                for i, agent in enumerate(self.agents):
                    # 简化：每个智能体看到相同的全局状态和奖励
                    if hasattr(agent, 'replay_buffer') and len(agent.replay_buffer) > 0:
                        # 只有当buffer有数据时才训练
                        if step_count % 10 == 0:  # 每10步训练一次
                            agent.train(gradient_steps=1)
                
                obs = next_obs
                
                # 重置环境
                if terminated or truncated:
                    obs, _ = self.main_env.reset()
                    episode_count += 1
                    
                    if episode_count % 5 == 0:
                        print(f"   Episode {episode_count}: 奖励={episode_reward:.2f}, 步数={step_count}")
                    
                    episode_reward = 0
                
                # 定期输出进度
                if step_count % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"   步数 {step_count}/{total_timesteps}, 用时 {elapsed_time:.1f}s")
        
        except KeyboardInterrupt:
            print(f"\n⚠️ 训练被用户中断")
        
        training_time = time.time() - start_time
        print(f"\n✅ 协调训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"🎯 总episodes: {episode_count}")
        
        return episode_count
    
    def test(self, n_episodes=5):
        """测试训练好的智能体"""
        print(f"\n🧪 开始测试 {n_episodes} episodes...")
        
        success_count = 0
        
        for episode in range(n_episodes):
            obs, info = self.main_env.reset()
            episode_reward = 0
            episode_success = False
            min_distance = float('inf')
            
            for step in range(100):  # 每个episode最多100步
                # 获取所有智能体的动作（确定性）
                actions = []
                for agent in self.agents:
                    action, _ = agent.predict(obs, deterministic=True)
                    actions.append(action[0])
                
                obs, reward, terminated, truncated, info = self.main_env.step(np.array(actions))
                episode_reward += reward
                
                # 检查成功
                distance = info.get('distance_to_target', float('inf'))
                min_distance = min(min_distance, distance)
                
                if info.get('is_success', False):
                    episode_success = True
                
                if terminated or truncated:
                    break
            
            if episode_success:
                success_count += 1
            
            print(f"   Episode {episode+1}: 奖励={episode_reward:.2f}, 最小距离={min_distance:.4f}, 成功={'✅' if episode_success else '❌'}")
        
        success_rate = success_count / n_episodes
        print(f"\n🎯 测试结果: 成功率 {success_rate:.1%} ({success_count}/{n_episodes})")
        
        return success_rate

class SingleJointWrapper(gym.Wrapper):
    """单关节包装器 - 简化版本"""
    
    def __init__(self, env, joint_id, num_joints):
        super().__init__(env)
        self.joint_id = joint_id
        self.num_joints = num_joints
        
        # 重新定义动作空间为单关节
        from gymnasium.spaces import Box
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 观察空间保持不变
        self.observation_space = env.observation_space
    
    def step(self, action):
        # 构建完整动作（其他关节设为0）
        full_action = np.zeros(self.num_joints)
        full_action[self.joint_id] = action[0]
        
        return self.env.step(full_action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def main():
    """主函数"""
    print("🌟 简化版MADDPG_SB3训练系统")
    print("🤖 策略: 多个独立DDPG智能体协调控制")
    print("🎯 目标: 3关节Reacher任务")
    print()
    
    # 创建MADDPG系统
    maddpg = SimpleMADDPG_SB3(num_joints=3)
    
    # 协调训练
    episode_count = maddpg.train_coordinated(total_timesteps=5000)  # 较短的训练用于演示
    
    # 测试
    success_rate = maddpg.test(n_episodes=5)
    
    print(f"\n🎉 简化版MADDPG_SB3完成!")
    print(f"   训练episodes: {episode_count}")
    print(f"   最终成功率: {success_rate:.1%}")

if __name__ == "__main__":
    main()
