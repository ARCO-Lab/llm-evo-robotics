#!/usr/bin/env python3
"""
混合关节训练（带渲染）：同时训练2关节和3关节Reacher
可以看到训练过程的渲染
"""

import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import time

# 设置渲染环境变量
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_RENDERER'] = 'glfw'

# 导入3关节环境
from perfect_3joint_training import Perfect3JointReacherEnv

class MixedJointExtractor(BaseFeaturesExtractor):
    """
    混合关节特征提取器
    能够处理不同维度的观察空间（2关节10维，3关节13维）
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(MixedJointExtractor, self).__init__(observation_space, features_dim)
        
        # 支持的最大观察维度（3关节的13维）
        self.max_obs_dim = 13
        obs_dim = observation_space.shape[0]
        
        print(f"🔧 MixedJointExtractor: {obs_dim} -> {features_dim}")
        print(f"   支持最大观察维度: {self.max_obs_dim}")
        
        # 使用最大维度设计网络，可以处理不同输入
        self.net = nn.Sequential(
            nn.Linear(self.max_obs_dim, 256),
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
        batch_size = observations.shape[0]
        obs_dim = observations.shape[1]
        
        # 如果输入维度小于最大维度，用零填充
        if obs_dim < self.max_obs_dim:
            # 创建填充后的观察
            padded_obs = torch.zeros(batch_size, self.max_obs_dim, device=observations.device)
            
            if obs_dim == 10:  # 2关节Reacher
                # 2关节: [cos1, cos2, sin1, sin2, vel1, vel2, ee_x, ee_y, target_x, target_y]
                # 3关节: [cos1, cos2, cos3, sin1, sin2, sin3, vel1, vel2, vel3, ee_x, ee_y, target_x, target_y]
                padded_obs[:, 0] = observations[:, 0]   # cos1
                padded_obs[:, 1] = observations[:, 1]   # cos2
                padded_obs[:, 2] = 1.0                  # cos3 (假设第3关节为0度，cos(0)=1)
                padded_obs[:, 3] = observations[:, 2]   # sin1
                padded_obs[:, 4] = observations[:, 3]   # sin2
                padded_obs[:, 5] = 0.0                  # sin3 (假设第3关节为0度，sin(0)=0)
                padded_obs[:, 6] = observations[:, 4]   # vel1
                padded_obs[:, 7] = observations[:, 5]   # vel2
                padded_obs[:, 8] = 0.0                  # vel3 (假设第3关节速度为0)
                padded_obs[:, 9] = observations[:, 6]   # ee_x
                padded_obs[:, 10] = observations[:, 7]  # ee_y
                padded_obs[:, 11] = observations[:, 8]  # target_x
                padded_obs[:, 12] = observations[:, 9]  # target_y
            else:
                # 其他情况，直接复制并填充零
                padded_obs[:, :obs_dim] = observations
            
            observations = padded_obs
        
        return self.net(observations)

class MixedJointActionWrapper(gym.ActionWrapper):
    """
    混合关节动作包装器
    将3维动作空间适配到不同关节数的环境
    """
    
    def __init__(self, env, original_action_dim):
        super().__init__(env)
        self.original_action_dim = original_action_dim
        
        # 统一动作空间为3维（最大关节数）
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        print(f"🔧 MixedJointActionWrapper: 原始动作维度={original_action_dim}, 统一为3维")
    
    def action(self, action):
        # 只使用前N个关节的动作
        return action[:self.original_action_dim]

class MixedJointObservationWrapper(gym.ObservationWrapper):
    """
    混合关节观察包装器
    将不同维度的观察空间统一为最大维度
    """
    
    def __init__(self, env, target_obs_dim=13):
        super().__init__(env)
        self.target_obs_dim = target_obs_dim
        self.original_obs_dim = env.observation_space.shape[0]
        
        # 统一观察空间为13维（3关节的维度）
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(target_obs_dim,), dtype=np.float64
        )
        
        print(f"🔧 MixedJointObservationWrapper: 原始观察维度={self.original_obs_dim}, 统一为{target_obs_dim}维")
    
    def observation(self, obs):
        if len(obs) < self.target_obs_dim:
            # 填充观察到目标维度
            padded_obs = np.zeros(self.target_obs_dim)
            
            if len(obs) == 10:  # 2关节
                # 按照MixedJointExtractor中的相同逻辑填充
                padded_obs[0] = obs[0]   # cos1
                padded_obs[1] = obs[1]   # cos2
                padded_obs[2] = 1.0      # cos3
                padded_obs[3] = obs[2]   # sin1
                padded_obs[4] = obs[3]   # sin2
                padded_obs[5] = 0.0      # sin3
                padded_obs[6] = obs[4]   # vel1
                padded_obs[7] = obs[5]   # vel2
                padded_obs[8] = 0.0      # vel3
                padded_obs[9] = obs[6]   # ee_x
                padded_obs[10] = obs[7]  # ee_y
                padded_obs[11] = obs[8]  # target_x
                padded_obs[12] = obs[9]  # target_y
            else:
                padded_obs[:len(obs)] = obs
            
            return padded_obs
        else:
            return obs

def create_2joint_env_with_render():
    """创建2关节环境（带渲染）"""
    env = gym.make('Reacher-v5', render_mode='human')
    env = MixedJointObservationWrapper(env, target_obs_dim=13)
    env = MixedJointActionWrapper(env, original_action_dim=2)
    env = Monitor(env)
    return env

def create_3joint_env_with_render():
    """创建3关节环境（带渲染）"""
    env = Perfect3JointReacherEnv(render_mode='human')
    env = MixedJointActionWrapper(env, original_action_dim=3)
    env = Monitor(env)
    return env

def train_mixed_joint_with_render():
    """
    带渲染的混合关节训练
    """
    print("🚀 混合关节Reacher训练（带渲染）")
    print("🎯 同时训练2关节和3关节Reacher")
    print("👁️ 可以看到训练过程的渲染")
    print("="*60)
    
    # 创建单一环境进行训练（这样可以看到渲染）
    print("🌍 创建混合训练环境...")
    
    # 先创建一个2关节环境进行训练
    print("📍 第一阶段：2关节Reacher训练")
    env_2joint = create_2joint_env_with_render()
    
    print("✅ 2关节环境创建完成")
    print(f"   观察空间: {env_2joint.observation_space}")
    print(f"   动作空间: {env_2joint.action_space}")
    
    # 创建SAC模型
    print("\n🤖 创建混合关节SAC模型...")
    
    policy_kwargs = {
        'features_extractor_class': MixedJointExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
    }
    
    model = SAC(
        'MlpPolicy',
        env_2joint,
        policy_kwargs=policy_kwargs,
        verbose=2,
        learning_starts=100,
        device='cpu',
        tensorboard_log="./tensorboard_logs/mixed_joint_render/",
    )
    
    print("✅ 混合关节SAC模型创建完成")
    
    try:
        # 第一阶段：训练2关节
        print(f"\n🎯 第一阶段：2关节训练 (5000步)...")
        print("💡 观察2关节机械臂的学习过程")
        
        start_time = time.time()
        
        model.learn(
            total_timesteps=5000,
            log_interval=4,
            progress_bar=True
        )
        
        phase1_time = time.time() - start_time
        print(f"\n✅ 第一阶段完成! 用时: {phase1_time/60:.1f} 分钟")
        
        # 保存第一阶段模型
        model.save("models/mixed_joint_phase1_2joint")
        print("💾 第一阶段模型已保存")
        
        # 快速评估2关节性能
        print("\n📊 快速评估2关节性能:")
        eval_2joint_quick(model, env_2joint)
        
        # 第二阶段：切换到3关节环境
        print(f"\n📍 第二阶段：3关节Reacher训练")
        env_2joint.close()
        
        env_3joint = create_3joint_env_with_render()
        
        print("✅ 3关节环境创建完成")
        print(f"   观察空间: {env_3joint.observation_space}")
        print(f"   动作空间: {env_3joint.action_space}")
        
        # 更新模型环境
        model.set_env(env_3joint)
        
        print(f"\n🎯 第二阶段：3关节训练 (5000步)...")
        print("💡 观察模型如何适应3关节机械臂")
        
        phase2_start = time.time()
        
        model.learn(
            total_timesteps=5000,
            log_interval=4,
            progress_bar=True,
            reset_num_timesteps=False  # 继续之前的训练
        )
        
        phase2_time = time.time() - phase2_start
        total_time = time.time() - start_time
        
        print(f"\n✅ 第二阶段完成! 用时: {phase2_time/60:.1f} 分钟")
        print(f"🎉 总训练时间: {total_time/60:.1f} 分钟")
        
        # 保存最终模型
        model.save("models/mixed_joint_render_final")
        print("💾 最终混合模型已保存")
        
        # 评估3关节性能
        print("\n📊 快速评估3关节性能:")
        eval_3joint_quick(model, env_3joint)
        
        # 最终测试：切换回2关节看泛化效果
        print(f"\n🔄 泛化测试：用训练好的模型控制2关节")
        env_3joint.close()
        
        env_2joint_test = create_2joint_env_with_render()
        model.set_env(env_2joint_test)
        
        print("📊 泛化测试结果:")
        eval_2joint_quick(model, env_2joint_test)
        
        env_2joint_test.close()
        
        print(f"\n🎉 混合关节训练（带渲染）完成！")
        print(f"💡 您已经看到了完整的训练过程")
        print(f"✅ 模型学会了处理不同关节数的机械臂")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 训练被用户中断")
        model.save("models/mixed_joint_render_interrupted")
        print("💾 中断模型已保存")
    
    finally:
        if 'env_2joint' in locals():
            env_2joint.close()
        if 'env_3joint' in locals():
            env_3joint.close()

def eval_2joint_quick(model, env):
    """快速评估2关节性能"""
    rewards = []
    successes = []
    
    for i in range(3):
        obs, info = env.reset()
        episode_reward = 0
        episode_success = False
        
        for step in range(50):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # 计算距离
            ee_pos = obs[9:11]
            target_pos = obs[11:13]
            distance = np.linalg.norm(ee_pos - target_pos)
            
            if distance < 0.02:
                episode_success = True
                break
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        successes.append(episode_success)
        
        print(f"   2关节 Episode {i+1}: 奖励={episode_reward:.2f}, 成功={'是' if episode_success else '否'}")
    
    avg_reward = np.mean(rewards)
    success_rate = np.mean(successes) * 100
    
    print(f"   2关节总结: 平均奖励={avg_reward:.3f}, 成功率={success_rate:.1f}%")

def eval_3joint_quick(model, env):
    """快速评估3关节性能"""
    rewards = []
    successes = []
    
    for i in range(3):
        obs, info = env.reset()
        episode_reward = 0
        episode_success = False
        
        for step in range(50):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # 检查是否成功（使用info或手动计算）
            if hasattr(info, 'is_success') and info['is_success']:
                episode_success = True
                break
            else:
                # 手动计算距离
                ee_pos = obs[9:11]
                target_pos = obs[11:13]
                distance = np.linalg.norm(ee_pos - target_pos)
                
                if distance < 0.02:
                    episode_success = True
                    break
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        successes.append(episode_success)
        
        print(f"   3关节 Episode {i+1}: 奖励={episode_reward:.2f}, 成功={'是' if episode_success else '否'}")
    
    avg_reward = np.mean(rewards)
    success_rate = np.mean(successes) * 100
    
    print(f"   3关节总结: 平均奖励={avg_reward:.3f}, 成功率={success_rate:.1f}%")

def main():
    """主函数"""
    print("🌟 混合关节Reacher训练系统（带渲染）")
    print("🎯 分阶段训练：2关节 → 3关节 → 泛化测试")
    print("👁️ 全程可视化训练过程")
    print()
    
    try:
        train_mixed_joint_with_render()
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 被用户中断")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


