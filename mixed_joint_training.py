#!/usr/bin/env python3
"""
混合关节训练：同时训练2关节和3关节Reacher
测试模型是否能够泛化到不同关节数
"""

import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import time
import tempfile

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

def create_2joint_env(render_mode='human'):
    """创建2关节环境"""
    env = gym.make('Reacher-v5', render_mode=render_mode)
    env = MixedJointObservationWrapper(env, target_obs_dim=13)
    env = MixedJointActionWrapper(env, original_action_dim=2)
    env = Monitor(env)
    return env

def create_3joint_env(render_mode='human'):
    """创建3关节环境"""
    env = Perfect3JointReacherEnv(render_mode=render_mode)
    env = MixedJointActionWrapper(env, original_action_dim=3)
    env = Monitor(env)
    return env

class MixedJointCallback(BaseCallback):
    """
    混合关节训练回调
    监控不同环境的性能
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards_2joint = []
        self.episode_rewards_3joint = []
        self.episode_successes_2joint = []
        self.episode_successes_3joint = []
    
    def _on_step(self) -> bool:
        # 这里可以添加特定的监控逻辑
        return True

def train_mixed_joint_reacher(total_timesteps: int = 50000):
    """
    训练混合关节Reacher模型
    同时学习2关节和3关节的控制
    """
    print("🚀 混合关节Reacher训练")
    print("🎯 同时训练2关节和3关节Reacher")
    print("💡 测试模型的跨关节泛化能力")
    print("="*60)
    
    # 创建混合环境
    print("🌍 创建混合训练环境...")
    
    # 方法1: 使用多个环境的向量化 (带渲染)
    def make_2joint_env():
        return create_2joint_env(render_mode='human')
    
    def make_3joint_env():
        return create_3joint_env(render_mode='human')
    
    # 创建混合环境：50%的2关节，50%的3关节
    env_fns = []
    for i in range(2):  # 2个2关节环境
        env_fns.append(make_2joint_env)
    for i in range(2):  # 2个3关节环境
        env_fns.append(make_3joint_env)
    
    # 使用向量化环境
    vec_env = DummyVecEnv(env_fns)
    
    print("✅ 混合环境创建完成")
    print(f"   环境数量: {len(env_fns)} (2个2关节 + 2个3关节)")
    print(f"   统一观察空间: {vec_env.observation_space}")
    print(f"   统一动作空间: {vec_env.action_space}")
    
    # 创建评估环境 (带渲染)
    eval_env_2joint = create_2joint_env(render_mode='human')
    eval_env_3joint = create_3joint_env(render_mode='human')
    
    # 创建SAC模型
    print("\n🤖 创建混合关节SAC模型...")
    
    policy_kwargs = {
        'features_extractor_class': MixedJointExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
    }
    
    model = SAC(
        'MlpPolicy',
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=2,
        learning_starts=200,  # 更多环境需要更多启动步数
        device='cpu',
        tensorboard_log="./tensorboard_logs/mixed_joint/",
    )
    
    print("✅ 混合关节SAC模型创建完成")
    print("   ✅ 自定义特征提取器: MixedJointExtractor")
    print("   ✅ 统一观察/动作空间处理")
    print("   ✅ 向量化环境训练")
    
    # 开始训练
    print(f"\n🎯 开始混合训练 ({total_timesteps}步)...")
    print("💡 模型将同时学习2关节和3关节的控制策略")
    
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 混合训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {total_timesteps/training_time:.1f}")
        
        # 保存模型
        model.save("models/mixed_joint_reacher_sac")
        print("💾 混合模型已保存: models/mixed_joint_reacher_sac")
        
        # 分别评估2关节和3关节性能
        print("\n📊 分别评估不同关节数的性能...")
        
        # 评估2关节
        print("\n🔍 评估2关节Reacher性能:")
        eval_2joint_results = evaluate_mixed_model(model, eval_env_2joint, "2关节", episodes=10)
        
        # 评估3关节
        print("\n🔍 评估3关节Reacher性能:")
        eval_3joint_results = evaluate_mixed_model(model, eval_env_3joint, "3关节", episodes=10)
        
        # 对比分析
        print(f"\n📈 混合训练效果对比:")
        print(f"   2关节成功率: {eval_2joint_results['success_rate']:.1f}%")
        print(f"   3关节成功率: {eval_3joint_results['success_rate']:.1f}%")
        print(f"   2关节平均奖励: {eval_2joint_results['avg_reward']:.3f}")
        print(f"   3关节平均奖励: {eval_3joint_results['avg_reward']:.3f}")
        
        # 泛化能力分析
        if eval_2joint_results['success_rate'] > 20 and eval_3joint_results['success_rate'] > 20:
            print("   🎉 优秀的跨关节泛化能力!")
        elif eval_2joint_results['success_rate'] > 10 or eval_3joint_results['success_rate'] > 10:
            print("   ✅ 良好的跨关节泛化能力!")
        else:
            print("   🔶 泛化能力有限，需要进一步优化")
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⚠️ 训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model.save("models/mixed_joint_reacher_sac_interrupted")
        print("💾 中断模型已保存")
    
    finally:
        vec_env.close()
        eval_env_2joint.close()
        eval_env_3joint.close()

def evaluate_mixed_model(model, env, env_name, episodes=10):
    """评估混合模型在特定环境上的性能"""
    
    all_rewards = []
    all_successes = []
    all_distances = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_success = False
        
        for step in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            
            # 计算距离（根据环境类型）
            if hasattr(info, 'distance_to_target'):
                distance = info['distance_to_target']
            else:
                # 手动计算距离
                if len(obs) >= 10:  # 确保有足够的观察维度
                    if len(obs) == 13:  # 3关节
                        ee_pos = obs[9:11]
                        target_pos = obs[11:13]
                    else:  # 2关节（填充后）
                        ee_pos = obs[9:11]
                        target_pos = obs[11:13]
                    distance = np.linalg.norm(ee_pos - target_pos)
                else:
                    distance = 1.0  # 默认值
            
            if distance < 0.02:
                episode_success = True
                break
            
            if terminated or truncated:
                break
        
        all_rewards.append(episode_reward)
        all_successes.append(episode_success)
        all_distances.append(distance)
        
        if episode % 5 == 0:
            print(f"   {env_name} Episode {episode+1}: 奖励={episode_reward:.2f}, 成功={'是' if episode_success else '否'}")
    
    results = {
        'avg_reward': np.mean(all_rewards),
        'success_rate': np.mean(all_successes) * 100,
        'avg_distance': np.mean(all_distances)
    }
    
    print(f"   {env_name} 总结: 平均奖励={results['avg_reward']:.3f}, 成功率={results['success_rate']:.1f}%")
    
    return results

def main():
    """主函数"""
    print("🌟 混合关节Reacher训练系统")
    print("🎯 同时训练2关节和3关节，测试泛化能力")
    print("💡 使用统一的网络架构和向量化环境")
    print()
    
    try:
        train_mixed_joint_reacher(total_timesteps=30000)
        
        print(f"\n🎉 混合关节训练完成！")
        print(f"💡 现在您有了一个能够处理不同关节数的通用模型")
        print(f"✅ 可以测试模型在2关节和3关节上的泛化性能")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 被用户中断")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
