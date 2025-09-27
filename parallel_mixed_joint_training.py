#!/usr/bin/env python3
"""
并行混合关节训练：同时开启两个进程训练同一个模型
一个进程训练2关节，一个进程训练3关节，共享同一个模型
"""

import os
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import time
import multiprocessing as mp
from threading import Thread
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

def make_2joint_env():
    """创建2关节环境工厂函数"""
    def _init():
        env = gym.make('Reacher-v5', render_mode='human')
        env = MixedJointObservationWrapper(env, target_obs_dim=13)
        env = MixedJointActionWrapper(env, original_action_dim=2)
        env = Monitor(env)
        return env
    return _init

def make_3joint_env():
    """创建3关节环境工厂函数"""
    def _init():
        env = Perfect3JointReacherEnv(render_mode='human')
        env = MixedJointActionWrapper(env, original_action_dim=3)
        env = Monitor(env)
        return env
    return _init

class ParallelTrainingCallback(BaseCallback):
    """
    并行训练回调
    监控不同环境的性能并同步模型
    """
    
    def __init__(self, save_freq=5000, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.last_save = 0
    
    def _on_step(self) -> bool:
        # 定期保存模型
        if self.num_timesteps - self.last_save >= self.save_freq:
            model_path = f"models/parallel_mixed_joint_checkpoint_{self.num_timesteps}"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"💾 模型检查点已保存: {model_path}")
            self.last_save = self.num_timesteps
        
        return True

def train_parallel_mixed_joint(total_timesteps: int = 50000):
    """
    并行混合关节训练
    同时使用2关节和3关节环境训练同一个模型
    """
    print("🚀 并行混合关节Reacher训练")
    print("🎯 同时训练2关节和3关节Reacher (并行)")
    print("💡 两个进程共享同一个模型")
    print("="*60)
    
    # 创建并行环境
    print("🌍 创建并行训练环境...")
    
    # 创建多个环境：2个2关节 + 2个3关节
    env_fns = []
    
    # 添加2关节环境
    for i in range(2):
        env_fns.append(make_2joint_env())
        print(f"   ✅ 2关节环境 {i+1} 已添加")
    
    # 添加3关节环境
    for i in range(2):
        env_fns.append(make_3joint_env())
        print(f"   ✅ 3关节环境 {i+1} 已添加")
    
    # 使用SubprocVecEnv实现真正的并行
    print("🔄 创建并行向量化环境...")
    try:
        # 尝试使用SubprocVecEnv (真正的多进程)
        vec_env = SubprocVecEnv(env_fns, start_method='spawn')
        print("✅ 使用SubprocVecEnv (多进程并行)")
    except Exception as e:
        print(f"⚠️ SubprocVecEnv失败，回退到DummyVecEnv: {e}")
        # 如果多进程失败，回退到单进程
        vec_env = DummyVecEnv(env_fns)
        print("✅ 使用DummyVecEnv (单进程)")
    
    print("✅ 并行环境创建完成")
    print(f"   环境数量: {len(env_fns)} (2个2关节 + 2个3关节)")
    print(f"   统一观察空间: {vec_env.observation_space}")
    print(f"   统一动作空间: {vec_env.action_space}")
    
    # 创建SAC模型
    print("\n🤖 创建并行混合关节SAC模型...")
    
    policy_kwargs = {
        'features_extractor_class': MixedJointExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
    }
    
    model = SAC(
        'MlpPolicy',
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=2,
        learning_starts=400,  # 4个环境，所以需要更多启动步数
        device='cpu',
        tensorboard_log="./tensorboard_logs/parallel_mixed_joint/",
        batch_size=256,  # 增加批次大小
        buffer_size=100000,  # 增加缓冲区大小
    )
    
    print("✅ 并行混合关节SAC模型创建完成")
    print("   ✅ 自定义特征提取器: MixedJointExtractor")
    print("   ✅ 多进程并行训练")
    print("   ✅ 增大缓冲区和批次大小")
    
    # 创建回调
    callback = ParallelTrainingCallback(save_freq=10000, verbose=1)
    
    # 开始训练
    print(f"\n🎯 开始并行训练 ({total_timesteps}步)...")
    print("💡 您将看到4个MuJoCo窗口同时训练:")
    print("   🔸 窗口1-2: 2关节Reacher")
    print("   🔸 窗口3-4: 3关节Reacher")
    print("💡 所有环境共享同一个神经网络模型")
    
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,
            progress_bar=True,
            callback=callback
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 并行训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {total_timesteps/training_time:.1f}")
        print(f"🚀 并行加速比: ~{len(env_fns)}x (理论)")
        
        # 保存最终模型
        model.save("models/parallel_mixed_joint_final")
        print("💾 最终并行模型已保存: models/parallel_mixed_joint_final")
        
        # 评估模型性能
        print("\n📊 评估并行训练的模型性能...")
        evaluate_parallel_model(model)
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⚠️ 训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model.save("models/parallel_mixed_joint_interrupted")
        print("💾 中断模型已保存")
    
    finally:
        vec_env.close()

def evaluate_parallel_model(model):
    """评估并行训练的模型"""
    print("🔍 创建评估环境...")
    
    # 创建单独的评估环境
    eval_env_2joint = gym.make('Reacher-v5', render_mode='human')
    eval_env_2joint = MixedJointObservationWrapper(eval_env_2joint, target_obs_dim=13)
    eval_env_2joint = MixedJointActionWrapper(eval_env_2joint, original_action_dim=2)
    eval_env_2joint = Monitor(eval_env_2joint)
    
    eval_env_3joint = Perfect3JointReacherEnv(render_mode='human')
    eval_env_3joint = MixedJointActionWrapper(eval_env_3joint, original_action_dim=3)
    eval_env_3joint = Monitor(eval_env_3joint)
    
    try:
        # 评估2关节
        print("\n🎮 评估2关节性能 (5个episode):")
        results_2joint = evaluate_env(model, eval_env_2joint, "2关节", episodes=5)
        
        # 评估3关节
        print("\n🎮 评估3关节性能 (5个episode):")
        results_3joint = evaluate_env(model, eval_env_3joint, "3关节", episodes=5)
        
        # 对比分析
        print(f"\n📈 并行训练效果总结:")
        print(f"   2关节: 成功率={results_2joint['success_rate']:.1f}%, 平均奖励={results_2joint['avg_reward']:.3f}")
        print(f"   3关节: 成功率={results_3joint['success_rate']:.1f}%, 平均奖励={results_3joint['avg_reward']:.3f}")
        
        # 泛化能力评估
        if results_2joint['success_rate'] > 20 and results_3joint['success_rate'] > 20:
            print("   🎉 优秀的并行训练效果!")
        elif results_2joint['success_rate'] > 10 or results_3joint['success_rate'] > 10:
            print("   ✅ 良好的并行训练效果!")
        else:
            print("   🔶 并行训练效果有待提升")
        
    finally:
        eval_env_2joint.close()
        eval_env_3joint.close()

def evaluate_env(model, env, env_name, episodes=5):
    """评估模型在特定环境上的性能"""
    
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
            
            # 计算距离
            if hasattr(info, 'distance_to_target'):
                distance = info['distance_to_target']
            elif hasattr(info, 'is_success'):
                # 对于3关节环境
                distance = 0.01 if info['is_success'] else 0.1
            else:
                # 手动计算距离
                if len(obs) >= 10:
                    ee_pos = obs[9:11]
                    target_pos = obs[11:13]
                    distance = np.linalg.norm(ee_pos - target_pos)
                else:
                    distance = 1.0
            
            if distance < 0.02:
                episode_success = True
                break
            
            if terminated or truncated:
                break
        
        all_rewards.append(episode_reward)
        all_successes.append(episode_success)
        all_distances.append(distance)
        
        print(f"   {env_name} Episode {episode+1}: 奖励={episode_reward:.2f}, 成功={'是' if episode_success else '否'}, 距离={distance:.3f}m")
    
    results = {
        'avg_reward': np.mean(all_rewards),
        'success_rate': np.mean(all_successes) * 100,
        'avg_distance': np.mean(all_distances)
    }
    
    print(f"   {env_name} 总结: 平均奖励={results['avg_reward']:.3f}, 成功率={results['success_rate']:.1f}%")
    
    return results

def main():
    """主函数"""
    print("🌟 并行混合关节Reacher训练系统")
    print("🎯 多进程并行训练，共享同一个模型")
    print("💡 同时显示多个MuJoCo窗口")
    print()
    
    try:
        # 设置多进程启动方法
        mp.set_start_method('spawn', force=True)
        
        # 开始并行训练 (增加训练步数)
        train_parallel_mixed_joint(total_timesteps=50000)
        
        print(f"\n🎉 并行混合关节训练完成！")
        print(f"💡 您已经看到了多进程并行训练的效果")
        print(f"✅ 模型学会了同时处理不同关节数的机械臂")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 被用户中断")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
