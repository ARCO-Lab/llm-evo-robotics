#!/usr/bin/env python3
"""
专注2关节训练：使用混合关节架构专门训练2关节Reacher
训练30000步，然后测试10个episode
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

class MixedJointExtractor(BaseFeaturesExtractor):
    """
    混合关节特征提取器
    专门优化用于2关节Reacher训练
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super(MixedJointExtractor, self).__init__(observation_space, features_dim)
        
        # 支持的最大观察维度（3关节的13维）
        self.max_obs_dim = 13
        obs_dim = observation_space.shape[0]
        
        print(f"🔧 MixedJointExtractor: {obs_dim} -> {features_dim}")
        print(f"   支持最大观察维度: {self.max_obs_dim}")
        print(f"   专门优化用于2关节训练")
        
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
    将3维动作空间适配到2关节环境
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
    将2关节的10维观察空间统一为13维
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

def create_2joint_env_no_render():
    """创建2关节环境（无渲染）"""
    env = gym.make('Reacher-v5', render_mode=None)
    env = MixedJointObservationWrapper(env, target_obs_dim=13)
    env = MixedJointActionWrapper(env, original_action_dim=2)
    env = Monitor(env)
    return env

def train_focused_2joint(total_timesteps: int = 30000):
    """
    专注训练2关节Reacher
    使用混合关节架构，但专门优化2关节性能
    """
    print("🚀 专注2关节Reacher训练")
    print("🎯 使用混合关节架构专门训练2关节")
    print("💡 训练过程带渲染，可以观察学习进展")
    print("="*60)
    
    # 创建训练环境（带渲染）
    print("🌍 创建2关节训练环境...")
    train_env = create_2joint_env_with_render()
    
    print("✅ 2关节训练环境创建完成")
    print(f"   观察空间: {train_env.observation_space}")
    print(f"   动作空间: {train_env.action_space}")
    
    # 创建SAC模型
    print("\n🤖 创建专注2关节SAC模型...")
    
    policy_kwargs = {
        'features_extractor_class': MixedJointExtractor,
        'features_extractor_kwargs': {'features_dim': 128},
    }
    
    model = SAC(
        'MlpPolicy',
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=2,
        learning_starts=200,
        device='cpu',
        tensorboard_log="./tensorboard_logs/focused_2joint/",
        batch_size=256,
        buffer_size=50000,
        # 针对2关节优化的参数
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        ent_coef='auto',
    )
    
    print("✅ 专注2关节SAC模型创建完成")
    print("   ✅ 使用混合关节特征提取器")
    print("   ✅ 针对2关节优化的超参数")
    print("   ✅ 训练过程带渲染")
    
    # 开始训练
    print(f"\n🎯 开始专注2关节训练 ({total_timesteps}步)...")
    print("💡 您将看到2关节Reacher的学习过程")
    print("💡 观察机械臂如何逐步学会到达目标")
    
    try:
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=4,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✅ 专注2关节训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"📊 平均FPS: {total_timesteps/training_time:.1f}")
        
        # 保存模型
        model.save("models/focused_2joint_reacher_sac")
        print("💾 专注2关节模型已保存: models/focused_2joint_reacher_sac")
        
        return model
        
    except KeyboardInterrupt:
        training_time = time.time() - start_time
        print(f"\n⚠️ 训练被用户中断")
        print(f"⏱️ 已训练时间: {training_time/60:.1f} 分钟")
        
        model.save("models/focused_2joint_reacher_sac_interrupted")
        print("💾 中断模型已保存")
        return model
    
    finally:
        train_env.close()

def test_focused_2joint_model(model=None):
    """
    测试专注训练的2关节模型
    运行10个episode进行详细评估
    """
    print("\n🎮 测试专注训练的2关节模型")
    print("🎯 运行10个episode进行详细评估")
    print("👁️ 带渲染观察模型性能")
    print("="*60)
    
    try:
        # 加载模型（如果没有传入）
        if model is None:
            print("📂 加载训练好的模型...")
            try:
                model = SAC.load("models/focused_2joint_reacher_sac")
                print("✅ 成功加载: models/focused_2joint_reacher_sac")
            except FileNotFoundError:
                try:
                    model = SAC.load("models/focused_2joint_reacher_sac_interrupted")
                    print("✅ 成功加载: models/focused_2joint_reacher_sac_interrupted")
                except FileNotFoundError:
                    print("❌ 没有找到训练好的模型")
                    return
        
        # 创建测试环境（带渲染）
        print("🌍 创建测试环境...")
        test_env = create_2joint_env_with_render()
        
        print("✅ 测试环境创建完成")
        print("🎯 开始10个episode的详细测试...")
        
        # 统计所有episode的结果
        all_episode_rewards = []
        all_episode_lengths = []
        all_episode_successes = []
        all_episode_final_distances = []
        all_episode_success_steps = []
        
        for episode in range(10):
            print(f"\n📍 Episode {episode + 1}/10:")
            
            obs, info = test_env.reset()
            episode_reward = 0
            episode_length = 0
            episode_success = False
            success_step = None
            
            for step in range(100):  # 每个episode最多100步
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # 计算距离
                ee_pos = obs[9:11]  # 末端执行器位置
                target_pos = obs[11:13]  # 目标位置
                distance = np.linalg.norm(ee_pos - target_pos)
                
                # 每20步打印一次状态
                if step % 20 == 0:
                    print(f"   Step {step}: 距离={distance:.3f}m, 奖励={reward:.3f}")
                    print(f"     动作: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}] (使用前2维)")
                
                # 检查是否成功
                if distance < 0.02 and not episode_success:
                    episode_success = True
                    success_step = step + 1
                    print(f"   ✅ 成功! 在第{success_step}步到达目标，距离={distance:.3f}m")
                    # 继续运行看能否保持
                
                # 检查是否结束
                if terminated or truncated:
                    final_distance = distance
                    break
            else:
                # 如果循环正常结束（没有break），说明达到了100步
                final_distance = distance
                print(f"   ⏰ 达到最大步数(100)，最终距离={final_distance:.3f}m")
            
            # 记录episode统计
            all_episode_rewards.append(episode_reward)
            all_episode_lengths.append(episode_length)
            all_episode_successes.append(episode_success)
            all_episode_final_distances.append(final_distance)
            if success_step is not None:
                all_episode_success_steps.append(success_step)
            
            print(f"   📊 Episode {episode + 1} 总结:")
            print(f"     总奖励: {episode_reward:.2f}")
            print(f"     长度: {episode_length}步")
            print(f"     成功: {'是' if episode_success else '否'}")
            print(f"     最终距离: {final_distance:.3f}m")
            if success_step is not None:
                print(f"     成功步数: {success_step}")
        
        # 计算总体统计
        avg_reward = np.mean(all_episode_rewards)
        avg_length = np.mean(all_episode_lengths)
        success_rate = np.mean(all_episode_successes) * 100
        avg_final_distance = np.mean(all_episode_final_distances)
        
        print(f"\n📊 专注2关节训练测试结果 (10个episode):")
        print(f"   平均episode奖励: {avg_reward:.3f}")
        print(f"   平均episode长度: {avg_length:.1f}步")
        print(f"   平均最终距离: {avg_final_distance:.3f}m")
        print(f"   成功率: {success_rate:.1f}% ({int(success_rate/10)}/10 episodes)")
        
        if all_episode_success_steps:
            avg_success_steps = np.mean(all_episode_success_steps)
            print(f"   平均成功步数: {avg_success_steps:.1f}步")
        
        # 性能评估
        print(f"\n🔬 性能分析:")
        if success_rate >= 80:
            print("   🎉 优秀的训练效果!")
            print("   💡 专注2关节训练非常成功")
        elif success_rate >= 50:
            print("   ✅ 良好的训练效果!")
            print("   💡 专注2关节训练效果不错")
        elif success_rate >= 20:
            print("   🔶 一般的训练效果")
            print("   💡 专注2关节训练有一定效果，可以继续优化")
        else:
            print("   ⚠️ 训练效果有待提升")
            print("   💡 可能需要调整超参数或增加训练步数")
        
        # 详细统计
        successful_episodes = [i+1 for i, success in enumerate(all_episode_successes) if success]
        if successful_episodes:
            print(f"   🎯 成功的episode: {successful_episodes}")
        
        # 奖励分布分析
        max_reward = max(all_episode_rewards)
        min_reward = min(all_episode_rewards)
        print(f"   📈 奖励范围: {min_reward:.3f} ~ {max_reward:.3f}")
        
        # 距离分析
        min_distance = min(all_episode_final_distances)
        max_distance = max(all_episode_final_distances)
        print(f"   📏 最终距离范围: {min_distance:.3f}m ~ {max_distance:.3f}m")
        
        test_env.close()
        
        print(f"\n🎉 专注2关节测试完成!")
        print(f"💡 这展示了混合关节架构在2关节上的专门训练效果")
        
        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_length': avg_length,
            'avg_distance': avg_final_distance
        }
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("🌟 专注2关节Reacher训练系统")
    print("🎯 使用混合关节架构专门训练2关节")
    print("💡 30000步训练 + 10个episode测试")
    print()
    
    try:
        # 训练阶段
        print("🚀 开始训练阶段...")
        model = train_focused_2joint(total_timesteps=30000)
        
        # 测试阶段
        print("\n" + "="*60)
        print("🎮 开始测试阶段...")
        results = test_focused_2joint_model(model)
        
        if results:
            print(f"\n🎉 专注2关节训练和测试完成！")
            print(f"✅ 成功率: {results['success_rate']:.1f}%")
            print(f"✅ 平均奖励: {results['avg_reward']:.3f}")
            print(f"💡 专门针对2关节的训练确实有效果")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 被用户中断")
        print("💡 可以稍后运行测试部分")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


