#!/usr/bin/env python3
"""
纯净的 Stable Baselines3 SAC 训练 MuJoCo Reacher
直接使用官方实现，无任何自定义适配器
"""

import gymnasium as gym
import numpy as np
import torch
import time
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

def baseline_sac_training():
    print("🚀 纯净的 Stable Baselines3 SAC 训练 MuJoCo Reacher")
    print("📚 参考文档: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html")
    print("🎯 环境: https://gymnasium.farama.org/environments/mujoco/reacher/")
    print("=" * 70)
    
    # 创建原生 MuJoCo Reacher 环境
    print("🏭 创建 MuJoCo Reacher-v5 环境...")
    env = gym.make('Reacher-v5', render_mode='human')
    env = Monitor(env)  # 添加监控包装器
    
    print(f"✅ 环境创建完成")
    print(f"🎮 动作空间: {env.action_space}")
    print(f"👁️ 观察空间: {env.observation_space}")
    print(f"📏 观察维度: {env.observation_space.shape}")
    
    # 创建评估环境
    eval_env = gym.make('Reacher-v5')
    eval_env = Monitor(eval_env)
    
    print("=" * 70)
    
    # 创建 SAC 模型 - 使用官方推荐参数
    print("🤖 创建 SAC 模型...")
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,          # 官方默认学习率
        buffer_size=1000000,         # 1M 缓冲区
        learning_starts=100,         # 100 步后开始学习
        batch_size=256,              # 批次大小
        tau=0.005,                   # 软更新系数
        gamma=0.99,                  # 折扣因子
        train_freq=1,                # 每步训练
        gradient_steps=1,            # 每次训练1个梯度步
        ent_coef='auto',             # 自动调整熵系数
        target_update_interval=1,    # 目标网络更新间隔
        use_sde=False,               # 不使用状态依赖探索
        verbose=1,                   # 详细输出
        device='cpu'                 # 使用 CPU
    )
    
    print("✅ SAC 模型创建完成")
    print(f"📊 模型参数:")
    print(f"   策略: MlpPolicy")
    print(f"   学习率: 3e-4")
    print(f"   缓冲区大小: 1,000,000")
    print(f"   批次大小: 256")
    print(f"   熵系数: auto")
    
    print("=" * 70)
    
    # 创建评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./sac_reacher_best/',
        log_path='./sac_reacher_logs/',
        eval_freq=5000,              # 每5000步评估一次
        n_eval_episodes=10,          # 每次评估10个episodes
        deterministic=True,          # 评估时使用确定性策略
        render=False                 # 评估时不渲染
    )
    
    # 开始训练
    print("🎯 开始训练...")
    print("📊 训练配置:")
    print("   总步数: 50,000")
    print("   评估频率: 每 5,000 步")
    print("   日志间隔: 每 1,000 步")
    print("=" * 70)
    
    start_time = time.time()
    
    # 训练模型
    model.learn(
        total_timesteps=50000,       # 训练50k步
        callback=eval_callback,      # 评估回调
        log_interval=10,             # 每10个episodes记录一次
        progress_bar=True            # 显示进度条
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("🏆 训练完成!")
    print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
    print("=" * 70)
    
    # 保存模型
    model.save("sac_reacher_final")
    print("💾 模型已保存为: sac_reacher_final.zip")
    
    # 最终评估
    print("\n🔍 最终评估 (20 episodes)...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )
    
    print(f"📊 最终评估结果:")
    print(f"   平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 性能评估
    if mean_reward > -5:
        print("🥇 优秀! SAC 学会了 Reacher 任务")
    elif mean_reward > -10:
        print("🥈 良好! SAC 有不错的表现")
    elif mean_reward > -20:
        print("🥉 一般! SAC 有一定学习效果")
    else:
        print("⚠️ 需要更多训练或参数调整")
    
    # 演示训练好的模型
    print("\n🎮 演示训练好的模型 (10 episodes)...")
    demo_env = gym.make('Reacher-v5', render_mode='human')
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(10):
        obs, info = demo_env.reset()
        episode_reward = 0
        episode_length = 0
        min_distance = float('inf')
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = demo_env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # 计算到目标的距离 (MuJoCo Reacher 的奖励函数包含距离信息)
            # 距离可以从奖励推算，或者从观察中获取
            if hasattr(info, 'distance') or 'distance' in info:
                distance = info.get('distance', 0)
            else:
                # MuJoCo Reacher 的观察包含目标向量，可以计算距离
                target_vector = obs[-3:-1]  # 最后几个维度是目标向量
                distance = np.linalg.norm(target_vector)
            
            min_distance = min(min_distance, distance)
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 判断成功 (距离小于某个阈值或奖励足够高)
        if episode_reward > -5 or min_distance < 0.05:  # MuJoCo 单位是米
            success_count += 1
            print(f"🎯 Episode {episode+1}: 成功! 奖励={episode_reward:.2f}, 长度={episode_length}")
        else:
            print(f"📊 Episode {episode+1}: 奖励={episode_reward:.2f}, 长度={episode_length}")
    
    demo_env.close()
    
    # 演示统计
    print("\n" + "=" * 70)
    print("📊 演示统计:")
    print(f"   成功率: {success_count/10:.1%} ({success_count}/10)")
    print(f"   平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"   平均长度: {np.mean(episode_lengths):.1f}")
    print(f"   奖励标准差: {np.std(episode_rewards):.2f}")
    
    # 与 MuJoCo 基准比较
    print(f"\n🏅 性能评估:")
    print(f"   MuJoCo Reacher 基准奖励通常在 -5 到 -50 之间")
    print(f"   您的模型平均奖励: {np.mean(episode_rewards):.2f}")
    
    if np.mean(episode_rewards) > -10:
        print("   🎉 表现优秀，超过了大多数基准!")
    elif np.mean(episode_rewards) > -20:
        print("   👍 表现良好，达到了合理水平!")
    else:
        print("   📈 有改进空间，可以尝试更长时间训练")
    
    print("\n✅ Baseline SAC 训练完成!")
    
    # 清理
    env.close()
    eval_env.close()
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'training_time': training_time,
        'demo_success_rate': success_count / 10,
        'demo_avg_reward': np.mean(episode_rewards)
    }

if __name__ == "__main__":
    print("🔥 开始 Baseline SAC + MuJoCo Reacher 训练")
    print("📖 这是一个纯净的实现，直接使用官方库")
    print()
    
    try:
        results = baseline_sac_training()
        
        print(f"\n🎊 训练结果总结:")
        print(f"   最终评估奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"   训练时间: {results['training_time']/60:.1f} 分钟")
        print(f"   演示成功率: {results['demo_success_rate']:.1%}")
        print(f"   演示平均奖励: {results['demo_avg_reward']:.2f}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        print("💡 请检查是否正确安装了 MuJoCo 和相关依赖")
        print("   pip install gymnasium[mujoco]")
        print("   pip install stable-baselines3[extra]")
