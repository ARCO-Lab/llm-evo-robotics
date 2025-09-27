#!/usr/bin/env python3
"""
简化版真实多关节 SAC 训练
不使用自定义特征提取器，直接使用标准 MlpPolicy
"""

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# 导入真实多关节环境
from real_multi_joint_reacher import RealMultiJointWrapper

def simple_train_real_multi_joint_sac(num_joints: int = 2):
    """简化版真实多关节 SAC 训练"""
    
    print(f"\n{'='*60}")
    print(f"🚀 简化版真实 {num_joints} 关节 Reacher SAC 训练")
    print(f"{'='*60}")
    
    # 创建真实多关节环境
    print(f"🌍 创建真实 {num_joints} 关节环境...")
    env = RealMultiJointWrapper(
        num_joints=num_joints,
        link_lengths=[0.1] * num_joints,
        render_mode=None
    )
    env = Monitor(env)
    
    print(f"✅ 环境创建完成")
    print(f"   观察空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    
    # 创建标准 SAC 模型 (不使用自定义特征提取器)
    print(f"🤖 创建标准 SAC 模型...")
    model = SAC(
        'MlpPolicy',  # 使用标准 MlpPolicy
        env,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=100,      # 早点开始学习
        batch_size=64,
        verbose=2,                # 详细输出
        device='cpu'
    )
    
    print(f"✅ SAC 模型创建完成")
    print(f"🚀 开始训练 (2000 steps)...")
    
    # 训练
    model.learn(
        total_timesteps=2000,
        progress_bar=True
    )
    
    print(f"✅ 训练完成")
    
    # 简单评估
    print(f"📈 评估模型...")
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=5, deterministic=True
    )
    
    print(f"📊 评估结果:")
    print(f"   平均奖励: {mean_reward:.3f} ± {std_reward:.3f}")
    
    env.close()
    return mean_reward

def main():
    """测试不同关节数"""
    print("🌟 简化版真实多关节 SAC 训练测试")
    print("💡 使用标准 MlpPolicy，不使用自定义特征提取器")
    
    # 测试 2 关节
    try:
        result_2j = simple_train_real_multi_joint_sac(num_joints=2)
        print(f"✅ 2 关节训练成功，平均奖励: {result_2j:.3f}")
    except Exception as e:
        print(f"❌ 2 关节训练失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试 3 关节
    try:
        result_3j = simple_train_real_multi_joint_sac(num_joints=3)
        print(f"✅ 3 关节训练成功，平均奖励: {result_3j:.3f}")
    except Exception as e:
        print(f"❌ 3 关节训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


