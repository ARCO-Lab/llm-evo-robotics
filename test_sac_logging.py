#!/usr/bin/env python3
"""
测试 SAC 日志输出
对比不同配置下的日志行为
"""

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from real_multi_joint_reacher import RealMultiJointWrapper

def test_standard_reacher():
    """测试标准 Reacher-v5"""
    print("🧪 测试标准 Reacher-v5 日志输出")
    
    env = gym.make('Reacher-v5')
    env = Monitor(env)
    
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=200,
        batch_size=64,
        verbose=2,
        device='cpu'
    )
    
    print("🎯 开始训练标准 Reacher...")
    model.learn(
        total_timesteps=3000,
        log_interval=4,
        progress_bar=True
    )
    
    env.close()
    print("✅ 标准 Reacher 训练完成\n")

def test_real_multi_joint():
    """测试真实多关节环境"""
    print("🧪 测试真实多关节环境日志输出")
    
    env = RealMultiJointWrapper(
        num_joints=2,
        link_lengths=[0.1, 0.1],
        render_mode=None
    )
    env = Monitor(env)
    
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=200,
        batch_size=64,
        verbose=2,
        device='cpu'
    )
    
    print("🎯 开始训练真实多关节...")
    model.learn(
        total_timesteps=3000,
        log_interval=4,
        progress_bar=True
    )
    
    env.close()
    print("✅ 真实多关节训练完成\n")

def test_longer_training():
    """测试更长时间的训练"""
    print("🧪 测试更长时间训练的日志输出")
    
    env = RealMultiJointWrapper(
        num_joints=2,
        link_lengths=[0.1, 0.1],
        render_mode=None
    )
    env = Monitor(env)
    
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=200,
        batch_size=64,
        verbose=2,
        device='cpu'
    )
    
    print("🎯 开始更长时间训练...")
    model.learn(
        total_timesteps=10000,  # 增加到 10000 步
        log_interval=2,         # 更频繁的日志输出
        progress_bar=True
    )
    
    env.close()
    print("✅ 更长时间训练完成\n")

if __name__ == "__main__":
    print("🌟 SAC 日志输出测试")
    print("💡 对比不同环境和配置的日志行为\n")
    
    try:
        # 测试1：标准 Reacher
        test_standard_reacher()
        
        # 测试2：真实多关节环境
        test_real_multi_joint()
        
        # 测试3：更长时间训练
        test_longer_training()
        
        print("🎉 所有测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


