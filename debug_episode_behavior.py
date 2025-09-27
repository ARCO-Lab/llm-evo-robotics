#!/usr/bin/env python3
"""
调试 episode 行为
检查真实多关节环境的 episode 长度和重置机制
"""

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from real_multi_joint_reacher import RealMultiJointWrapper

def debug_standard_reacher():
    """调试标准 Reacher 的 episode 行为"""
    print("🔍 调试标准 Reacher-v5 episode 行为")
    
    env = gym.make('Reacher-v5')
    env = Monitor(env)
    
    episode_lengths = []
    
    for episode in range(5):
        obs, info = env.reset()
        episode_length = 0
        
        while True:
            action = env.action_space.sample()  # 随机动作
            obs, reward, terminated, truncated, info = env.step(action)
            episode_length += 1
            
            if terminated or truncated:
                break
            
            if episode_length > 1000:  # 防止无限循环
                print(f"   Episode {episode+1}: 超过1000步，强制结束")
                break
        
        episode_lengths.append(episode_length)
        print(f"   Episode {episode+1}: {episode_length} 步")
    
    print(f"   平均 episode 长度: {sum(episode_lengths)/len(episode_lengths):.1f}")
    env.close()
    print()

def debug_real_multi_joint():
    """调试真实多关节环境的 episode 行为"""
    print("🔍 调试真实多关节环境 episode 行为")
    
    env = RealMultiJointWrapper(
        num_joints=2,
        link_lengths=[0.1, 0.1],
        render_mode=None
    )
    env = Monitor(env)
    
    episode_lengths = []
    
    for episode in range(5):
        obs, info = env.reset()
        episode_length = 0
        
        while True:
            action = env.action_space.sample()  # 随机动作
            obs, reward, terminated, truncated, info = env.step(action)
            episode_length += 1
            
            if terminated or truncated:
                break
            
            if episode_length > 1000:  # 防止无限循环
                print(f"   Episode {episode+1}: 超过1000步，强制结束")
                break
        
        episode_lengths.append(episode_length)
        print(f"   Episode {episode+1}: {episode_length} 步")
    
    print(f"   平均 episode 长度: {sum(episode_lengths)/len(episode_lengths):.1f}")
    env.close()
    print()

def debug_sac_episode_counting():
    """调试 SAC 的 episode 计数机制"""
    print("🔍 调试 SAC episode 计数机制")
    
    from stable_baselines3 import SAC
    
    # 标准环境
    print("📊 标准 Reacher-v5:")
    env1 = gym.make('Reacher-v5')
    env1 = Monitor(env1)
    
    model1 = SAC('MlpPolicy', env1, verbose=0, device='cpu')
    
    # 训练少量步数并检查
    model1.learn(total_timesteps=500, log_interval=1)
    print(f"   训练500步后的统计信息已输出")
    env1.close()
    
    print("\n📊 真实多关节环境:")
    env2 = RealMultiJointWrapper(num_joints=2, link_lengths=[0.1, 0.1])
    env2 = Monitor(env2)
    
    model2 = SAC('MlpPolicy', env2, verbose=0, device='cpu')
    
    # 训练少量步数并检查
    model2.learn(total_timesteps=500, log_interval=1)
    print(f"   训练500步后的统计信息已输出")
    env2.close()

if __name__ == "__main__":
    print("🌟 Episode 行为调试")
    print("💡 检查不同环境的 episode 长度和重置机制\n")
    
    try:
        debug_standard_reacher()
        debug_real_multi_joint()
        debug_sac_episode_counting()
        
        print("🎉 调试完成！")
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()


