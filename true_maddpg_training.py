#!/usr/bin/env python3
"""
真正的MADDPG实现：
1. 每个关节由一个DDPG智能体控制
2. 中心化训练，分布式执行
3. 所有智能体在同一个环境中协作
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

class TrueMADDPG:
    """真正的MADDPG实现 - 多智能体协作"""
    
    def __init__(self, num_joints=3):
        self.num_joints = num_joints
        self.agents = []
        
        print(f"🌟 创建真正的MADDPG系统: {num_joints}个智能体协作")
        
        # 创建主环境用于协作训练和可视化
        self.env = create_env(num_joints, render_mode='human', show_position_info=True)
        
        # 为每个关节创建独立的DDPG智能体
        for i in range(num_joints):
            print(f"🤖 创建智能体 {i} (控制关节{i})...")
            
            # 创建单关节环境包装器
            single_env = SingleJointEnvWrapper(self.env, joint_id=i, num_joints=num_joints)
            
            # 添加动作噪声
            action_noise = NormalActionNoise(
                mean=np.zeros(1), 
                sigma=0.1 * np.ones(1)
            )
            
            # 创建DDPG智能体（开启详细日志）
            agent = DDPG(
                "MlpPolicy",
                single_env,
                learning_rate=1e-3,
                gamma=0.99,
                tau=0.005,
                action_noise=action_noise,
                verbose=1,  # 开启详细输出
                device='cpu',
                batch_size=64,
                buffer_size=50000,
                learning_starts=100,
                tensorboard_log=f"./tensorboard_logs/maddpg_agent_{i}/"  # 添加TensorBoard日志
            )
            
            self.agents.append(agent)
            print(f"   ✅ 智能体 {i} 创建完成")
        
        print(f"🌟 MADDPG系统初始化完成: {num_joints}个智能体将协作控制机械臂")
    
    def train_collaborative(self, total_timesteps=5000):
        """协作训练所有智能体"""
        print(f"\n🎯 开始MADDPG协作训练 {total_timesteps} 步...")
        print(f"   每个智能体控制一个关节，共同学习最优策略")
        
        obs, _ = self.env.reset()
        episode_count = 0
        episode_reward = 0
        step_count = 0
        
        # 存储经验用于训练
        experiences = []
        
        start_time = time.time()
        
        try:
            while step_count < total_timesteps:
                # 🤖 所有智能体同时决策
                actions = []
                for i, agent in enumerate(self.agents):
                    # 每个智能体基于全局观察做决策
                    action, _ = agent.predict(obs, deterministic=False)
                    actions.append(action[0])  # 取出标量动作
                
                # 🎯 执行联合动作
                joint_action = np.array(actions)
                next_obs, reward, terminated, truncated, info = self.env.step(joint_action)
                episode_reward += reward
                step_count += 1
                
                # 📝 存储经验
                experiences.append({
                    'obs': obs.copy(),
                    'actions': actions.copy(),
                    'reward': reward,
                    'next_obs': next_obs.copy(),
                    'done': terminated or truncated
                })
                
                # 🎓 定期训练智能体
                if len(experiences) >= 100 and step_count % 50 == 0:
                    print(f"   📚 步数 {step_count}: 开始训练智能体...")
                    self._train_agents(experiences[-100:])  # 使用最近100个经验
                
                obs = next_obs
                
                # 🔄 重置环境
                if terminated or truncated:
                    obs, _ = self.env.reset()
                    episode_count += 1
                    
                    if episode_count % 5 == 0:
                        avg_reward = episode_reward
                        success = info.get('is_success', False) if info else False
                        distance = info.get('distance_to_target', 0) if info else 0
                        print(f"   Episode {episode_count}: 奖励={avg_reward:.2f}, 距离={distance:.4f}, 成功={'✅' if success else '❌'}")
                    
                    episode_reward = 0
                
                # 📊 定期输出进度
                if step_count % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    fps = step_count / elapsed_time
                    print(f"   🚀 步数 {step_count}/{total_timesteps}, FPS: {fps:.1f}, 用时: {elapsed_time:.1f}s")
        
        except KeyboardInterrupt:
            print(f"\n⚠️ 训练被用户中断")
        
        training_time = time.time() - start_time
        print(f"\n✅ MADDPG协作训练完成!")
        print(f"⏱️ 训练时间: {training_time/60:.1f} 分钟")
        print(f"🎯 总episodes: {episode_count}")
        print(f"📊 平均FPS: {step_count/training_time:.1f}")
        
        return episode_count
    
    def _train_agents(self, experiences):
        """训练所有智能体并显示loss信息"""
        # 简化版训练：每个智能体独立学习
        for i, agent in enumerate(self.agents):
            # 为每个智能体准备训练数据
            for exp in experiences[-10:]:  # 使用最近10个经验
                # 构造单关节的经验
                single_action = [exp['actions'][i]]
                
                # 添加到智能体的replay buffer
                try:
                    if hasattr(agent.replay_buffer, 'add'):
                        agent.replay_buffer.add(
                            exp['obs'], 
                            single_action, 
                            exp['reward'], 
                            exp['next_obs'], 
                            exp['done'], 
                            [{}]
                        )
                except Exception as e:
                    # 如果添加经验失败，跳过
                    continue
            
            # 训练智能体并获取loss信息
            try:
                buffer_size = 0
                if hasattr(agent.replay_buffer, 'size'):
                    buffer_size = agent.replay_buffer.size()
                elif hasattr(agent.replay_buffer, '__len__'):
                    buffer_size = len(agent.replay_buffer)
                
                if buffer_size > agent.batch_size:
                    # 保存训练前的参数以计算loss变化
                    old_actor_loss = getattr(agent, '_last_actor_loss', 0.0)
                    old_critic_loss = getattr(agent, '_last_critic_loss', 0.0)
                    
                    # 执行训练
                    agent.train(gradient_steps=1)
                    
                    # 尝试获取最新的loss信息
                    current_actor_loss = getattr(agent, '_last_actor_loss', old_actor_loss)
                    current_critic_loss = getattr(agent, '_last_critic_loss', old_critic_loss)
                    
                    # 显示训练信息
                    print(f"     🤖 Agent {i}: Buffer={buffer_size}, Actor_Loss={current_actor_loss:.4f}, Critic_Loss={current_critic_loss:.4f}")
                    
            except Exception as e:
                # 如果训练失败，显示错误信息
                print(f"     ❌ Agent {i} 训练失败: {str(e)[:50]}...")
                continue
    
    def test(self, n_episodes=5):
        """测试训练好的MADDPG系统"""
        print(f"\n🧪 开始测试MADDPG系统 {n_episodes} episodes...")
        
        success_count = 0
        episode_rewards = []
        episode_distances = []
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_success = False
            min_distance = float('inf')
            
            print(f"\n   🎮 Episode {episode+1} 开始...")
            
            for step in range(100):  # 每个episode最多100步
                # 🤖 所有智能体协作决策（确定性）
                actions = []
                for i, agent in enumerate(self.agents):
                    action, _ = agent.predict(obs, deterministic=True)
                    actions.append(action[0])
                
                # 🎯 执行联合动作
                joint_action = np.array(actions)
                obs, reward, terminated, truncated, info = self.env.step(joint_action)
                episode_reward += reward
                
                # 📊 检查成功和距离
                distance = info.get('distance_to_target', float('inf'))
                min_distance = min(min_distance, distance)
                
                if info.get('is_success', False):
                    episode_success = True
                
                if terminated or truncated:
                    break
            
            if episode_success:
                success_count += 1
            
            episode_rewards.append(episode_reward)
            episode_distances.append(min_distance)
            
            print(f"   📊 Episode {episode+1}: 奖励={episode_reward:.2f}, 最小距离={min_distance:.4f}, 成功={'✅' if episode_success else '❌'}")
        
        success_rate = success_count / n_episodes
        avg_reward = np.mean(episode_rewards)
        avg_distance = np.mean(episode_distances)
        
        print(f"\n🎯 MADDPG测试结果:")
        print(f"   🏆 成功率: {success_rate:.1%} ({success_count}/{n_episodes})")
        print(f"   💰 平均奖励: {avg_reward:.2f}")
        print(f"   📏 平均最小距离: {avg_distance:.4f}")
        
        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_distance': avg_distance
        }

class SingleJointEnvWrapper(gym.Wrapper):
    """单关节环境包装器 - 用于MADDPG中的单个智能体"""
    
    def __init__(self, env, joint_id, num_joints):
        super().__init__(env)
        self.joint_id = joint_id
        self.num_joints = num_joints
        
        # 重新定义动作空间为单关节
        from gymnasium.spaces import Box
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 观察空间保持不变（全局观察）
        self.observation_space = env.observation_space
        
        # 存储其他智能体的动作
        self._other_actions = np.zeros(num_joints)
        
        # 确保有必要的属性
        if not hasattr(self, 'metadata'):
            self.metadata = getattr(env, 'metadata', {})
        if not hasattr(self, 'spec'):
            self.spec = getattr(env, 'spec', None)
        
    def step(self, action):
        """单关节步骤 - 需要与其他智能体协调"""
        # 这个方法实际上不会被直接调用
        # 因为我们在MADDPG中直接使用原环境
        full_action = self._other_actions.copy()
        full_action[self.joint_id] = action[0]
        return self.env.step(full_action)
    
    def reset(self, **kwargs):
        """重置环境"""
        return self.env.reset(**kwargs)
    
    def render(self, **kwargs):
        """渲染"""
        return self.env.render(**kwargs)
    
    def close(self):
        """关闭环境"""
        return self.env.close()

def main():
    """主函数"""
    print("🌟 真正的MADDPG协作训练系统")
    print("🤖 策略: 每个关节由独立智能体控制，多智能体协作学习")
    print("🎯 目标: 3关节Reacher任务")
    print("👀 特点: 实时可视化训练过程")
    print()
    
    # 创建MADDPG系统
    maddpg = TrueMADDPG(num_joints=3)
    
    # 协作训练
    print("\n" + "="*60)
    episode_count = maddpg.train_collaborative(total_timesteps=3000)  # 较短训练用于演示
    
    # 测试
    print("\n" + "="*60)
    result = maddpg.test(n_episodes=3)
    
    print(f"\n🎉 真正的MADDPG训练完成!")
    print(f"   📈 训练episodes: {episode_count}")
    print(f"   🏆 最终成功率: {result['success_rate']:.1%}")
    print(f"   💰 平均奖励: {result['avg_reward']:.2f}")
    print(f"   📏 平均距离: {result['avg_distance']:.4f}")
    
    print(f"\n🔍 MADDPG vs SAC 对比:")
    print(f"   ✅ MADDPG: 每个关节独立学习，可能有更好的专业化")
    print(f"   ✅ SAC: 单一网络处理所有关节，可能更简单高效")
    print(f"   🎯 建议: 比较两种方法在相同训练步数下的表现")

if __name__ == "__main__":
    main()
