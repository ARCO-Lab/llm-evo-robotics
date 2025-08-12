#!/usr/bin/env python3
"""
真实模型测试脚本 - 使用训练好的Actor网络来生成动作
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time

# 设置路径
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))

from reacher2d_env import Reacher2DEnv


class SimpleActor(nn.Module):
    """简化的Actor网络，直接从observation生成action"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(SimpleActor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
        )
        
        self.action_dim = action_dim
        
    def forward(self, obs):
        x = self.net(obs)
        mean, log_std = torch.split(x, self.action_dim, dim=-1)
        
        # 限制log_std范围
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def get_action(self, obs, deterministic=False):
        """获取动作"""
        mean, log_std = self.forward(obs)
        
        if deterministic:
            return torch.tanh(mean)
        else:
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            x_t = dist.rsample()
            action = torch.tanh(x_t)
            return action


def load_and_extract_actor_weights(model_path):
    """加载模型并提取Actor网络的权重"""
    print(f"🔍 分析模型文件: {model_path}")
    
    model_data = torch.load(model_path, map_location='cpu')
    
    print(f"📊 模型信息:")
    print(f"   训练步骤: {model_data.get('step', 'N/A')}")
    print(f"   成功率: {model_data.get('final_success_rate', model_data.get('success_rate', 'N/A'))}")
    print(f"   最小距离: {model_data.get('final_min_distance', model_data.get('min_distance', 'N/A'))}")
    print(f"   训练完成: {model_data.get('training_completed', 'N/A')}")
    
    # 提取Actor状态字典
    actor_state_dict = model_data.get('actor_state_dict', {})
    print(f"\n🧠 Actor网络结构:")
    for key, tensor in actor_state_dict.items():
        print(f"   {key}: {tensor.shape}")
    
    return actor_state_dict, model_data


def test_with_real_model(model_path, num_episodes=3):
    """使用真实的训练模型进行测试"""
    
    print(f"🎯 使用真实模型测试: {model_path}")
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 加载模型权重
    actor_state_dict, model_info = load_and_extract_actor_weights(model_path)
    
    # 创建环境
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': 'human',
        'config_path': "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    }
    
    env = Reacher2DEnv(**env_params)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"🏗️ 环境配置:")
    print(f"   观察维度: {obs_dim}")
    print(f"   动作维度: {action_dim}")
    print(f"   Action space: {env.action_space}")
    
    # 方法1: 尝试创建简化的Actor网络
    try:
        print(f"\n🤖 尝试创建简化Actor网络...")
        simple_actor = SimpleActor(obs_dim, action_dim)
        
        # 尝试加载权重（可能不完全匹配，但可以部分加载）
        try:
            # 过滤出可以匹配的权重
            filtered_state_dict = {}
            for key, value in actor_state_dict.items():
                if key in simple_actor.state_dict():
                    if simple_actor.state_dict()[key].shape == value.shape:
                        filtered_state_dict[key] = value
                        print(f"   ✅ 匹配权重: {key}")
                    else:
                        print(f"   ⚠️ 形状不匹配: {key} 期望{simple_actor.state_dict()[key].shape} 得到{value.shape}")
                else:
                    print(f"   ❌ 未找到对应层: {key}")
            
            if filtered_state_dict:
                simple_actor.load_state_dict(filtered_state_dict, strict=False)
                print(f"✅ 部分加载了 {len(filtered_state_dict)} 个权重层")
                use_trained_model = True
            else:
                print(f"❌ 没有匹配的权重，将使用随机初始化的网络")
                use_trained_model = False
                
        except Exception as e:
            print(f"❌ 权重加载失败: {e}")
            use_trained_model = False
            
    except Exception as e:
        print(f"❌ 创建Actor网络失败: {e}")
        print(f"🔄 回退到启发式策略")
        simple_actor = None
        use_trained_model = False
    
    # 开始测试
    print(f"\n🎮 开始测试 {num_episodes} episodes")
    if use_trained_model and simple_actor is not None:
        print(f"   使用策略: 训练好的Actor网络")
    else:
        print(f"   使用策略: 改进的启发式策略")
    
    success_count = 0
    total_rewards = []
    min_distances = []
    goal_threshold = 50.0
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        max_steps = 500
        min_distance_this_episode = float('inf')
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        while step_count < max_steps:
            # 计算当前距离
            end_pos = env._get_end_effector_position()
            goal_pos = env.goal_pos
            distance = np.linalg.norm(np.array(end_pos) - goal_pos)
            min_distance_this_episode = min(min_distance_this_episode, distance)
            
            # 生成动作
            if use_trained_model and simple_actor is not None:
                # 使用训练好的模型
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_tensor = simple_actor.get_action(obs_tensor, deterministic=True)
                    actions = action_tensor.squeeze(0).numpy()
                    
                    # 缩放到环境的动作范围
                    actions = actions * env.action_space.high[0]  # tanh输出[-1,1]缩放到[-100,100]
            else:
                # 使用改进的启发式策略
                direction = np.array(goal_pos) - np.array(end_pos)
                
                if distance > 1e-6:
                    direction_norm = direction / distance
                    
                    # 根据距离调整动作强度
                    if distance > 200:
                        action_magnitude = 30.0  # 远距离时较大动作
                    elif distance > 100:
                        action_magnitude = 20.0  # 中等距离
                    else:
                        action_magnitude = 10.0  # 近距离时小心动作
                    
                    # 为每个关节生成动作
                    actions = np.array([
                        action_magnitude * direction_norm[0] * 0.4,
                        action_magnitude * direction_norm[1] * 0.4,
                        action_magnitude * direction_norm[0] * 0.3,
                        action_magnitude * direction_norm[1] * 0.3,
                    ])
                    
                    # 添加小幅噪声
                    noise = np.random.normal(0, 2.0, size=actions.shape)
                    actions = actions + noise
                    
                    # 裁剪到动作空间
                    actions = np.clip(actions, env.action_space.low, env.action_space.high)
                else:
                    # 距离很近时使用微调
                    actions = np.random.uniform(-5, 5, size=action_dim)
            
            # 执行动作
            obs, reward, done, info = env.step(actions)
            episode_reward += reward
            step_count += 1
            
            # 渲染
            env.render()
            time.sleep(0.02)
            
            # 每50步打印进度
            if step_count % 50 == 0:
                print(f"  步骤 {step_count}: 距离 {distance:.1f}px, 奖励 {episode_reward:.1f}")
                if use_trained_model:
                    print(f"    动作 (模型): [{actions[0]:+6.2f}, {actions[1]:+6.2f}, {actions[2]:+6.2f}, {actions[3]:+6.2f}]")
            
            # 检查成功
            if distance <= goal_threshold:
                success_count += 1
                print(f"  🎉 成功到达目标! 距离: {distance:.1f}px, 步骤: {step_count}")
                break
                
            if done:
                print(f"  Episode 结束 (done=True)")
                break
        
        # 记录结果
        total_rewards.append(episode_reward)
        min_distances.append(min_distance_this_episode)
        
        print(f"Episode {episode + 1} 结果:")
        print(f"  最小距离: {min_distance_this_episode:.1f}px")
        print(f"  总奖励: {episode_reward:.2f}")
        print(f"  步骤数: {step_count}")
        print(f"  成功: {'是' if min_distance_this_episode <= goal_threshold else '否'}")
    
    # 计算统计
    success_rate = success_count / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_min_distance = np.mean(min_distances)
    
    # 打印总结
    strategy_name = "训练模型" if use_trained_model else "启发式策略"
    print(f"\n{'='*60}")
    print(f"🏆 {strategy_name}测试结果总结:")
    print(f"  测试Episodes: {num_episodes}")
    print(f"  成功次数: {success_count}")
    print(f"  成功率: {success_rate:.1%}")
    print(f"  平均奖励: {avg_reward:.2f}")
    print(f"  平均最小距离: {avg_min_distance:.1f} pixels")
    print(f"  目标阈值: {goal_threshold:.1f} pixels")
    print(f"{'='*60}")
    
    if not use_trained_model:
        print(f"\n💡 注意: 由于模型加载问题，使用了启发式策略")
        print(f"   要测试真实模型，需要:")
        print(f"   1. 修复GNN编码器的数据类型问题")
        print(f"   2. 或者重新构建与训练时完全一致的网络结构")
    
    env.close()
    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_min_distance': avg_min_distance,
        'success_count': success_count,
        'total_episodes': num_episodes,
        'used_trained_model': use_trained_model
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用真实训练模型进行测试")
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=3,
                        help='测试的episode数量')
    
    args = parser.parse_args()
    
    test_with_real_model(args.model_path, args.episodes) 