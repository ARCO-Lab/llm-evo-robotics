#!/usr/bin/env python3
"""
增强版模型评估脚本
功能：
- 多个评估指标：成功率、平均奖励、最小距离、路径效率等
- 多个测试场景：不同起始位置、不同目标位置
- 可视化选项：显示轨迹、保存结果图表
- 性能分析：动作分布、角度使用情况等

使用方式: 
python evaluate_model.py --model-path ./trained_models/reacher2d/test/*/best_models/final_model_step_19999.pth --episodes 20
"""

import sys
import os
import torch
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import json

# 添加路径
base_dir = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(base_dir)
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/attn_model'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/sac'))
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/utils'))

from reacher2d_env import Reacher2DEnv
from attn_model import AttnModel
from sac_model import AttentionSACWithBuffer
from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder


class ModelEvaluator:
    def __init__(self, model_path, config_path=None):
        self.model_path = model_path
        self.config_path = config_path or "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
        
        # 环境配置
        self.env_params = {
            'num_links': 4,
            'link_lengths': [80, 80, 80, 60],
            'render_mode': None,  # 默认不渲染，可在评估时开启
            'config_path': self.config_path
        }
        
        # 初始化环境和模型
        self.env = None
        self.sac = None
        self.gnn_embed = None
        self.model_info = {}
        
        self._setup_model()
    
    def _setup_model(self):
        """设置模型和环境"""
        # 创建环境
        self.env = Reacher2DEnv(**self.env_params)
        num_joints = self.env.action_space.shape[0]
        
        # 创建GNN编码器
        reacher2d_encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
        self.gnn_embed = reacher2d_encoder.get_gnn_embeds(
            num_links=num_joints, 
            link_lengths=self.env_params['link_lengths']
        )
        
        # 创建SAC模型
        attn_model = AttnModel(128, 128, 130, 4)
        self.sac = AttentionSACWithBuffer(attn_model, num_joints, env_type='reacher2d')
        
        # 加载模型
        if os.path.exists(self.model_path):
            model_data = torch.load(self.model_path, map_location='cpu')
            self.sac.actor.load_state_dict(model_data['actor_state_dict'])
            
            # 提取模型信息
            self.model_info = {
                'step': model_data.get('step', 'N/A'),
                'success_rate': model_data.get('success_rate', 'N/A'),
                'min_distance': model_data.get('min_distance', 'N/A'),
                'timestamp': model_data.get('timestamp', 'N/A'),
                'final_success_rate': model_data.get('final_success_rate', 'N/A'),
                'final_min_distance': model_data.get('final_min_distance', 'N/A'),
                'training_completed': model_data.get('training_completed', False)
            }
            
            print(f"✅ 模型加载成功:")
            for key, value in self.model_info.items():
                print(f"   {key}: {value}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
    
    def evaluate_basic_performance(self, num_episodes=20, max_steps=1000, goal_threshold=50.0, render=False):
        """基本性能评估"""
        print(f"\n🎯 开始基本性能评估 ({num_episodes} episodes)")
        
        if render:
            self.env_params['render_mode'] = 'human'
            self.env = Reacher2DEnv(**self.env_params)
        
        results = {
            'episodes': [],
            'success_count': 0,
            'total_rewards': [],
            'min_distances': [],
            'step_counts': [],
            'action_stats': defaultdict(list),
            'trajectories': []
        }
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            step_count = 0
            min_distance_this_episode = float('inf')
            trajectory = []
            episode_actions = []
            
            print(f"  Episode {episode + 1}/{num_episodes}", end="")
            
            while step_count < max_steps:
                # 获取动作
                action = self.sac.get_action(
                    torch.from_numpy(obs).float(),
                    self.gnn_embed.squeeze(0),
                    num_joints=self.env.action_space.shape[0],
                    deterministic=True
                )
                
                # 记录轨迹
                end_pos = self.env._get_end_effector_position()
                trajectory.append(end_pos.copy())
                episode_actions.append(action.cpu().numpy().copy())
                
                # 执行动作
                obs, reward, done, info = self.env.step(action.cpu().numpy())
                episode_reward += reward
                step_count += 1
                
                # 计算距离
                goal_pos = self.env.goal_pos
                distance = np.linalg.norm(np.array(end_pos) - goal_pos)
                min_distance_this_episode = min(min_distance_this_episode, distance)
                
                if render:
                    self.env.render()
                    time.sleep(0.01)  # 控制渲染速度
                
                # 检查成功
                if distance <= goal_threshold:
                    results['success_count'] += 1
                    print(f" ✅ 成功! (距离: {distance:.1f}px, 步骤: {step_count})")
                    break
                    
                if done:
                    break
            
            if min_distance_this_episode > goal_threshold:
                print(f" ❌ 失败 (最小距离: {min_distance_this_episode:.1f}px)")
            
            # 记录结果
            results['episodes'].append({
                'episode': episode + 1,
                'reward': episode_reward,
                'min_distance': min_distance_this_episode,
                'steps': step_count,
                'success': min_distance_this_episode <= goal_threshold
            })
            
            results['total_rewards'].append(episode_reward)
            results['min_distances'].append(min_distance_this_episode)
            results['step_counts'].append(step_count)
            results['trajectories'].append(trajectory)
            
            # 动作统计
            episode_actions = np.array(episode_actions)
            for i in range(episode_actions.shape[1]):
                results['action_stats'][f'joint_{i+1}'].extend(episode_actions[:, i])
        
        # 计算总体统计
        success_rate = results['success_count'] / num_episodes
        avg_reward = np.mean(results['total_rewards'])
        avg_min_distance = np.mean(results['min_distances'])
        avg_steps = np.mean(results['step_counts'])
        
        results['summary'] = {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_min_distance': avg_min_distance,
            'avg_steps': avg_steps,
            'goal_threshold': goal_threshold
        }
        
        return results
    
    def evaluate_robustness(self, num_tests=10):
        """鲁棒性评估：测试不同初始条件"""
        print(f"\n🔄 开始鲁棒性评估 ({num_tests} 测试)")
        
        results = []
        
        # 测试不同的初始角度配置
        initial_angles = [
            [0, 0, 0, 0],           # 水平
            [45, -45, 45, -45],     # 之字形
            [90, 0, -90, 0],        # 交替
            [-45, -45, -45, -45],   # 全向下
            [30, 60, -30, -60],     # 渐变
        ]
        
        for test_idx, angles in enumerate(initial_angles):
            if test_idx >= num_tests:
                break
                
            print(f"  测试 {test_idx + 1}: 初始角度 {angles}")
            
            # 设置初始角度
            obs = self.env.reset()
            for i, angle in enumerate(angles):
                if i < len(self.env.bodies):
                    self.env.bodies[i].angle = np.radians(angle)
            
            # 运行一个episode
            episode_reward = 0
            step_count = 0
            max_steps = 500
            
            while step_count < max_steps:
                action = self.sac.get_action(
                    torch.from_numpy(obs).float(),
                    self.gnn_embed.squeeze(0),
                    num_joints=self.env.action_space.shape[0],
                    deterministic=True
                )
                
                obs, reward, done, info = self.env.step(action.cpu().numpy())
                episode_reward += reward
                step_count += 1
                
                end_pos = self.env._get_end_effector_position()
                distance = np.linalg.norm(np.array(end_pos) - self.env.goal_pos)
                
                if distance <= 50.0:
                    print(f"    ✅ 成功到达目标 (步骤: {step_count})")
                    break
                    
                if done:
                    break
            
            results.append({
                'test': test_idx + 1,
                'initial_angles': angles,
                'final_reward': episode_reward,
                'steps': step_count,
                'success': distance <= 50.0
            })
        
        return results
    
    def analyze_actions(self, num_episodes=5):
        """动作分析：分析模型的动作模式"""
        print(f"\n📊 开始动作分析 ({num_episodes} episodes)")
        
        all_actions = []
        all_observations = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_actions = []
            episode_obs = []
            
            for step in range(200):  # 分析前200步
                action = self.sac.get_action(
                    torch.from_numpy(obs).float(),
                    self.gnn_embed.squeeze(0),
                    num_joints=self.env.action_space.shape[0],
                    deterministic=True
                )
                
                episode_actions.append(action.cpu().numpy())
                episode_obs.append(obs.copy())
                
                obs, reward, done, info = self.env.step(action.cpu().numpy())
                
                if done:
                    break
            
            all_actions.extend(episode_actions)
            all_observations.extend(episode_obs)
        
        # 分析动作统计
        all_actions = np.array(all_actions)
        analysis = {
            'action_means': np.mean(all_actions, axis=0),
            'action_stds': np.std(all_actions, axis=0),
            'action_mins': np.min(all_actions, axis=0),
            'action_maxs': np.max(all_actions, axis=0),
            'action_ranges': np.max(all_actions, axis=0) - np.min(all_actions, axis=0)
        }
        
        return analysis
    
    def save_results(self, results, output_dir="evaluation_results"):
        """保存评估结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON结果
        json_file = os.path.join(output_dir, "evaluation_results.json")
        with open(json_file, 'w') as f:
            # 处理numpy类型
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return obj
            
            # 递归转换numpy类型
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {k: recursive_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(v) for v in obj]
                else:
                    return convert_numpy(obj)
            
            json.dump(recursive_convert(results), f, indent=2)
        
        print(f"✅ 结果已保存到: {json_file}")
        
        return json_file
    
    def print_summary(self, results):
        """打印评估总结"""
        print(f"\n{'='*60}")
        print(f"🏆 模型评估总结")
        print(f"{'='*60}")
        
        if 'summary' in results:
            summary = results['summary']
            print(f"📊 基本性能:")
            print(f"   成功率: {summary['success_rate']:.1%}")
            print(f"   平均奖励: {summary['avg_reward']:.2f}")
            print(f"   平均最小距离: {summary['avg_min_distance']:.1f} pixels")
            print(f"   平均步骤数: {summary['avg_steps']:.1f}")
            print(f"   目标阈值: {summary['goal_threshold']:.1f} pixels")
        
        print(f"\n🤖 模型信息:")
        for key, value in self.model_info.items():
            print(f"   {key}: {value}")
        
        print(f"{'='*60}")

    def close(self):
        """清理资源"""
        if self.env:
            self.env.close()


def main():
    parser = argparse.ArgumentParser(description="增强版模型评估脚本")
    parser.add_argument('--model-path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--episodes', type=int, default=20,
                        help='基本评估的episode数量')
    parser.add_argument('--render', action='store_true',
                        help='是否显示渲染')
    parser.add_argument('--robustness-tests', type=int, default=5,
                        help='鲁棒性测试数量')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='结果输出目录')
    parser.add_argument('--goal-threshold', type=float, default=50.0,
                        help='目标成功阈值(像素)')
    
    args = parser.parse_args()
    
    print(f"🧪 开始评估模型: {args.model_path}")
    print(f"📊 评估配置:")
    print(f"   Episodes: {args.episodes}")
    print(f"   目标阈值: {args.goal_threshold} pixels")
    print(f"   鲁棒性测试: {args.robustness_tests}")
    print(f"   渲染: {'是' if args.render else '否'}")
    
    try:
        # 创建评估器
        evaluator = ModelEvaluator(args.model_path)
        
        # 基本性能评估
        basic_results = evaluator.evaluate_basic_performance(
            num_episodes=args.episodes,
            goal_threshold=args.goal_threshold,
            render=args.render
        )
        
        # 鲁棒性评估
        robustness_results = evaluator.evaluate_robustness(args.robustness_tests)
        
        # 动作分析
        action_analysis = evaluator.analyze_actions(num_episodes=5)
        
        # 汇总结果
        full_results = {
            'model_info': evaluator.model_info,
            'basic_performance': basic_results,
            'robustness': robustness_results,
            'action_analysis': action_analysis,
            'evaluation_config': {
                'episodes': args.episodes,
                'goal_threshold': args.goal_threshold,
                'robustness_tests': args.robustness_tests
            }
        }
        
        # 打印总结
        evaluator.print_summary(basic_results)
        
        # 保存结果
        output_file = evaluator.save_results(full_results, args.output_dir)
        
        print(f"\n📋 详细分析:")
        print(f"   鲁棒性测试成功率: {sum(1 for r in robustness_results if r['success']) / len(robustness_results):.1%}")
        print(f"   动作范围分析: joint1: {action_analysis['action_ranges'][0]:.2f}, joint2: {action_analysis['action_ranges'][1]:.2f}")
        print(f"   评估结果已保存到: {output_file}")
        
        # 清理
        evaluator.close()
        
    except Exception as e:
        print(f"❌ 评估过程中发生错误: {e}")
        raise e


if __name__ == "__main__":
    main() 