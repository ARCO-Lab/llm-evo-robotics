#!/usr/bin/env python3
"""
增强版训练脚本 - PPO版本
代码结构清晰，功能模块化，保持所有核心功能
"""

import sys
import os
import time
import numpy as np
import torch
import argparse
import logging
from collections import deque

# === 路径设置 ===
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.extend([
    base_dir,
    os.path.join(base_dir, 'examples/2d_reacher/envs'),
    os.path.join(base_dir, 'examples/surrogate_model/gnn_encoder'),
    os.path.join(base_dir, 'examples/rl/train'),
    os.path.join(base_dir, 'examples/rl/common'),
    os.path.join(base_dir, 'examples/rl/environments'),
    os.path.join(base_dir, 'examples/rl')
])

# === 核心导入 ===
import gym
gym.logger.set_level(40)

if not hasattr(np, 'bool'):
    np.bool = bool

from training_logger import TrainingLogger, RealTimeMonitor
from gnn_encoder import GNN_Encoder
from attn_model.attn_model import AttnModel
# from sac.ppo_model import AttentionPPOWithBuffer
from sac.universal_ppo_model import UniversalPPOWithBuffer
from env_config.env_wrapper import make_reacher2d_vec_envs
from reacher2d_env import Reacher2DEnv

# RL相关导入
import environments
from common import *
from attn_dataset.sim_data_handler import DataHandler

# === 配置常量 ===
SILENT_MODE = True
GOAL_THRESHOLD = 20.0
DEFAULT_CONFIG = {
    'num_links': 3,
    'link_lengths': [90,90,90],
    'config_path': None
}

# === 自定义参数解析器 ===
def create_training_parser():
    """创建训练专用的参数解析器"""
    parser = argparse.ArgumentParser(description='Enhanced PPO Training for Reacher2D')
    
    # 基本参数
    parser.add_argument('--env-name', default='reacher2d', help='环境名称')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num-processes', type=int, default=1, help='并行进程数')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--save-dir', default='./trained_models/reacher2d/enhanced_test/', help='保存目录')
    
    # 渲染控制参数
    parser.add_argument('--render', action='store_true', default=False, help='是否显示可视化窗口')
    parser.add_argument('--no-render', action='store_true', default=False, help='强制禁用可视化窗口')
    
    # PPO特定参数
    parser.add_argument('--clip-epsilon', type=float, default=0.1, help='PPO裁剪参数')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='熵系数')
    parser.add_argument('--value-coef', type=float, default=0.25, help='值函数损失系数')
    parser.add_argument('--ppo-epochs', type=int, default=4, help='PPO更新轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--buffer-size', type=int, default=2048, help='缓冲区容量')
    
    # 恢复训练参数
    parser.add_argument('--resume-checkpoint', type=str, default=None, help='检查点路径')
    parser.add_argument('--resume-lr', type=float, default=None, help='恢复时的学习率')
    
    # 🔧 MAP-Elites机器人配置参数
    parser.add_argument('--num-joints', type=int, default=3, help='机器人关节数量')
    parser.add_argument('--link-lengths', nargs='+', type=float, default=[90.0, 90.0, 90.0], help='机器人链节长度')
    parser.add_argument('--total-steps', type=int, default=10000, help='总训练步数')
    
    # 兼容性参数（用于其他环境）
    parser.add_argument('--grammar-file', type=str, default='/home/xli149/Documents/repos/RoboGrammar/data/designs/grammar_jan21.dot', help='语法文件')
    parser.add_argument('--rule-sequence', nargs='+', default=['0'], help='规则序列')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='禁用CUDA')
    
    return parser

# === 工具函数 ===
def smart_print(*args, **kwargs):
    """智能打印函数"""
    if not SILENT_MODE:
        print(*args, **kwargs)

def get_time_stamp():
    """获取时间戳"""
    return time.strftime('%m-%d-%Y-%H-%M-%S')

# === 模型管理器 ===
class ModelManager:
    """模型保存和加载管理器"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.best_models_dir = os.path.join(save_dir, 'best_models')
        os.makedirs(self.best_models_dir, exist_ok=True)
    
    def save_best_model(self, ppo, success_rate, min_distance, step):
        """保存最佳PPO模型"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            model_data = {
                'step': step,
                'success_rate': success_rate,
                'min_distance': min_distance,
                'timestamp': timestamp,
                'actor_state_dict': ppo.actor.state_dict(),
                'critic_state_dict': ppo.critic.state_dict(),
                'actor_optimizer_state_dict': ppo.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': ppo.critic_optimizer.state_dict(),
                'update_count': ppo.update_count,
                'model_type': 'PPO'
            }
            
            model_file = os.path.join(self.best_models_dir, f'best_ppo_model_step_{step}_{timestamp}.pth')
            torch.save(model_data, model_file)
            
            latest_file = os.path.join(self.best_models_dir, 'latest_best_model.pth')
            torch.save(model_data, latest_file)
            
            print(f"🏆 保存最佳PPO模型: 成功率 {success_rate:.3f}, 距离 {min_distance:.1f}, 步骤 {step}")
            return True
        except Exception as e:
            print(f"❌ 保存PPO模型失败: {e}")
            return False
    
    def save_checkpoint(self, ppo, step, **kwargs):
        """智能保存检查点 - 只保存最佳模型"""
        try:
            current_best_distance = kwargs.get('current_best_distance', float('inf'))
            best_min_distance = kwargs.get('best_min_distance', float('inf'))
            
            # 只有在性能改善时才保存
            if current_best_distance < best_min_distance:
                success_rate = kwargs.get('best_success_rate', 0.0)
                print(f"🏆 发现更好性能，保存最佳模型: {current_best_distance:.1f}px")
                return self.save_best_model(ppo, success_rate, current_best_distance, step)
            else:
                print(f"⏭️  性能未改善 ({current_best_distance:.1f}px >= {best_min_distance:.1f}px)，跳过保存")
                return False
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False

    def save_final_model(self, ppo, step, **kwargs):
        """保存最终PPO模型"""
        try:
            final_model_data = {
                'step': step,
                'training_completed': True,
                'actor_state_dict': ppo.actor.state_dict(),
                'critic_state_dict': ppo.critic.state_dict(),
                'actor_optimizer_state_dict': ppo.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': ppo.critic_optimizer.state_dict(),
                'update_count': ppo.update_count,
                'model_type': 'PPO',
                **kwargs
            }
            
            final_path = os.path.join(self.best_models_dir, f'final_ppo_model_step_{step}.pth')
            torch.save(final_model_data, final_path)
            print(f"💾 保存最终PPO模型: {final_path}")
            return True
        except Exception as e:
            print(f"❌ 保存最终PPO模型失败: {e}")
            return False

    def load_checkpoint(self, ppo, checkpoint_path, device='cpu'):
        """加载PPO检查点"""
        try:
            if not os.path.exists(checkpoint_path):
                print(f"❌ 检查点文件不存在: {checkpoint_path}")
                return 0
            
            print(f"🔄 Loading PPO checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 加载网络状态
            if 'actor_state_dict' in checkpoint:
                ppo.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
                print("✅ PPO Actor loaded")
            
            if 'critic_state_dict' in checkpoint:
                ppo.critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)
                print("✅ PPO Critic loaded")
            
            # 加载优化器状态
            if 'actor_optimizer_state_dict' in checkpoint:
                ppo.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                print("✅ Actor optimizer loaded")
            
            if 'critic_optimizer_state_dict' in checkpoint:
                ppo.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                print("✅ Critic optimizer loaded")
            
            # 加载更新计数
            if 'update_count' in checkpoint:
                ppo.update_count = checkpoint['update_count']
                print(f"✅ Update count loaded: {ppo.update_count}")
            
            start_step = checkpoint.get('step', 0)
            print(f"✅ PPO Checkpoint loaded successfully! Starting step: {start_step}")
            
            return start_step
            
        except Exception as e:
            print(f"❌ Failed to load PPO checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return 0

# === 环境设置管理器 ===
class EnvironmentSetup:
    """环境设置管理器"""
    
    @staticmethod
    def create_reacher2d_env(args):
        """创建Reacher2D环境"""
        # 获取环境配置
        if hasattr(args, 'num_joints') and hasattr(args, 'link_lengths'):
            num_links = args.num_joints
            link_lengths = args.link_lengths
            print(f"🤖 使用MAP-Elites配置: {num_links}关节, 长度={link_lengths}")
        else:
            num_links = DEFAULT_CONFIG['num_links']
            link_lengths = DEFAULT_CONFIG['link_lengths']
            print(f"🤖 使用默认配置: {num_links}关节, 长度={link_lengths}")

        should_render = False
        if hasattr(args, 'render') and args.render:
            should_render = True
        elif hasattr(args, 'no_render') and args.no_render:
            should_render = False
        else:
            should_render = False
            print("🎨 渲染设置: 默认禁用 (无渲染参数)")

        env_params = {
            'num_links': num_links,
            'link_lengths': link_lengths,
            'render_mode': 'human' if args.num_processes == 1 else None,
            'config_path': DEFAULT_CONFIG['config_path']
        }
        
        print(f"环境参数: {env_params}")
        print(f"🎨 渲染设置: {'启用' if should_render else '禁用'}")
        
        # 创建训练环境
        envs = make_reacher2d_vec_envs(
            env_params=env_params,
            seed=args.seed,
            num_processes=args.num_processes,
            gamma=args.gamma,
            log_dir=None,
            device=torch.device('cpu'),
            allow_early_resets=False,
        )
        
        # 根据参数决定是否创建渲染环境
        if should_render:
            render_env_params = env_params.copy()
            render_env_params['render_mode'] = 'human'
            sync_env = Reacher2DEnv(**render_env_params)
            print(f"✅ 训练环境已创建（进程数: {args.num_processes}，带渲染）")
        else:
            sync_env = None
            print(f"✅ 训练环境已创建（进程数: {args.num_processes}，无渲染）")
            
        return envs, sync_env, env_params

# === 训练管理器 ===
class TrainingManager:
    """训练过程管理器 - PPO版本"""
    
    def __init__(self, args, ppo, logger, model_manager):
        self.args = args
        self.ppo = ppo
        self.logger = logger
        self.model_manager = model_manager
        
        # 训练状态
        self.best_success_rate = 0.0
        self.best_min_distance = float('inf')
        self.consecutive_success_count = 0
        self.min_consecutive_successes = 2

        # Episodes控制 - 2个episodes × 120k步
        self.current_episodes = 0
        self.max_episodes = 2
        self.steps_per_episode = 120000
        self.current_episode_steps = 0
        self.total_training_steps = 0
        self.episode_results = []
        self.current_episode_start_step = 0
        self.current_episode_start_time = time.time()
        
        # 追踪每个episode的最佳表现
        self.current_episode_best_distance = float('inf')
        self.current_episode_best_reward = float('-inf')
        self.current_episode_min_distance_step = 0
        
        print(f"🎯 PPO训练配置: 2个episodes × 120,000步/episode = 总计240,000步")

    def update_episode_tracking(self, episode_step, infos, episode_rewards):
        """在每个训练步骤中更新episode追踪"""
        for proc_id in range(len(infos)):
            if len(infos) > proc_id and isinstance(infos[proc_id], dict):
                info = infos[proc_id]
                
                # 提取当前距离
                current_distance = float('inf')
                if 'goal' in info:
                    current_distance = info['goal'].get('distance_to_goal', float('inf'))
                elif 'distance' in info:
                    current_distance = info['distance']
                
                # 更新最佳距离
                if current_distance < self.current_episode_best_distance:
                    self.current_episode_best_distance = current_distance
                    self.current_episode_best_reward = episode_rewards[proc_id]
                    self.current_episode_min_distance_step = episode_step

    def _classify_episode_result(self, goal_reached, distance, steps, reward):
        """分类episode结果"""
        if goal_reached:
            if steps < 200:
                return {
                    'type': 'PERFECT_SUCCESS',
                    'score': 1.0,
                    'description': '快速精确到达'
                }
            elif steps < 400:
                return {
                    'type': 'GOOD_SUCCESS', 
                    'score': 0.9,
                    'description': '正常速度到达'
                }
            else:
                return {
                    'type': 'SLOW_SUCCESS',
                    'score': 0.7,
                    'description': '缓慢但成功到达'
                }
        else:
            if distance < 50:
                return {
                    'type': 'NEAR_SUCCESS',
                    'score': 0.5,
                    'description': '接近成功'
                }
            elif distance < 100:
                return {
                    'type': 'TIMEOUT_CLOSE',
                    'score': 0.3,
                    'description': '超时但较接近'
                }
            elif reward > -100:
                return {
                    'type': 'TIMEOUT_MEDIUM',
                    'score': 0.2,
                    'description': '超时中等表现'
                }
            else:
                return {
                    'type': 'COMPLETE_FAILURE',
                    'score': 0.0,
                    'description': '完全失败'
                }

    def _check_episode_stopping_conditions(self, step):
        """检查是否应该停止训练"""
        # 🔧 修复：增加episodes数量，适合MAP-Elites训练
        if self.current_episodes >= 20:  # 从2增加到20个episodes
            print(f"🏁 完成{self.current_episodes}个episodes，训练结束")
            return True
        
        # 🔧 修复：减少单个episode步数限制，适合快速评估
        episode_steps = step - self.current_episode_start_step
        if episode_steps >= 500:  # 从120,000减少到500步
            print(f"⏰ 当前episode达到500步限制")
            return False  # 不是整体结束，只是当前episode结束
        
        return False

    def _generate_final_fitness_report(self):
        """生成最终fitness报告"""
        if len(self.episode_results) == 0:
            print("⚠️ 没有episode结果数据")
            return
        
        print("\n" + "="*50)
        print("🎯 最终PPO训练结果报告")
        print("="*50)
        
        # 计算综合指标（基于最佳距离）
        success_count = sum(1 for ep in self.episode_results if ep['success'])
        success_rate = success_count / len(self.episode_results)
        avg_best_distance = sum(ep['best_distance'] for ep in self.episode_results) / len(self.episode_results)
        avg_end_distance = sum(ep['end_distance'] for ep in self.episode_results) / len(self.episode_results)
        avg_steps = sum(ep['steps'] for ep in self.episode_results) / len(self.episode_results)
        avg_score = sum(ep['score'] for ep in self.episode_results) / len(self.episode_results)
        
        print(f"📊 Episodes完成: {len(self.episode_results)}/2")
        print(f"🎯 成功率: {success_rate:.1%} ({success_count}/{len(self.episode_results)})")
        print(f"🏆 平均最佳距离: {avg_best_distance:.1f}px")
        print(f"📏 平均结束距离: {avg_end_distance:.1f}px")
        print(f"⏱️  平均步数: {avg_steps:.0f}")
        print(f"⭐ 平均得分: {avg_score:.2f}")
        
        # 详细episode信息
        print("\n📋 详细Episode结果:")
        for i, ep in enumerate(self.episode_results, 1):
            status = "✅" if ep['success'] else "❌"
            print(f"   Episode {i}: {status} {ep['type']} - "
                  f"最佳距离:{ep['best_distance']:.1f}px@{ep['best_distance_step']}步, "
                  f"结束距离:{ep['end_distance']:.1f}px, "
                  f"得分:{ep['score']:.2f}")
        
        print("="*50)

    def handle_episode_end(self, proc_id, step, episode_rewards, infos):
        """处理episode结束逻辑 - 支持维持检查"""
        if len(infos) <= proc_id or not isinstance(infos[proc_id], dict):
            episode_rewards[proc_id] = 0.0
            return False
        
        info = infos[proc_id]
        
        # 计算episode详细信息
        episode_steps = step - self.current_episode_start_step
        episode_duration = time.time() - self.current_episode_start_time if hasattr(self, 'current_episode_start_time') else 0
        episode_reward = episode_rewards[proc_id]
        
        # 使用最佳距离而不是结束时距离
        best_distance = self.current_episode_best_distance
        best_reward = self.current_episode_best_reward
        goal_reached = best_distance < 20.0
        
        # 检查维持完成情况
        maintain_completed = False
        maintain_counter = 0
        maintain_target = 500
        
        # 尝试从环境获取维持信息
        try:
            maintain_completed = info.get('maintain_completed', False)
            maintain_counter = info.get('maintain_counter', 0)
        except:
            pass
        
        # 获取结束时的距离用于对比
        end_distance = float('inf')
        if 'goal' in info:
            end_distance = info['goal'].get('distance_to_goal', float('inf'))
        elif 'distance' in info:
            end_distance = info['distance']
        
        # 分类episode结果（基于维持完成情况）
        if maintain_completed:
            episode_type = {
                'type': 'MAINTAIN_SUCCESS',
                'score': 1.0,
                'description': f'成功维持{maintain_counter}步'
            }
            goal_reached = True  # 维持完成算作成功
        else:
            episode_type = self._classify_episode_result(goal_reached, best_distance, episode_steps, best_reward)
        
        # 存储episode结果
        episode_result = {
            'episode_num': self.current_episodes + 1,
            'type': episode_type['type'],
            'success': maintain_completed or goal_reached,
            'maintain_completed': maintain_completed,
            'maintain_counter': maintain_counter,
            'best_distance': best_distance,
            'end_distance': end_distance,
            'best_distance_step': self.current_episode_min_distance_step,
            'steps': episode_steps,
            'duration': episode_duration,
            'reward': episode_reward,
            'best_reward': best_reward,
            'score': episode_type['score'],
            'description': episode_type['description']
        }
        
        self.episode_results.append(episode_result)
        self.current_episodes += 1
        
        # 打印episode结果（显示维持信息）
        print(f"📊 Episode {self.current_episodes}/2 完成:")
        print(f"   类型: {episode_type['type']} ({episode_type['description']})")
        print(f"   成功: {'✅' if episode_result['success'] else '❌'}")
        if maintain_completed:
            print(f"   🎊 维持完成: {maintain_counter}/{maintain_target} 步")
        print(f"   最佳距离: {best_distance:.1f}px (步数: {self.current_episode_min_distance_step})")
        print(f"   结束距离: {end_distance:.1f}px")
        print(f"   总步数: {episode_steps}")
        print(f"   最终奖励: {episode_reward:.2f}")
        print(f"   得分: {episode_type['score']:.2f}")
        
        # 重置episode追踪
        self.current_episode_best_distance = float('inf')
        self.current_episode_best_reward = float('-inf')
        self.current_episode_min_distance_step = 0
        self.current_episode_start_step = step
        self.current_episode_start_time = time.time()
        
        # 🔧 修复：更新全局最佳成功率和距离
        if episode_result['success']:
            self.consecutive_success_count += 1
            # 更新全局最佳距离
            if best_distance < self.best_min_distance:
                self.best_min_distance = best_distance
            
            # 计算当前成功率
            success_count = sum(1 for ep in self.episode_results if ep['success'])
            current_success_rate = success_count / len(self.episode_results)
            
            # 更新全局最佳成功率
            if current_success_rate > self.best_success_rate:
                self.best_success_rate = current_success_rate
                print(f"🎯 更新最佳成功率: {self.best_success_rate:.1%}")
        else:
            self.consecutive_success_count = 0
        
        # 检查停止条件
        should_stop = self._check_episode_stopping_conditions(step)
        if should_stop:
            self._generate_final_fitness_report()
        
        episode_rewards[proc_id] = 0.0
        return should_stop
    
    def should_update_model(self, step):
        """PPO每个episode结束后更新"""
        # PPO不需要warmup，buffer满了就可以更新
        # return len(self.ppo.buffer.joint_q) >= self.ppo.batch_size
        return len(self.ppo.buffer.experiences) >= self.ppo.batch_size
    # def update_and_log(self, step, next_obs=None, next_gnn_embeds=None, num_joints=12):
    #     """PPO更新并记录"""
    #     metrics = self.ppo.update(next_obs, next_gnn_embeds, num_joints)
        
    #     if metrics:
    #         enhanced_metrics = metrics.copy()
    #         enhanced_metrics.update({
    #             'step': step,
    #             'buffer_size': len(self.ppo.buffer.joint_q),
    #             'learning_rate': self.ppo.actor_optimizer.param_groups[0]['lr'],
    #             'update_count': metrics['update_count']
    #         })
            
    #         self.logger.log_step(step, enhanced_metrics, episode=step//100)
            
    #         if step % 100 == 0:
    #             print(f"Step {step}: "
    #                   f"Actor Loss: {metrics['actor_loss']:.4f}, "
    #                   f"Critic Loss: {metrics['critic_loss']:.4f}, "
    #                   f"Entropy: {metrics['entropy']:.4f}")


    def update_and_log(self, step, next_obs=None, next_gnn_embeds=None, num_joints=12):
        """PPO更新并记录 - 增强版loss打印"""
        metrics = self.ppo.update(next_obs, next_gnn_embeds, num_joints)
        
        if metrics:
            enhanced_metrics = metrics.copy()
            enhanced_metrics.update({
                'step': step,
                # 'buffer_size': len(self.ppo.buffer.joint_q),
                'buffer_size': len(self.ppo.buffer.experiences),
                'learning_rate': self.ppo.actor_optimizer.param_groups[0]['lr'],
                'update_count': metrics['update_count']
            })
            
            self.logger.log_step(step, enhanced_metrics, episode=step//100)
            
            # 🔧 增强版loss打印 - 每次更新都打印
            print(f"\n🔥 PPO网络Loss更新 [Step {step}]:")
            print(f"   📊 Actor Loss: {metrics['actor_loss']:.6f}")
            print(f"   📊 Critic Loss: {metrics['critic_loss']:.6f}")
            print(f"   📊 总Loss: {metrics['actor_loss'] + metrics['critic_loss']:.6f}")
            print(f"   🎭 Entropy: {metrics['entropy']:.6f}")
            print(f"   📈 学习率: {self.ppo.actor_optimizer.param_groups[0]['lr']:.2e}")
            print(f"   🔄 更新次数: {metrics['update_count']}")
            # print(f"   💾 Buffer大小: {len(self.ppo.buffer.joint_q)}")
            print(f"   💾 Buffer大小: {len(self.ppo.buffer.experiences)}")
            
            # 🔧 添加梯度范数信息（如果可用）
            if 'actor_grad_norm' in metrics:
                print(f"   ⚡ Actor梯度范数: {metrics['actor_grad_norm']:.6f}")
            if 'critic_grad_norm' in metrics:
                print(f"   ⚡ Critic梯度范数: {metrics['critic_grad_norm']:.6f}")
            
            # 🔧 添加PPO特定指标
            if 'policy_ratio' in metrics:
                print(f"   🎯 策略比率: {metrics['policy_ratio']:.4f}")
            if 'clip_fraction' in metrics:
                print(f"   ✂️  裁剪比例: {metrics['clip_fraction']:.4f}")
            if 'kl_divergence' in metrics:
                print(f"   📏 KL散度: {metrics['kl_divergence']:.6f}")
            if 'explained_variance' in metrics:
                print(f"   📊 解释方差: {metrics['explained_variance']:.4f}")
            
            # 🔧 Loss趋势分析
            if hasattr(self, 'loss_history'):
                if len(self.loss_history) >= 3:
                    recent_losses = self.loss_history[-3:]
                    if recent_losses[-1] < recent_losses[0]:
                        trend = "📉 下降"
                    elif recent_losses[-1] > recent_losses[0]:
                        trend = "📈 上升"
                    else:
                        trend = "➡️  平稳"
                    print(f"   📈 Loss趋势: {trend}")
            else:
                self.loss_history = []
            
            # 记录loss历史
            total_loss = metrics['actor_loss'] + metrics['critic_loss']
            self.loss_history.append(total_loss)
            if len(self.loss_history) > 10:  # 只保留最近10次
                self.loss_history = self.loss_history[-10:]
            
            print(f"   {'='*50}")

def main(args):
    """主训练函数 - PPO版本"""
    print("🚀 开始PPO训练...")
    
    # 设置随机种子和设备
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device('cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建训练日志
    training_log_path = os.path.join(args.save_dir, 'logs.txt')
    with open(training_log_path, 'w') as f:
        pass
    
    # 创建环境
    if args.env_name == 'reacher2d':
        print("使用 reacher2d 环境")
        env_setup = EnvironmentSetup()
        envs, sync_env, env_params = env_setup.create_reacher2d_env(args)
        args.env_type = 'reacher2d'
    else:
        print(f"使用 bullet 环境: {args.env_name}")
        from a2c_ppo_acktr.envs import make_vec_envs
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes, 
                            args.gamma, None, device, False, args=args)
        sync_env = None
        args.env_type = 'bullet'

    # 获取关节数量和创建数据处理器
    num_joints = envs.action_space.shape[0]
    print(f"关节数量: {num_joints}")
    data_handler = DataHandler(num_joints, args.env_type)

    # 创建GNN编码器
    if args.env_type == 'reacher2d':
        sys.path.append(os.path.join(base_dir, 'examples/2d_reacher/utils'))
        from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder
        
        print("🤖 初始化 Reacher2D GNN 编码器...")
        # reacher2d_encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
        reacher2d_encoder = Reacher2D_GNN_Encoder(max_nodes=num_joints, num_joints=num_joints)
        single_gnn_embed = reacher2d_encoder.get_gnn_embeds(
            num_links=num_joints, 
            link_lengths=env_params['link_lengths']
        )
        print(f"✅ Reacher2D GNN 嵌入生成成功，形状: {single_gnn_embed.shape}")
    else:
        rule_sequence = [int(s.strip(",")) for s in args.rule_sequence]
        gnn_encoder = GNN_Encoder(args.grammar_file, rule_sequence, 70, num_joints)
        gnn_graph = gnn_encoder.get_graph(rule_sequence)
        single_gnn_embed = gnn_encoder.get_gnn_embeds(gnn_graph)

    # 创建PPO模型
    attn_model = AttnModel(128, 130, 130, 4)
    # ppo = AttentionPPOWithBuffer(
    #     attn_model, num_joints, 
    #     buffer_size=args.buffer_size, 
    #     batch_size=args.batch_size,
    #     lr=args.lr, 
    #     gamma=args.gamma,
    #     clip_epsilon=args.clip_epsilon,
    #     entropy_coef=args.entropy_coef,
    #     value_coef=args.value_coef,
    #     env_type=args.env_type
    #     # 🔧 移除了 joint_embed_dim 参数
    # )
    # 创建通用PPO模型
    print("🎯 初始化通用PPO模型...")
    ppo = UniversalPPOWithBuffer(
        buffer_size=args.buffer_size, 
        batch_size=args.batch_size,
        lr=args.lr, 
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=0.5,
        device=device,
        env_type=args.env_type
    )
    print("✅ 通用PPO模型初始化完成")
    # PPO特定参数设置
    print(f"🎯 PPO配置: clip_epsilon={args.clip_epsilon}, entropy_coef={args.entropy_coef}")
    
    # 创建训练监控系统
    experiment_name = f"reacher2d_ppo_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # 配置信息
    hyperparams = {
        'learning_rate': args.lr,
        'clip_epsilon': args.clip_epsilon,
        'entropy_coef': args.entropy_coef,
        'value_coef': args.value_coef,
        'ppo_epochs': args.ppo_epochs,
        'batch_size': args.batch_size,
        'buffer_size': args.buffer_size,
        'gamma': args.gamma,
        'seed': args.seed,
        'num_processes': args.num_processes,
        'total_steps': 240000,
        'optimizer': 'Adam',
        'algorithm': 'PPO',
        'network_architecture': {
            'attn_model_dims': '128-130-130-4',
            'action_dim': num_joints,
        }
    }
    
    env_config = {
        'env_name': args.env_name,
        'env_type': args.env_type,
        'goal_threshold': GOAL_THRESHOLD,
        'action_dim': num_joints,
    }
    
    if args.env_type == 'reacher2d':
        env_config.update({
            'num_links': env_params['num_links'],
            'link_lengths': env_params['link_lengths'],
            'config_path': env_params.get('config_path', 'N/A'),
            'render_mode': env_params.get('render_mode', 'human'),
            'reward_function': '距离+进度+成功+碰撞+方向奖励',
            'physics_engine': 'PyMunk',
            'obstacle_type': 'zigzag',
            'action_space': 'continuous',
            'observation_space': 'joint_angles_and_positions'
        })
    
    logger = TrainingLogger(
        log_dir=os.path.join(args.save_dir, 'training_logs'),
        experiment_name=experiment_name,
        hyperparams=hyperparams,
        env_config=env_config
    )
    
    print(f"📊 PPO训练监控系统已初始化: {logger.experiment_dir}")
    
    # 创建管理器
    model_manager = ModelManager(args.save_dir)
    training_manager = TrainingManager(args, ppo, logger, model_manager)
    
    # 处理checkpoint恢复
    start_step = 0
    if args.resume_checkpoint:
        print(f"🔄 从检查点恢复PPO训练: {args.resume_checkpoint}")
        start_step = model_manager.load_checkpoint(ppo, args.resume_checkpoint)

        if start_step > 0:
            print(f"成功加载PPO checkpoint, 从step {start_step} 开始训练")

            # 更新学习率
            if args.resume_lr:
                for param_group in ppo.actor_optimizer.param_groups:
                    param_group['lr'] = args.resume_lr
                for param_group in ppo.critic_optimizer.param_groups:
                    param_group['lr'] = args.resume_lr
                print(f"更新学习率为 {args.resume_lr}")
            
    # 运行训练循环
    run_training_loop(args, envs, sync_env, ppo, single_gnn_embed, training_manager, num_joints, start_step)
    training_results = collect_training_results(training_manager)
    
    # 清理资源
    cleanup_resources(sync_env, logger, model_manager, training_manager)
    return training_results

def collect_training_results(training_manager):
    """收集训练结果用于fitness计算"""
    import numpy as np
    
    if not hasattr(training_manager, 'episode_results') or not training_manager.episode_results:
        print("⚠️ 没有找到episode_results，返回默认结果")
        return {
            'success': False,
            'error': 'No episode results available',
            'episodes_completed': 0,
            'success_rate': 0.0,
            'avg_best_distance': float('inf'),
            'avg_score': 0.0,
            'total_training_time': 0.0,
            'episode_details': [],
            'learning_progress': 0.0,
            'avg_steps_to_best': 120000
        }
    
    episodes = training_manager.episode_results
    print(f"📊 收集到 {len(episodes)} 个episode的结果")
    
    # 计算基础统计
    success_count = sum(1 for ep in episodes if ep.get('success', False))
    total_episodes = len(episodes)
    
    # 计算平均最佳距离
    distances = [ep.get('best_distance', float('inf')) for ep in episodes]
    avg_best_distance = np.mean([d for d in distances if d != float('inf')]) if distances else float('inf')
    
    # 计算学习进步
    if len(episodes) >= 2:
        first_score = episodes[0].get('score', 0)
        last_score = episodes[-1].get('score', 0)
        learning_progress = last_score - first_score
    else:
        learning_progress = 0.0
    
    # 计算平均到达最佳距离的步数
    steps_to_best = [ep.get('best_distance_step', 120000) for ep in episodes]
    avg_steps_to_best = np.mean(steps_to_best)
    
    # 计算总训练时间
    durations = [ep.get('duration', 0) for ep in episodes]
    total_training_time = sum(durations)
    
    result = {
        'success': True,
        'episodes_completed': total_episodes,
        'success_rate': success_count / total_episodes if total_episodes > 0 else 0.0,
        'avg_best_distance': avg_best_distance,
        'avg_score': np.mean([ep.get('score', 0) for ep in episodes]),
        'total_training_time': total_training_time,
        'episode_details': episodes,
        'learning_progress': learning_progress,
        'avg_steps_to_best': avg_steps_to_best,
        'episode_results': episodes
    }
    
    print(f"✅ PPO训练结果收集完成:")
    print(f"   Episodes: {total_episodes}")
    print(f"   成功率: {result['success_rate']:.1%}")
    print(f"   平均最佳距离: {result['avg_best_distance']:.1f}px")
    print(f"   学习进步: {result['learning_progress']:+.3f}")
    
    return result

def run_training_loop(args, envs, sync_env, ppo, single_gnn_embed, training_manager, num_joints, start_step=0):
    """运行PPO训练循环 - Episodes版本"""
    current_obs = envs.reset()
    print(f"初始观察: {current_obs.shape}")
    
    # 重置渲染环境
    if sync_env:
        sync_env.reset()
        print("🔧 sync_env 已重置")
    
    current_gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)
    episode_rewards = [0.0] * args.num_processes
    
    # Episodes控制参数
    max_episodes = 2
    steps_per_episode = 120000
    
    print(f"开始PPO训练: {max_episodes}个episodes × {steps_per_episode}步/episode")

    training_completed = False
    early_termination_reason = ""
    global_step = start_step  # 全局步数计数器

    try:
        # Episodes循环
        for episode_num in range(max_episodes):
            print(f"\n🎯 开始PPO Episode {episode_num + 1}/{max_episodes}")
            
            print(f"🔄 重置环境开始Episode {episode_num + 1}...")
            current_obs = envs.reset()
            if sync_env:
                sync_env.reset()
                print("🔧 sync_env 已重置")
            current_gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)
            episode_rewards = [0.0] * args.num_processes
            
            # 重置episode追踪
            training_manager.current_episode_start_step = global_step
            training_manager.current_episode_start_time = time.time()
            training_manager.current_episode_best_distance = float('inf')
            training_manager.current_episode_best_reward = float('-inf')
            training_manager.current_episode_min_distance_step = 0
            
            episode_step = 0
            episode_completed = False
            
            # 单个Episode的训练循环
            while episode_step < steps_per_episode and not episode_completed:
                # 进度显示
                if episode_step % 100 == 0:
                    # smart_print(f"PPO Episode {episode_num+1}, Step {episode_step}/{steps_per_episode}: Buffer size: {len(ppo.buffer.joint_q)}")
                    smart_print(f"PPO Episode {episode_num+1}, Step {episode_step}/{steps_per_episode}: Buffer size: {len(ppo.buffer.experiences)}")

                # 获取动作 - PPO版本
                if global_step < 1000:  # PPO的简单warmup
                    action_batch = torch.from_numpy(np.array([
                        envs.action_space.sample() for _ in range(args.num_processes)
                    ]))
                    log_prob_batch = None
                    value_batch = None
                else:
                    actions = []
                    log_probs = []
                    values = []
                    for proc_id in range(args.num_processes):
                        action, log_prob, value = ppo.get_action(
                            current_obs[proc_id],
                            current_gnn_embeds[proc_id],
                            num_joints=envs.action_space.shape[0],
                            deterministic=False
                        )
                        actions.append(action)
                        log_probs.append(log_prob)
                        values.append(value)
                    
                    action_batch = torch.stack(actions)
                    log_prob_batch = torch.stack(log_probs) if log_probs[0] is not None else None
                    value_batch = torch.stack(values)

                # 动作分析（调试用）
                if episode_step % 50 == 0 or episode_step < 20:
                    if hasattr(envs, 'envs') and len(envs.envs) > 0:
                        env_goal = getattr(envs.envs[0], 'goal_pos', 'NOT FOUND')
                        print(f"🎯 [PPO Episode {episode_num+1}] Step {episode_step} - 环境goal_pos: {env_goal}")

                # 执行动作
                next_obs, reward, done, infos = envs.step(action_batch)

                # 渲染处理
                if sync_env:
                    sync_action = action_batch[0].cpu().numpy() if hasattr(action_batch, 'cpu') else action_batch[0]
                    sync_env.step(sync_action)
                    sync_env.render()

                next_gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)

                # PPO经验存储
                for proc_id in range(args.num_processes):
                    if global_step >= 1000:  # 只在非warmup期间存储
                        ppo.store_experience(
                            obs=current_obs[proc_id],
                            gnn_embeds=current_gnn_embeds[proc_id],
                            action=action_batch[proc_id],
                            reward=reward[proc_id].item(),
                            done=done[proc_id].item(),
                            log_prob=log_prob_batch[proc_id] if log_prob_batch is not None else None,
                            value=value_batch[proc_id] if value_batch is not None else None,
                            num_joints=num_joints
                        )
                    episode_rewards[proc_id] += reward[proc_id].item()

                current_obs = next_obs.clone()
                current_gnn_embeds = next_gnn_embeds.clone()

                # 更新episode追踪
                training_manager.update_episode_tracking(global_step, infos, episode_rewards)

                # 处理episode结束
                for proc_id in range(args.num_processes):
                    is_done = done[proc_id].item() if torch.is_tensor(done[proc_id]) else bool(done[proc_id])
                    if is_done:
                        print(f"🔍 [DEBUG] PPO Episode结束检测: proc_id={proc_id}, 当前episodes={training_manager.current_episodes}")
                        
                        should_end = training_manager.handle_episode_end(proc_id, episode_step, episode_rewards, infos)
                        print(f"🔍 [DEBUG] handle_episode_end返回: should_end={should_end}, 新的current_episodes={training_manager.current_episodes}")
                        
                        # 检查维持完成情况（从环境直接获取）
                        maintain_completed = False
                        if len(infos) > proc_id and isinstance(infos[proc_id], dict):
                            # 从环境info获取维持信息
                            maintain_info = infos[proc_id].get('maintain', {})
                            maintain_completed = maintain_info.get('maintain_completed', False)
                            maintain_counter = maintain_info.get('maintain_counter', 0)
                            maintain_target = maintain_info.get('maintain_target', 500)
                            
                            print(f"🏆 [DEBUG] 维持检查: {maintain_counter}/{maintain_target} 步, 完成: {maintain_completed}")
                        
                        # 检查是否到达目标（但未维持够时间）
                        goal_reached = infos[proc_id].get('goal', {}).get('distance_to_goal', float('inf')) < 20.0
                        print(f"🔍 [DEBUG] 目标检查: goal_reached={goal_reached}")
                        
                        if should_end:  # 完成2个训练episodes
                            print(f"🔍 [DEBUG] 触发should_end，整个PPO训练结束")
                            training_completed = True
                            early_termination_reason = f"完成{training_manager.current_episodes}个episodes"
                            episode_completed = True
                            break
                        elif maintain_completed:  # 维持10秒完成，结束当前训练episode
                            print(f"🔍 [DEBUG] 触发maintain_completed，当前PPO训练episode结束")
                            print(f"🎊 PPO训练Episode {episode_num+1} 维持成功完成！开始下一个episode...")
                            episode_completed = True  # 结束当前训练episode
                            break
                        elif goal_reached:  # 到达目标但未维持够时间，继续训练
                            print(f"🎯 [DEBUG] 到达目标但需继续维持，继续当前PPO episode")
                            # 关键：不break，让机器人继续学习维持
                            pass
                        
                        # 环境重置（继续当前训练episode）
                        if hasattr(envs, 'reset_one'):
                            current_obs[proc_id] = envs.reset_one(proc_id)
                            current_gnn_embeds[proc_id] = single_gnn_embed

                # PPO模型更新 - 在episode结束时更新
                if training_manager.should_update_model(global_step):
                    # PPO需要下一个状态的值函数
                    training_manager.update_and_log(
                        global_step, 
                        next_obs=current_obs[0] if args.num_processes > 0 else None,
                        next_gnn_embeds=current_gnn_embeds[0] if args.num_processes > 0 else None,
                        num_joints=num_joints
                    )
                
                # 定期保存和绘图
                if global_step % 200 == 0 and global_step > 0:
                    # 获取当前最佳距离
                    current_best_distance = training_manager.current_episode_best_distance
                    
                    # 传递当前距离用于比较
                    saved = training_manager.model_manager.save_checkpoint(
                        ppo, global_step,
                        best_success_rate=training_manager.best_success_rate,
                        best_min_distance=training_manager.best_min_distance,
                        current_best_distance=current_best_distance,
                        consecutive_success_count=training_manager.consecutive_success_count,
                        current_episode=episode_num + 1,
                        episode_step=episode_step
                    )
                    
                    # 如果保存成功，更新最佳记录
                    if saved:
                        training_manager.best_min_distance = current_best_distance
                        print(f"📈 更新全局最佳距离: {current_best_distance:.1f}px")

                # 低频日志记录
                if global_step % 2000 == 0 and global_step > 0:
                    training_manager.logger.plot_losses(recent_steps=2000, show=False)
                    print(f"📊 PPO Step {global_step}: 当前最佳距离 {training_manager.best_min_distance:.1f}px")
                
                episode_step += 1  # episode内步数递增
                global_step += args.num_processes  # 全局步数递增
                
                if training_completed:
                    break
            
            print(f"📊 PPO Episode {episode_num + 1} 完成: {episode_step} 步")
            
            if training_completed:
                print(f"🏁 PPO训练提前终止: {early_termination_reason}")
                break

    except Exception as e:
        print(f"🔴 PPO训练过程中发生错误: {e}")
        training_manager.logger.save_logs()
        training_manager.logger.generate_report()
        raise e

def cleanup_resources(sync_env, logger, model_manager, training_manager):
    """清理资源"""
    if sync_env:
        sync_env.close()
    
    # 🔧 修复：保存最终成功的模型
    if training_manager.best_success_rate > 0:
        print(f"💾 保存最终成功模型...")
        model_manager.save_final_model(
            training_manager.ppo, 
            step=training_manager.total_training_steps,
            final_success_rate=training_manager.best_success_rate,
            final_min_distance=training_manager.best_min_distance,
            final_consecutive_successes=training_manager.consecutive_success_count,
            episode_results=training_manager.episode_results
        )
            
    # 生成最终报告
    print(f"\n{'='*60}")
    print(f"🏁 PPO训练完成总结:")
    print(f"  最佳成功率: {training_manager.best_success_rate:.3f}")
    print(f"  最佳最小距离: {training_manager.best_min_distance:.1f} pixels")
    print(f"  当前连续成功次数: {training_manager.consecutive_success_count}")
    
    logger.generate_report()
    logger.plot_losses(show=False)
    print(f"📊 完整PPO训练日志已保存到: {logger.experiment_dir}")
    print(f"{'='*60}")

def test_trained_model(model_path, num_episodes=10, render=True):
    """测试训练好的PPO模型性能"""
    print(f"🧪 开始测试PPO模型: {model_path}")
    
    # 环境配置
    env_params = {
        'num_links': DEFAULT_CONFIG['num_links'],
        'link_lengths': DEFAULT_CONFIG['link_lengths'],
        'render_mode': 'human' if render else None,
        'config_path': DEFAULT_CONFIG['config_path']
    }
    
    print(f"🔧 测试环境配置: {env_params}")
    
    # 创建环境
    env = Reacher2DEnv(**env_params)
    num_joints = env.action_space.shape[0]
    
    print(f"🔍 测试环境验证:")
    print(f"   起始位置: {getattr(env, 'anchor_point', 'N/A')}")
    print(f"   目标位置: {getattr(env, 'goal_pos', 'N/A')}")
    print(f"   关节数量: {num_joints}")
    print(f"   动作空间: {env.action_space}")
    
    # 创建GNN编码器
    sys.path.append(os.path.join(base_dir, 'examples/2d_reacher/utils'))
    from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder
    
    reacher2d_encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
    gnn_embed = reacher2d_encoder.get_gnn_embeds(
        num_links=num_joints, 
        link_lengths=env_params['link_lengths']
    )
    
    print(f"   GNN嵌入形状: {gnn_embed.shape}")
    
    # 创建PPO模型 - 🔧 修复：使用与训练时相同的学习率
    attn_model = AttnModel(128, 130, 130, 4)
    # ppo = AttentionPPOWithBuffer(attn_model, num_joints, 
    #                             buffer_size=2048, batch_size=64,
    #                             lr=2e-4, env_type='reacher2d')  # 修复: 使用训练时的学习率
    # 创建通用PPO模型用于测试
    print("🎯 初始化通用PPO模型用于测试...")
    ppo = UniversalPPOWithBuffer(
        buffer_size=2048, 
        batch_size=64,
        lr=2e-4, 
        device=torch.device('cpu'),
        env_type='reacher2d'
    )
    print("✅ 通用PPO测试模型初始化完成")
    # 加载PPO模型
    try:
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return None
            
        print(f"🔄 加载PPO模型: {model_path}")
        model_data = torch.load(model_path, map_location='cpu')
        
        if 'actor_state_dict' in model_data:
            ppo.actor.load_state_dict(model_data['actor_state_dict'], strict=False)
            print("✅ PPO Actor 加载成功")
        
        if 'critic_state_dict' in model_data:
            ppo.critic.load_state_dict(model_data['critic_state_dict'], strict=False)
            print("✅ PPO Critic 加载成功")
        
        # 🔧 关键修复：设置模型为评估模式
        ppo.actor.eval()
        ppo.critic.eval()
        print("🎯 模型已设置为评估模式")
        
        # 模型验证
        print(f"🔍 PPO模型验证:")
        print(f"   Actor参数数量: {sum(p.numel() for p in ppo.actor.parameters())}")
        print(f"   Critic参数数量: {sum(p.numel() for p in ppo.critic.parameters())}")
        
        print(f"📋 PPO模型信息:")
        print(f"   训练步数: {model_data.get('step', 'N/A')}")
        print(f"   时间戳: {model_data.get('timestamp', 'N/A')}")
        print(f"   模型类型: {model_data.get('model_type', 'N/A')}")
        if 'success_rate' in model_data:
            print(f"   训练时成功率: {model_data.get('success_rate', 'N/A'):.3f}")
        if 'min_distance' in model_data:
            print(f"   训练时最小距离: {model_data.get('min_distance', 'N/A'):.1f}")
            
    except Exception as e:
        print(f"❌ 加载PPO模型失败: {e}")
        return None
    
    # 测试多个episode
    success_count = 0
    total_rewards = []
    min_distances = []
    episode_lengths = []
    
    print(f"\n🎮 开始测试PPO模型 {num_episodes} 个episodes...")
    print(f"🎯 目标阈值: {GOAL_THRESHOLD} pixels")
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        max_steps = 5000
        min_distance_this_episode = float('inf')
        episode_success = False
        
        print(f"\n📍 Episode {episode + 1}/{num_episodes}")
        print(f"   初始观察形状: {obs.shape}")
        print(f"   初始末端位置: {env._get_end_effector_position()}")
        
        # 计算初始距离
        initial_distance = np.linalg.norm(np.array(env._get_end_effector_position()) - env.goal_pos)
        print(f"   初始目标距离: {initial_distance:.1f}px")
        
        while step_count < max_steps:
            # 添加动作调试
            if step_count % 100 == 0 or step_count < 5:
                # 在测试循环中使用PPO获取动作
                action, _, _ = ppo.get_action(
                    torch.from_numpy(obs).float(),
                    gnn_embed.squeeze(0),
                    num_joints=num_joints,
                    deterministic=True  # 测试时使用确定性策略
                )
                print(f"   Step {step_count}: PPO Action = {action.detach().cpu().numpy()}")
                print(f"   Step {step_count}: 末端位置 = {env._get_end_effector_position()}")
                
                # 计算距离
                end_pos = env._get_end_effector_position()
                goal_pos = env.goal_pos
                distance = np.linalg.norm(np.array(end_pos) - goal_pos)
                print(f"   Step {step_count}: 距离 = {distance:.1f}px")
            else:
                # 获取动作（使用确定性策略）
                action, _, _ = ppo.get_action(
                    torch.from_numpy(obs).float(),
                    gnn_embed.squeeze(0),
                    num_joints=num_joints,
                    deterministic=True
                )
            
            # 执行动作
            obs, reward, done, info = env.step(action.cpu().numpy())
            episode_reward += reward
            step_count += 1
            
            # 检查距离
            end_pos = env._get_end_effector_position()
            goal_pos = env.goal_pos
            distance = np.linalg.norm(np.array(end_pos) - goal_pos)
            min_distance_this_episode = min(min_distance_this_episode, distance)
            
            # 渲染
            if render:
                env.render()
                time.sleep(0.02)
            
            # 检查是否到达目标
            if done:
                if not episode_success:
                    success_count += 1
                    episode_success = True
                    print(f"  🎉 PPO目标到达! 距离: {distance:.1f} pixels, 步骤: {step_count}")
                break
        
        total_rewards.append(episode_reward)
        min_distances.append(min_distance_this_episode)
        episode_lengths.append(step_count)
        
        print(f"  📊 Episode {episode + 1} 结果:")
        print(f"    奖励: {episode_reward:.2f}")
        print(f"    最小距离: {min_distance_this_episode:.1f} pixels")
        print(f"    步骤数: {step_count}")
        print(f"    成功: {'✅ 是' if episode_success else '❌ 否'}")
        print(f"    距离改善: {initial_distance - min_distance_this_episode:.1f}px")
    
    # 测试总结
    success_rate = success_count / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_min_distance = np.mean(min_distances)
    avg_episode_length = np.mean(episode_lengths)
    
    print(f"\n{'='*60}")
    print(f"🏆 PPO测试结果总结:")
    print(f"  测试Episodes: {num_episodes}")
    print(f"  成功次数: {success_count}")
    print(f"  成功率: {success_rate:.1%}")
    print(f"  平均奖励: {avg_reward:.2f}")
    print(f"  平均最小距离: {avg_min_distance:.1f} pixels")
    print(f"  平均Episode长度: {avg_episode_length:.1f} steps")
    print(f"  目标阈值: {GOAL_THRESHOLD:.1f} pixels")
    
    # 性能评价
    print(f"\n📋 PPO性能评价:")
    if success_rate >= 0.8:
        print(f"  🏆 优秀! 成功率 >= 80%")
    elif success_rate >= 0.5:
        print(f"  👍 良好! 成功率 >= 50%")
    elif success_rate >= 0.2:
        print(f"  ⚠️  一般! 成功率 >= 20%")
    else:
        print(f"  ❌ 需要改进! 成功率 < 20%")
        
    if avg_min_distance <= GOAL_THRESHOLD:
        print(f"  ✅ 平均最小距离达到目标阈值")
    else:
        print(f"  ⚠️  平均最小距离超出目标阈值 {avg_min_distance - GOAL_THRESHOLD:.1f} pixels")
    
    print(f"{'='*60}")
    
    env.close()
    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward, 
        'avg_min_distance': avg_min_distance,
        'avg_episode_length': avg_episode_length,
        'success_count': success_count,
        'total_episodes': num_episodes
    }

def find_latest_model():
    """查找最新的模型文件"""
    base_path = "./trained_models/reacher2d/enhanced_test"
    
    if not os.path.exists(base_path):
        print(f"❌ 训练模型目录不存在: {base_path}")
        return None
    
    model_candidates = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.pth') and ('final_model' in file or 'checkpoint' in file or 'ppo' in file):
                full_path = os.path.join(root, file)
                mtime = os.path.getmtime(full_path)
                model_candidates.append((full_path, mtime))
    
    if not model_candidates:
        print(f"❌ 在 {base_path} 中未找到PPO模型文件")
        return None
    
    model_candidates.sort(key=lambda x: x[1], reverse=True)
    
    print(f"🔍 找到 {len(model_candidates)} 个模型文件:")
    for i, (path, mtime) in enumerate(model_candidates[:5]):
        import datetime
        time_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {i+1}. {os.path.basename(path)} ({time_str})")
    
    latest_model = model_candidates[0][0]
    print(f"✅ 选择最新PPO模型: {latest_model}")
    return latest_model

# === 主程序入口 ===
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    
    # 检查是否是测试模式
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("🧪 进入PPO测试模式")
        
        model_path = None
        num_episodes = 5
        render = True
        
        for i, arg in enumerate(sys.argv):
            if arg == '--model-path' and i + 1 < len(sys.argv):
                model_path = sys.argv[i + 1]
            elif arg == '--episodes' and i + 1 < len(sys.argv):
                num_episodes = int(sys.argv[i + 1])
            elif arg == '--no-render':
                render = False
            elif arg == '--latest':
                model_path = 'latest'
        
        if model_path is None or model_path == 'latest':
            print("🔍 自动查找最新PPO模型...")
            model_path = find_latest_model()
        
        if model_path:
            print(f"🎯 PPO测试参数: episodes={num_episodes}, render={render}")
            result = test_trained_model(model_path, num_episodes, render)
            
            if result:
                print(f"\n🎯 PPO快速结论:")
                if result['success_rate'] >= 0.8:
                    print(f"  ✅ PPO模型表现优秀! 继续当前训练策略")
                elif result['success_rate'] >= 0.3:
                    print(f"  ⚠️  PPO模型表现一般，建议继续训练或调整参数")
                else:
                    print(f"  ❌ PPO模型表现较差，需要重新审视奖励函数或网络结构")
        else:
            print("❌ 未找到可测试的PPO模型")
        
        exit(0)
    
    # 训练模式 - 参数解析
    parser = create_training_parser()
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save_dir = os.path.join(args.save_dir, get_time_stamp())
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 保存参数
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        f.write(str(sys.argv))

    main(args)