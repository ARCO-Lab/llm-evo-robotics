#!/usr/bin/env python3
"""
增强版训练脚本 - 重构版
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
from sac.sac_model import AttentionSACWithBuffer
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
    'num_links': 2,
    'link_lengths': [90,90],
    # 'config_path': "/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
    'config_path': None
}

# === 自定义参数解析器 ===
def create_training_parser():
    """创建训练专用的参数解析器"""
    parser = argparse.ArgumentParser(description='Enhanced SAC Training for Reacher2D')
    
    # 基本参数
    parser.add_argument('--env-name', default='reacher2d', help='环境名称')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num-processes', type=int, default=1, help='并行进程数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--alpha', type=float, default=0.1, help='SAC熵系数')
    parser.add_argument('--save-dir', default='./trained_models/reacher2d/enhanced_test/', help='保存目录')
    
      # 🆕 添加渲染控制参数
    parser.add_argument('--render', action='store_true', default=False, help='是否显示可视化窗口')
    parser.add_argument('--no-render', action='store_true', default=False, help='强制禁用可视化窗口')
    # SAC特定参数
    parser.add_argument('--tau', type=float, default=0.005, help='软更新参数')
    parser.add_argument('--warmup-steps', type=int, default=1000, help='热身步数')
    parser.add_argument('--target-entropy-factor', type=float, default=0.8, help='目标熵系数')
    parser.add_argument('--update-frequency', type=int, default=2, help='网络更新频率')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--buffer-capacity', type=int, default=10000, help='缓冲区容量')
    
    # 恢复训练参数
    parser.add_argument('--resume-checkpoint', type=str, default=None, help='检查点路径')
    parser.add_argument('--resume-lr', type=float, default=None, help='恢复时的学习率')
    parser.add_argument('--resume-alpha', type=float, default=None, help='恢复时的alpha值')
    
    # MAP-Elites机器人配置参数
    parser.add_argument('--num-joints', type=int, default=3, help='机器人关节数量')
    parser.add_argument('--link-lengths', nargs='+', type=float, default=[90.0, 90.0, 90.0], help='机器人链节长度')
    parser.add_argument('--total-steps', type=int, default=10000, help='总训练步数')
    parser.add_argument('--individual-id', type=str, default='', help='MAP-Elites个体ID')
    parser.add_argument('--generation', type=int, default=0, help='当前进化代数')
    
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
    
    def save_best_model(self, sac, success_rate, min_distance, step):
        """保存最佳模型"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            model_data = {
                'step': step,
                'success_rate': success_rate,
                'min_distance': min_distance,
                'timestamp': timestamp,
                'actor_state_dict': sac.actor.state_dict(),
                'critic1_state_dict': sac.critic1.state_dict(),
                'critic2_state_dict': sac.critic2.state_dict(),
                'target_critic1_state_dict': sac.target_critic1.state_dict(),
                'target_critic2_state_dict': sac.target_critic2.state_dict(),
                'actor_optimizer_state_dict': sac.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': sac.critic_optimizer.state_dict(),
                'alpha_optimizer_state_dict': sac.alpha_optimizer.state_dict(),
                'alpha': sac.alpha.item() if torch.is_tensor(sac.alpha) else sac.alpha,
                'log_alpha': sac.log_alpha.item() if torch.is_tensor(sac.log_alpha) else sac.log_alpha,
            }
            
            model_file = os.path.join(self.best_models_dir, f'best_model_step_{step}_{timestamp}.pth')
            torch.save(model_data, model_file)
            
            latest_file = os.path.join(self.best_models_dir, 'latest_best_model.pth')
            torch.save(model_data, latest_file)
            
            print(f"🏆 保存最佳模型: 成功率 {success_rate:.3f}, 距离 {min_distance:.1f}, 步骤 {step}")
            return True
        except Exception as e:
            print(f"❌ 保存模型失败: {e}")
            return False
    
    # 修复 save_checkpoint 方法 (第145-159行)
    # def save_checkpoint(self, sac, step, **kwargs):
    #     """保存检查点 - 完整版"""
    #     try:
    #         checkpoint_data = {
    #             'step': step,
    #             'actor_state_dict': sac.actor.state_dict(),
    #             'critic1_state_dict': sac.critic1.state_dict(),
    #             'critic2_state_dict': sac.critic2.state_dict(),
    #             'target_critic1_state_dict': sac.target_critic1.state_dict(),
    #             'target_critic2_state_dict': sac.target_critic2.state_dict(),
    #             # 🔧 添加优化器状态
    #             'actor_optimizer_state_dict': sac.actor_optimizer.state_dict(),
    #             'critic_optimizer_state_dict': sac.critic_optimizer.state_dict(),
    #             'alpha_optimizer_state_dict': sac.alpha_optimizer.state_dict(),
    #             # 🔧 添加alpha值
    #             'alpha': sac.alpha.item() if torch.is_tensor(sac.alpha) else sac.alpha,
    #             'log_alpha': sac.log_alpha.item() if torch.is_tensor(sac.log_alpha) else sac.log_alpha,
    #             # 🔧 添加训练状态
    #             'buffer_size': len(sac.memory),
    #             'warmup_steps': sac.warmup_steps,
    #             **kwargs
    #         }
            
    #         checkpoint_path = os.path.join(self.best_models_dir, f'checkpoint_step_{step}.pth')
    #         torch.save(checkpoint_data, checkpoint_path)
    #         print(f"💾 保存检查点: step {step}, buffer size: {len(sac.memory)}")
    #         return True
    #     except Exception as e:
    #         print(f"❌ 保存检查点失败: {e}")
    #         return False
    def save_checkpoint(self, sac, step, **kwargs):
        """智能保存检查点 - 只保存最佳模型"""
        try:
            current_best_distance = kwargs.get('current_best_distance', float('inf'))
            best_min_distance = kwargs.get('best_min_distance', float('inf'))
            
            # 只有在性能改善时才保存
            if current_best_distance < best_min_distance:
                success_rate = kwargs.get('best_success_rate', 0.0)
                print(f"🏆 发现更好性能，保存最佳模型: {current_best_distance:.1f}px")
                return self.save_best_model(sac, success_rate, current_best_distance, step)
            else:
                print(f"⏭️  性能未改善 ({current_best_distance:.1f}px >= {best_min_distance:.1f}px)，跳过保存")
                return False
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False
    # 修复 save_final_model 方法 (第161-176行)
    def save_final_model(self, sac, step, **kwargs):
        """保存最终模型 - 完整版"""
        try:
            final_model_data = {
                'step': step,
                'training_completed': True,
                'actor_state_dict': sac.actor.state_dict(),
                'critic1_state_dict': sac.critic1.state_dict(),
                'critic2_state_dict': sac.critic2.state_dict(),
                'target_critic1_state_dict': sac.target_critic1.state_dict(),
                'target_critic2_state_dict': sac.target_critic2.state_dict(),
                # 🔧 添加优化器状态
                'actor_optimizer_state_dict': sac.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': sac.critic_optimizer.state_dict(),
                'alpha_optimizer_state_dict': sac.alpha_optimizer.state_dict(),
                # 🔧 添加alpha值
                'alpha': sac.alpha.item() if torch.is_tensor(sac.alpha) else sac.alpha,
                'log_alpha': sac.log_alpha.item() if torch.is_tensor(sac.log_alpha) else sac.log_alpha,
                **kwargs
            }
            
            final_path = os.path.join(self.best_models_dir, f'final_model_step_{step}.pth')
            torch.save(final_model_data, final_path)
            print(f"💾 保存最终模型: {final_path}")
            return True
        except Exception as e:
            print(f"❌ 保存最终模型失败: {e}")
            return False

    # 修复 load_checkpoint 方法 (第178-226行)
    def load_checkpoint(self, sac, checkpoint_path, device='cpu'):
        """加载检查点 - 增强版"""
        try:
            if not os.path.exists(checkpoint_path):
                print(f"❌ 检查点文件不存在: {checkpoint_path}")
                return 0
            
            print(f"🔄 Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            print(f"📋 Checkpoint contains: {list(checkpoint.keys())}")
            
            # 🔧 加载网络状态 - 增强错误处理
            networks = ['actor', 'critic1', 'critic2', 'target_critic1', 'target_critic2']
            for network_name in networks:
                state_dict_key = f'{network_name}_state_dict'
                if state_dict_key in checkpoint:
                    try:
                        network = getattr(sac, network_name)
                        state_dict = checkpoint[state_dict_key]
                        
                        # 🔧 处理proj_dict问题
                        if 'proj_dict' in str(state_dict.keys()):
                            print(f"⚠️ {network_name} 包含proj_dict，使用strict=False加载")
                            missing_keys, unexpected_keys = network.load_state_dict(state_dict, strict=False)
                            if unexpected_keys:
                                print(f"   忽略的键: {unexpected_keys[:3]}...")  # 只显示前3个
                        else:
                            network.load_state_dict(state_dict)
                        
                        print(f"✅ {network_name} loaded")
                    except Exception as e:
                        print(f"⚠️ {network_name} 加载失败: {e}")
                        print("   将跳过该网络，使用初始化权重")
                else:
                    print(f"⚠️ 未找到 {state_dict_key}")
            
            # 🔧 加载优化器状态 - 可选加载
            load_optimizers = True  # 可以设为False如果想用新的学习率
            if load_optimizers:
                optimizers = ['actor_optimizer', 'critic_optimizer', 'alpha_optimizer']
                for opt_name in optimizers:
                    opt_key = f'{opt_name}_state_dict'
                    if opt_key in checkpoint:
                        try:
                            optimizer = getattr(sac, opt_name)
                            optimizer.load_state_dict(checkpoint[opt_key])
                            print(f"✅ {opt_name} loaded")
                        except Exception as e:
                            print(f"⚠️ {opt_name} 加载失败: {e}")
                            print("   将使用当前优化器状态")
            
            # 🔧 加载alpha值
            if 'alpha' in checkpoint:
                try:
                    alpha_val = checkpoint['alpha']
                    if isinstance(alpha_val, (int, float)):
                        sac.alpha = alpha_val
                    else:
                        sac.alpha = alpha_val.item()
                    print(f"✅ Alpha loaded: {sac.alpha}")
                except Exception as e:
                    print(f"⚠️ Alpha 加载失败: {e}")
                    
            if 'log_alpha' in checkpoint:
                try:
                    log_alpha_val = checkpoint['log_alpha']
                    if isinstance(log_alpha_val, (int, float)):
                        sac.log_alpha.data.fill_(log_alpha_val)
                    else:
                        sac.log_alpha.data.fill_(log_alpha_val.item())
                    print(f"✅ Log Alpha loaded: {sac.log_alpha.item()}")
                except Exception as e:
                    print(f"⚠️ Log Alpha 加载失败: {e}")
            
            # 🔧 显示额外信息
            start_step = checkpoint.get('step', 0)
            buffer_size = checkpoint.get('buffer_size', 'N/A')
            warmup_steps = checkpoint.get('warmup_steps', 'N/A')
            
            print(f"✅ Checkpoint loaded successfully!")
            print(f"   Starting step: {start_step}")
            print(f"   Buffer size: {buffer_size}")
            print(f"   Warmup steps: {warmup_steps}")
            
            return start_step
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            print("Training will start from scratch...")
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


        should_render = True  # 🔧 默认启用渲染
        if hasattr(args, 'render') and args.render:
            should_render = True
        elif hasattr(args, 'no_render') and args.no_render:
            should_render = False
        else:
            should_render = True  # 🔧 默认启用渲染
            print("🎨 渲染设置: 默认启用")

        env_params = {
            'num_links': num_links,
            'link_lengths': link_lengths,
            'render_mode': 'human' if should_render and args.num_processes == 1 else None,
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
        
        # 🔧 不使用sync_env，让训练环境直接渲染
        sync_env = None
        if should_render:
            print(f"✅ 训练环境已创建（进程数: {args.num_processes}，直接渲染）")
            # 确保第一个环境有渲染模式
            if hasattr(envs, 'envs') and len(envs.envs) > 0:
                if not hasattr(envs.envs[0], 'render_mode') or envs.envs[0].render_mode != 'human':
                    envs.envs[0].render_mode = 'human'
                    print(f"🔧 设置第一个训练环境为渲染模式")
                # 强制初始化渲染
                if hasattr(envs.envs[0], '_init_rendering'):
                    envs.envs[0]._init_rendering()
                    print(f"🎨 初始化训练环境渲染")
                # 强制第一次渲染以显示窗口
                try:
                    envs.envs[0].render()
                    print(f"🖼️ 强制第一次渲染，显示pygame窗口")
                except Exception as e:
                    print(f"⚠️ 渲染初始化错误: {e}")
        else:
            print(f"✅ 训练环境已创建（进程数: {args.num_processes}，无渲染）")
            
        return envs, sync_env, env_params

# === 训练管理器 ===
class TrainingManager:
    """训练过程管理器"""
    
    def __init__(self, args, sac, logger, model_manager):
        self.args = args
        self.sac = sac
        self.logger = logger
        self.model_manager = model_manager
        
        # 训练状态
        self.best_success_rate = 0.0
        self.best_min_distance = float('inf')
        self.consecutive_success_count = 0
        self.min_consecutive_successes = 2

                # 🆕 Episodes控制 - 2个episodes × 120k步
        self.current_episodes = 0
        self.max_episodes = 2
        self.steps_per_episode = 120000
        self.current_episode_steps = 0
        self.total_training_steps = 0
        self.episode_results = []
        self.current_episode_start_step = 0
        self.current_episode_start_time = time.time()
        
        # 🆕 追踪每个episode的最佳表现
        self.current_episode_best_distance = float('inf')
        self.current_episode_best_reward = float('-inf')
        self.current_episode_min_distance_step = 0
        
        print(f"🎯 训练配置: 2个episodes × 120,000步/episode = 总计240,000步")

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
                    self.current_episode_min_distance_step = episode_step  # 🔧 直接使用episode_step
            
    # def update_episode_tracking(self, step, infos, episode_rewards):
    #     """在每个训练步骤中更新episode追踪"""
    #     for proc_id in range(len(infos)):
    #         if len(infos) > proc_id and isinstance(infos[proc_id], dict):
    #             info = infos[proc_id]
                
    #             # 提取当前距离
    #             current_distance = float('inf')
    #             if 'goal' in info:
    #                 current_distance = info['goal'].get('distance_to_goal', float('inf'))
    #             elif 'distance' in info:
    #                 current_distance = info['distance']
                
    #             # 更新最佳距离
    #             if current_distance < self.current_episode_best_distance:
    #                 self.current_episode_best_distance = current_distance
    #                 self.current_episode_best_reward = episode_rewards[proc_id]
    #                 self.current_episode_min_distance_step = step - self.current_episode_start_step

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
        # 完成2个episodes就停止
        if self.current_episodes >= 2:
            print(f"🏁 完成{self.current_episodes}个episodes，训练结束")
            return True
        
        # 检查当前episode步数限制
        episode_steps = step - self.current_episode_start_step
        if episode_steps >= 120000:
            print(f"⏰ 当前episode达到120,000步限制")
            return False  # 不是整体结束，只是当前episode结束
        
        return False

    def _generate_final_fitness_report(self):
        """生成最终fitness报告"""
        if len(self.episode_results) == 0:
            print("⚠️ 没有episode结果数据")
            return
        
        print("\n" + "="*50)
        print("🎯 最终训练结果报告")
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

    # def handle_episode_end(self, proc_id, step, episode_rewards, infos):
    #     """处理episode结束逻辑 - 使用最佳距离版本"""
    #     if len(infos) <= proc_id or not isinstance(infos[proc_id], dict):
    #         episode_rewards[proc_id] = 0.0
    #         return False
        
    #     info = infos[proc_id]
        
    #     # 🎯 计算episode详细信息
    #     episode_steps = step - self.current_episode_start_step
    #     episode_duration = time.time() - self.current_episode_start_time if hasattr(self, 'current_episode_start_time') else 0
    #     episode_reward = episode_rewards[proc_id]
        
    #     # 🎯 使用最佳距离而不是结束时距离
    #     best_distance = self.current_episode_best_distance
    #     best_reward = self.current_episode_best_reward
    #     goal_reached = best_distance < 20.0
        
    #     # 获取结束时的距离用于对比
    #     end_distance = float('inf')
    #     if 'goal' in info:
    #         end_distance = info['goal'].get('distance_to_goal', float('inf'))
    #     elif 'distance' in info:
    #         end_distance = info['distance']
        
    #     # 🎯 分类episode结果（基于最佳距离）
    #     episode_type = self._classify_episode_result(goal_reached, best_distance, episode_steps, best_reward)
        
    #     # 存储episode结果
    #     episode_result = {
    #         'episode_num': self.current_episodes + 1,
    #         'type': episode_type['type'],
    #         'success': goal_reached,
    #         'best_distance': best_distance,      # 🎯 最佳距离
    #         'end_distance': end_distance,        # 🎯 结束距离
    #         'best_distance_step': self.current_episode_min_distance_step,  # 🎯 达到最佳距离的步数
    #         'steps': episode_steps,
    #         'duration': episode_duration,
    #         'reward': episode_reward,
    #         'best_reward': best_reward,          # 🎯 达到最佳距离时的奖励
    #         'score': episode_type['score'],
    #         'description': episode_type['description']
    #     }
        
    #     self.episode_results.append(episode_result)
    #     self.current_episodes += 1
        
    #     # 🎯 打印episode结果（显示最佳距离）
    #     print(f"📊 Episode {self.current_episodes}/2 完成:")
    #     print(f"   类型: {episode_type['type']} ({episode_type['description']})")
    #     print(f"   成功: {'✅' if goal_reached else '❌'}")
    #     print(f"   最佳距离: {best_distance:.1f}px (步数: {self.current_episode_min_distance_step})")
    #     print(f"   结束距离: {end_distance:.1f}px")
    #     print(f"   总步数: {episode_steps}")
    #     print(f"   最终奖励: {episode_reward:.2f}")
    #     print(f"   得分: {episode_type['score']:.2f}")
        
    #     # 🎯 重置episode追踪
    #     self.current_episode_best_distance = float('inf')
    #     self.current_episode_best_reward = float('-inf')
    #     self.current_episode_min_distance_step = 0
    #     self.current_episode_start_step = step
    #     self.current_episode_start_time = time.time()
        
    #     # 检查停止条件
    #     should_stop = self._check_episode_stopping_conditions(step)
    #     if should_stop:
    #         self._generate_final_fitness_report()
        
    #     episode_rewards[proc_id] = 0.0
    #     return should_stop
   # 在第547行左右，修改handle_episode_end方法：

    def handle_episode_end(self, proc_id, step, episode_rewards, infos):
        """处理episode结束逻辑 - 支持维持检查"""
        if len(infos) <= proc_id or not isinstance(infos[proc_id], dict):
            episode_rewards[proc_id] = 0.0
            return False
        
        info = infos[proc_id]
        
        # 🎯 计算episode详细信息
        episode_steps = step - self.current_episode_start_step
        episode_duration = time.time() - self.current_episode_start_time if hasattr(self, 'current_episode_start_time') else 0
        episode_reward = episode_rewards[proc_id]
        
        # 🎯 使用最佳距离而不是结束时距离
        best_distance = self.current_episode_best_distance
        best_reward = self.current_episode_best_reward
        goal_reached = best_distance < 20.0
        
        # 🆕 检查维持完成情况
        maintain_completed = False
        maintain_counter = 0
        maintain_target = 500
        
        # 尝试从环境获取维持信息
        try:
            # 这里需要访问实际的环境实例来获取维持信息
            # 由于我们在训练循环中已经检查了，这里主要是记录
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
        
        # 🎯 分类episode结果（基于维持完成情况）
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
            'success': maintain_completed or goal_reached,  # 🆕 维持完成或到达目标都算成功
            'maintain_completed': maintain_completed,  # 🆕 维持完成标志
            'maintain_counter': maintain_counter,  # 🆕 维持步数
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
        
        # 🎯 打印episode结果（显示维持信息）
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
        
        # 🎯 重置episode追踪
        self.current_episode_best_distance = float('inf')
        self.current_episode_best_reward = float('-inf')
        self.current_episode_min_distance_step = 0
        self.current_episode_start_step = step
        self.current_episode_start_time = time.time()
        
        # 检查停止条件
        should_stop = self._check_episode_stopping_conditions(step)
        if should_stop:
            self._generate_final_fitness_report()
        
        episode_rewards[proc_id] = 0.0
        return should_stop
    
    def should_update_model(self, step):
        """判断是否应该更新模型"""
        return (step >= self.sac.warmup_steps and 
                step % self.args.update_frequency == 0 and 
                self.sac.memory.can_sample(self.sac.batch_size))
    
    def update_and_log(self, step, total_steps):
        """更新模型并记录"""
        metrics = self.sac.update()
        
        if metrics:
            enhanced_metrics = metrics.copy()
            enhanced_metrics.update({
                'step': step,
                'buffer_size': len(self.sac.memory),
                'learning_rate': self.sac.actor_optimizer.param_groups[0]['lr'],
                'warmup_progress': min(1.0, step / max(self.sac.warmup_steps, 1))
            })
            
            self.logger.log_step(step, enhanced_metrics, episode=step//100)
            
            if step % 100 == 0:
                # 原始格式输出（保持兼容性）
                print(f"Step {step} (total_steps {total_steps}): "
                      f"Learning Rate: {metrics['lr']:.6f}, "
                      f"Critic Loss: {metrics['critic_loss']:.4f}, "
                      f"Actor Loss: {metrics['actor_loss']:.4f}, "
                      f"Alpha: {metrics['alpha']:.4f}, "
                      f"Buffer Size: {len(self.sac.memory)}")
                
                # 🆕 更新熵权重调度 (每100步调用一次)
                if step % 100 == 0:
                    self.sac.update_alpha_schedule(step, total_steps)
                    # 同步更新metrics中的alpha值
                    metrics['alpha'] = self.sac.alpha
                
                # 🆕 添加标准化的损失输出格式（供损失提取器捕获）
                print(f"🔥 SAC网络Loss更新 [Step {step}]:")
                print(f"📊 Actor Loss: {metrics['actor_loss']:.6f}")
                print(f"📊 Critic Loss: {metrics['critic_loss']:.6f}")
                print(f"📊 Alpha Loss: {metrics.get('alpha_loss', 0.0):.6f}")
                print(f"📊 Alpha: {metrics['alpha']:.6f} (调度后)")
                print(f"📊 Q1均值: {metrics.get('q1_mean', 0.0):.6f}")
                print(f"📊 Q2均值: {metrics.get('q2_mean', 0.0):.6f}")
                print(f"📊 Q1标准差: {metrics.get('q1_std', 0.0):.6f}")
                print(f"📊 Q2标准差: {metrics.get('q2_std', 0.0):.6f}")
                print(f"📊 熵项: {metrics.get('entropy_term', 0.0):.6f}")
                print(f"📊 Q值项: {metrics.get('q_term', 0.0):.6f}")
                print(f"📈 学习率: {metrics['lr']:.2e}")
                print(f"💾 Buffer大小: {len(self.sac.memory)}")
                
                # 🆕 添加Attention网络损失信息
                if any(key.startswith('attention_') for key in metrics.keys()):
                    print(f"\n🔥 Attention网络Loss更新 [Step {step}]:")
                    # 🆕 独立显示三个attention网络的信息
                    if 'attention_actor_loss' in metrics:
                        print(f"📊 Actor Attention Loss: {metrics['attention_actor_loss']:.6f}")
                    if 'attention_critic_main_loss' in metrics:
                        print(f"📊 Critic Main Attention Loss: {metrics['attention_critic_main_loss']:.6f}")
                    if 'attention_critic_value_loss' in metrics:
                        print(f"📊 Critic Value Attention Loss: {metrics['attention_critic_value_loss']:.6f}")
                    if 'attention_total_loss' in metrics:
                        print(f"📊 Attention总损失: {metrics['attention_total_loss']:.6f}")
                    
                    # 🆕 显示梯度范数（更详细的信息）
                    if 'attention_actor_grad_norm' in metrics:
                        print(f"🔍 Actor Attention梯度范数: {metrics['attention_actor_grad_norm']:.6f}")
                    if 'attention_critic_main_grad_norm' in metrics:
                        print(f"🔍 Critic Main Attention梯度范数: {metrics['attention_critic_main_grad_norm']:.6f}")
                    if 'attention_critic_value_grad_norm' in metrics:
                        print(f"🔍 Critic Value Attention梯度范数: {metrics['attention_critic_value_grad_norm']:.6f}")
                    
                    # 🆕 分别显示Actor和Critic参数统计
                    if 'attention_actor_param_mean' in metrics:
                        print(f"📊 Actor Attention参数: 均值={metrics['attention_actor_param_mean']:.6f}, 标准差={metrics.get('attention_actor_param_std', 0):.6f}")
                    if 'attention_critic_param_mean' in metrics:
                        print(f"📊 Critic Attention参数: 均值={metrics['attention_critic_param_mean']:.6f}, 标准差={metrics.get('attention_critic_param_std', 0):.6f}")
                    
                    # 🆕 显示attention网络的总体参数统计
                    if 'attention_param_mean' in metrics:
                        print(f"📊 Attention参数均值: {metrics['attention_param_mean']:.6f}")
                    if 'attention_param_std' in metrics:
                        print(f"📊 Attention参数标准差: {metrics['attention_param_std']:.6f}")
                    
                    # 🆕 显示关节关注度分析
                    if 'most_important_joint' in metrics:
                        print(f"🎯 最重要关节: Joint {metrics['most_important_joint']}")
                    if 'max_joint_importance' in metrics:
                        print(f"🎯 最重要关节: Joint {metrics.get('most_important_joint', 'N/A')} (重要性: {metrics['max_joint_importance']:.6f})")
                    if 'importance_concentration' in metrics:
                        print(f"📊 重要性集中度: {metrics['importance_concentration']:.6f}")
                    if 'importance_entropy' in metrics:
                        print(f"📊 重要性熵值: {metrics['importance_entropy']:.6f}")
                    if 'robot_num_joints' in metrics:
                        print(f"🤖 机器人结构: {metrics['robot_num_joints']}关节")
                    if 'robot_structure_info' in metrics:
                        print(f"🤖 机器人结构: {metrics['robot_num_joints']}关节 ({metrics['robot_structure_info']})")
                    
                    # 🆕 显示关节活跃度和重要性（只显示存在的关节）
                    if 'robot_num_joints' in metrics:
                        num_joints = metrics['robot_num_joints']
                        joint_activities = []
                        joint_importances = []
                        joint_angles = []
                        joint_velocities = []
                        link_lengths = []
                        
                        for i in range(min(num_joints, 20)):
                            activity = metrics.get(f'joint_{i}_activity', -1)
                            importance = metrics.get(f'joint_{i}_importance', -1)
                            angle_mag = metrics.get(f'joint_{i}_angle_magnitude', -1)
                            vel_mag = metrics.get(f'joint_{i}_velocity_magnitude', -1)
                            link_len = metrics.get(f'link_{i}_length', -1)
                            
                            if activity != -1:
                                joint_activities.append(f"J{i}:{activity:.3f}")
                            if importance != -1:
                                joint_importances.append(f"J{i}:{importance:.3f}")
                            if angle_mag != -1:
                                joint_angles.append(f"J{i}:{angle_mag:.3f}")
                            if vel_mag != -1:
                                joint_velocities.append(f"J{i}:{vel_mag:.3f}")
                            if link_len != -1:
                                link_lengths.append(f"L{i}:{link_len:.1f}px")
                        
                        if joint_activities:
                            print(f"🔍 关节活跃度: {', '.join(joint_activities)}")
                        if joint_importances:
                            print(f"🎯 关节重要性: {', '.join(joint_importances)}")
                        if joint_angles:
                            print(f"📐 关节角度幅度: {', '.join(joint_angles)}")
                        if joint_velocities:
                            print(f"⚡ 关节速度幅度: {', '.join(joint_velocities)}")
                        if link_lengths:
                            print(f"📏 Link长度: {', '.join(link_lengths)}")
                
                # 🆕 添加标准化的成功率报告（每500步输出一次）
                if step % 500 == 0 and len(self.episode_results) > 0:
                    success_count = sum(1 for ep in self.episode_results if ep.get('success', False))
                    current_success_rate = (success_count / len(self.episode_results)) * 100
                    
                    print(f"📊 SAC训练进度报告 [Step {step}]:")
                    print(f"✅ 当前成功率: {current_success_rate:.1f}%")
                    print(f"🏆 当前最佳距离: {self.best_min_distance:.1f}px")
                    print(f"📊 当前Episode最佳距离: {self.current_episode_best_distance:.1f}px")
                    print(f"🔄 连续成功次数: {self.consecutive_success_count}")
                    print(f"📋 已完成Episodes: {len(self.episode_results)}")



def main(args):
    """主训练函数"""
    print("🚀 开始训练...")
    
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
        reacher2d_encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
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

    # 🔧 创建优化的SAC模型 - 专门为Critic稳定性优化
    print("🔧 Critic稳定性优化配置:")
    optimized_batch_size = max(args.batch_size, 256)  # 增加到256，提高稳定性
    print(f"   批次大小: {args.batch_size} → {optimized_batch_size}")
    print(f"   学习率策略: Actor={args.lr:.2e}, Critic={args.lr*1.5:.2e} (1.5倍，更稳定)")
    print(f"   Tau参数: 将自动调整到至少0.01")
    
    attn_model = AttnModel(128, 130, 130, 4)
    sac = AttentionSACWithBuffer(
        attn_model, num_joints, 
        buffer_capacity=args.buffer_capacity, 
        batch_size=optimized_batch_size,  # 使用优化的批次大小
        lr=args.lr, 
        tau=args.tau,  # tau会在SAC内部自动调整
        gamma=args.gamma,
        alpha=args.alpha,
        env_type=args.env_type
    )
    
    # 添加SAC特定参数
    sac.warmup_steps = args.warmup_steps
    sac.alpha = torch.tensor(args.alpha)
    sac.min_alpha = 0.05
    print(f"🔒 Alpha衰减下限设置为: {sac.min_alpha}")
    
    # 🆕 设置individual_id到SAC模型
    if hasattr(args, 'individual_id') and args.individual_id:
        sac.individual_id = args.individual_id
        print(f"🆔 设置Individual ID: {args.individual_id}")
        
        # 🆕 直接设置环境属性
        generation = getattr(args, 'generation', 0)
        
        # 设置同步环境
        if sync_env:
            sync_env.current_generation = generation
            sync_env.individual_id = args.individual_id
            print(f"🆔 设置同步环境上下文: 个体={args.individual_id}, 代数={generation}")
        
        # 设置向量环境
        if hasattr(envs, 'envs'):
            for i, env_wrapper in enumerate(envs.envs):
                # 🔧 递归设置所有层级的环境属性
                current_env = env_wrapper
                while hasattr(current_env, 'env'):
                    if hasattr(current_env, 'current_generation'):
                        current_env.current_generation = generation
                        current_env.individual_id = args.individual_id
                    current_env = current_env.env
                
                # 最终的环境对象
                if hasattr(current_env, '__class__'):
                    current_env.current_generation = generation
                    current_env.individual_id = args.individual_id
                    print(f"🆔 设置环境{i}上下文: 个体={args.individual_id}, 代数={generation} (类型: {current_env.__class__.__name__})")

    if hasattr(sac, 'target_entropy'):
        sac.target_entropy = -num_joints * args.target_entropy_factor
    
    # 创建训练监控系统
    experiment_name = f"reacher2d_sac_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # 配置信息
    hyperparams = {
        'learning_rate': args.lr,
        'alpha': args.alpha,
        'warmup_steps': args.warmup_steps,
        'target_entropy_factor': args.target_entropy_factor,
        'batch_size': args.batch_size,
        'buffer_capacity': args.buffer_capacity,
        'gamma': args.gamma,
        'seed': args.seed,
        'num_processes': args.num_processes,
        'update_frequency': args.update_frequency,
        'total_steps': 120000,
        'optimizer': 'Adam',
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
    
    # monitor = RealTimeMonitor(logger, alert_thresholds={
    #     'critic_loss': {'max': 50.0, 'nan_check': True},
    #     'actor_loss': {'max': 10.0, 'nan_check': True},
    #     'alpha_loss': {'max': 5.0, 'nan_check': True},
    #     'alpha': {'min': 0.01, 'max': 2.0, 'nan_check': True}
    # })
    
    print(f"📊 训练监控系统已初始化: {logger.experiment_dir}")
    
    # 创建管理器
    model_manager = ModelManager(args.save_dir)
    training_manager = TrainingManager(args, sac, logger, model_manager)
    
    # 处理checkpoint恢复
    start_step = 0
    if args.resume_checkpoint:
        print(f"🔄 从检查点恢复训练: {args.resume_checkpoint}")
        start_step = model_manager.load_checkpoint(sac, args.resume_checkpoint)

        if start_step > 0:
            print(f"成功加载checkpoint, 从step {start_step} 开始训练")
            sac.warmup_steps = 0

            # 更新学习率和alpha
            if args.resume_lr:
                for param_group in sac.actor_optimizer.param_groups:
                    param_group['lr'] = args.resume_lr
                for param_group in sac.critic_optimizer.param_groups:
                    param_group['lr'] = args.resume_lr
                print(f"更新学习率为 {args.resume_lr}")

            if args.resume_alpha:
                sac.alpha = args.resume_alpha
                print(f"更新alpha为 {args.resume_alpha}")
            
    # 运行训练循环
    run_training_loop(args, envs, sync_env, sac, single_gnn_embed, training_manager, num_joints, start_step)
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
        first_score = episodes[0].get('episode_score', 0)
        last_score = episodes[-1].get('episode_score', 0)
        learning_progress = last_score - first_score
    else:
        learning_progress = 0.0
    
    # 计算平均到达最佳距离的步数
    steps_to_best = [ep.get('steps_to_best', 120000) for ep in episodes]
    avg_steps_to_best = np.mean(steps_to_best)
    
    # 计算总训练时间
    durations = [ep.get('duration', 0) for ep in episodes]
    total_training_time = sum(durations)
    
    result = {
        'success': True,
        'episodes_completed': total_episodes,
        'success_rate': success_count / total_episodes if total_episodes > 0 else 0.0,
        'avg_best_distance': avg_best_distance,
        'avg_score': np.mean([ep.get('episode_score', 0) for ep in episodes]),
        'total_training_time': total_training_time,
        'episode_details': episodes,
        'learning_progress': learning_progress,
        'avg_steps_to_best': avg_steps_to_best,
        'episode_results': episodes
    }
    
    print(f"✅ 训练结果收集完成:")
    print(f"   Episodes: {total_episodes}")
    print(f"   成功率: {result['success_rate']:.1%}")
    print(f"   平均最佳距离: {result['avg_best_distance']:.1f}px")
    print(f"   学习进步: {result['learning_progress']:+.3f}")
    
    return result
def run_training_loop(args, envs, sync_env, sac, single_gnn_embed, training_manager, num_joints, start_step=0):
    """运行训练循环 - Episodes版本"""
    current_obs = envs.reset()
    print(f"初始观察: {current_obs.shape}")
    
    # 🔧 不使用sync_env，训练环境直接渲染
    print("🔧 训练环境已重置（直接渲染模式）")
    
    current_gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)
    episode_rewards = [0.0] * args.num_processes
    
    # 🆕 Episodes控制参数
    max_episodes = 2  # 🔧 修改为2个episodes
    steps_per_episode = 120000
    
    print(f"开始训练: warmup {sac.warmup_steps} 步")
    print(f"训练配置: {max_episodes}个episodes × {steps_per_episode}步/episode")
    
    training_completed = False
    early_termination_reason = ""
    global_step = start_step  # 全局步数计数器（用于模型更新等）

    try:
        # 🆕 Episodes循环
        for episode_num in range(max_episodes):
            print(f"\n🎯 开始Episode {episode_num + 1}/{max_episodes}")
            
            print(f"🔄 重置环境开始Episode {episode_num + 1}...")
            current_obs = envs.reset()
            if sync_env:
                sync_env.reset()
                print("🔧 sync_env 已重置")
            current_gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)
            episode_rewards = [0.0] * args.num_processes  # 重置episode奖励
            # 重置episode追踪
            training_manager.current_episode_start_step = global_step
            training_manager.current_episode_start_time = time.time()
            training_manager.current_episode_best_distance = float('inf')
            training_manager.current_episode_best_reward = float('-inf')
            training_manager.current_episode_min_distance_step = 0
            
            episode_step = 0  # 🎯 每个episode内的步数计数
            episode_completed = False
            
            # 🆕 单个Episode的训练循环
            while episode_step < steps_per_episode and not episode_completed:
                # 进度显示
                if episode_step % 100 == 0:
                    if global_step < sac.warmup_steps:
                        smart_print(f"Episode {episode_num+1}, Step {episode_step}/{steps_per_episode}: Warmup phase ({global_step}/{sac.warmup_steps})")
                    else:
                        smart_print(f"Episode {episode_num+1}, Step {episode_step}/{steps_per_episode}: Training phase, Buffer size: {len(sac.memory)}")

                # 获取动作
                if global_step < sac.warmup_steps:
                    action_batch = torch.from_numpy(np.array([
                        envs.action_space.sample() for _ in range(args.num_processes)
                    ]))
                else:
                    actions = []
                    for proc_id in range(args.num_processes):
                        # 🆕 计算距离以启用距离自适应控制
                        current_obs_np = current_obs[proc_id].cpu().numpy()
                        # 从观察中提取末端位置和目标位置
                        if len(current_obs_np) >= 8:  # reacher2d观察格式
                            end_pos = current_obs_np[-5:-3]  # 末端位置
                            goal_pos = current_obs_np[-3:-1]  # 目标位置
                            distance_to_goal = np.linalg.norm(end_pos - goal_pos)
                        else:
                            distance_to_goal = None
                        
                        action = sac.get_action(
                            current_obs[proc_id],
                            current_gnn_embeds[proc_id],
                            num_joints=envs.action_space.shape[0],
                            deterministic=False,
                            distance_to_goal=distance_to_goal  # 🆕 传递距离信息
                        )
                        actions.append(action)
                    action_batch = torch.stack(actions)

                # 动作分析（调试用）
                if episode_step % 50 == 0 or episode_step < 20:
                    if hasattr(envs, 'envs') and len(envs.envs) > 0:
                        env_goal = getattr(envs.envs[0], 'goal_pos', 'NOT FOUND')
                        # 🔧 显示环境内部的真实episode计数
                        env_episode = getattr(envs.envs[0], 'current_episode', episode_num+1)
                        print(f"🎯 [Episode {env_episode}] Step {episode_step} - 环境goal_pos: {env_goal}")

                # 执行动作
                next_obs, reward, done, infos = envs.step(action_batch)

                # 🔧 直接使用训练环境进行渲染（不使用sync_env）
                if hasattr(envs, 'envs') and len(envs.envs) > 0 and hasattr(envs.envs[0], 'render_mode') and envs.envs[0].render_mode == 'human':
                    try:
                        envs.envs[0].render()
                        # 每100步显示一次渲染状态
                        if episode_step % 100 == 0:
                            print(f"🖼️ [Step {episode_step}] 训练环境渲染更新")
                    except Exception as e:
                        if episode_step % 500 == 0:  # 减少错误消息频率
                            print(f"⚠️ 渲染错误: {e}")

                next_gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)

                # 存储经验
                for proc_id in range(args.num_processes):
                    sac.store_experience(
                        obs=current_obs[proc_id],
                        gnn_embeds=current_gnn_embeds[proc_id],
                        action=action_batch[proc_id],
                        reward=reward[proc_id],
                        next_obs=next_obs[proc_id],
                        next_gnn_embeds=next_gnn_embeds[proc_id],
                        done=done[proc_id],
                        num_joints=num_joints
                    )
                    episode_rewards[proc_id] += reward[proc_id].item()

                current_obs = next_obs.clone()
                current_gnn_embeds = next_gnn_embeds.clone()

                # 🆕 更新episode追踪
                training_manager.update_episode_tracking(global_step, infos, episode_rewards)

                # 处理episode结束
                # 替换第1055-1083行的代码：

                # 🔧 删除重复的done检测循环（移动到下面统一处理）

                # 🔧 只在第一次done时处理episode结束，避免重复触发
                episode_end_handled = False
                for proc_id in range(args.num_processes):
                        is_done = done[proc_id].item() if torch.is_tensor(done[proc_id]) else bool(done[proc_id])
                        if is_done and not episode_end_handled:
                            print(f"🔍 [DEBUG] Episode结束检测: proc_id={proc_id}, 当前episodes={training_manager.current_episodes}")
                            
                            should_end = training_manager.handle_episode_end(proc_id, episode_step, episode_rewards, infos)
                            print(f"🔍 [DEBUG] handle_episode_end返回: should_end={should_end}, 新的current_episodes={training_manager.current_episodes}")
                            episode_end_handled = True  # 标记已处理，避免重复
                            
                            # 🔧 检查是否到达目标
                            goal_reached = infos[proc_id].get('goal', {}).get('distance_to_goal', float('inf')) < 20.0
                            print(f"🔍 [DEBUG] 目标检查: goal_reached={goal_reached}")
                            
                            if should_end:  # 完成2个训练episodes
                                print(f"🔍 [DEBUG] 触发should_end，整个训练结束")
                                training_completed = True
                                early_termination_reason = f"完成{training_manager.current_episodes}个episodes"
                                episode_completed = True
                                break
                            elif goal_reached:  # 到达目标，结束当前训练episode
                                print(f"🔍 [DEBUG] 触发goal_reached，当前训练episode结束")
                                print(f"🎉 训练Episode {episode_num+1} 成功完成！开始下一个episode...")
                                episode_completed = True  # 结束当前训练episode，但不结束整个训练
                                break
                            
                            # 环境重置（继续当前训练episode）
                            if hasattr(envs, 'reset_one'):
                                current_obs[proc_id] = envs.reset_one(proc_id)
                                current_gnn_embeds[proc_id] = single_gnn_embed
                
                # 模型更新
                if training_manager.should_update_model(global_step):
                    training_manager.update_and_log(global_step, global_step)
                
                # 🆕 定期输出goal到达统计 (每500步)
                if global_step % 500 == 0 and global_step > 0:
                    # 获取所有环境的goal到达统计
                    total_goal_reaches = 0
                    total_steps = 0
                    for proc_id in range(args.num_processes):
                        if hasattr(envs, 'envs') and len(envs.envs) > proc_id:
                            env = envs.envs[proc_id]
                            if hasattr(env, 'get_goal_reach_stats'):
                                stats = env.get_goal_reach_stats()
                                total_goal_reaches += stats['goal_reach_count']
                                total_steps += stats['total_steps']
                    
                    if total_steps > 0:
                        goal_reach_percentage = (total_goal_reaches / total_steps) * 100
                        print(f"📊 Goal到达统计 [Step {global_step}]:")
                        print(f"   🎯 到达次数: {total_goal_reaches}")
                        print(f"   📈 总步数: {total_steps}")
                        print(f"   ✅ 到达率: {goal_reach_percentage:.2f}%")
                
                # 定期保存和绘图
                if global_step % 200 == 0 and global_step > 0:  # 🆕 改为200步检测
                    # 🆕 获取当前最佳距离
                    current_best_distance = training_manager.current_episode_best_distance
                    
                    # 🆕 传递当前距离用于比较
                    saved = training_manager.model_manager.save_checkpoint(
                        sac, global_step,
                        best_success_rate=training_manager.best_success_rate,
                        best_min_distance=training_manager.best_min_distance,
                        current_best_distance=current_best_distance,  # 🆕 关键参数
                        consecutive_success_count=training_manager.consecutive_success_count,
                        current_episode=episode_num + 1,
                        episode_step=episode_step
                    )
                    
                    # 🆕 如果保存成功，更新最佳记录
                    if saved:
                        training_manager.best_min_distance = current_best_distance
                        print(f"📈 更新全局最佳距离: {current_best_distance:.1f}px")

                # 🆕 低频日志记录
                if global_step % 2000 == 0 and global_step > 0:
                    training_manager.logger.plot_losses(recent_steps=2000, show=False)
                    print(f"📊 Step {global_step}: 当前最佳距离 {training_manager.best_min_distance:.1f}px")
                
                episode_step += 1  # 🎯 episode内步数递增
                global_step += args.num_processes  # 全局步数递增
                
                if training_completed:
                    break
            
            print(f"📊 Episode {episode_num + 1} 完成: {episode_step} 步")
            
            # 🆕 输出当前episode的goal到达统计
            if hasattr(envs, 'envs') and len(envs.envs) > 0:
                env = envs.envs[0]  # 获取第一个环境的统计
                if hasattr(env, 'get_goal_reach_stats'):
                    stats = env.get_goal_reach_stats()
                    print(f"   🎯 Goal到达统计: {stats['goal_reach_count']}次")
                    print(f"   📈 到达率: {stats['goal_reach_percentage']:.2f}%")
                    if stats['max_maintain_streak'] > 0:
                        print(f"   🏆 最长维持: {stats['max_maintain_streak']}步")
            
            if training_completed:
                print(f"🏁 训练提前终止: {early_termination_reason}")
                break

    except Exception as e:
        print(f"🔴 训练过程中发生错误: {e}")
        training_manager.logger.save_logs()
        training_manager.logger.generate_report()
        raise e
# def run_training_loop(args, envs, sync_env, sac, single_gnn_embed, training_manager, num_joints, start_step=0):
#     """运行训练循环"""
#     current_obs = envs.reset()
#     print(f"初始观察: {current_obs.shape}")
    
#     # 重置渲染环境
#     if sync_env:
#         sync_env.reset()
#         print("🔧 sync_env 已重置")
    
#     current_gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)
#     episode_rewards = [0.0] * args.num_processes
#     num_step = 120000
#     total_steps = 0
    
#     print(f"开始训练: warmup {sac.warmup_steps} 步")
#     print(f"总训练步数: {num_step}, 更新频率: {args.update_frequency}")
#     if start_step > 0:
#         print(f"从步骤 {start_step} 恢复训练")
#     else:
#         print(f"预期warmup完成步骤: {sac.warmup_steps}")

#     training_completed = False
#     early_termination_reason = ""

#     try:
#         for step in range(start_step, num_step):
#             # 进度显示
#             if step % 100 == 0:
#                 if step < sac.warmup_steps:
#                     smart_print(f"Step {step}/{num_step}: Warmup phase ({step}/{sac.warmup_steps})")
#                 else:
#                     smart_print(f"Step {step}/{num_step}: Training phase, Buffer size: {len(sac.memory)}")

#             # 获取动作
#             if step < sac.warmup_steps:
#                 action_batch = torch.from_numpy(np.array([
#                     envs.action_space.sample() for _ in range(args.num_processes)
#                 ]))
#             else:
#                 actions = []
#                 for proc_id in range(args.num_processes):
#                     action = sac.get_action(
#                         current_obs[proc_id],
#                                             current_gnn_embeds[proc_id],
#                                             num_joints=envs.action_space.shape[0],
#                         deterministic=False
#                     )
#                     actions.append(action)
#                 action_batch = torch.stack(actions)

#             # 动作分析（调试用）
#             if step % 50 == 0 or step < 20:
#                 if hasattr(envs, 'envs') and len(envs.envs) > 0:
#                     env_goal = getattr(envs.envs[0], 'goal_pos', 'NOT FOUND')
#                     print(f"🎯 [训练] Step {step} - 环境goal_pos: {env_goal}")
                
#                 smart_print(f"\n🎯 Step {step} Action Analysis:")
#                 action_numpy = action_batch.cpu().numpy() if hasattr(action_batch, 'cpu') else action_batch.numpy()
                
#                 for proc_id in range(min(args.num_processes, 2)):
#                     action_values = action_numpy[proc_id]
#                     action_str = ', '.join([f"{val:+6.2f}" for val in action_values])
#                     smart_print(f"  Process {proc_id}: Actions = [{action_str}]")
#                     smart_print(f"    Max action: {np.max(np.abs(action_values)):6.2f}, Mean abs: {np.mean(np.abs(action_values)):6.2f}")

#             # 执行动作
#             next_obs, reward, done, infos = envs.step(action_batch)

#             # 渲染处理
#             if sync_env:
#                 sync_action = action_batch[0].cpu().numpy() if hasattr(action_batch, 'cpu') else action_batch[0]
#                 sync_env.step(sync_action)
#                 sync_env.render()

#             next_gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)

#             # 存储经验
#             for proc_id in range(args.num_processes):
#                 sac.store_experience(
#                         obs=current_obs[proc_id],
#                         gnn_embeds=current_gnn_embeds[proc_id],
#                         action=action_batch[proc_id],
#                         reward=reward[proc_id],
#                         next_obs=next_obs[proc_id],
#                         next_gnn_embeds=next_gnn_embeds[proc_id],
#                         done=done[proc_id],
#                         num_joints=num_joints
#                 )
#                 episode_rewards[proc_id] += reward[proc_id].item()

#             current_obs = next_obs.clone()
#             current_gnn_embeds = next_gnn_embeds.clone()

#             # 处理episode结束
#             for proc_id in range(args.num_processes):
#                 is_done = done[proc_id].item() if torch.is_tensor(done[proc_id]) else bool(done[proc_id])
#                 if is_done:
#                     should_end = training_manager.handle_episode_end(proc_id, step, episode_rewards, infos)
#                     if should_end:
#                         training_completed = True
#                         early_termination_reason = f"连续成功{training_manager.consecutive_success_count}次，达到训练目标"
#                         break
                    
#                     if hasattr(envs, 'reset_one'):
#                         current_obs[proc_id] = envs.reset_one(proc_id)
#                         current_gnn_embeds[proc_id] = single_gnn_embed
            
#             # 模型更新
#             if training_manager.should_update_model(step):
#                 training_manager.update_and_log(step, total_steps)
            
#             # 定期保存和绘图
#             if step % 1000 == 0 and step > 0:
#                 training_manager.logger.plot_losses(recent_steps=2000, show=False)
#                 training_manager.model_manager.save_checkpoint(
#                     sac, step,
#                     best_success_rate=training_manager.best_success_rate,
#                     best_min_distance=training_manager.best_min_distance,
#                     consecutive_success_count=training_manager.consecutive_success_count
#                 )
            
#             total_steps += args.num_processes
            
#             if training_completed:
#                 print(f"🏁 训练提前终止: {early_termination_reason}")
#                 break

#     except Exception as e:
#         print(f"🔴 训练过程中发生错误: {e}")
#         training_manager.logger.save_logs()
#         training_manager.logger.generate_report()
#         raise e

def cleanup_resources(sync_env, logger, model_manager, training_manager):
    """清理资源"""
    if sync_env:
            sync_env.close()
            
    # 生成最终报告
    print(f"\n{'='*60}")
    print(f"🏁 训练完成总结:")
    print(f"  最佳成功率: {training_manager.best_success_rate:.3f}")
    print(f"  最佳最小距离: {training_manager.best_min_distance:.1f} pixels")
    print(f"  当前连续成功次数: {training_manager.consecutive_success_count}")
    
    logger.generate_report()
    logger.plot_losses(show=False)
    print(f"📊 完整训练日志已保存到: {logger.experiment_dir}")
    print(f"{'='*60}")

# === 测试功能 ===
# def test_trained_model(model_path, num_episodes=10, render=True):
#     """测试训练好的模型性能"""
#     print(f"🧪 开始测试模型: {model_path}")
    
#     # 环境配置
#     env_params = {
#         'num_links': DEFAULT_CONFIG['num_links'],
#         'link_lengths': DEFAULT_CONFIG['link_lengths'],
#         'render_mode': 'human' if render else None,
#         'config_path': DEFAULT_CONFIG['config_path']
#     }
    
#     # 创建环境
#     env = Reacher2DEnv(**env_params)
#     num_joints = env.action_space.shape[0]
    
#     # 创建GNN编码器
#     sys.path.append(os.path.join(base_dir, 'examples/2d_reacher/utils'))
#     from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder
    
#     reacher2d_encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
#     gnn_embed = reacher2d_encoder.get_gnn_embeds(
#         num_links=num_joints, 
#         link_lengths=env_params['link_lengths']
#     )
    
#     # 创建SAC模型
#     attn_model = AttnModel(128, 130, 130, 4)
#     sac = AttentionSACWithBuffer(attn_model, num_joints, 
#                                 buffer_capacity=10000, batch_size=64,
#                                 lr=1e-5, env_type='reacher2d')
    
#     # 加载模型
#     try:
#         if not os.path.exists(model_path):
#             print(f"❌ 模型文件不存在: {model_path}")
#             return None
            
#         print(f"🔄 加载模型: {model_path}")
#         model_data = torch.load(model_path, map_location='cpu')
        
#         if 'actor_state_dict' in model_data:
#             sac.actor.load_state_dict(model_data['actor_state_dict'], strict=False)
#             print("✅ Actor 加载成功")
        
#         print(f"📋 模型信息:")
#         print(f"   训练步数: {model_data.get('step', 'N/A')}")
#         print(f"   时间戳: {model_data.get('timestamp', 'N/A')}")
#         if 'success_rate' in model_data:
#             print(f"   训练时成功率: {model_data.get('success_rate', 'N/A'):.3f}")
#         if 'min_distance' in model_data:
#             print(f"   训练时最小距离: {model_data.get('min_distance', 'N/A'):.1f}")
            
#     except Exception as e:
#         print(f"❌ 加载模型失败: {e}")
#         return None
    
#     # 测试多个episode
#     success_count = 0
#     total_rewards = []
#     min_distances = []
#     episode_lengths = []
    
#     print(f"\n🎮 开始测试 {num_episodes} 个episodes...")
#     print(f"🎯 目标阈值: {GOAL_THRESHOLD} pixels")
    
#     for episode in range(num_episodes):
#         obs = env.reset()
#         episode_reward = 0
#         step_count = 0
#         max_steps = 2500
#         min_distance_this_episode = float('inf')
#         episode_success = False
        
#         print(f"\n📍 Episode {episode + 1}/{num_episodes}")
        
#         while step_count < max_steps:
#             # 获取动作（使用确定性策略）
#             action = sac.get_action(
#                 torch.from_numpy(obs).float(),
#                 gnn_embed.squeeze(0),
#                 num_joints=num_joints,
#                 deterministic=True
#             )
            
#             # 执行动作
#             obs, reward, done, info = env.step(action.cpu().numpy())
#             episode_reward += reward
#             step_count += 1
            
#             # 检查距离
#             end_pos = env._get_end_effector_position()
#             goal_pos = env.goal_pos
#             distance = np.linalg.norm(np.array(end_pos) - goal_pos)
#             min_distance_this_episode = min(min_distance_this_episode, distance)
            
#             # 渲染
#             if render:
#                 env.render()
#                 time.sleep(0.02)
            
#             # 检查是否到达目标
#             if done:
#                 if not episode_success:
#                     success_count += 1
#                     episode_success = True
#                     print(f"  🎉 目标到达! 距离: {distance:.1f} pixels, 步骤: {step_count}")
#                 break
        
#         total_rewards.append(episode_reward)
#         min_distances.append(min_distance_this_episode)
#         episode_lengths.append(step_count)
        
#         print(f"  📊 Episode {episode + 1} 结果:")
#         print(f"    奖励: {episode_reward:.2f}")
#         print(f"    最小距离: {min_distance_this_episode:.1f} pixels")
#         print(f"    步骤数: {step_count}")
#         print(f"    成功: {'✅ 是' if episode_success else '❌ 否'}")
    
#     # 测试总结
#     success_rate = success_count / num_episodes
#     avg_reward = np.mean(total_rewards)
#     avg_min_distance = np.mean(min_distances)
#     avg_episode_length = np.mean(episode_lengths)
    
#     print(f"\n{'='*60}")
#     print(f"🏆 测试结果总结:")
#     print(f"  测试Episodes: {num_episodes}")
#     print(f"  成功次数: {success_count}")
#     print(f"  成功率: {success_rate:.1%}")
#     print(f"  平均奖励: {avg_reward:.2f}")
#     print(f"  平均最小距离: {avg_min_distance:.1f} pixels")
#     print(f"  平均Episode长度: {avg_episode_length:.1f} steps")
#     print(f"  目标阈值: {GOAL_THRESHOLD:.1f} pixels")
    
#     # 性能评价
#     print(f"\n📋 性能评价:")
#     if success_rate >= 0.8:
#         print(f"  🏆 优秀! 成功率 >= 80%")
#     elif success_rate >= 0.5:
#         print(f"  👍 良好! 成功率 >= 50%")
#     elif success_rate >= 0.2:
#         print(f"  ⚠️  一般! 成功率 >= 20%")
#     else:
#         print(f"  ❌ 需要改进! 成功率 < 20%")
        
#     if avg_min_distance <= GOAL_THRESHOLD:
#         print(f"  ✅ 平均最小距离达到目标阈值")
#     else:
#         print(f"  ⚠️  平均最小距离超出目标阈值 {avg_min_distance - GOAL_THRESHOLD:.1f} pixels")
    
#     print(f"{'='*60}")
    
#     env.close()
#     return {
#         'success_rate': success_rate,
#         'avg_reward': avg_reward, 
#         'avg_min_distance': avg_min_distance,
#         'avg_episode_length': avg_episode_length,
#         'success_count': success_count,
#         'total_episodes': num_episodes
#     }
def test_trained_model(model_path, num_episodes=10, render=True):
    """测试训练好的模型性能"""
    print(f"🧪 开始测试模型: {model_path}")
    
    # 环境配置
    env_params = {
        'num_links': DEFAULT_CONFIG['num_links'],
        'link_lengths': DEFAULT_CONFIG['link_lengths'],
        'render_mode': 'human' if render else None,
        'config_path': DEFAULT_CONFIG['config_path']
    }
    
    # 🔧 添加环境配置调试
    print(f"🔧 测试环境配置: {env_params}")
    
    # 创建环境
    env = Reacher2DEnv(**env_params)
    num_joints = env.action_space.shape[0]
    
    # 🔧 添加环境验证
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
    
    # 创建SAC模型
    attn_model = AttnModel(128, 130, 130, 4)
    sac = AttentionSACWithBuffer(attn_model, num_joints, 
                                buffer_capacity=10000, batch_size=64,
                                lr=1e-5, env_type='reacher2d')
    
    # 加载模型
    try:
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return None
            
        print(f"🔄 加载模型: {model_path}")
        model_data = torch.load(model_path, map_location='cpu')
        
        if 'actor_state_dict' in model_data:
            sac.actor.load_state_dict(model_data['actor_state_dict'], strict=False)
            print("✅ Actor 加载成功")
        
        # 🔧 修复模型验证代码
        print(f"🔍 模型验证:")
        print(f"   Actor参数数量: {sum(p.numel() for p in sac.actor.parameters())}")
        
        # 🔧 安全的权重检查
        try:
            first_param = next(iter(sac.actor.parameters()))
            if first_param.numel() > 5:  # 确保有足够的元素
                print(f"   Actor第一层权重示例: {first_param.flatten()[:5]}")
            else:
                print(f"   Actor第一层权重: {first_param}")
        except Exception as e:
            print(f"   权重检查跳过: {e}")
        
        print(f"📋 模型信息:")
        print(f"   训练步数: {model_data.get('step', 'N/A')}")
        print(f"   时间戳: {model_data.get('timestamp', 'N/A')}")
        if 'success_rate' in model_data:
            print(f"   训练时成功率: {model_data.get('success_rate', 'N/A'):.3f}")
        if 'min_distance' in model_data:
            print(f"   训练时最小距离: {model_data.get('min_distance', 'N/A'):.1f}")
            
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None
    
    # 测试多个episode
    success_count = 0
    total_rewards = []
    min_distances = []
    episode_lengths = []
    
    print(f"\n🎮 开始测试 {num_episodes} 个episodes...")
    print(f"🎯 目标阈值: {GOAL_THRESHOLD} pixels")
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        max_steps = 2500
        min_distance_this_episode = float('inf')
        episode_success = False
        
        print(f"\n📍 Episode {episode + 1}/{num_episodes}")
        print(f"   初始观察形状: {obs.shape}")
        print(f"   初始末端位置: {env._get_end_effector_position()}")
        
        # 计算初始距离
        initial_distance = np.linalg.norm(np.array(env._get_end_effector_position()) - env.goal_pos)
        print(f"   初始目标距离: {initial_distance:.1f}px")
        
        while step_count < max_steps:
            # 🔧 添加动作调试
            if step_count % 100 == 0 or step_count < 5:
                # 获取动作（使用确定性策略）
                action = sac.get_action(
                    torch.from_numpy(obs).float(),
                    gnn_embed.squeeze(0),
                    num_joints=num_joints,
                    deterministic=True
                )
                print(f"   Step {step_count}: Action = {action.detach().cpu().numpy()}")
                print(f"   Step {step_count}: 末端位置 = {env._get_end_effector_position()}")
                
                # 计算距离
                end_pos = env._get_end_effector_position()
                goal_pos = env.goal_pos
                distance = np.linalg.norm(np.array(end_pos) - goal_pos)
                print(f"   Step {step_count}: 距离 = {distance:.1f}px")
                
                # 🔧 尝试非确定性动作进行对比
                if step_count == 0:  # 只在第一步对比
                    action_random = sac.get_action(
                        torch.from_numpy(obs).float(),
                        gnn_embed.squeeze(0),
                        num_joints=num_joints,
                        deterministic=False
                    )
                    print(f"   Step {step_count}: 随机Action = {action_random.detach().cpu().numpy()}")
            else:
                # 获取动作（使用确定性策略）
                action = sac.get_action(
                    torch.from_numpy(obs).float(),
                    gnn_embed.squeeze(0),
                    num_joints=num_joints,
                    deterministic=True  # 🔧 可以改为False试试
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
                    print(f"  🎉 目标到达! 距离: {distance:.1f} pixels, 步骤: {step_count}")
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
    print(f"🏆 测试结果总结:")
    print(f"  测试Episodes: {num_episodes}")
    print(f"  成功次数: {success_count}")
    print(f"  成功率: {success_rate:.1%}")
    print(f"  平均奖励: {avg_reward:.2f}")
    print(f"  平均最小距离: {avg_min_distance:.1f} pixels")
    print(f"  平均Episode长度: {avg_episode_length:.1f} steps")
    print(f"  目标阈值: {GOAL_THRESHOLD:.1f} pixels")
    
    # 性能评价
    print(f"\n📋 性能评价:")
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
            if file.endswith('.pth') and ('final_model' in file or 'checkpoint' in file):
                full_path = os.path.join(root, file)
                mtime = os.path.getmtime(full_path)
                model_candidates.append((full_path, mtime))
    
    if not model_candidates:
        print(f"❌ 在 {base_path} 中未找到模型文件")
        return None
    
    model_candidates.sort(key=lambda x: x[1], reverse=True)
    
    print(f"🔍 找到 {len(model_candidates)} 个模型文件:")
    for i, (path, mtime) in enumerate(model_candidates[:5]):
        import datetime
        time_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {i+1}. {os.path.basename(path)} ({time_str})")
    
    latest_model = model_candidates[0][0]
    print(f"✅ 选择最新模型: {latest_model}")
    return latest_model

# === 主程序入口 ===
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    
    # 检查是否是测试模式
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("🧪 进入测试模式")
        
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
            print("🔍 自动查找最新模型...")
            model_path = find_latest_model()
        
        if model_path:
            print(f"🎯 测试参数: episodes={num_episodes}, render={render}")
            result = test_trained_model(model_path, num_episodes, render)
            
            if result:
                print(f"\n🎯 快速结论:")
                if result['success_rate'] >= 0.8:
                    print(f"  ✅ 模型表现优秀! 继续当前训练策略")
                elif result['success_rate'] >= 0.3:
                    print(f"  ⚠️  模型表现一般，建议继续训练或调整参数")
                else:
                    print(f"  ❌ 模型表现较差，需要重新审视奖励函数或网络结构")
        else:
            print("❌ 未找到可测试的模型")
        
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