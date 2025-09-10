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
    'num_links': 4,
    'link_lengths': [80, 80, 80, 60],
    'config_path': "/home/xli149/Documents/repos/test_robo/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
}

# === 自定义参数解析器 ===
def create_training_parser():
    """创建训练专用的参数解析器"""
    parser = argparse.ArgumentParser(description='Enhanced SAC Training for Reacher2D')
    
    # 基本参数
    parser.add_argument('--env-name', default='reacher2d', help='环境名称')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num-processes', type=int, default=2, help='并行进程数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--alpha', type=float, default=0.1, help='SAC熵系数')
    parser.add_argument('--save-dir', default='./trained_models/reacher2d/enhanced_test/', help='保存目录')
    
      # 🆕 添加渲染控制参数
    parser.add_argument('--render', action='store_true', default=False, help='是否显示可视化窗口')
    parser.add_argument('--no-render', action='store_true', default=False, help='强制禁用可视化窗口')
    # SAC特定参数
    parser.add_argument('--warmup-steps', type=int, default=1000, help='热身步数')
    parser.add_argument('--target-entropy-factor', type=float, default=0.8, help='目标熵系数')
    parser.add_argument('--update-frequency', type=int, default=2, help='网络更新频率')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--buffer-capacity', type=int, default=10000, help='缓冲区容量')
    
    # 恢复训练参数
    parser.add_argument('--resume-checkpoint', type=str, default=None, help='检查点路径')
    parser.add_argument('--resume-lr', type=float, default=None, help='恢复时的学习率')
    parser.add_argument('--resume-alpha', type=float, default=None, help='恢复时的alpha值')
    
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
    def save_checkpoint(self, sac, step, **kwargs):
        """保存检查点 - 完整版"""
        try:
            checkpoint_data = {
                'step': step,
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
                # 🔧 添加训练状态
                'buffer_size': len(sac.memory),
                'warmup_steps': sac.warmup_steps,
                **kwargs
            }
            
            checkpoint_path = os.path.join(self.best_models_dir, f'checkpoint_step_{step}.pth')
            torch.save(checkpoint_data, checkpoint_path)
            print(f"💾 保存检查点: step {step}, buffer size: {len(sac.memory)}")
            return True
        except Exception as e:
            print(f"❌ 保存检查点失败: {e}")
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


        should_render = False
        if hasattr(args, 'render') and args.render:
            should_render = True
        elif hasattr(args, 'no_render') and args.no_render:
            should_render = False
        else:
            should_render = False  # 🆕 默认不显示，除非明确指定
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
        
        # 🆕 根据参数决定是否创建渲染环境
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
        self.min_consecutive_successes = 3
        
    def handle_episode_end(self, proc_id, step, episode_rewards, infos):
        """处理episode结束逻辑"""
        if len(infos) <= proc_id or not isinstance(infos[proc_id], dict):
            episode_rewards[proc_id] = 0.0
            return False
        
        info = infos[proc_id]
        goal_reached = False
        distance = float('inf')
        
        # 检查目标信息
        if 'goal' in info:
            goal_info = info['goal']
            distance = goal_info.get('distance_to_goal', float('inf'))
            goal_reached = goal_info.get('goal_reached', False)
            
            print(f"Episode {step} 结束: 奖励 {episode_rewards[proc_id]:.2f}, 距离 {distance:.1f}")
            
            if goal_reached:
                print(f"🎉 成功到达目标! 距离: {distance:.1f}")
                self.consecutive_success_count += 1
                
                # 🔧 统一的保存逻辑
                if distance < self.best_min_distance:
                    self.best_min_distance = distance
                    success_rate = self.consecutive_success_count / max(1, step // 100)
                    self.best_success_rate = max(success_rate, self.best_success_rate)
                    
                    # 保存最佳模型（包含完整状态）
                    self.model_manager.save_best_model(
                        self.sac, success_rate, distance, step
                    )
                
                # 检查是否达到训练目标
                if self.consecutive_success_count >= self.min_consecutive_successes and step > 5000:
                    print(f"🏁 连续成功{self.consecutive_success_count}次，训练达到目标!")
                    
                    # 🔧 只需要标记训练完成，不需要重复保存
                    print(f"✅ 最佳模型已保存，训练目标达成！")
                    return True  # 结束训练
        else:
                self.consecutive_success_count = 0
        
        # 记录episode指标
        episode_metrics = {
            'reward': episode_rewards[proc_id],
            'length': step,
            'distance_to_goal': distance,
            'goal_reached': goal_reached
        }
        self.logger.log_episode(step // 100, episode_metrics)
        episode_rewards[proc_id] = 0.0
        
        return False
    
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
                print(f"Step {step} (total_steps {total_steps}): "
                      f"Learning Rate: {metrics['lr']:.6f}, "
                      f"Critic Loss: {metrics['critic_loss']:.4f}, "
                      f"Actor Loss: {metrics['actor_loss']:.4f}, "
                      f"Alpha: {metrics['alpha']:.4f}, "
                      f"Buffer Size: {len(self.sac.memory)}")



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

    # 创建SAC模型
    attn_model = AttnModel(128, 130, 130, 4)
    sac = AttentionSACWithBuffer(
        attn_model, num_joints, 
        buffer_capacity=args.buffer_capacity, batch_size=args.batch_size,
        lr=args.lr, env_type=args.env_type
    )
    
    # 添加SAC特定参数
    sac.warmup_steps = args.warmup_steps
    sac.alpha = torch.tensor(args.alpha)
    sac.min_alpha = 0.05
    print(f"🔒 Alpha衰减下限设置为: {sac.min_alpha}")

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
    
    # 清理资源
    cleanup_resources(sync_env, logger, model_manager, training_manager)

def run_training_loop(args, envs, sync_env, sac, single_gnn_embed, training_manager, num_joints, start_step=0):
    """运行训练循环"""
    current_obs = envs.reset()
    print(f"初始观察: {current_obs.shape}")
    
    # 重置渲染环境
    if sync_env:
        sync_env.reset()
        print("🔧 sync_env 已重置")
    
    current_gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)
    episode_rewards = [0.0] * args.num_processes
    num_step = 120000
    total_steps = 0
    
    print(f"开始训练: warmup {sac.warmup_steps} 步")
    print(f"总训练步数: {num_step}, 更新频率: {args.update_frequency}")
    if start_step > 0:
        print(f"从步骤 {start_step} 恢复训练")
    else:
        print(f"预期warmup完成步骤: {sac.warmup_steps}")

    training_completed = False
    early_termination_reason = ""

    try:
        for step in range(start_step, num_step):
            # 进度显示
            if step % 100 == 0:
                if step < sac.warmup_steps:
                    smart_print(f"Step {step}/{num_step}: Warmup phase ({step}/{sac.warmup_steps})")
                else:
                    smart_print(f"Step {step}/{num_step}: Training phase, Buffer size: {len(sac.memory)}")

            # 获取动作
            if step < sac.warmup_steps:
                action_batch = torch.from_numpy(np.array([
                    envs.action_space.sample() for _ in range(args.num_processes)
                ]))
            else:
                actions = []
                for proc_id in range(args.num_processes):
                    action = sac.get_action(
                        current_obs[proc_id],
                                            current_gnn_embeds[proc_id],
                                            num_joints=envs.action_space.shape[0],
                        deterministic=False
                    )
                    actions.append(action)
                action_batch = torch.stack(actions)

            # 动作分析（调试用）
            if step % 50 == 0 or step < 20:
                if hasattr(envs, 'envs') and len(envs.envs) > 0:
                    env_goal = getattr(envs.envs[0], 'goal_pos', 'NOT FOUND')
                    print(f"🎯 [训练] Step {step} - 环境goal_pos: {env_goal}")
                
                smart_print(f"\n🎯 Step {step} Action Analysis:")
                action_numpy = action_batch.cpu().numpy() if hasattr(action_batch, 'cpu') else action_batch.numpy()
                
                for proc_id in range(min(args.num_processes, 2)):
                    action_values = action_numpy[proc_id]
                    action_str = ', '.join([f"{val:+6.2f}" for val in action_values])
                    smart_print(f"  Process {proc_id}: Actions = [{action_str}]")
                    smart_print(f"    Max action: {np.max(np.abs(action_values)):6.2f}, Mean abs: {np.mean(np.abs(action_values)):6.2f}")

            # 执行动作
            next_obs, reward, done, infos = envs.step(action_batch)

            # 渲染处理
            if sync_env:
                sync_action = action_batch[0].cpu().numpy() if hasattr(action_batch, 'cpu') else action_batch[0]
                sync_env.step(sync_action)
                sync_env.render()

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

            # 处理episode结束
            for proc_id in range(args.num_processes):
                is_done = done[proc_id].item() if torch.is_tensor(done[proc_id]) else bool(done[proc_id])
                if is_done:
                    should_end = training_manager.handle_episode_end(proc_id, step, episode_rewards, infos)
                    if should_end:
                        training_completed = True
                        early_termination_reason = f"连续成功{training_manager.consecutive_success_count}次，达到训练目标"
                        break
                    
                    if hasattr(envs, 'reset_one'):
                        current_obs[proc_id] = envs.reset_one(proc_id)
                        current_gnn_embeds[proc_id] = single_gnn_embed
            
            # 模型更新
            if training_manager.should_update_model(step):
                training_manager.update_and_log(step, total_steps)
            
            # 定期保存和绘图
            if step % 1000 == 0 and step > 0:
                training_manager.logger.plot_losses(recent_steps=2000, show=False)
                training_manager.model_manager.save_checkpoint(
                    sac, step,
                    best_success_rate=training_manager.best_success_rate,
                    best_min_distance=training_manager.best_min_distance,
                    consecutive_success_count=training_manager.consecutive_success_count
                )
            
            total_steps += args.num_processes
            
            if training_completed:
                print(f"🏁 训练提前终止: {early_termination_reason}")
                break

    except Exception as e:
        print(f"🔴 训练过程中发生错误: {e}")
        training_manager.logger.save_logs()
        training_manager.logger.generate_report()
        raise e

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
    
    # 创建环境
    env = Reacher2DEnv(**env_params)
    num_joints = env.action_space.shape[0]
    
    # 创建GNN编码器
    sys.path.append(os.path.join(base_dir, 'examples/2d_reacher/utils'))
    from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder
    
    reacher2d_encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
    gnn_embed = reacher2d_encoder.get_gnn_embeds(
        num_links=num_joints, 
        link_lengths=env_params['link_lengths']
    )
    
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
        
        while step_count < max_steps:
            # 获取动作（使用确定性策略）
            action = sac.get_action(
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