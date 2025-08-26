#!/usr/bin/env python3
"""
增强版训练脚本 - 集成损失监控系统
基于原始train.py，添加了完整的损失记录和监控功能
"""

# 保持原有的所有导入
import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)

sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/gnn_encoder'))
sys.path.insert(0, os.path.join(base_dir, 'examples/rl/train'))  
sys.path.insert(0, os.path.join(base_dir, 'examples/rl/common'))
sys.path.insert(0, os.path.join(base_dir, 'examples/rl/environments'))
sys.path.append(os.path.join(base_dir, 'examples/rl'))

import numpy as np
import time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import argparse

gym.logger.set_level(40)

# 导入训练监控系统
from training_logger import TrainingLogger, RealTimeMonitor

# 原有的训练导入
from gnn_encoder import GNN_Encoder

if not hasattr(np, 'bool'):
    np.bool = bool

import environments
from arguments import get_parser
from utils import solve_argv_conflict
from common import *
from evaluation import render
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs, make_env
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from attn_dataset.sim_data_handler import DataHandler
from attn_model.attn_model import AttnModel
from sac.sac_model import AttentionSACWithBuffer
from env_config.env_wrapper import make_reacher2d_vec_envs, make_smart_reacher2d_vec_envs
from reacher2d_env import Reacher2DEnv
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
from async_renderer import AsyncRenderer, StateExtractor
import logging


SILENT_MODE = True

def smart_print(*args, **kwargs):
    if not SILENT_MODE:
        print(*args, **kwargs)

def check_goal_reached(env, goal_threshold=50.0):
    """检查是否到达目标"""
    try:
        if hasattr(env, '_get_end_effector_position') and hasattr(env, 'goal_pos'):
            end_pos = env._get_end_effector_position()
            goal_pos = env.goal_pos
            distance = np.linalg.norm(np.array(end_pos) - goal_pos)
            return distance <= goal_threshold, distance
    except Exception as e:
        print(f"目标检测失败: {e}")
    return False, float('inf')


def save_best_model(sac, model_save_path, success_rate, min_distance, step):
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
        
        model_file = os.path.join(model_save_path, f'best_model_step_{step}_{timestamp}.pth')
        torch.save(model_data, model_file)
        
        latest_file = os.path.join(model_save_path, 'latest_best_model.pth')
        torch.save(model_data, latest_file)
        
        smart_print(f"🏆 保存最佳模型: {model_file}")
        smart_print(f"   成功率: {success_rate:.3f}, 最小距离: {min_distance:.1f}, 步骤: {step}")
        
        return True
    except Exception as e:
        smart_print(f"❌ 保存模型失败: {e}")
        return False
    

# def load_checkpoint(sac, checkpoint_path, device='cpu'):
#     """从检查点加载SAC模型"""
#     try:
#         if not os.path.exists(checkpoint_path):
#             print(f"❌ 检查点文件不存在: {checkpoint_path}")
#             return 0
#         else:
#             print(f"🔄 Loading checkpoint from: {checkpoint_path}")

#         smart_print(f"🔄 Loading checkpoint from: {checkpoint_path}")
#         checkpoint = torch.load(checkpoint_path, map_location=device)
        
#         # 打印checkpoint内容
#         smart_print(f"📋 Checkpoint contains: {list(checkpoint.keys())}")
        
#         # 加载各个网络的状态
#         if 'actor_state_dict' in checkpoint:
#             sac.actor.load_state_dict(checkpoint['actor_state_dict'])
#             smart_print("✅ Actor loaded")
        
#         if 'critic1_state_dict' in checkpoint:
#             sac.critic1.load_state_dict(checkpoint['critic1_state_dict'])
#             smart_print("✅ Critic1 loaded")
            
#         if 'critic2_state_dict' in checkpoint:
#             sac.critic2.load_state_dict(checkpoint['critic2_state_dict'])
#             smart_print("✅ Critic2 loaded")
        
#         # 加载target networks（如果存在）
#         if 'target_critic1_state_dict' in checkpoint:
#             sac.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
#             smart_print("✅ Target Critic1 loaded")
#         else:
#             # 如果没有保存target networks，从main networks复制
#             sac.target_critic1.load_state_dict(sac.critic1.state_dict())
#             smart_print("⚠️ Target Critic1 copied from Critic1")
            
#         if 'target_critic2_state_dict' in checkpoint:
#             sac.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
#             smart_print("✅ Target Critic2 loaded")
#         else:
#             sac.target_critic2.load_state_dict(sac.critic2.state_dict())
#             smart_print("⚠️ Target Critic2 copied from Critic2")
        
#         # 加载优化器状态（可选）
#         load_optimizers = True  # 设为False如果你想用新的学习率
#         if load_optimizers:
#             if 'actor_optimizer_state_dict' in checkpoint:
#                 sac.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
#                 print("✅ Actor optimizer loaded")
            
#             if 'critic_optimizer_state_dict' in checkpoint:
#                 sac.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
#                 smart_print("✅ Critic optimizer loaded")
                
#             if 'alpha_optimizer_state_dict' in checkpoint:
#                 sac.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
#                 smart_print("✅ Alpha optimizer loaded")
        
#         # 加载alpha值
#         if 'alpha' in checkpoint:
#             if isinstance(checkpoint['alpha'], torch.Tensor):
#                 sac.alpha = checkpoint['alpha'].item()
#             else:
#                 sac.alpha = checkpoint['alpha']
#             smart_print(f"✅ Alpha loaded: {sac.alpha}")
            
#         if 'log_alpha' in checkpoint:
#             if isinstance(checkpoint['log_alpha'], torch.Tensor):
#                 sac.log_alpha.data.fill_(checkpoint['log_alpha'].item())
#             else:
#                 sac.log_alpha.data.fill_(checkpoint['log_alpha'])
#             smart_print(f"✅ Log Alpha loaded: {sac.log_alpha.item()}")
        
#         # 返回训练步数
#         start_step = checkpoint.get('step', 0)
#         smart_print(f"✅ Checkpoint loaded successfully! Starting from step: {start_step}")
        
#         return start_step
        
#     except Exception as e:
#         print(f"❌ Failed to load checkpoint: {e}")
#         print("Training will start from scratch...")
#         return 0

def load_checkpoint(sac, checkpoint_path, device='cpu'):
    """完全修复proj_dict的checkpoint加载"""
    try:
        if not os.path.exists(checkpoint_path):
            print(f"❌ 检查点文件不存在: {checkpoint_path}")
            return 0
        
        print(f"🔄 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        print(f"📋 Checkpoint contains: {list(checkpoint.keys())}")
        
        # 先加载非proj_dict的部分
        for network_name in ['critic1', 'critic2', 'target_critic1', 'target_critic2']:
            state_dict_key = f'{network_name}_state_dict'
            if state_dict_key in checkpoint:
                state_dict = checkpoint[state_dict_key]
                network = getattr(sac, network_name)
                
                # 分离main部分和proj_dict部分
                main_state = {}
                proj_dict_state = {}
                
                for key, value in state_dict.items():
                    if key.startswith('proj_dict.'):
                        proj_dict_state[key] = value
                    else:
                        main_state[key] = value
                
                # 加载主要部分
                missing_keys, unexpected_keys = network.load_state_dict(main_state, strict=False)
                if unexpected_keys:
                    print(f"⚠️ {network_name}: 忽略不匹配的层: {unexpected_keys}")
                print(f"✅ {network_name} main networks loaded")
                
                # 重建proj_dict
                proj_layers = {}
                for key, tensor in proj_dict_state.items():
                    if '.weight' in key:
                        # 从 "proj_dict.8.weight" 提取维度 "8"
                        dim_key = key.split('.')[1]
                        input_dim = tensor.shape[1]
                        output_dim = tensor.shape[0]
                        
                        # 创建新的线性层
                        proj_layers[dim_key] = nn.Linear(input_dim, output_dim).to(device)
                        proj_layers[dim_key].weight.data = tensor.clone()
                        
                        print(f"  📌 重建 proj_dict['{dim_key}']: {input_dim} → {output_dim}")
                
                # 加载bias
                for key, tensor in proj_dict_state.items():
                    if '.bias' in key:
                        dim_key = key.split('.')[1]
                        if dim_key in proj_layers:
                            proj_layers[dim_key].bias.data = tensor.clone()
                
                # 将重建的proj_dict赋值给网络
                if proj_layers:
                    network.proj_dict = nn.ModuleDict(proj_layers)
                    print(f"✅ {network_name} proj_dict 完全恢复: {list(proj_layers.keys())}")
                else:
                    print(f"ℹ️ {network_name}: 没有找到proj_dict层")
        
        # 加载Actor（通常没有proj_dict问题）
        if 'actor_state_dict' in checkpoint:
            missing_keys, unexpected_keys = sac.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
            if unexpected_keys:
                print(f"⚠️ Actor: 忽略不匹配的层: {unexpected_keys}")
            print("✅ Actor loaded")
        
        # 加载优化器状态（可选）
        load_optimizers = True  # 设为False如果你想用新的学习率
        if load_optimizers:
            if 'actor_optimizer_state_dict' in checkpoint:
                try:
                    sac.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                    print("✅ Actor optimizer loaded")
                except Exception as e:
                    print(f"⚠️ Actor optimizer 加载失败，将使用新的优化器状态: {e}")
            
            if 'critic_optimizer_state_dict' in checkpoint:
                try:
                    sac.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                    print("✅ Critic optimizer loaded")
                except Exception as e:
                    print(f"⚠️ Critic optimizer 加载失败，将使用新的优化器状态: {e}")
                
            if 'alpha_optimizer_state_dict' in checkpoint:
                try:
                    sac.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                    print("✅ Alpha optimizer loaded")
                except Exception as e:
                    print(f"⚠️ Alpha optimizer 加载失败，将使用新的优化器状态: {e}")
        
        # 加载alpha值
        if 'alpha' in checkpoint:
            if isinstance(checkpoint['alpha'], torch.Tensor):
                sac.alpha = checkpoint['alpha'].item()
            else:
                sac.alpha = checkpoint['alpha']
            print(f"✅ Alpha loaded: {sac.alpha}")
            
        if 'log_alpha' in checkpoint:
            if isinstance(checkpoint['log_alpha'], torch.Tensor):
                sac.log_alpha.data.fill_(checkpoint['log_alpha'].item())
            else:
                sac.log_alpha.data.fill_(checkpoint['log_alpha'])
            print(f"✅ Log Alpha loaded: {sac.log_alpha.item()}")
        
        # 返回训练步数
        start_step = checkpoint.get('step', 0)
        print(f"✅ Checkpoint 完全修复加载! Starting from step: {start_step}")
        
        return start_step
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        print("Training will start from scratch...")
        return 0

def setup_logging(args):
    train_log_level = getattr(args, 'train_log_level',  'INFO')
    if args.quiet:
        train_log_level = 'SILENT'
    logger = logging.getLogger('TrainingScript')
    logger.handlers.clear()

    if train_log_level != 'SILENT':
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        log_level = level_map.get(train_log_level.upper(), logging.INFO)
        logger.setLevel(log_level)

        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
    else:
        logger.setLevel(logging.CRITICAL + 10)
    
    return logger, train_log_level == 'SILENT'


def main(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device('cpu')

    os.makedirs(args.save_dir, exist_ok=True)

    # 🚀 NEW: 初始化训练监控系统
    # experiment_name = f"reacher2d_sac_{time.strftime('%Y%m%d_%H%M%S')}"
    # logger = TrainingLogger(
    #     log_dir=os.path.join(args.save_dir, 'training_logs'),
    #     experiment_name=experiment_name
    # )
    
    # # 设置监控阈值
    # monitor = RealTimeMonitor(logger, alert_thresholds={
    #     'critic_loss': {'max': 50.0, 'nan_check': True},
    #     'actor_loss': {'max': 10.0, 'nan_check': True},
    #     'alpha_loss': {'max': 5.0, 'nan_check': True},
    #     'alpha': {'min': 0.01, 'max': 2.0, 'nan_check': True}
    # })
    
    # smart_print(f"📊 训练监控系统已初始化: {logger.experiment_dir}")


    hyperparams = {
        'learning_rate': args.lr,
        'alpha': args.alpha,
        'warmup_steps': args.warmup_steps,
        'target_entropy_factor': args.target_entropy_factor,
        'batch_size': 64,  # 从SAC初始化中获取
        'buffer_capacity': 10000,
        'gamma': args.gamma,
        'seed': args.seed,
        'num_processes': args.num_processes,
        'update_frequency': args.update_frequency,
        'total_steps': 120000,  # num_step会在后面定义
        'optimizer': 'Adam',
        'network_architecture': {
            'attn_model_dims': '128-130-130-4',
            'action_dim': 'dynamic',  # 会在后面更新
            'critic_hidden_layers': '待确定',
            'actor_hidden_layers': '待确定'
        }
    }


    training_log_path = os.path.join(args.save_dir, 'logs.txt')
    fp_log = open(training_log_path, 'w')
    fp_log.close()

    if args.env_name == 'reacher2d':
        smart_print("use reacher2d env")

        if hasattr(args, 'num_joints') and hasattr(args, 'link_lengths'):
            # 使用MAP-Elites传入的配置
            num_links = args.num_joints
            link_lengths = args.link_lengths
            smart_print(f"🤖 使用MAP-Elites配置: {num_links}关节, 长度={link_lengths}")
        else:
            # 使用默认配置
            num_links = 4
            link_lengths = [80, 80, 80, 60]
            smart_print(f"🤖 使用默认配置: {num_links}关节, 长度={link_lengths}")


        env_params = {
            'num_links': num_links,
            'link_lengths': link_lengths,
            'render_mode': 'human',
            'config_path': "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
        }
        smart_print(f"num links: {env_params['num_links']}")
        smart_print(f"link lengths: {env_params['link_lengths']}")

        env_config = {
            'env_name': args.env_name,
            'env_type': 'reacher2d',
            'num_links': env_params['num_links'],
            'link_lengths': env_params['link_lengths'],
            'config_path': env_params.get('config_path', 'N/A'),
            'render_mode': env_params.get('render_mode', 'human'),
            'reward_function': '距离+进度+成功+碰撞+方向奖励',
            'physics_engine': 'PyMunk',
            'goal_threshold': 35.0,
            'obstacle_type': 'zigzag',
            'action_space': 'continuous',
            'observation_space': 'joint_angles_and_positions'
        }

        # 🎨 异步渲染模式：多进程训练 + 独立渲染
        async_renderer = None
        sync_env = None
        
        if args.num_processes > 1:
            smart_print("🚀 多进程模式：启用异步渲染")
            
            train_env_params = env_params.copy()
            train_env_params['render_mode'] = None
            
            envs = make_reacher2d_vec_envs(
                env_params=train_env_params,
                seed=args.seed,
                num_processes=args.num_processes,
                gamma=args.gamma,
                log_dir=None,
                device=device,
                allow_early_resets=False,
            )
            
            async_renderer = AsyncRenderer(env_params)
            async_renderer.start()
            
            sync_env = Reacher2DEnv(**train_env_params)
            smart_print(f"✅ 异步渲染器已启动 (PID: {async_renderer.render_process.pid})")
            
        else:   
            smart_print("🏃 单进程模式：直接渲染")
            envs = make_reacher2d_vec_envs(
                env_params=env_params,
                seed=args.seed,
                num_processes=args.num_processes,
                gamma=args.gamma,
                log_dir=None,
                device=device,
                allow_early_resets=False
            )

        smart_print("✅ 环境创建成功")
        args.env_type = 'reacher2d'
        
    else:
        smart_print(f"use bullet env: {args.env_name}")
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes, 
                            args.gamma, None, device, False, args=args)
        render_env = gym.make(args.env_name, args=args)
        render_env.seed(args.seed)
        args.env_type = 'bullet'


        env_config = {
            'env_name': args.env_name,
            'env_type': 'bullet',
            'action_space': 'continuous',
            'physics_engine': 'Bullet'
        }


    num_joints = envs.action_space.shape[0]
    smart_print(f"Number of joints: {num_joints}")
    num_updates = 5
    num_step = 120000
    data_handler = DataHandler(num_joints, args.env_type)

    hyperparams['network_architecture']['action_dim'] = num_joints
    hyperparams['total_steps'] = num_step
    env_config['action_dim'] = num_joints

    if args.env_type == 'reacher2d':
        sys.path.append(os.path.join(os.path.dirname(__file__), '../2d_reacher/utils'))
        from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder
        
        smart_print("🤖 初始化 Reacher2D GNN 编码器...")
        reacher2d_encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
        single_gnn_embed = reacher2d_encoder.get_gnn_embeds(
            num_links=num_joints, 
            link_lengths=env_params['link_lengths']
        )
        smart_print(f"✅ Reacher2D GNN 嵌入生成成功，形状: {single_gnn_embed.shape}")
    else:
        rule_sequence = [int(s.strip(",")) for s in args.rule_sequence]
        gnn_encoder = GNN_Encoder(args.grammar_file, rule_sequence, 70, num_joints)
        gnn_graph = gnn_encoder.get_graph(rule_sequence)
        single_gnn_embed = gnn_encoder.get_gnn_embeds(gnn_graph)

    action_dim = num_joints
    attn_model = AttnModel(128, 130, 130, 4)
    sac = AttentionSACWithBuffer(attn_model, action_dim, 
                                buffer_capacity=10000, batch_size=64,
                                lr=args.lr,
                                env_type=args.env_type)
    
    
    sac.warmup_steps = args.warmup_steps
    # sac.alpha = args.alpha
    sac.alpha = torch.tensor(0.1)

    if hasattr(sac, 'target_entropy'):
        sac.target_entropy = -action_dim * args.target_entropy_factor


    sac.min_alpha = 0.05  # 设置下限为0.05
    print(f"🔒 Alpha衰减下限设置为: {sac.min_alpha}")
    
    experiment_name = f"reacher2d_sac_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(
        log_dir=os.path.join(args.save_dir, 'training_logs'),
        experiment_name=experiment_name,
        hyperparams=hyperparams,  # ← 【新增参数】
        env_config=env_config     # ← 【新增参数】
    )
    
    # 设置监控阈值
    monitor = RealTimeMonitor(logger, alert_thresholds={
        'critic_loss': {'max': 50.0, 'nan_check': True},
        'actor_loss': {'max': 10.0, 'nan_check': True},
        'alpha_loss': {'max': 5.0, 'nan_check': True},
        'alpha': {'min': 0.01, 'max': 2.0, 'nan_check': True}
    })
    
    smart_print(f"📊 训练监控系统已初始化: {logger.experiment_dir}")

    start_step = 0
    if hasattr(args,  'resume_checkpoint') and  args.resume_checkpoint:
        print(f"🔄 从检查点恢复训练: {args.resume_checkpoint}")
        start_step = load_checkpoint(sac, args.resume_checkpoint)

        if start_step > 0:
            print(f"成功加载checkpoint, 从step {start_step} 开始训练")
            sac.warmup_steps = 0

            if hasattr(args, 'resume_lr') and args.resume_lr:
                for param_group in sac.actor_optimizer.param_groups:
                    param_group['lr'] = args.resume_lr
                for param_group in sac.critic_optimizer.param_groups:
                    param_group['lr'] = args.resume_lr
                print(f"更新学习率为 {args.resume_lr}")

            if hasattr(args, 'resume_alpha') and args.resume_alpha:
                sac.alpha = args.resume_alpha
                print(f"更新alpha为 {args.resume_alpha}")
            
            resume_hyperparams = {
                'is_resumed_training': True,
                'resume_checkpoint': args.resume_checkpoint,
                'resume_step': start_step,
                'resume_lr': getattr(args, 'resume_lr', None),
                'resume_alpha': getattr(args, 'resume_alpha', None),
                'additional_steps': getattr(args, 'additional_steps', None),
                'original_lr': args.lr,
                'original_alpha': args.alpha,
                'warmup_skipped': True
            }
            logger.update_hyperparams(resume_hyperparams)
            
            experiment_name = f"resumed_{experiment_name}_from_step_{start_step}"
            logger.experiment_name = experiment_name
        else:
            print("❌ 无法加载checkpoint, 从开始训练")
            start_step = 0
    
    current_obs = envs.reset()
    current_gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)
    total_steps = 0
    episode_rewards = [0.0] * args.num_processes
    eval_frequency = 200
    
    # 🏆 添加最佳模型保存相关变量
    best_success_rate = 0.0
    best_min_distance = float('inf')
    goal_threshold = 35.0
    consecutive_success_count = 0
    min_consecutive_successes = 3
    model_save_path = os.path.join(args.save_dir, 'best_models')
    os.makedirs(model_save_path, exist_ok=True)
    
    if not hasattr(args, 'update_frequency'):
        args.update_frequency = args.update_frequency

    # smart_print(f"start training, warmup {sac.warmup_steps} steps")
    # smart_print(f"Total training steps: {num_step}, Update frequency: {args.update_frequency}")
    # smart_print(f"Expected warmup completion at step: {sac.warmup_steps}")

    # if start_step > 0 and hasattr(args, 'additional_steps') and args.additional_steps:
    #     num_step = start_step + args.additional_steps
    #     smart_print(f"🔄 从step {start_step} 开始训练, 额外训练 {args.additional_steps} 步")
    # else:
    #     num_step = 120000
    print(f"start training, warmup {sac.warmup_steps} steps")
    print(f"Total training steps: {num_step}, Update frequency: {args.update_frequency}")
    if start_step > 0:
        print(f"Resume from step: {start_step}")
    else:
        print(f"Expected warmup completion at step: {sac.warmup_steps}")

    training_completed = False
    early_termination_reason = ""

    try:
        for step in range(start_step, num_step):
            
            # 🚀 NEW: 记录每个episode的性能指标
            if step % 100 == 0:
                if step < sac.warmup_steps:
                    smart_print(f"Step {step}/{num_step}: Warmup phase ({step}/{sac.warmup_steps})")
                else:
                    smart_print(f"Step {step}/{num_step}: Training phase, Buffer size: {len(sac.memory)}")

                if async_renderer:
                    stats = async_renderer.get_stats()
                    smart_print(f"   🎨 渲染FPS: {stats.get('fps', 0):.1f}")

            if step < sac.warmup_steps:
                action_batch = torch.from_numpy(np.array([envs.action_space.sample() for _ in range(args.num_processes)]))
            else:
                actions = []
                for proc_id in range(args.num_processes):
                    action = sac.get_action(current_obs[proc_id],
                                            current_gnn_embeds[proc_id],
                                            num_joints=envs.action_space.shape[0],
                                            deterministic=False)
                    actions.append(action)
                action_batch = torch.stack(actions)

            # 🔍 Action监控
            if step % 50 == 0 or step < 20:
                if hasattr(envs, 'envs') and len(envs.envs) > 0:
                    env_goal = getattr(envs.envs[0], 'goal_pos', 'NOT FOUND')
                    print(f"🎯 [训练] Step {step} - 环境goal_pos: {env_goal}")
                smart_print(f"\n🎯 Step {step} Action Analysis:")
                action_numpy = action_batch.cpu().numpy() if hasattr(action_batch, 'cpu') else action_batch.numpy()
                
                for proc_id in range(min(args.num_processes, 2)):
                    action_values = action_numpy[proc_id]
                    # smart_print(f"  Process {proc_id}: Actions = [{action_values[0]:+6.2f}, {action_values[1]:+6.2f}, {action_values[2]:+6.2f}, {action_values[3]:+6.2f}]")
                    action_str = ', '.join([f"{val:+6.2f}" for val in action_values])
                    smart_print(f"  Process {proc_id}: Actions = [{action_str}]")
                    smart_print(f"    Max action: {np.max(np.abs(action_values)):6.2f}, Mean abs: {np.mean(np.abs(action_values)):6.2f}")

            next_obs, reward, done, infos = envs.step(action_batch)
            
            # 🎨 异步渲染
            if async_renderer and sync_env:
                sync_action = action_batch[0].cpu().numpy() if hasattr(action_batch, 'cpu') else action_batch[0]
                sync_env.step(sync_action)
                robot_state = StateExtractor.extract_robot_state(sync_env, step)
                async_renderer.render_frame(robot_state)

            next_gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)

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

            for proc_id in range(args.num_processes):
                is_done = done[proc_id].item() if torch.is_tensor(done[proc_id]) else bool(done[proc_id])
                if is_done:
                    print(f"Episode {step} finished with reward {episode_rewards[proc_id]:.2f}")

                    # 🔧 从infos中获取环境状态信息
                    if len(infos) > proc_id and isinstance(infos[proc_id], dict):
                        info = infos[proc_id]
                        
                        # 检查goal相关信息
                        if 'goal' in info:
                            goal_info = info['goal']
                            distance = goal_info.get('distance_to_goal', float('inf'))
                            goal_reached = goal_info.get('goal_reached', False)
                            
                            print(f"  🎯 距离目标: {distance:.1f} pixels")
                            
                            if goal_reached or distance <= 35.0:
                                print(f"🎉 成功到达目标! 距离: {distance:.1f}")
                                consecutive_success_count += 1
                                
                                # 计算成功率
                                if hasattr(envs, 'episode_count'):
                                    envs.episode_count += 1
                                else:
                                    envs.episode_count = 1
                                
                                success_rate = consecutive_success_count / max(envs.episode_count, 1)
                                
                                # 🏆 保存成功模型
                                if success_rate > best_success_rate or distance < best_min_distance:
                                    best_success_rate = max(success_rate, best_success_rate)
                                    best_min_distance = min(distance, best_min_distance)
                                    save_best_model(sac, model_save_path, success_rate, distance, step)
                                
                                # 🎯 检查是否应该停止训练
                                if consecutive_success_count >= min_consecutive_successes:
                                    print(f"🏁 连续成功{consecutive_success_count}次，训练达到目标!")
                                    print(f"   成功率: {success_rate:.3f}")
                                    print(f"   最小距离: {best_min_distance:.1f}")
                                    
                                    # 保存最终成功模型
                                    final_model_path = os.path.join(model_save_path, f'final_successful_model_step_{step}.pth')
                                    final_model_data = {
                                        'step': step,
                                        'final_success_rate': success_rate,
                                        'final_min_distance': best_min_distance,
                                        'consecutive_successes': consecutive_success_count,
                                        'training_completed': True,
                                        'reason': 'Reached target consecutive successes',
                                        'actor_state_dict': sac.actor.state_dict(),
                                        'critic1_state_dict': sac.critic1.state_dict(),
                                        'critic2_state_dict': sac.critic2.state_dict(),
                                    }
                                    torch.save(final_model_data, final_model_path)
                                    print(f"💾 保存最终成功模型: {final_model_path}")
                                    
                                    # 可以选择是否退出训练
                                    if step > 5000:  # 至少训练5000步后才允许提前退出
                                        print("🎯 提前结束训练 - 已达到训练目标")
                                        training_completed = True
                                        early_termination_reason = f"连续成功{consecutive_success_count}次，达到训练目标"
                                        break  # 这会退出当前for循环，但不会退出整个训练循环
                            else:
                                # 重置连续成功计数
                                consecutive_success_count = 0
                        
                        # 可选：显示其他有用信息
                        if 'robot' in info:
                            robot_info = info['robot']
                            print(f"  🤖 步骤: {robot_info.get('step_count', 0)}")
                    
                    # 🚀 NEW: 记录episode指标
                    episode_metrics = {
                        'reward': episode_rewards[proc_id],
                        'length': step,
                        'distance_to_goal': distance if 'distance' in locals() else float('inf'),
                        'goal_reached': goal_reached if 'goal_reached' in locals() else False
                    }
                    logger.log_episode(step // 100, episode_metrics)
                    
                    episode_rewards[proc_id] = 0.0
                    
                    if hasattr(envs, 'reset_one'):
                        current_obs[proc_id] = envs.reset_one(proc_id)
                        current_gnn_embeds[proc_id] = single_gnn_embed
            if step % 100 == 0:  # 每100步检查一次
                    # 计算最近的成功率
                recent_successes = 0
                recent_episodes = 0
                
                # 这里需要维护一个滑动窗口的成功记录
                if hasattr(envs, 'recent_success_history'):
                    recent_successes = sum(envs.recent_success_history[-100:])  # 最近100步
                    recent_episodes = len(envs.recent_success_history[-100:])
                    recent_success_rate = recent_successes / max(recent_episodes, 1)
                    
                    # 如果最近成功率很高，可以考虑暂停训练
                    if recent_success_rate >= 0.8 and step > 10000:  # 至少训练10000步
                        print(f"�� 最近100步成功率: {recent_success_rate:.3f} >= 0.8")
                        print(f"   建议暂停训练，保存当前模型")
                        
                        # 保存最终模型
                        final_model_path = os.path.join(model_save_path, f'final_successful_model_step_{step}.pth')
                        final_model_data = {
                            'step': step,
                            'final_success_rate': recent_success_rate,
                            'training_completed': True,
                            'reason': 'High success rate achieved',
                            'actor_state_dict': sac.actor.state_dict(),
                            'critic1_state_dict': sac.critic1.state_dict(),
                            'critic2_state_dict': sac.critic2.state_dict(),
                        }
                        torch.save(final_model_data, final_model_path)
                        print(f"💾 保存成功模型: {final_model_path}")

            # 🚀 NEW: 训练更新和损失记录
            if (step >= sac.warmup_steps and 
                step % args.update_frequency == 0 and 
                sac.memory.can_sample(sac.batch_size)):
                
                metrics = sac.update()
                
                if metrics:
                    # 🚀 NEW: 记录损失到监控系统
                    enhanced_metrics = metrics.copy()
                    enhanced_metrics.update({
                        'step': step,
                        'buffer_size': len(sac.memory),
                        'learning_rate': sac.actor_optimizer.param_groups[0]['lr'],
                        'warmup_progress': min(1.0, step / max(sac.warmup_steps, 1))  # 🚀 FIX: 防止除零
                    })
                    
                    logger.log_step(step, enhanced_metrics, episode=step//100)
                    alerts = monitor.check_alerts(step, enhanced_metrics)
                    
                    if step % 100 == 0:
                        print(f"Step {step} (total_steps {total_steps}): "
                            f"Critic Loss: {metrics['critic_loss']:.4f}, "
                            f"Actor Loss: {metrics['actor_loss']:.4f}, "
                            f"Alpha: {metrics['alpha']:.4f}, "
                            f"Buffer Size: {len(sac.memory)}")
                        
                        # 🚀 NEW: 使用新的统计打印
                        logger.print_current_stats(step, detailed=(step % 500 == 0))
                        
                        if 'entropy_term' in metrics:
                            smart_print(f"  Actor Loss 组件分析:")
                            smart_print(f"    Entropy Term (α*log_π): {metrics['entropy_term']:.4f}")
                            smart_print(f"    Q Term (Q值): {metrics['q_term']:.4f}")
                            smart_print(f"    Actor Loss = {metrics['entropy_term']:.4f} - {metrics['q_term']:.4f} = {metrics['actor_loss']:.4f}")
                            
                            if metrics['actor_loss'] < 0:
                                smart_print(f"    ✓ 负数Actor Loss = 高Q值 = 好的策略!")
                            else:
                                smart_print(f"    ⚠ 正数Actor Loss = 低Q值 = 策略需要改进")
            
            # 🚀 NEW: 定期生成损失曲线
            if step % 1000 == 0 and step > 0:
                logger.plot_losses(recent_steps=2000, show=False)
                
                checkpoint_path = os.path.join(model_save_path, f'checkpoint_step_{step}.pth')
                checkpoint_data = {
                    'step': step,
                    'best_success_rate': best_success_rate,
                    'best_min_distance': best_min_distance,
                    'consecutive_success_count': consecutive_success_count,
                    'actor_state_dict': sac.actor.state_dict(),
                    'critic1_state_dict': sac.critic1.state_dict(),
                    'critic2_state_dict': sac.critic2.state_dict(),
                }
                torch.save(checkpoint_data, checkpoint_path)
                smart_print(f"💾 保存检查点模型: {checkpoint_path}")
            
            total_steps += args.num_processes
            if training_completed:
                print(f"🏁 训练提前终止: {early_termination_reason}")
                break

    except Exception as e:
        smart_print(f"🔴 训练过程中发生错误: {e}")
        
        # 🚀 NEW: 在异常时也保存日志
        logger.save_logs()
        logger.generate_report()
        raise e

    finally:
        # 清理资源
        if 'async_renderer' in locals() and async_renderer:
            async_renderer.stop()
        if 'sync_env' in locals() and sync_env:
            sync_env.close()
            
        # 🚀 NEW: 最终报告和图表生成
        print(f"\n{'='*60}")
        print(f"🏁 训练完成总结:")
        print(f"  总步数: {step}")
        print(f"  最佳成功率: {best_success_rate:.3f}")
        print(f"  最佳最小距离: {best_min_distance:.1f} pixels")
        print(f"  当前连续成功次数: {consecutive_success_count}")
        
        # 🚀 NEW: 生成完整的训练报告
        logger.generate_report()
        logger.plot_losses(show=False)
        
        print(f"📊 完整训练日志已保存到: {logger.experiment_dir}")
        
        # 保存最终模型
        final_model_path = os.path.join(model_save_path, f'final_model_step_{step}.pth')
        final_model_data = {
            'step': step,
            'final_success_rate': best_success_rate,
            'final_min_distance': best_min_distance,
            'final_consecutive_successes': consecutive_success_count,
            'training_completed': True,
            'experiment_name': experiment_name,  # 🚀 NEW: 关联实验名称
            'actor_state_dict': sac.actor.state_dict(),
            'critic1_state_dict': sac.critic1.state_dict(),
            'critic2_state_dict': sac.critic2.state_dict(),
            'target_critic1_state_dict': sac.target_critic1.state_dict(),
            'target_critic2_state_dict': sac.target_critic2.state_dict(),
        }
        torch.save(final_model_data, final_model_path)
        print(f"💾 保存最终模型: {final_model_path}")
        print(f"{'='*60}")
def test_trained_model(model_path, num_episodes=10, render=True):
    """测试训练好的模型性能"""
    print(f"🧪 开始测试模型: {model_path}")
    
    # 环境配置 - 与训练时保持一致
    env_params = {
        'num_links': 4,
        'link_lengths': [80, 80, 80, 60],
        'render_mode': 'human' if render else None,
        'config_path': "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml",
        'debug_level': 'SILENT'
    }
    
    # 创建环境
    env = Reacher2DEnv(**env_params)
    num_joints = env.action_space.shape[0]
    
    # 创建GNN编码器 - 与训练时保持一致
    sys.path.append(os.path.join(base_dir, 'examples/2d_reacher/utils'))
    from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder
    
    reacher2d_encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
    gnn_embed = reacher2d_encoder.get_gnn_embeds(
        num_links=num_joints, 
        link_lengths=env_params['link_lengths']
    )
    
    # 创建SAC模型 - 与训练时保持一致
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
        
        # 加载网络状态
        if 'actor_state_dict' in model_data:
            sac.actor.load_state_dict(model_data['actor_state_dict'], strict=False)
            print("✅ Actor 加载成功")
        
        # 显示模型信息
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
    goal_threshold = 35.0  # 与奖励函数中的阈值保持一致
    
    print(f"\n🎮 开始测试 {num_episodes} 个episodes...")
    print(f"🎯 目标阈值: {goal_threshold} pixels")
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        max_steps = 500  # 限制最大步数
        min_distance_this_episode = float('inf')
        episode_success = False
        
        print(f"\n📍 Episode {episode + 1}/{num_episodes}")
        
        while step_count < max_steps:
            # 获取动作（使用确定性策略）
            action = sac.get_action(
                torch.from_numpy(obs).float(),
                gnn_embed.squeeze(0),
                num_joints=num_joints,
                deterministic=True  # 测试时使用确定性策略
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
                time.sleep(0.02)  # 稍微减慢渲染速度以便观察
            
            # 检查是否到达目标
            if distance <= goal_threshold:
                if not episode_success:  # 避免重复计数
                    success_count += 1
                    episode_success = True
                    print(f"  🎉 目标到达! 距离: {distance:.1f} pixels, 步骤: {step_count}")
                    if not render:  # 如果没有渲染，在到达目标后等待几步再结束
                        if step_count > max_steps - 50:  # 最后50步内到达就结束
                            break
                    else:
                        break
                        
            if done:
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
    print(f"  目标阈值: {goal_threshold:.1f} pixels")
    
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
        
    if avg_min_distance <= goal_threshold:
        print(f"  ✅ 平均最小距离达到目标阈值")
    else:
        print(f"  ⚠️  平均最小距离超出目标阈值 {avg_min_distance - goal_threshold:.1f} pixels")
    
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
    
    # 查找所有可能的模型文件
    model_candidates = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.pth') and ('final_model' in file or 'checkpoint' in file):
                full_path = os.path.join(root, file)
                # 获取文件修改时间
                mtime = os.path.getmtime(full_path)
                model_candidates.append((full_path, mtime))
    
    if not model_candidates:
        print(f"❌ 在 {base_path} 中未找到模型文件")
        return None
    
    # 按修改时间排序，最新的在前
    model_candidates.sort(key=lambda x: x[1], reverse=True)
    
    print(f"🔍 找到 {len(model_candidates)} 个模型文件:")
    for i, (path, mtime) in enumerate(model_candidates[:5]):  # 只显示前5个
        import datetime
        time_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {i+1}. {os.path.basename(path)} ({time_str})")
    
    latest_model = model_candidates[0][0]
    print(f"✅ 选择最新模型: {latest_model}")
    return latest_model

 
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    
    # 🚀 NEW: 检查是否是测试模式
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("🧪 进入测试模式")
        
        # 解析测试参数
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
                model_path = 'latest'  # 标记使用最新模型
        
        # 如果没有指定模型路径，自动查找最新模型
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
    
    # 🚀 NEW: 检查是否是训练模式 (默认)
    elif len(sys.argv) > 1 and sys.argv[1] == '--train':
        print("🚀 进入训练模式")
        # 移除 --train 参数，继续使用原有的参数解析
        sys.argv.pop(1)  # 移除 --train
    
    # 🔧 MODIFIED: 原有的测试命令保持兼容性
    if len(sys.argv) > 1 and sys.argv[1] == '--test-reacher2d':
        print("🤖 启动 Reacher2D 环境训练 (兼容模式)")
        args_list = ['--env-name', 'reacher2d',
                     '--num-processes', '2',
                     '--lr', '3e-4',
                     '--gamma', '0.99',
                     '--seed', '42',
                     '--save-dir', './trained_models/reacher2d/enhanced_test/',
                     '--grammar-file', '/home/xli149/Documents/repos/RoboGrammar/data/designs/grammar_jan21.dot',
                     '--rule-sequence', '0']
        test_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # 🚀 NEW: 默认训练模式 - 使用 reacher2d 配置
    elif len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] not in ['--test', '--train', '--test-reacher2d']):
        print("🤖 默认启动 Reacher2D 环境训练")
        args_list = ['--env-name', 'reacher2d',
                     '--num-processes', '2',
                     '--lr', '3e-4',
                     '--gamma', '0.99',
                     '--seed', '42',
                     '--save-dir', './trained_models/reacher2d/enhanced_test/',
                     '--grammar-file', '/home/xli149/Documents/repos/RoboGrammar/data/designs/grammar_jan21.dot',
                     '--rule-sequence', '0']
        test_args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    else:
        # 其他环境的训练配置 (bullet env等)
        args_list = ['--env-name', 'RobotLocomotion-v0',
                     '--task', 'FlatTerrainTask',
                     '--grammar-file', '../../data/designs/grammar_jan21.dot',
                     '--algo', 'ppo',
                     '--use-gae',
                     '--log-interval', '5',
                     '--num-steps', '1024',
                     '--num-processes', '2',
                     '--lr', '3e-4',
                     '--entropy-coef', '0',
                     '--value-loss-coef', '0.5',
                     '--ppo-epoch', '10',
                     '--num-mini-batch', '32',
                     '--gamma', '0.995',
                     '--gae-lambda', '0.95',
                     '--num-env-steps', '30000000',
                     '--use-linear-lr-decay',
                     '--use-proper-time-limits',
                     '--save-interval', '100',
                     '--seed', '2',
                     '--save-dir', './trained_models/RobotLocomotion-v0/enhanced_test/',
                     '--render-interval', '80']
        test_args = sys.argv[1:]
    
    parser = get_parser()

    # 训练相关参数
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                       help='从指定checkpoint继续训练的文件路径')
    parser.add_argument('--resume-lr', type=float, default=None,
                       help='恢复训练时使用的新学习率')
    parser.add_argument('--resume-alpha', type=float, default=None,
                       help='恢复训练时使用的新alpha值')
    parser.add_argument('--additional-steps', type=int, default=None,
                       help='恢复训练时额外的训练步数')
    

    parser.add_argument('--batch-size', type=int, default=64,
                    help='训练批次大小')
    parser.add_argument('--buffer-capacity', type=int, default=10000,
                    help='经验回放缓冲区大小')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                    help='热身步数')
    parser.add_argument('--target-entropy-factor', type=float, default=0.8,
                    help='目标熵系数 (target_entropy = -action_dim * factor)')
    parser.add_argument('--update-frequency', type=int, default=2,
                    help='网络更新频率 (每N步更新一次)')
    
    args = parser.parse_args(args_list + test_args)

    solve_argv_conflict(args_list)
    args = parser.parse_args(args_list + test_args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.save_dir = os.path.join(args.save_dir, get_time_stamp())
    try:
        os.makedirs(args.save_dir, exist_ok=True)
    except OSError:
        pass

    fp = open(os.path.join(args.save_dir, 'args.txt'), 'w')
    fp.write(str(args_list + test_args))
    fp.close()

    main(args)