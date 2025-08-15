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
        
        print(f"🏆 保存最佳模型: {model_file}")
        print(f"   成功率: {success_rate:.3f}, 最小距离: {min_distance:.1f}, 步骤: {step}")
        
        return True
    except Exception as e:
        print(f"❌ 保存模型失败: {e}")
        return False
    

def load_checkpoint(sac, checkpoint_path, device='cpu'):
    """从检查点加载SAC模型"""
    try:
        print(f"🔄 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 打印checkpoint内容
        print(f"📋 Checkpoint contains: {list(checkpoint.keys())}")
        
        # 加载各个网络的状态
        if 'actor_state_dict' in checkpoint:
            sac.actor.load_state_dict(checkpoint['actor_state_dict'])
            print("✅ Actor loaded")
        
        if 'critic1_state_dict' in checkpoint:
            sac.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            print("✅ Critic1 loaded")
            
        if 'critic2_state_dict' in checkpoint:
            sac.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            print("✅ Critic2 loaded")
        
        # 加载target networks（如果存在）
        if 'target_critic1_state_dict' in checkpoint:
            sac.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
            print("✅ Target Critic1 loaded")
        else:
            # 如果没有保存target networks，从main networks复制
            sac.target_critic1.load_state_dict(sac.critic1.state_dict())
            print("⚠️ Target Critic1 copied from Critic1")
            
        if 'target_critic2_state_dict' in checkpoint:
            sac.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
            print("✅ Target Critic2 loaded")
        else:
            sac.target_critic2.load_state_dict(sac.critic2.state_dict())
            print("⚠️ Target Critic2 copied from Critic2")
        
        # 加载优化器状态（可选）
        load_optimizers = True  # 设为False如果你想用新的学习率
        if load_optimizers:
            if 'actor_optimizer_state_dict' in checkpoint:
                sac.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                print("✅ Actor optimizer loaded")
            
            if 'critic_optimizer_state_dict' in checkpoint:
                sac.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                print("✅ Critic optimizer loaded")
                
            if 'alpha_optimizer_state_dict' in checkpoint:
                sac.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
                print("✅ Alpha optimizer loaded")
        
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
        print(f"✅ Checkpoint loaded successfully! Starting from step: {start_step}")
        
        return start_step
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print("Training will start from scratch...")
        return 0


def main(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device('cpu')

    os.makedirs(args.save_dir, exist_ok=True)

    # 🚀 NEW: 初始化训练监控系统
    experiment_name = f"reacher2d_sac_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(
        log_dir=os.path.join(args.save_dir, 'training_logs'),
        experiment_name=experiment_name
    )
    
    # 设置监控阈值
    monitor = RealTimeMonitor(logger, alert_thresholds={
        'critic_loss': {'max': 50.0, 'nan_check': True},
        'actor_loss': {'max': 10.0, 'nan_check': True},
        'alpha_loss': {'max': 5.0, 'nan_check': True},
        'alpha': {'min': 0.01, 'max': 2.0, 'nan_check': True}
    })
    
    print(f"📊 训练监控系统已初始化: {logger.experiment_dir}")

    training_log_path = os.path.join(args.save_dir, 'logs.txt')
    fp_log = open(training_log_path, 'w')
    fp_log.close()

    if args.env_name == 'reacher2d':
        print("use reacher2d env")

        env_params = {
            'num_links': 4,
            'link_lengths': [80, 80, 80, 60],
            'render_mode': 'human',
            'config_path': "/home/xli149/Documents/repos/RoboGrammar/examples/2d_reacher/configs/reacher_with_zigzag_obstacles.yaml"
        }
        print(f"num links: {env_params['num_links']}")
        print(f"link lengths: {env_params['link_lengths']}")

        # 🎨 异步渲染模式：多进程训练 + 独立渲染
        async_renderer = None
        sync_env = None
        
        if args.num_processes > 1:
            print("🚀 多进程模式：启用异步渲染")
            
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
            print(f"✅ 异步渲染器已启动 (PID: {async_renderer.render_process.pid})")
            
        else:
            print("🏃 单进程模式：直接渲染")
            envs = make_reacher2d_vec_envs(
                env_params=env_params,
                seed=args.seed,
                num_processes=args.num_processes,
                gamma=args.gamma,
                log_dir=None,
                device=device,
                allow_early_resets=False
            )

        print(f"✅ 环境创建成功")
        args.env_type = 'reacher2d'
        
    else:
        print(f"use bullet env: {args.env_name}")
        envs = make_vec_envs(args.env_name, args.seed, args.num_processes, 
                            args.gamma, None, device, False, args=args)
        render_env = gym.make(args.env_name, args=args)
        render_env.seed(args.seed)
        args.env_type = 'bullet'

    num_joints = envs.action_space.shape[0]
    print(f"Number of joints: {num_joints}")
    num_updates = 5
    num_step = 120000
    data_handler = DataHandler(num_joints, args.env_type)

    if args.env_type == 'reacher2d':
        sys.path.append(os.path.join(os.path.dirname(__file__), '../2d_reacher/utils'))
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

    action_dim = num_joints
    attn_model = AttnModel(128, 130, 130, 4)
    sac = AttentionSACWithBuffer(attn_model, action_dim, 
                                buffer_capacity=10000, batch_size=64,
                                lr=1e-5,
                                env_type=args.env_type)
    
    sac.warmup_steps = 1000
    sac.alpha = 0.15
    if hasattr(sac, 'target_entropy'):
        sac.target_entropy = -action_dim * 0.8
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
        args.update_frequency = 2

    print(f"start training, warmup {sac.warmup_steps} steps")
    print(f"Total training steps: {num_step}, Update frequency: {args.update_frequency}")
    print(f"Expected warmup completion at step: {sac.warmup_steps}")

    try:
        for step in range(num_step):
            
            # 🚀 NEW: 记录每个episode的性能指标
            if step % 100 == 0:
                if step < sac.warmup_steps:
                    print(f"Step {step}/{num_step}: Warmup phase ({step}/{sac.warmup_steps})")
                else:
                    print(f"Step {step}/{num_step}: Training phase, Buffer size: {len(sac.memory)}")

                if async_renderer:
                    stats = async_renderer.get_stats()
                    print(f"   🎨 渲染FPS: {stats.get('fps', 0):.1f}")

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
                print(f"\n🎯 Step {step} Action Analysis:")
                action_numpy = action_batch.cpu().numpy() if hasattr(action_batch, 'cpu') else action_batch.numpy()
                
                for proc_id in range(min(args.num_processes, 2)):
                    action_values = action_numpy[proc_id]
                    print(f"  Process {proc_id}: Actions = [{action_values[0]:+6.2f}, {action_values[1]:+6.2f}, {action_values[2]:+6.2f}, {action_values[3]:+6.2f}]")
                    print(f"    Max action: {np.max(np.abs(action_values)):6.2f}, Mean abs: {np.mean(np.abs(action_values)):6.2f}")

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
                    
                    # 🚀 NEW: 记录episode指标
                    episode_metrics = {
                        'reward': episode_rewards[proc_id],
                        'length': step  # 简化版本，实际应该记录episode长度
                    }
                    logger.log_episode(step // 100, episode_metrics)  # 近似episode数
                    
                    episode_rewards[proc_id] = 0.0
                    
                    if hasattr(envs, 'reset_one'):
                        current_obs[proc_id] = envs.reset_one(proc_id)
                        current_gnn_embeds[proc_id] = single_gnn_embed

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
                        'warmup_progress': min(1.0, step / sac.warmup_steps)
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
                            print(f"  Actor Loss 组件分析:")
                            print(f"    Entropy Term (α*log_π): {metrics['entropy_term']:.4f}")
                            print(f"    Q Term (Q值): {metrics['q_term']:.4f}")
                            print(f"    Actor Loss = {metrics['entropy_term']:.4f} - {metrics['q_term']:.4f} = {metrics['actor_loss']:.4f}")
                            
                            if metrics['actor_loss'] < 0:
                                print(f"    ✓ 负数Actor Loss = 高Q值 = 好的策略!")
                            else:
                                print(f"    ⚠ 正数Actor Loss = 低Q值 = 策略需要改进")
            
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
                print(f"💾 保存检查点模型: {checkpoint_path}")
            
            total_steps += args.num_processes

    except Exception as e:
        print(f"🔴 训练过程中发生错误: {e}")
        
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

 
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test-reacher2d':
        print("🤖 启动 Reacher2D 环境测试")
        args_list = ['--env-name', 'reacher2d',
                     '--num-processes', '2',
                     '--lr', '3e-4',
                     '--gamma', '0.99',
                     '--seed', '42',
                     '--save-dir', './trained_models/reacher2d/enhanced_test/',
                     '--grammar-file', '/home/xli149/Documents/repos/RoboGrammar/data/designs/grammar_jan21.dot',
                     '--rule-sequence', '0']
        test_args = sys.argv[2:] if len(sys.argv) > 2 else []
    else:
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
    args = parser.parse_args(args_list + test_args)

    solve_argv_conflict(args_list)
    parser = get_parser()
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