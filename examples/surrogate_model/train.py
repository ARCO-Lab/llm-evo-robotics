import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)

sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
sys.path.insert(0, os.path.join(base_dir, 'examples/surrogate_model/gnn_encoder'))
sys.path.insert(0, os.path.join(base_dir, 'examples/rl/train'))  # 使用insert确保优先级
sys.path.insert(0, os.path.join(base_dir, 'examples/rl/common'))
sys.path.insert(0, os.path.join(base_dir, 'examples/rl/environments'))
sys.path.append(os.path.join(base_dir, 'examples/rl'))

import numpy as np
import time
from collections import deque

from gnn_encoder import GNN_Encoder

if not hasattr(np, 'bool'):
    np.bool = bool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

gym.logger.set_level(40)


# 直接导入，现在environments在路径中
import environments

# 直接导入模块，不使用rl.前缀
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

# 修改第50行的导入
from env_config.env_wrapper import make_reacher2d_vec_envs, make_smart_reacher2d_vec_envs
from reacher2d_env import Reacher2DEnv
# 在 train.py 第6行后添加
sys.path.insert(0, os.path.join(base_dir, 'examples/2d_reacher/envs'))
from async_renderer import AsyncRenderer, StateExtractor  # 🎨 添加这行


def check_goal_reached(env, goal_threshold=50.0):  # 调整默认阈值为50.0
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
        
        # 保存SAC所有组件
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
        
        # 保存文件
        model_file = os.path.join(model_save_path, f'best_model_step_{step}_{timestamp}.pth')
        torch.save(model_data, model_file)
        
        # 同时保存一个"latest_best"版本便于加载
        latest_file = os.path.join(model_save_path, 'latest_best_model.pth')
        torch.save(model_data, latest_file)
        
        print(f"🏆 保存最佳模型: {model_file}")
        print(f"   成功率: {success_rate:.3f}, 最小距离: {min_distance:.1f}, 步骤: {step}")
        
        return True
    except Exception as e:
        print(f"❌ 保存模型失败: {e}")
        return False

def load_best_model(sac, model_save_path):
    """加载最佳模型"""
    try:
        latest_file = os.path.join(model_save_path, 'latest_best_model.pth')
        if os.path.exists(latest_file):
            model_data = torch.load(latest_file, map_location=sac.device)
            
            sac.actor.load_state_dict(model_data['actor_state_dict'])
            sac.critic1.load_state_dict(model_data['critic1_state_dict'])
            sac.critic2.load_state_dict(model_data['critic2_state_dict'])
            sac.target_critic1.load_state_dict(model_data['target_critic1_state_dict'])
            sac.target_critic2.load_state_dict(model_data['target_critic2_state_dict'])
            
            print(f"✅ 加载最佳模型成功: Step {model_data['step']}, 成功率: {model_data['success_rate']:.3f}")
            return True
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
    return False


def main(args):

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device('cpu')

    os.makedirs(args.save_dir, exist_ok = True)

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
            
            # 创建无渲染的训练环境
            train_env_params = env_params.copy()
            train_env_params['render_mode'] = None  # 训练环境不渲染
            
            envs = make_reacher2d_vec_envs(
                env_params=train_env_params,
                seed=args.seed,
                num_processes=args.num_processes,
                gamma=args.gamma,
                log_dir=None,
                device=device,
                allow_early_resets=False,
            )
            
            # 创建异步渲染器
       
            async_renderer = AsyncRenderer(env_params)  # 使用原始参数（包含渲染）
            async_renderer.start()
            
            # 创建状态同步环境
            sync_env = Reacher2DEnv(**train_env_params)
            print(f"✅ 异步渲染器已启动 (PID: {async_renderer.render_process.pid})")
            
        else:
            print("🏃 单进程模式：直接渲染")
            # 单进程直接渲染
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
                            args.gamma, None, device, False, args = args)

        render_env = gym.make(args.env_name, args = args)
        render_env.seed(args.seed)
        args.env_type = 'bullet'

    num_joints = envs.action_space.shape[0]  # 这就是关节数量！
    print(f"Number of joints: {num_joints}")
    num_updates = 5
    num_step = 50000  # 从2000增加到5000，给更多学习时间
    data_handler = DataHandler(num_joints, args.env_type)



    #TODO: 需要修改，现在reacher2d是不支持rule_sequence的

    # 在第115行 data_handler = DataHandler(num_joints, args.env_type) 之后添加：

    if args.env_type == 'reacher2d':
        # 🔸 使用 Reacher2D GNN 编码器
        sys.path.append(os.path.join(os.path.dirname(__file__), '../2d_reacher/utils'))
        from reacher2d_gnn_encoder import Reacher2D_GNN_Encoder
        
        print("🤖 初始化 Reacher2D GNN 编码器...")
        reacher2d_encoder = Reacher2D_GNN_Encoder(max_nodes=20, num_joints=num_joints)
        single_gnn_embed = reacher2d_encoder.get_gnn_embeds(
            num_links=num_joints, 
            # link_lengths=[80, 80, 80, 60]  # 或者从 env_params 获取
            link_lengths = env_params['link_lengths']
        )
        print(f"✅ Reacher2D GNN 嵌入生成成功，形状: {single_gnn_embed.shape}")
    else:
        # 🔸 使用原有的 Bullet GNN 编码器
        rule_sequence = [int(s.strip(",")) for s in args.rule_sequence]
        gnn_encoder = GNN_Encoder(args.grammar_file, rule_sequence, 70, num_joints)
        gnn_graph = gnn_encoder.get_graph(rule_sequence)
        single_gnn_embed = gnn_encoder.get_gnn_embeds(gnn_graph)

    # 然后删除或注释掉原来的第117-121行：
    # rule_sequence = [int(s.strip(",")) for s in args.rule_sequence]
    # gnn_encoder = GNN_Encoder(args.grammar_file, rule_sequence, 70, num_joints)
    # gnn_graph = gnn_encoder.get_graph(rule_sequence)
    # single_gnn_embed = gnn_encoder.get_gnn_embeds(gnn_graph)
    # rule_sequence = [int(s.strip(",")) for s in args.rule_sequence]
    # gnn_encoder = GNN_Encoder(args.grammar_file, rule_sequence, 70, num_joints)
    
    # gnn_graph = gnn_encoder.get_graph(rule_sequence)
    # single_gnn_embed = gnn_encoder.get_gnn_embeds(gnn_graph)  # [1, N, D]


    action_dim = num_joints  # 使用实际的关节数，而不是硬编码12
    attn_model = AttnModel(128, 130, 130, 4)
    sac = AttentionSACWithBuffer(attn_model, action_dim, 
                                buffer_capacity=10000, batch_size=32,  # 从64减少到32
                                lr=1e-4,  # 降低学习率从3e-4到1e-4
                                env_type=args.env_type)
    
    # 🔧 重新优化SAC参数以平衡探索和利用
    sac.warmup_steps = 1000   # 从500增加到1000，更充分的探索
    sac.alpha = 0.2          # 从0.1增加到0.2，增加探索性
    if hasattr(sac, 'target_entropy'):
        sac.target_entropy = -action_dim * 0.8  # 从0.5增加到0.8，鼓励更多样化的策略
    current_obs = envs.reset()
    current_gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)  # [B, N, D]
    total_steps =0
    episode_rewards = [0.0] * args.num_processes
    eval_frequency = 200  # 增加评估间隔
    
    # 🏆 添加最佳模型保存相关变量
    best_success_rate = 0.0
    best_min_distance = float('inf')
    goal_threshold = 35.0  # 从25.0调整到35.0像素，更容易达成成功
    consecutive_success_count = 0
    min_consecutive_successes = 3  # 连续成功次数要求
    model_save_path = os.path.join(args.save_dir, 'best_models')
    os.makedirs(model_save_path, exist_ok=True)
    
    # 添加缺少的参数
    if not hasattr(args, 'update_frequency'):
        args.update_frequency = 2  # 从4减少到2，更频繁更新
    
    print(f"start training, warmup {sac.warmup_steps} steps")
    print(f"Total training steps: {num_step}, Update frequency: {args.update_frequency}")
    print(f"Expected warmup completion at step: {sac.warmup_steps}")

    try:

        for step in range(num_step):
            
            # 添加进度信息
            if step % 100 == 0:
                if step < sac.warmup_steps:
                    print(f"Step {step}/{num_step}: Warmup phase ({step}/{sac.warmup_steps})")
                else:
                    print(f"Step {step}/{num_step}: Training phase, Buffer size: {len(sac.memory)}")

                if async_renderer:
                    stats = async_renderer.get_stats()
                    print(f"   🎨 渲染FPS: {stats.get('fps', 0):.1f}")

            if step < sac.warmup_steps:  # 使用step而不是total_steps来判断预热期
                action_batch = torch.from_numpy(np.array([envs.action_space.sample() for _ in range(args.num_processes)]))
            else:
                actions = []
                for proc_id in range(args.num_processes):
                    action = sac.get_action(current_obs[proc_id],
                                            current_gnn_embeds[proc_id],
                                            num_joints = envs.action_space.shape[0],
                                            deterministic = False)
                    actions.append(action)

                action_batch = torch.stack(actions)

            # action_batch = torch.from_numpy(np.random.uniform(-100, 100, (args.num_processes, num_joints)))

            # 🔍 添加Action监控 - 每50步详细打印action值
            if step % 50 == 0 or step < 20:  # 前20步和每50步
                print(f"\n🎯 Step {step} Action Analysis:")
                action_numpy = action_batch.cpu().numpy() if hasattr(action_batch, 'cpu') else action_batch.numpy()
                
                for proc_id in range(min(args.num_processes, 2)):  # 只打印前2个进程
                    action_values = action_numpy[proc_id]
                    print(f"  Process {proc_id}: Actions = [{action_values[0]:+6.2f}, {action_values[1]:+6.2f}, {action_values[2]:+6.2f}, {action_values[3]:+6.2f}]")
                    print(f"    Max action: {np.max(np.abs(action_values)):6.2f}, Mean abs: {np.mean(np.abs(action_values)):6.2f}")
                
                # 打印action统计
                all_actions = action_numpy.flatten()
                print(f"  📊 All Actions Stats:")
                print(f"    Range: [{np.min(all_actions):+6.2f}, {np.max(all_actions):+6.2f}]")
                print(f"    Mean: {np.mean(all_actions):+6.2f}, Std: {np.std(all_actions):6.2f}")
                print(f"    Action space limit: ±{envs.action_space.high[0]:.1f}")
                
                # 检查action是否饱和
                saturated = np.sum(np.abs(all_actions) > envs.action_space.high[0] * 0.9)
                print(f"    Actions near saturation (>90% limit): {saturated}/{len(all_actions)}")

            next_obs, reward, done, infos = envs.step(action_batch)
            
            # 🔍 添加距离和位置监控 - 每步都检查
            if step % 10 == 0 or step < 30:  # 前30步和每10步监控距离
                # 🎯 获取机器人末端位置和目标距离
                if async_renderer and sync_env:
                    # 多进程模式：使用sync_env获取准确的状态信息
                    end_pos = sync_env._get_end_effector_position()
                    goal_pos = sync_env.goal_pos
                    distance = np.linalg.norm(np.array(end_pos) - goal_pos)
                    
                    print(f"  🎯 Distance Monitoring (Step {step}):")
                    print(f"    End Effector: [{end_pos[0]:7.1f}, {end_pos[1]:7.1f}]")
                    print(f"    Goal Position: [{goal_pos[0]:7.1f}, {goal_pos[1]:7.1f}]")
                    print(f"    Distance to Goal: {distance:7.1f} pixels")
                    
                    # 🏆 检查是否到达目标
                    goal_reached, current_distance = check_goal_reached(sync_env, goal_threshold)
                    if goal_reached:
                        consecutive_success_count += 1
                        print(f"    🎉 目标到达! 连续成功次数: {consecutive_success_count}")
                        
                        # 更新最佳距离
                        if current_distance < best_min_distance:
                            best_min_distance = current_distance
                        
                        # 连续成功达到要求时保存模型
                        if consecutive_success_count >= min_consecutive_successes:
                            current_success_rate = consecutive_success_count / min_consecutive_successes
                            if current_success_rate > best_success_rate:
                                best_success_rate = current_success_rate
                                save_best_model(sac, model_save_path, best_success_rate, best_min_distance, step)
                    else:
                        consecutive_success_count = 0  # 重置连续成功计数
                        
                elif args.num_processes == 1:
                    # 🔧 单进程模式：直接从envs获取状态
                    # 需要访问底层环境
                    if hasattr(envs, 'envs') and len(envs.envs) > 0:
                        base_env = envs.envs[0]
                        # 查找真正的环境实例
                        while hasattr(base_env, 'env'):
                            base_env = base_env.env
                        
                        if hasattr(base_env, '_get_end_effector_position'):
                            end_pos = base_env._get_end_effector_position()
                            goal_pos = base_env.goal_pos
                            distance = np.linalg.norm(np.array(end_pos) - goal_pos)
                            
                            print(f"  🎯 Distance Monitoring (Step {step}) [Single Process]:")
                            print(f"    End Effector: [{end_pos[0]:7.1f}, {end_pos[1]:7.1f}]")
                            print(f"    Goal Position: [{goal_pos[0]:7.1f}, {goal_pos[1]:7.1f}]")
                            print(f"    Distance to Goal: {distance:7.1f} pixels")
                            
                            # 🏆 检查是否到达目标（单进程版本）
                            goal_reached, current_distance = check_goal_reached(base_env, goal_threshold)
                            if goal_reached:
                                consecutive_success_count += 1
                                print(f"    🎉 目标到达! 连续成功次数: {consecutive_success_count}")
                                
                                if current_distance < best_min_distance:
                                    best_min_distance = current_distance
                                
                                if consecutive_success_count >= min_consecutive_successes:
                                    current_success_rate = consecutive_success_count / min_consecutive_successes
                                    if current_success_rate > best_success_rate:
                                        best_success_rate = current_success_rate
                                        save_best_model(sac, model_save_path, best_success_rate, best_min_distance, step)
                            else:
                                consecutive_success_count = 0
                    
                # 记录距离变化趋势（公共部分）
                if 'end_pos' in locals():
                    if not hasattr(main, 'prev_distances'):
                        main.prev_distances = []
                    main.prev_distances.append(distance)
                    
                    # 计算距离变化趋势（最近5步）
                    if len(main.prev_distances) >= 5:
                        recent_distances = main.prev_distances[-5:]
                        distance_trend = recent_distances[-1] - recent_distances[0]  # 正值=远离，负值=接近
                        avg_distance = np.mean(recent_distances)
                        print(f"    Distance Trend (last 5 steps): {distance_trend:+6.1f} ({'🔴 Moving Away' if distance_trend > 10 else '🟢 Getting Closer' if distance_trend < -10 else '🟡 No Clear Trend'})")
                        print(f"    Average Distance (last 5): {avg_distance:7.1f}")
                        
                        # 如果距离数据太多，保留最近50个
                        if len(main.prev_distances) > 50:
                            main.prev_distances = main.prev_distances[-50:]
            
            # 🔍 添加Reward监控 - 显示reward变化
            if step % 50 == 0 or step < 20:
                # 安全处理reward数据类型
                if hasattr(reward, 'cpu'):
                    reward_numpy = reward.cpu().numpy()
                elif hasattr(reward, 'numpy'):
                    reward_numpy = reward.numpy()
                else:
                    reward_numpy = reward
                    
                # 安全地处理reward值，转换为标量
                reward_0 = float(reward_numpy[0]) if len(reward_numpy) > 0 else 0.0
                reward_1 = float(reward_numpy[1]) if len(reward_numpy) > 1 else 0.0
                reward_min = float(np.min(reward_numpy))
                reward_max = float(np.max(reward_numpy))
                
                print(f"  💰 Rewards: [{reward_0:+7.3f}, {reward_1:+7.3f}] (Process 0, 1)")
                print(f"    Reward range: [{reward_min:+7.3f}, {reward_max:+7.3f}]")
                
                # 检查done状态
                if hasattr(done, 'cpu'):
                    done_numpy = done.cpu().numpy()
                elif hasattr(done, 'numpy'):
                    done_numpy = done.numpy()
                else:
                    done_numpy = done
                    
                done_count = np.sum(done_numpy)
                if done_count > 0:
                    print(f"  🏁 Episodes completed: {done_count}/{len(done_numpy)}")

                    # 🎯 新增：从info中获取碰撞信息
                if 'infos' in locals() and len(infos) > 0:  
                    first_env_info = infos[0]
                    if 'collisions' in first_env_info:
                        collision_info = first_env_info['collisions']
                        goal_info = first_env_info['goal']
                        
                        print(f"  💥 Collision Monitoring:")
                        print(f"    Total Collisions: {collision_info['total_count']}")
                        print(f"    Episode Collisions: {collision_info['collisions_this_episode']}")
                        print(f"    Collision Rate: {collision_info['collision_rate']:.4f}")
                        print(f"    Collision Penalty: {collision_info['collision_penalty']:.2f}")
                        
                        print(f"  🎯 Goal Monitoring:")
                        print(f"    Distance: {goal_info['distance_to_goal']:.1f} pixels")
                        print(f"    Goal Reached: {'✅' if goal_info['goal_reached'] else '❌'}")
                        
                        # 如果有奖励分解信息
                        if 'reward_breakdown' in first_env_info:
                            breakdown = first_env_info['reward_breakdown']
                            print(f"  💰 Reward Breakdown:")
                            for key, value in breakdown.items():
                                print(f"    {key}: {value:.3f}")
            
            # 🎨 异步渲染
            if async_renderer and sync_env:
                sync_action = action_batch[0].cpu().numpy() if hasattr(action_batch, 'cpu') else action_batch[0]
                
                # 🔧 关键修复：确保渲染环境也使用缩放后的action！
                # SAC在get_action中已经缩放了action，但这里需要确认
                if args.env_type == 'reacher2d':
                    # action_batch已经是缩放后的值，直接使用
                    print(f"  🎨 渲染Action: [{sync_action[0]:+6.1f}, {sync_action[1]:+6.1f}, {sync_action[2]:+6.1f}, {sync_action[3]:+6.1f}]")
                
                sync_env.step(sync_action)
                robot_state = StateExtractor.extract_robot_state(sync_env, step)
                async_renderer.render_frame(robot_state)
            #TODO: 需要修改 成更灵活的gnn_embeds
            next_gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)  # [B, N, D]

            for proc_id in range(args.num_processes):
                sac.store_experience(
                        obs = current_obs[proc_id],
                        gnn_embeds = current_gnn_embeds[proc_id],
                        action = action_batch[proc_id],
                        reward = reward[proc_id],
                        next_obs = next_obs[proc_id],
                        next_gnn_embeds = next_gnn_embeds[proc_id],
                        done = done[proc_id],
                        num_joints = num_joints
                )
                episode_rewards[proc_id] += reward[proc_id].item()  # 将tensor转换为标量

            current_obs = next_obs.clone()
            current_gnn_embeds = next_gnn_embeds.clone()


            for proc_id in range(args.num_processes):
                # 安全检查done状态
                is_done = done[proc_id].item() if torch.is_tensor(done[proc_id]) else bool(done[proc_id])
                if is_done:
                    print(f"Episode {step} finished with reward {episode_rewards[proc_id]:.2f}")


                    episode_rewards[proc_id] = 0.0
                    
                    # 重置环境（如果需要）
                    if hasattr(envs, 'reset_one'):
                        current_obs[proc_id] = envs.reset_one(proc_id)
                        current_gnn_embeds[proc_id] = single_gnn_embed

            if (step >= sac.warmup_steps and 
                step % args.update_frequency == 0 and 
                sac.memory.can_sample(sac.batch_size)):
                
                metrics = sac.update()
                
                if metrics and step % 100 == 0:
                    print(f"Step {step} (total_steps {total_steps}): "
                        f"Critic Loss: {metrics['critic_loss']:.4f}, "
                        f"Actor Loss: {metrics['actor_loss']:.4f}, "
                        f"Alpha: {metrics['alpha']:.4f}, "
                        f"Buffer Size: {len(sac.memory)}")
                    
                    # 🏆 添加训练状态报告
                    print(f"  🎯 Training Status:")
                    print(f"    Best Success Rate: {best_success_rate:.3f}")
                    print(f"    Best Min Distance: {best_min_distance:.1f} pixels")
                    print(f"    Consecutive Successes: {consecutive_success_count}")
                    print(f"    Goal Threshold: {goal_threshold:.1f} pixels")
                    
                    # 添加详细的loss分析
                    if 'entropy_term' in metrics:
                        print(f"  Actor Loss 组件分析:")
                        print(f"    Entropy Term (α*log_π): {metrics['entropy_term']:.4f}")
                        print(f"    Q Term (Q值): {metrics['q_term']:.4f}")
                        print(f"    Actor Loss = {metrics['entropy_term']:.4f} - {metrics['q_term']:.4f} = {metrics['actor_loss']:.4f}")
                        
                        if metrics['actor_loss'] < 0:
                            print(f"    ✓ 负数Actor Loss = 高Q值 = 好的策略!")
                        else:
                            print(f"    ⚠ 正数Actor Loss = 低Q值 = 策略需要改进")
            
            # 🏆 定期保存检查点模型（不管是否到达目标）
            if step % 1000 == 0 and step > 0:
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
            
            total_steps += args.num_processes  # 并行环境步数累加

    except Exception as e:
        print(f"🔴 训练过程中发生错误: {e}")
        raise e

    finally:
        # 清理资源
        if 'async_renderer' in locals() and async_renderer:
            async_renderer.stop()
        if 'sync_env' in locals() and sync_env:
            sync_env.close()
            
        # 🏆 训练结束时保存最终模型
        print(f"\n{'='*60}")
        print(f"🏁 训练完成总结:")
        print(f"  总步数: {step}")
        print(f"  最佳成功率: {best_success_rate:.3f}")
        print(f"  最佳最小距离: {best_min_distance:.1f} pixels")
        print(f"  当前连续成功次数: {consecutive_success_count}")
        
        # 保存最终模型
        final_model_path = os.path.join(model_save_path, f'final_model_step_{step}.pth')
        final_model_data = {
            'step': step,
            'final_success_rate': best_success_rate,
            'final_min_distance': best_min_distance,
            'final_consecutive_successes': consecutive_success_count,
            'training_completed': True,
            'actor_state_dict': sac.actor.state_dict(),
            'critic1_state_dict': sac.critic1.state_dict(),
            'critic2_state_dict': sac.critic2.state_dict(),
            'target_critic1_state_dict': sac.target_critic1.state_dict(),
            'target_critic2_state_dict': sac.target_critic2.state_dict(),
        }
        torch.save(final_model_data, final_model_path)
        print(f"💾 保存最终模型: {final_model_path}")
        print(f"{'='*60}")
        
        # 8. 定期评估
        # if step % eval_frequency == 0:
        #     eval_reward = evaluate_sac(envs, sac, single_gnn_embed, args)
        #     print(f"Evaluation at step {total_steps}: Average reward = {eval_reward:.2f}")

# def evaluate_sac(envs, sac, single_gnn_embed, args, num_episodes=5):
#         """评估SAC性能"""
#         total_rewards = []
        
#         for episode in range(num_episodes):
#             gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)
#             episode_reward = 0
#             done = False
            
#             while not done:
#                 # 使用确定性策略
#                 actions = []
#                 for proc_id in range(args.num_processes):
#                     action = sac.get_action(obs[proc_id], gnn_embeds[proc_id], deterministic=True)
#                     actions.append(action)
                
#                 action_batch = torch.stack(actions)
#                 obs, rewards, dones, _ = envs.step(action_batch)
#                 episode_reward += rewards.mean().item()
#                 done = dones.any().item()
            
#             total_rewards.append(episode_reward)
        
#         return np.mean(total_rewards)


# def train(args):
#     torch.manual_seed(args.seed)
#     torch.set_num_threads(1)
#     device = torch.device('cpu')

#     os.makedirs(args.save_dir, exist_ok = True)

#     training_log_path = os.path.join(args.save_dir, 'logs.txt')
#     fp_log = open(training_log_path, 'w')
#     fp_log.close()

#     envs = make_vec_envs(args.env_name, args.seed, args.num_processes, 
#                         args.gamma, None, device, False, args = args)

#     render_env = gym.make(args.env_name, args = args)
#     render_env.seed(args.seed)
#     num_joints = envs.action_space.shape[0]  # 这就是关节数量！
#     print(f"Number of joints: {num_joints}")
#     obs = envs.reset()
#     num_updates = 5
#     num_step = 1000
    
#     data_handler = DataHandler(num_joints)
#     rule_sequence = [int(s.strip(",")) for s in args.rule_sequence]
#     gnn_encoder = GNN_Encoder(args.grammar_file, rule_sequence, 70, num_joints)
    
#     gnn_graph = gnn_encoder.get_graph(rule_sequence)
#     single_gnn_embed = gnn_encoder.get_gnn_embeds(gnn_graph)  # [1, N, D]
#     attn_model = AttnModel()
#     for i in range(num_updates):
#         for step in range(num_step):
#             action = torch.from_numpy(np.array([envs.action_space.sample() for _ in range(args.num_processes)]))
#             obs, reward, done, infos = envs.step(action)
           
#             gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)  # [B, N, D]
#             data_handler.save_data(obs, action, reward, gnn_embeds, done)

#         data_loader = data_handler.get_data_loader()
#     print(data_handler.get_data_length())

#     for batch in data_loader:
#         joint_q = batch["joint_q"]
#         vertex_k = batch["vertex_k"]
#         vertex_v = batch["vertex_v"]
#         vertex_mask = batch["vertex_mask"]
#         attn_output = attn_model(joint_q, vertex_k, vertex_v, vertex_mask)


# if __name__ == "__main__":

#     torch.set_default_dtype(torch.float64)
#     args_list = ['--env-name', 'RobotLocomotion-v0',
#                  '--task', 'FlatTerrainTask',
#                  '--grammar-file', '../../data/designs/grammar_jan21.dot',
#                  '--algo', 'ppo',
#                  '--use-gae',
#                  '--log-interval', '5',
#                  '--num-steps', '1024',
#                  '--num-processes', '8',
#                  '--lr', '3e-4',
#                  '--entropy-coef', '0',
#                  '--value-loss-coef', '0.5',
#                  '--ppo-epoch', '10',
#                  '--num-mini-batch', '32',
#                  '--gamma', '0.995',
#                  '--gae-lambda', '0.95',
#                  '--num-env-steps', '30000000',
#                  '--use-linear-lr-decay',
#                  '--use-proper-time-limits',
#                  '--save-interval', '100',
#                  '--seed', '2',
#                  '--save-dir', './trained_models/RobotLocomotion-v0/test/',
#                  '--render-interval', '80']
#     parser = get_parser()
#     args = parser.parse_args(args_list + sys.argv[1:])

        
#     solve_argv_conflict(args_list)
#     parser = get_parser()
#     args = parser.parse_args(args_list + sys.argv[1:])

#     args.cuda = not args.no_cuda and torch.cuda.is_available()

#     args.save_dir = os.path.join(args.save_dir, get_time_stamp())
#     try:
#         os.makedirs(args.save_dir, exist_ok = True)
#     except OSError:
#         pass

#     fp = open(os.path.join(args.save_dir, 'args.txt'), 'w')
#     fp.write(str(args_list + sys.argv[1:]))
#     fp.close()

#     # train(args)
#     main(args)


 
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    
    # 🎯 检查是否要测试 Reacher2D
    if len(sys.argv) > 1 and sys.argv[1] == '--test-reacher2d':
        print("🤖 启动 Reacher2D 环境测试")
        args_list = ['--env-name', 'reacher2d',
                     '--num-processes', '2',  # 恢复多进程便于并行训练
                     '--lr', '3e-4',
                     '--gamma', '0.99',
                     '--seed', '42',
                     '--save-dir', './trained_models/reacher2d/test/',
                     '--grammar-file', '/home/xli149/Documents/repos/RoboGrammar/data/designs/grammar_jan21.dot',
                     '--rule-sequence', '0']
        # 移除测试标志，避免解析错误
        test_args = sys.argv[2:] if len(sys.argv) > 2 else []
    else:
        # 原来的 RobotLocomotion-v0 配置
        args_list = ['--env-name', 'RobotLocomotion-v0',
                     '--task', 'FlatTerrainTask',
                     '--grammar-file', '../../data/designs/grammar_jan21.dot',
                     '--algo', 'ppo',
                     '--use-gae',
                     '--log-interval', '5',
                     '--num-steps', '1024',
                     '--num-processes', '2',  # 恢复多进程便于并行训练
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
                     '--save-dir', './trained_models/RobotLocomotion-v0/test/',
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