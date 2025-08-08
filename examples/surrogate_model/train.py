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

from env_config.env_wrapper import make_reacher2d_vec_envs
from reacher2d_env import Reacher2DEnv


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


        envs = make_reacher2d_vec_envs(
            env_params = env_params,
            seed = args.seed,
            num_processes = args.num_processes,
            gamma = args.gamma,
            log_dir = None,
            device = device,
            allow_early_resets = False
        )

        print(f"✅ 多进程向量化环境创建成功")

        render_env = Reacher2DEnv(
            num_links = env_params['num_links'],
            link_lengths = env_params['link_lengths'],
            render_mode = env_params['render_mode'],
            config_path = env_params['config_path']

        )
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
    num_step = 5000  # 减少总训练步数，方便调试
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
    attn_model = AttnModel(128, 128, 130, 4)
    sac = AttentionSACWithBuffer(attn_model, action_dim, buffer_capacity=10000, batch_size=64, env_type=args.env_type)
    
    # 手动设置较短的预热期
    sac.warmup_steps = 500  # 预热500步，约为总步数的10%
    current_obs = envs.reset()
    current_gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)  # [B, N, D]
    total_steps =0
    episode_rewards = [0.0] * args.num_processes
    eval_frequency = 200  # 增加评估间隔
    # 添加缺少的参数
    if not hasattr(args, 'update_frequency'):
        args.update_frequency = 4
    
    print(f"start training, warmup {sac.warmup_steps} steps")
    print(f"Total training steps: {num_step}, Update frequency: {args.update_frequency}")
    print(f"Expected warmup completion at step: {sac.warmup_steps}")

    for step in range(num_step):
        
        # 添加进度信息
        if step % 100 == 0:
            if step < sac.warmup_steps:
                print(f"Step {step}/{num_step}: Warmup phase ({step}/{sac.warmup_steps})")
            else:
                print(f"Step {step}/{num_step}: Training phase, Buffer size: {len(sac.memory)}")

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


        next_obs, reward, done, infos = envs.step(action_batch)
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
        
        total_steps += args.num_processes  # 并行环境步数累加
        
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
                     '--num-processes', '2',  # 减少进程数便于调试
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
                     '--num-processes', '8',
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