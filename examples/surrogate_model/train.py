import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
sys.path.append(base_dir)

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

def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device('cpu')

    os.makedirs(args.save_dir, exist_ok = True)

    training_log_path = os.path.join(args.save_dir, 'logs.txt')
    fp_log = open(training_log_path, 'w')
    fp_log.close()

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes, 
                        args.gamma, None, device, False, args = args)

    render_env = gym.make(args.env_name, args = args)
    render_env.seed(args.seed)
    num_joints = envs.action_space.shape[0]  # 这就是关节数量！
    print(f"Number of joints: {num_joints}")
    obs = envs.reset()
    num_updates = 5
    num_step = 1000
    data_handler = DataHandler(num_joints)
    rule_sequence = [int(s.strip(",")) for s in args.rule_sequence]
    gnn_encoder = GNN_Encoder(args.grammar_file, rule_sequence, 70, num_joints)
    
    gnn_graph = gnn_encoder.get_graph(rule_sequence)
    single_gnn_embed = gnn_encoder.get_gnn_embeds(gnn_graph)  # [1, N, D]
    attn_model = AttnModel()
    for i in range(num_updates):
        for step in range(num_step):
            action = torch.from_numpy(np.array([envs.action_space.sample() for _ in range(args.num_processes)]))
            obs, reward, done, infos = envs.step(action)
           
            gnn_embeds = single_gnn_embed.repeat(args.num_processes, 1, 1)  # [B, N, D]
            data_handler.save_data(obs, action, reward, gnn_embeds, done)

        data_loader = data_handler.get_data_loader()
    print(data_handler.get_data_length())

    for batch in data_loader:
        joint_q = batch["joint_q"]
        vertex_k = batch["vertex_k"]
        vertex_v = batch["vertex_v"]
        vertex_mask = batch["vertex_mask"]
        attn_output = attn_model(joint_q, vertex_k, vertex_v, vertex_mask)
        print(attn_output.shape)


if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)
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
    parser = get_parser()
    args = parser.parse_args(args_list + sys.argv[1:])

        
    solve_argv_conflict(args_list)
    parser = get_parser()
    args = parser.parse_args(args_list + sys.argv[1:])

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.save_dir = os.path.join(args.save_dir, get_time_stamp())
    try:
        os.makedirs(args.save_dir, exist_ok = True)
    except OSError:
        pass

    fp = open(os.path.join(args.save_dir, 'args.txt'), 'w')
    fp.write(str(args_list + sys.argv[1:]))
    fp.close()

    train(args)