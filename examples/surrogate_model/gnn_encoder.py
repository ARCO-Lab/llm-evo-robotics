### TODO: This file is used to encode the robot design into a graph structure
import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'graph_learning'))
sys.path.append(os.path.join(base_dir, 'design_search'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from Preprocessor import Preprocessor
from Net import GNN
import pyrobotdesign as rd
from RobotGrammarEnv import RobotGrammarEnv
from tasks import FlatTerrainTask
import time
from design_search import RobotDesignEnv, make_graph, build_normalized_robot, presimulate, simulate
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

def feature_extraction(robot_graph):
    adj_matrix, features, masks = preprocessor.preprocess(robot_graph)
    return adj_matrix, features, masks



class Graph_Net(torch.nn.Module):
    def __init__(self, max_nodes, num_channels, num_outputs, max_joints=20):
        super(Graph_Net, self).__init__()
        
        batch_normalization = False

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(num_channels, 64, num_nodes, batch_normalization = batch_normalization, add_loop=True)
        self.gnn1_embed = GNN(num_channels, 64, 64, batch_normalization = batch_normalization, add_loop=True, lin=False)
        
        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes, batch_normalization = batch_normalization)
        self.gnn2_embed = GNN(3 * 64, 64, 64, batch_normalization = batch_normalization, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, batch_normalization = batch_normalization, lin=False)

        # 结构信息处理
        self.struct_dim = 128
        self.struct_processor = torch.nn.Linear(3 * 64, self.struct_dim)
        
        # 改为动态生成查询向量的网络
        self.query_generator = torch.nn.Linear(self.struct_dim, self.struct_dim)
        self.max_joints = max_joints
        
        # 注意力机制
        self.attention = torch.nn.MultiheadAttention(embed_dim=self.struct_dim, num_heads=8, batch_first=True)
        
        # 最终输出层
        self.output_projection = torch.nn.Linear(self.struct_dim, 1)
        
    def forward(self, x, adj, mask=None, num_joints=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)  # [batch, nodes, 192]
        
        # 处理结构信息
        global_struct = x.mean(dim=1)  # [batch, 192]
        processed_struct = F.relu(self.struct_processor(global_struct))  # [batch, 128]
        
        # 动态确定关节数量
        if num_joints is None:
            # 如果没有提供，可以从图结构推断或使用默认值
            num_joints = min(x.size(1), self.max_joints)  # 使用节点数或最大关节数
        
        batch_size = processed_struct.size(0)
        
        # 动态生成查询向量
        base_query = self.query_generator(processed_struct)  # [batch, 128]
        
        # 为每个关节生成稍微不同的查询
        joint_indices = torch.arange(num_joints, device=x.device, dtype=torch.float32)
        joint_indices = joint_indices.unsqueeze(0).unsqueeze(2) / num_joints  # [1, num_joints, 1]
        
        # 广播并添加位置编码
        queries = base_query.unsqueeze(1).repeat(1, num_joints, 1)  # [batch, num_joints, 128]
        position_encoding = joint_indices.repeat(batch_size, 1, self.struct_dim)
        queries = queries + 0.1 * position_encoding  # 添加轻微的位置扰动
        
        # 键值都是处理过的结构信息
        keys = values = processed_struct.unsqueeze(1)  # [batch, 1, 128]
        
        # 注意力计算
        joint_embeds, attention_weights = self.attention(queries, keys, values)  # [batch, num_joints, 128]
        
        # 生成最终输出
        joint_outputs = self.output_projection(joint_embeds).squeeze(-1)  # [batch, num_joints]
        
        return joint_outputs, l1 + l2, e1 + e2, joint_embeds


class GNN_Encoder:
    def __init__(self, grammar_file, rule_sequence, max_nodes, num_joints):
        self.grammar_file = grammar_file
        self.rule_sequence = rule_sequence
        self.max_nodes = max_nodes
        self.num_joints = num_joints
        self.graphs =  rd.load_graphs(grammar_file)
        self.rules = self.get_rules()
        self.all_labels = self.get_all_labels()
        self.preprocessor = Preprocessor(all_labels = self.all_labels, max_nodes = self.max_nodes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn = Graph_Net(max_nodes = self.max_nodes, num_channels = 41, num_outputs = 1).to(self.device)
        self.gnn.eval()
    
    def get_rules(self):
        return [rd.create_rule_from_graph(g) for g in self.graphs]
    
    def get_all_labels(self):
        all_labels = set()
        for rule in self.rules:
            for node in rule.lhs.nodes:
                all_labels.add(node.attrs.require_label)
        all_labels = sorted(list(all_labels))
        return all_labels
    def get_graph(self, rule_sequence):
        return make_graph(self.rules, rule_sequence)
    def get_gnn_embeds(self, robot_graph):
        adj_matrix, features, masks = self.preprocessor.preprocess(robot_graph)
        masks_np = np.full(len(features), True)
       
        with torch.no_grad():
            features = torch.tensor(features).unsqueeze(0)
            adj_matrix = torch.tensor(adj_matrix).unsqueeze(0)
            masks = torch.tensor(masks_np).unsqueeze(0)
            x, link_loss, entropy_loss, gnn_embed = self.gnn(features, adj_matrix, masks, num_joints = self.num_joints)
        return gnn_embed
            

if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)
    grammar_file = "data/designs/grammar_apr30.dot"
    rule_sequence = [0, 7, 1, 13, 1, 2, 16, 12, 13, 6, 4, 19, 4, 17, 5, 3, 2, 16, 4, 5, 18, 9, 8, 9, 9, 8]
    max_nodes = 80
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs =  rd.load_graphs(grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]
    all_labels = set()
    for rule in rules:
        for node in rule.lhs.nodes:
            all_labels.add(node.attrs.require_label)
    all_labels = sorted(list(all_labels))
    global preprocessor
    preprocessor = Preprocessor(all_labels = all_labels)

    graph = make_graph(rules, rule_sequence)
    print(f'graph: {graph}')
    adj_matrix, features, _ = feature_extraction(graph)
    masks_np = np.full(len(features), True)

    
    print(f'adj_matrix: {adj_matrix}')
    print(f'features: {features}')

    gnn = Graph_Net(max_nodes = max_nodes, num_channels = features.shape[1], num_outputs = 1).to(device)
    gnn.eval()
    with torch.no_grad():
        features = torch.tensor(features).unsqueeze(0)
        adj_matrix = torch.tensor(adj_matrix).unsqueeze(0)
        masks = torch.tensor(masks_np).unsqueeze(0)
        x, link_loss, entropy_loss, gnn_embed = gnn(features, adj_matrix, masks)
    print(f'x: {x}')
    print(f'link_loss: {link_loss}')
    print(f'entropy_loss: {entropy_loss}')
    print(f'gnn_embed: {gnn_embed}')
