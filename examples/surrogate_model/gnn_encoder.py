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
    def __init__(self, max_nodes, num_channels, num_outputs):
        super(Graph_Net, self).__init__()
        
        batch_normalization = False

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(num_channels, 64, num_nodes, batch_normalization = batch_normalization, add_loop=True)
        self.gnn1_embed = GNN(num_channels, 64, 64, batch_normalization = batch_normalization, add_loop=True, lin=False)
        
        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes, batch_normalization = batch_normalization)
        self.gnn2_embed = GNN(3 * 64, 64, 64, batch_normalization = batch_normalization, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, batch_normalization = batch_normalization, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, num_outputs)
        
    def forward(self, x, adj, mask=None):
        
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)
        gnn_embed = x.mean(dim=1)
        x = F.relu(self.lin1(gnn_embed))

        x = self.lin2(x)
        
        return x, l1 + l2, e1 + e2, gnn_embed


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
  
    max_nodes = 80
    grammar_file = "data/designs/grammar_apr30.dot"
    task = FlatTerrainTask()
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs =  rd.load_graphs(grammar_file)
    rules = [rd.create_rule_from_graph(g) for g in graphs]
    rule_sequence = [0, 7, 1, 13, 1, 2, 16, 12, 13, 6, 4, 19, 4, 17, 5, 3, 2, 16, 4, 5, 18, 9, 8, 9, 9, 8]
    num_iterations = 1000
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
