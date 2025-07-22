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
from Net import Net
import pyrobotdesign as rd
from RobotGrammarEnv import RobotGrammarEnv
from tasks import FlatTerrainTask
import time
from design_search import RobotDesignEnv, make_graph, build_normalized_robot, presimulate, simulate
# class GNN_Encoder(nn.Module):
#     def __init__(self):
#         pass
#     def encode(self, robot_graph):
#         pass


def encode_graph(robot_graph):
    adj_matrix, features, masks = preprocessor.preprocess(robot_graph)
    return adj_matrix, features, masks




if __name__ == "__main__":

  
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
    sample_adj_matrix, sample_features, _ = encode_graph(graph)
    print(f'graph: {graph}')
    print(f'sample_adj_matrix: {sample_adj_matrix}')
    print(f'sample_features: {sample_features}')