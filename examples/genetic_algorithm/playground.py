import pyrobotdesign as rd
from design_search import make_graph, build_normalized_robot, presimulate, has_nonterminals
import os
import sys
import copy
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir,'graph_learning'))
sys.path.append(os.path.join(base_dir, 'design_search'))
from RobotGrammarEnv import RobotGrammarEnv


graphs = rd.load_graphs("data/designs/grammar_apr30.dot")
rules = [rd.create_rule_from_graph(g) for g in graphs]
rule_sequence = [ 0, 7, 1, 13, 1, 2, 16, 12, 13, 6, 4, 19, 4, 17, 5, 3, 2, 16, 4, 5, 9, 18, 9, 8, 9, 9, 8]
task = "FlatTerrainTask"
seed = 42
mpc_num_processes = 8
env = RobotGrammarEnv(task, rules, seed = seed, mpc_num_processes = mpc_num_processes)
copy_rule = rule_sequence.copy()
graph_test = make_graph(rules, copy_rule)

print(env.get_available_actions(graph_test))



