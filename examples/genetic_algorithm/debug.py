import copy
import time
import uuid
import pyrobotdesign as rd  
import random
import os
import sys
from typing import List, Dict, Set, Tuple, Optional

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'graph_learning'))
sys.path.append(os.path.join(base_dir, 'design_search'))

from design_search import make_graph, build_normalized_robot, presimulate, get_applicable_matches, has_nonterminals, make_initial_graph

def debug_rule_sequence(rule_sequence):
    """调试规则序列的应用过程"""
    print(f"\n=== 调试规则序列 {rule_sequence} ===")
    
    try:
        # 加载规则
        graphs = rd.load_graphs("data/designs/grammar_apr30.dot")
        rules = [rd.create_rule_from_graph(g) for g in graphs]
        print(f"加载了 {len(rules)} 个规则")
        
        # 创建初始图 - 使用正确的函数
        graph = make_initial_graph()
        print(f"初始图: {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")
        
        # 逐步应用规则
        for i, rule_idx in enumerate(rule_sequence):
            print(f"\n--- 步骤 {i+1}: 应用规则 {rule_idx} ---")
            
            if rule_idx >= len(rules):
                print(f"错误: 规则索引 {rule_idx} 超出范围 (最大: {len(rules)-1})")
                return False
            
            rule = rules[rule_idx]
            print(f"规则名称: {rule.name}")
            
            # 检查匹配
            matches = list(get_applicable_matches(rule, graph))
            print(f"找到 {len(matches)} 个匹配")
            
            if not matches:
                print("错误: 没有找到匹配")
                print("当前图状态:")
                print_graph_state(graph)
                return False
            
            # 应用规则
            graph = rd.apply_rule(rule, graph, matches[0])
            print(f"应用后: {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")
            print_graph_state(graph)
        
        # 检查最终状态
        print(f"\n=== 最终检查 ===")
        has_nt = has_nonterminals(graph)
        print(f"包含非终结符: {has_nt}")
        
        if not has_nt:
            try:
                robot = build_normalized_robot(graph)
                print(f"成功构建机器人: {len(robot.links)} 个连杆")
                return True
            except Exception as e:
                print(f"构建机器人失败: {e}")
                return False
        else:
            print("图仍包含非终结符，需要更多规则")
            return False
            
    except Exception as e:
        print(f"调试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_graph_state(graph):
    """打印图的状态"""
    print("节点:")
    for i, node in enumerate(graph.nodes):
        print(f"  {i}: {node.name} (label={node.attrs.label}, shape={node.attrs.shape})")
    
    print("边:")
    for i, edge in enumerate(graph.edges):
        print(f"  {i}: {edge.tail}->{edge.head} (joint_type={edge.attrs.joint_type})")

def find_valid_sequences():
    """寻找有效的规则序列"""
    print("=== 寻找有效的规则序列 ===")
    
    # 一些基本的有效序列
    test_sequences = [
        [0, 3, 6, 7],  # 最简单：robot -> body(无腿) -> 终结tail和head
        [0, 3, 6, 7, 10],  # 添加关节
        [0, 2, 8, 5, 6, 7],  # 带腿版本：robot -> body(有腿) -> 肢体 -> 终结
        [0, 2, 8, 5, 6, 7, 14],  # 添加关节
    ]
    
    valid_sequences = []
    for seq in test_sequences:
        print(f"\n测试序列: {seq}")
        if debug_rule_sequence(seq):
            valid_sequences.append(seq)
            print("✓ 有效")
        else:
            print("✗ 无效")
    
    return valid_sequences

if __name__ == "__main__":
    # 首先找到一些有效的序列
    valid_sequences = find_valid_sequences()
    
    print(f"\n找到 {len(valid_sequences)} 个有效序列:")
    for seq in valid_sequences:
        print(f"  {seq}")