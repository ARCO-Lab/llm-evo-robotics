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

class RobotIndividual:
    # 根据调试结果定义的规则分类
    RULE_CATEGORIES = {
        'start': [0],  # make_robot - 必须第一个
        'body_type': [2, 3],  # make_body_with_legs, make_body_without_legs
        'limb_shape': [8, 9],  # make_normal_limb_link, make_long_limb_link
        'termination': [5, 6, 7],  # end_limb, end_tail, end_head
        'body_joints': [10, 11, 12, 13],  # 身体关节类型
        'limb_joints': [14, 15, 16, 17, 18, 19]  # 肢体关节类型
    }
    
    # 有效的完整序列模板
    VALID_TEMPLATES = [
        # 简单机器人模板
        {
            'pattern': [0, 3, 6, 7],
            'description': '简单单体机器人',
            'extendable': True,
            'extension_points': [(2, 'body_joints')]  # 在位置2之后可以插入身体关节
        },
        # 带肢体的机器人模板
        {
            'pattern': [0, 2, 8, 5, 6, 7, 14],
            'description': '带单肢体机器人',
            'extendable': True,
            'extension_points': [
                (3, 'limb_shape'),  # 可以改变肢体形状
                (6, 'limb_joints')  # 可以改变关节类型
            ]
        }
    ]
    
    # 规则依赖和约束
    RULE_CONSTRAINTS = {
        # 规则0之后必须选择身体类型
        0: {'must_follow': [2, 3]},
        # 规则2之后需要处理肢体
        2: {'enables': [4, 8, 9], 'requires_termination': [5]},
        # 规则3之后可以直接终结
        3: {'can_terminate': True},
        # 肢体相关规则
        8: {'requires_termination': [5]},  # limb_link需要终结limb
        9: {'requires_termination': [5]},
        # 终结规则
        5: {'terminates': 'limb'},
        6: {'terminates': 'tail'},
        7: {'terminates': 'head'},
        # 关节规则只能在有边的情况下应用
        10: {'requires': 'body_edge'},
        11: {'requires': 'body_edge'},
        12: {'requires': 'body_edge'},
        13: {'requires': 'body_edge'},
        14: {'requires': 'limb_edge'},
        15: {'requires': 'limb_edge'},
        16: {'requires': 'limb_edge'},
        17: {'requires': 'limb_edge'},
        18: {'requires': 'limb_edge'},
        19: {'requires': 'limb_edge'},
    }

    def __init__(self, chromosome: List[int] = None):
        if chromosome is None:
            # 随机选择一个有效模板
            template = random.choice(self.VALID_TEMPLATES)
            self.chromosome = template['pattern'].copy()
        else:
            self.chromosome = chromosome.copy()
        
        self.fitness = 0
        self.number_of_rules = 20  # 根据调试结果，有20个规则
        self.id = str(uuid.uuid4())
        
        # 确保有效性
        self._ensure_valid_chromosome()
        
        # 加载语法规则（缓存）
        if not hasattr(RobotIndividual, '_rules_cache'):
            RobotIndividual._rules_cache = self._load_rules()
    
    def _load_rules(self):
        """加载并缓存语法规则"""
        try:
            graphs = rd.load_graphs("data/designs/grammar_apr30.dot")
            rules = [rd.create_rule_from_graph(g) for g in graphs]
            return rules
        except Exception as e:
            print(f"警告: 无法加载规则文件: {e}")
            return []
    
    def _ensure_valid_chromosome(self):
        """确保染色体是有效的"""
        if not self.chromosome or self.chromosome[0] != 0:
            # 如果没有以规则0开始，使用默认模板
            self.chromosome = [0, 3, 6, 7]
    
    def is_valid_sequence(self, sequence: List[int]) -> bool:
        """检查规则序列是否有效"""
        if not sequence or sequence[0] != 0:
            return False
        
        try:
            graph = make_initial_graph()
            rules = self._rules_cache
            
            if not rules:
                return False
            
            # 逐步应用规则
            for rule_idx in sequence:
                if rule_idx >= len(rules):
                    return False
                
                rule = rules[rule_idx]
                matches = list(get_applicable_matches(rule, graph))
                
                if not matches:
                    return False
                
                graph = rd.apply_rule(rule, graph, matches[0])
            
            # 检查最终状态
            if has_nonterminals(graph):
                return False
            
            # 尝试构建机器人
            try:
                robot = build_normalized_robot(graph)
                return len(robot.links) > 0
            except:
                return False
                
        except Exception:
            return False
    
    def complete_sequence(self, partial_sequence: List[int]) -> List[int]:
        """尝试完成不完整的序列"""
        if not partial_sequence or partial_sequence[0] != 0:
            return [0, 3, 6, 7]  # 返回默认有效序列
        
        # 如果已经是有效序列，直接返回
        if self.is_valid_sequence(partial_sequence):
            return partial_sequence
        
        # 尝试添加必要的终结规则
        completed = partial_sequence.copy()
        
        # 检查当前状态需要什么终结规则
        try:
            graph = make_initial_graph()
            rules = self._rules_cache
            
            if not rules:
                return [0, 3, 6, 7]
            
            # 应用现有规则
            for rule_idx in completed:
                if rule_idx < len(rules):
                    rule = rules[rule_idx]
                    matches = list(get_applicable_matches(rule, graph))
                    if matches:
                        graph = rd.apply_rule(rule, graph, matches[0])
            
            # 检查需要什么终结规则
            termination_needed = []
            for node in graph.nodes:
                if node.attrs.label == 'head':
                    termination_needed.append(7)
                elif node.attrs.label == 'tail':
                    termination_needed.append(6)
                elif node.attrs.label == 'limb':
                    termination_needed.append(5)
            
            # 添加终结规则
            for term_rule in termination_needed:
                if term_rule not in completed:
                    completed.append(term_rule)
            
            # 如果仍然无效，返回默认序列
            if not self.is_valid_sequence(completed):
                return [0, 3, 6, 7]
            
            return completed
            
        except Exception:
            return [0, 3, 6, 7]
    
    def smart_mutate(self):
        """智能变异：考虑规则依赖关系"""
        if len(self.chromosome) <= 1:
            return
        
        mutation_strategies = [
            self._mutate_replace_compatible,
            self._mutate_insert_compatible,
            self._mutate_remove_safe,
            self._mutate_swap_order
        ]
        
        # 随机选择变异策略
        strategy = random.choice(mutation_strategies)
        new_chromosome = strategy()
        
        # 如果变异后的序列有效，则使用它
        if self.is_valid_sequence(new_chromosome):
            self.chromosome = new_chromosome
        else:
            # 否则尝试完成序列
            completed = self.complete_sequence(new_chromosome)
            if self.is_valid_sequence(completed):
                self.chromosome = completed
    
    def _mutate_replace_compatible(self) -> List[int]:
        """替换为兼容的规则"""
        new_chromosome = self.chromosome.copy()
        
        # 不能替换第一个规则（必须是0）
        if len(new_chromosome) > 1:
            pos = random.randint(1, len(new_chromosome) - 1)
            current_rule = new_chromosome[pos]
            
            # 根据当前规则类型选择替换
            compatible_rules = []
            for category, rules in self.RULE_CATEGORIES.items():
                if current_rule in rules:
                    compatible_rules.extend([r for r in rules if r != current_rule])
                    break
            
            if compatible_rules:
                new_chromosome[pos] = random.choice(compatible_rules)
        
        return new_chromosome
    
    def _mutate_insert_compatible(self) -> List[int]:
        """插入兼容的规则"""
        new_chromosome = self.chromosome.copy()
        
        # 在随机位置插入规则
        pos = random.randint(1, len(new_chromosome))
        
        # 选择可能的规则
        possible_rules = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        new_rule = random.choice(possible_rules)
        
        new_chromosome.insert(pos, new_rule)
        return new_chromosome
    
    def _mutate_remove_safe(self) -> List[int]:
        """安全地移除规则"""
        if len(self.chromosome) <= 4:  # 保持最小长度
            return self.chromosome.copy()
        
        new_chromosome = self.chromosome.copy()
        
        # 不能移除关键规则（0, 终结规则）
        removable_positions = []
        for i, rule in enumerate(new_chromosome):
            if i > 0 and rule not in [0, 5, 6, 7]:  # 保留起始和终结规则
                removable_positions.append(i)
        
        if removable_positions:
            pos = random.choice(removable_positions)
            new_chromosome.pop(pos)
        
        return new_chromosome
    
    def _mutate_swap_order(self) -> List[int]:
        """交换规则顺序"""
        if len(self.chromosome) <= 2:
            return self.chromosome.copy()
        
        new_chromosome = self.chromosome.copy()
        
        # 不交换第一个规则
        pos1 = random.randint(1, len(new_chromosome) - 1)
        pos2 = random.randint(1, len(new_chromosome) - 1)
        
        if pos1 != pos2:
            new_chromosome[pos1], new_chromosome[pos2] = new_chromosome[pos2], new_chromosome[pos1]
        
        return new_chromosome
    
    def smart_crossover(self, other: 'RobotIndividual') -> Tuple['RobotIndividual', 'RobotIndividual']:
        """智能交叉：保持有效性"""
        # 基于模板的交叉
        parent1_seq = self.chromosome
        parent2_seq = other.chromosome
        
        # 找到共同的前缀
        common_prefix = []
        min_len = min(len(parent1_seq), len(parent2_seq))
        
        for i in range(min_len):
            if parent1_seq[i] == parent2_seq[i]:
                common_prefix.append(parent1_seq[i])
            else:
                break
        
        # 生成子代
        child1_seq = common_prefix + parent2_seq[len(common_prefix):]
        child2_seq = common_prefix + parent1_seq[len(common_prefix):]
        
        # 完成序列
        child1_seq = self.complete_sequence(child1_seq)
        child2_seq = self.complete_sequence(child2_seq)
        
        child1 = RobotIndividual(child1_seq)
        child2 = RobotIndividual(child2_seq)
        
        return child1, child2
    
    def build_robot(self):
        """构建机器人"""
        try:
            if not self.is_valid_sequence(self.chromosome):
                return None
            
            rules = self._rules_cache
            if not rules:
                return None
            
            graph = make_graph(rules, self.chromosome)
            robot = build_normalized_robot(graph)
            return robot
        except Exception as e:
            print(f"构建机器人失败: {e}")
            return None
    
    # 保持原有接口
    def get_id(self):
        return self.id
    
    def set_initial_position(self, initial_position):
        self.initial_position = initial_position
        
    def get_initial_position(self):
        return getattr(self, 'initial_position', None)
    
    def get_fitness(self):
        return self.fitness
    
    def get_chromosome(self):
        return self.chromosome
    
    def set_fitness(self, fitness):
        self.fitness = fitness
    
    def mutate(self):
        """保持原有接口的变异方法"""
        self.smart_mutate()
    
    def crossover_single_point(self, other):
        """保持原有接口的交叉方法"""
        return self.smart_crossover(other)
    
    def crossover(self, other):
        """保持原有接口的交叉方法"""
        return self.smart_crossover(other)

# 测试代码
if __name__ == "__main__":
    print("=== 测试智能个体类 ===")
    
    # 测试有效序列
    print("\n1. 测试有效序列:")
    test_sequences = [
        [0, 3, 6, 7],
        [0, 2, 8, 5, 6, 7, 14],
        [0, 3, 6, 7, 10]
    ]
    
    for seq in test_sequences:
        individual = RobotIndividual(seq)
        valid = individual.is_valid_sequence(seq)
        print(f"序列 {seq}: 有效性 = {valid}")
    
    # 测试序列完成
    print("\n2. 测试序列完成:")
    incomplete_seq = [0, 2]
    individual = RobotIndividual()
    completed = individual.complete_sequence(incomplete_seq)
    print(f"原始不完整序列: {incomplete_seq}")
    print(f"完成后序列: {completed}")
    print(f"完成后有效性: {individual.is_valid_sequence(completed)}")
    
    # 测试智能变异
    print("\n3. 测试智能变异:")
    individual = RobotIndividual([0, 3, 6, 7])
    print(f"原始: {individual.chromosome}")
    
    for i in range(3):
        individual_copy = RobotIndividual(individual.chromosome)
        individual_copy.smart_mutate()
        valid = individual_copy.is_valid_sequence(individual_copy.chromosome)
        print(f"变异{i+1}: {individual_copy.chromosome}, 有效: {valid}")
    
    # 测试智能交叉
    print("\n4. 测试智能交叉:")
    parent1 = RobotIndividual([0, 3, 6, 7])
    parent2 = RobotIndividual([0, 2, 8, 5, 6, 7, 14])
    print(f"父代1: {parent1.chromosome}")
    print(f"父代2: {parent2.chromosome}")
    
    child1, child2 = parent1.smart_crossover(parent2)
    print(f"子代1: {child1.chromosome}, 有效: {child1.is_valid_sequence(child1.chromosome)}")
    print(f"子代2: {child2.chromosome}, 有效: {child2.is_valid_sequence(child2.chromosome)}")
    
    # 测试机器人构建
    print("\n5. 测试机器人构建:")
    individual = RobotIndividual([0, 3, 6, 7])
    robot = individual.build_robot()
    if robot:
        print(f"机器人构建: 成功，{len(robot.links)} 个连杆")
    else:
        print("机器人构建: 失败")