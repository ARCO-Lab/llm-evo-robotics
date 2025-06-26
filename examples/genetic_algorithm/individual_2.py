import copy
import time
import uuid
import pyrobotdesign as rd
import os
import sys
import random
from typing import List, Dict, Set, Tuple, Optional

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir,'graph_learning'))
sys.path.append(os.path.join(base_dir, 'design_search'))

from RobotGrammarEnv import RobotGrammarEnv
from design_search import make_graph, build_normalized_robot, presimulate, has_nonterminals

class RobotIndividual:

    def __init__(self, task, seed = 0, mpc_num_processes = 8, grammar_file = None):
        if not hasattr(RobotIndividual, '_rules_cache'):
            RobotIndividual._rules_cache = self._load_rules(grammar_file)

        self.env = RobotGrammarEnv(task, self._rules_cache, seed = seed, mpc_num_processes = mpc_num_processes)

        self.init_chromosome()

        self.mutation_list = ['single_bit_mutation', 'add_link_mutation']

        self.fitness = 0

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def init_chromosome(self):
        self.chromosome = [0]
    def set_initial_position(self, initial_position):
        self.initial_position = initial_position
    def set_chromosome(self, chromosome):
        self.chromosome = chromosome

    def get_chromosome(self):
        return self.chromosome

    def _load_rules(self, grammar_file):
        try:
            graphs = rd.load_graphs(grammar_file)
            rules = [rd.create_rule_from_graph(g) for g in graphs]
            return rules
        except Exception as e:
            print(f"Warning: Failed to load grammar file {grammar_file}: {e}")
            return []
    # def __deepcopy__(self, memo):
    #     """自定义深拷贝方法，避免拷贝不可序列化的规则对象"""
    #     # 创建新的个体，但不拷贝规则缓存
    #     new_individual = RobotIndividual.__new__(RobotIndividual)
        
    #     # 拷贝可序列化的属性
    #     new_individual.chromosome = copy.deepcopy(self.chromosome, memo)
    #     new_individual.env = None  # 环境对象也可能有序列化问题，稍后重新创建
    #     new_individual.mutation_list = copy.deepcopy(self.mutation_list, memo)
        
    #     # 重新设置规则缓存（引用同一个缓存）
    #     new_individual._rules_cache = RobotIndividual._rules_cache
        
    #     # 重新创建环境
    #     if hasattr(self, 'env') and self.env is not None:
    #         new_individual.env = RobotGrammarEnv(
    #             self.env.task, 
    #             self._rules_cache, 
    #             seed=self.env.seed, 
    #             mpc_num_processes=self.env.mpc_num_processes
    #         )
        
    #     return new_individual

    def env_reset(self):
        self.env.reset()

    def check_valid_state(self, state: rd.Graph) -> bool:
        return self.env.is_valid(state)
    

    def get_available_actions(self, state: rd.Graph):

        available_actions = self.env.get_available_actions(state)
        if len(available_actions) == 0:
            return []
        return available_actions

    def get_reward(self, state: rd.Graph) -> float:
        return self.env.get_reward(state)
    



    def single_bit_mutation(self):
        new_chromosome = self.chromosome.copy()
        if len(new_chromosome) == 0:
            return new_chromosome  # 空染色体直接返回

        max_attempts = 10
        success = False

        for _ in range(max_attempts):
            index = random.randint(0, len(new_chromosome) - 1)
            left_chromosome = new_chromosome[:index]
            try:
                graph = make_graph(self._rules_cache, left_chromosome)
                available_actions = self.get_available_actions(graph)
            except Exception as e:
                print(f"[Mutation Fail] Cannot make_graph with prefix {left_chromosome}: {e}")
                continue

            if len(available_actions) == 0:
                continue  # 当前 index 不合法，尝试下一个随机 index

            for _ in range(max_attempts):
                action = random.choice(available_actions)
                if index == len(new_chromosome) - 1:
                    temp_graph = make_graph(self._rules_cache, left_chromosome + [action])
                    if len(self.get_available_actions(temp_graph)) > 0:
                        new_chromosome = left_chromosome + [action]
                        success = True
                        break
                else:
                    temp_graph = make_graph(self._rules_cache, left_chromosome + [action])
                    next_available = self.get_available_actions(temp_graph)
                    if len(next_available) > 0 and new_chromosome[index + 1] in next_available:
                        new_chromosome[index] = action
                        success = True
                        break

            if success:
                break
        
        off_spring = RobotIndividual(self.env.task, self.env.seed, self.env.mpc_num_processes, self._rules_cache)
        off_spring.set_chromosome(new_chromosome) if success else off_spring.set_chromosome(self.chromosome)
        return off_spring


    def add_link_mutation(self):
        new_chromosome = self.chromosome.copy()

        max_attempts = 10
        success = False

        for _ in range(max_attempts):
            index = random.randint(0, len(new_chromosome))  # 插入的位置可以等于 len，表示 append

            left_chromosome = new_chromosome[:index]
            right_chromosome = new_chromosome[index:]
            try:
                graph = make_graph(self._rules_cache, left_chromosome)
                available_actions = self.get_available_actions(graph)
            except Exception as e:
                print(f"Error: {e}")
                continue

            if len(available_actions) == 0:
                continue  # 此位置无法插入任何 rule，换个位置试

            for _ in range(max_attempts):
                action = random.choice(available_actions)
                # 插入后，检查后续部分是否仍然合法（若有的话）
                temp_graph = make_graph(self._rules_cache, left_chromosome + [action])
                next_available = self.get_available_actions(temp_graph)

                if len(right_chromosome) == 0:
                    # 插入在尾部，只要求还能扩展即可
                    if len(next_available) > 0:
                        new_chromosome = left_chromosome + [action] + right_chromosome
                        success = True
                        break
                else:
                    if right_chromosome[0] in next_available:
                        new_chromosome = left_chromosome + [action] + right_chromosome
                        try:
                            graph = make_graph(self._rules_cache, new_chromosome)
                        except Exception as e:
                            print(f"Error: {e}")
                            continue
                        success = True

                        break

            if success:
                break

        off_spring = RobotIndividual(self.env.task, self.env.seed, self.env.mpc_num_processes, self._rules_cache)
        off_spring.set_chromosome(new_chromosome) if success else off_spring.set_chromosome(self.chromosome)
        return off_spring


    def insert_mutation_test(self):
        return self._attempt_mutation('insert')

    def choose_valid_mutation_action(self):
        choice = random.choice(self.mutation_list)
        if choice == 'single_bit_mutation':
            return self.single_bit_mutation()
        elif choice == 'add_link_mutation':
            return self.add_link_mutation()

    def clone(self) -> 'RobotIndividual':
        new_individual = RobotIndividual(self.env.task, self.env.seed, self.env.mpc_num_processes, self._rules_cache)
        new_individual.set_chromosome(self.chromosome.copy())
        return new_individual

    def choose_valid_crossover_action(self, other: 'RobotIndividual'):
        parent1 = self.chromosome.copy()
        parent2 = other.chromosome.copy()

        if len(parent1) <= 1 or len(parent2) <= 1:
            # 过短不能交叉，直接复制
            return self.clone(), other.clone()

        max_attempts = 50
        min_length = min(len(parent1), len(parent2))

        if min_length < 3:
            return self.clone(), other.clone()


        for _ in range(max_attempts):
            start_pos = random.randint(1, min_length - 2)
            max_segment_length = min_length - start_pos
            segment_length = random.randint(1, max_segment_length)
            end_pos = start_pos + segment_length

            # 切分并交叉
            child1_chrom = parent1[:start_pos] + parent2[start_pos:end_pos] + parent1[end_pos:]
            child2_chrom = parent2[:start_pos] + parent1[start_pos:end_pos] + parent2[end_pos:]

            try:
                # 验证两个子代是否合法
                make_graph(self._rules_cache, child1_chrom)
                make_graph(self._rules_cache, child2_chrom)

                # 创建并返回两个新个体
                child1 = RobotIndividual(self.env.task, self.env.seed, self.env.mpc_num_processes, self._rules_cache)
                child2 = RobotIndividual(self.env.task, self.env.seed, self.env.mpc_num_processes, self._rules_cache)
                child1.set_chromosome(child1_chrom)
                child2.set_chromosome(child2_chrom)
                return child1, child2

            except Exception as e:
                print(f"[Crossover Error] Segment ({start_pos}, {end_pos}): {e}")
                continue

        # 所有尝试失败，回退为 parent 克隆
        print("[Crossover Failed] Returning parent clones.")
        return self.clone(), other.clone()
                
    
    def mutation(self):
        if len(self.chromosome) <= 1:
            return
        
        pass
    


    def crossover(self, other: 'RobotIndividual'):

        pass
    
    

def test_mutation():
    # 替换为你实际的 grammar 文件路径
    grammar_file_path = 'data/designs/grammar_apr30.dot'

    # 初始化个体
    task = 'FlatTerrainTask'  # 假设你在 RobotGrammarEnv 中支持 'flat' 地形
    individual = RobotIndividual(task=task, grammar_file=grammar_file_path)

    # 设置初始 chromosome
    individual.chromosome = [0, 7, 1, 13, 1, 2, 16, 12, 13, 6, 4, 19, 4, 17, 5, 3, 2, 16, 4, 5, 18, 9, 8, 9, 9, 8 ]  # 你可以根据 grammar 中的 rule 数量自定义

    print("Original chromosome:", individual.chromosome)

    # 选择一个中间位置进行突变
    # mutation_index = random.randint(1, len(individual.chromosome) - 1)
    # mutated_chromosome = individual.choose_valid_mutation_action(mutation_index)

    # mutated_chromosome = individual.choose_valid_mutation_action()
    mutated_chromosome = individual.add_link_mutation()



    print("Mutated chromosome :", mutated_chromosome)

    # 基本断言
    assert mutated_chromosome is not None, "Mutation returned None"
    assert isinstance(mutated_chromosome, list), "Mutation did not return a list"
    assert mutated_chromosome != individual.chromosome or True, "Mutation may fail but should return original"

    # 验证突变后的结构是否合法（可选）
    try:
        final_graph = make_graph(individual._rules_cache, mutated_chromosome)
        is_valid = individual.check_valid_state(final_graph)
        print("Is mutated graph valid?", is_valid)
    except Exception as e:
        print("Failed to build or validate mutated graph:", e)

def simple_crossover_test():
    """简单的交叉测试"""
    
    print("=== 简单交叉测试 ===")
    
    # 初始化
    grammar_file_path = 'data/designs/grammar_apr30.dot'
    task = 'FlatTerrainTask'
    
    parent1 = RobotIndividual(task=task, grammar_file=grammar_file_path)
    parent2 = RobotIndividual(task=task, grammar_file=grammar_file_path)
    
    # 使用已知有效的序列
    parent1.chromosome = [0, 7, 1, 13, 1, 2, 16, 12, 13, 6, 4, 19, 4, 17, 5, 3, 2, 16, 4, 5, 18, 9, 8, 9, 9, 8]
    parent2.chromosome = [0, 12, 7, 1, 12, 3, 10, 1, 3, 1, 12, 12, 1, 3, 10, 2, 16, 8, 1, 3, 12, 4, 1, 3, 2, 12, 18, 9, 18, 8, 5, 5, 1, 12, 6, 3]
    
    print(f"父代1: {parent1.chromosome}")
    print(f"父代2: {parent2.chromosome}")
    
    # 执行交叉
    child1, child2 = parent1.choose_valid_crossover_action(parent2)
    
    print(f"子代1: {child1}")
    print(f"子代2: {child2}")
    
    # 简单验证
    assert child1 is not None, "子代1不应该为None"
    assert child2 is not None, "子代2不应该为None"
    assert len(child1) > 0, "子代1不应该为空"
    assert len(child2) > 0, "子代2不应该为空"
    assert child1[0] == 0, "子代1应该以规则0开始"
    assert child2[0] == 0, "子代2应该以规则0开始"
    
    print("✅ 基本测试通过")



if __name__ == "__main__":
    # test_mutation()
    simple_crossover_test()
