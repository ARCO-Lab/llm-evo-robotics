import pyrobotdesign as rd
import numpy as np
import random
import copy
import time
import os
import sys
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import torch
import torch.multiprocessing as torch_mp
import traceback

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "design_search"))

from individual import RobotIndividual
import tasks
from design_search import make_graph, build_normalized_robot, presimulate, simulate
from RobotGrammarEnv import RobotGrammarEnv

class SimpleGeneticAlgorithm:
    def __init__(self, 
                 population_size: int = 20,
                 max_generation: int = 30,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8,
                 task_name: str = 'FlatTerrainTask',
                 parallel_mode: str = 'gpu_batch',
                 num_processes: int = 1,
                 batch_size: int =2,
                 grammar_file: str = 'data/designs/grammar_apr30.dot'):
        
        self.population_size = population_size
        self.max_generation = max_generation
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.task_name = task_name
        self.parallel_mode = parallel_mode
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grammar_file = grammar_file
        if num_processes is None:
            self.num_processes = min(mp.cpu_count(), self.population_size)
        else:
            self.num_processes = num_processes
        
        self.setup_evaluation_pool()

        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = 0


    def setup_evaluation_pool(self):
        print(f"initializing evaluation environment")
        task_class = getattr(tasks, self.task_name)
        #TODO: 需要修改 暂时用无噪声估计
        self.task = task_class(force_std=0.0, torque_std = 0.0)
        self.graphs = rd.load_graphs(self.grammar_file)
        self.rules = [rd.create_rule_from_graph(g) for g in self.graphs]
        print(f"initializing evaluation environment done")

       

    def initialize_population(self):
        # 定义多样化的初始模板
        valid_templates = [
            [0, 3, 6, 7],                    # 简单单体机器人
            [0, 2, 8, 5, 6, 7, 14],          # 带单肢体机器人
            [0, 2, 8, 5, 6, 7, 15],          # 带单肢体机器人（不同关节）
            [0, 2, 8, 5, 6, 7, 16],          # 带单肢体机器人（不同关节）
            [0, 2, 9, 5, 6, 7, 14],          # 长肢体机器人
            [0, 1, 3, 6, 7],                 # 多节身体简单机器人
            [0, 3, 6, 7, 10],                # 带身体关节的简单机器人
            [0, 3, 6, 7, 11],                # 带身体关节的简单机器人（不同类型）
        ]
        
        for i in range(self.population_size):
            # 按比例分配不同模板
            template_idx = i % len(valid_templates)
            base_chromosome = valid_templates[template_idx].copy()
            
            # 随机变异一些个体以增加多样性
            if random.random() < 0.3:  # 30%的概率进行轻微变异
                individual = RobotIndividual(base_chromosome)
                individual.smart_mutate()  # 轻微变异
            else:
                individual = RobotIndividual(base_chromosome)
            
            self.population.append(individual)
        
        print(f"initializing population done")

    def evaluate_population(self):
        print(f"GPU batch evaluation")
        fitness_scores = []
        for i in range(0, len(self.population)):
            batch = self.population[i:i+self.batch_size]
            print(f"batch length: {len(batch)}")
            batch_fitness = self._evaluate_batch(batch)
            fitness_scores.extend(batch_fitness)

            # print(f"Evaluated {i+1} of {len(self.population)} individuals")

        return fitness_scores

    def _evaluate_batch(self, batch):
        batch_fitness = []

        # print(f"batch: {batch}, length: {len(batch)}")
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures =  []
            for individual in batch:
                future = executor.submit(self._evaluate_single_individual, individual)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    fitness = future.result()
                    batch_fitness.append(fitness)
                except Exception as e:
                    # print(f"Error evaluating individual: {e}, fitness: {-10}")
                    # traceback.print_exc()
                    batch_fitness.append(-10.0)

        return batch_fitness
    
    def _evaluate_single_individual(self, individual):
        graph = make_graph(self.rules, individual.chromosome)

        robot = build_normalized_robot(graph)
        # print(f"individual: {individual.chromosome}")
        robot_init_pos, has_self_collision = presimulate(robot)
        # print(f"has_self_collision: {has_self_collision}")

        if has_self_collision:
            return -10.0
        
        individual.set_initial_position(robot_init_pos)
        _ , reward = simulate(robot, self.task, 42, 2, 1)
        # reward = 10.0
        print(f"reward: {reward}")
        return reward


    def selection(self, population, fitness_scores):
        """ tournament selection """
        selected_individuals = []
        tournament_size = 2

        for _ in range(len(population)):
            tournament_individuals_idx = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_individuals_idx]
            winner_idx = tournament_individuals_idx[np.argmax(tournament_fitness)]
            selected_individuals.append(copy.deepcopy(population[winner_idx]))

        return selected_individuals

    def crossover_and_mutation(self, population):
        """ crossover and mutation """
        new_population = []
        for _ in range(len(population)):
            parent1, parent2 = random.sample(population, 2)

            if random.random() < self.crossover_rate:
                # 修复：使用正确的交叉方法名
                child1, child2 = parent1.smart_crossover(parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            if random.random() < self.mutation_rate:
                child1.mutate()
            
            if random.random() < self.mutation_rate:
                child2.mutate()

            new_population.extend([child1, child2])

        # 确保种群大小一致
        new_population = random.choices(new_population, k=len(population))
        return new_population


    def visualize_best_individual(self, best_individual):
        graph = make_graph(self.rules, best_individual.chromosome)
        robot = build_normalized_robot(graph)
        robot_init_pos, has_self_collision = presimulate(robot)
        if not has_self_collision:
            main_sim = rd.BulletSimulation(self.task.time_step)
            self.task.add_terrain(main_sim)
            main_sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
            robot_idx = main_sim.find_robot_index(robot)
            viewer = rd.GLFWViewer()
            lower = np.zeros(3)
            upper = np.zeros(3)
            main_sim.get_robot_world_aabb(robot_idx, lower, upper)
            viewer.camera_params.position = 0.5 * (lower + upper)
            viewer.camera_params.yaw = -np.pi / 4
            viewer.camera_params.pitch = -np.pi / 6
            viewer.camera_params.distance = 1.5 * np.linalg.norm(upper - lower)
            sim_time = 0
            while sim_time < 2:
                main_sim.step()
                main_sim.get_robot_world_aabb(robot_idx, lower, upper)
                target_pos = 0.5 * (lower + upper)
                viewer.camera_params.position = target_pos
                viewer.update(self.task.time_step)
                viewer.render(main_sim)
                sim_time += self.task.time_step
                print(f"sim_time: {sim_time}")
                        
    def run_ga(self):
        print(f"Running GA")
        print(f"config: population_size={self.population_size}, max_generation={self.max_generation}, mutation_rate={self.mutation_rate}, crossover_rate={self.crossover_rate}")

        self.initialize_population()

        for generation in range(self.max_generation):
            print(f"Generation {generation+1} of {self.max_generation}")

            fitness_scores = self.evaluate_population()

            for individual, fitness in zip(self.population, fitness_scores):
                    print(f"individual: {individual.chromosome}, fitness: {fitness}")
                    individual.fitness = fitness
            
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]

                print(f"best_idx: {best_idx}")
                print(f"best_individual: {self.population}")
                self.best_individual = copy.deepcopy(self.population[best_idx])
                print(f"New best fitness: {self.best_fitness}")
                print(f"Best individual: {self.best_individual.chromosome}")
                self.visualize_best_individual(self.best_individual)
            avg_fitness = np.mean(fitness_scores)
            std_fitness = np.std(fitness_scores)
            self.fitness_history.append({
                'generation': generation,
                'avg_fitness': avg_fitness,
                'std_fitness': std_fitness,
                'best_fitness': self.best_fitness
            })
            
            print(f" best fitness: {self.best_fitness}, avg fitness: {avg_fitness}, std fitness: {std_fitness}")

            if generation < self.max_generation - 1:
                selected_population = self.selection(self.population, fitness_scores)
                self.population = self.crossover_and_mutation(selected_population)

        return self.best_individual, self.fitness_history

if __name__ == "__main__":

    torch_mp.set_start_method("spawn", force=True)

    ga = SimpleGeneticAlgorithm(population_size=20, 
                                max_generation=300, 
                                mutation_rate=0.2,
                                crossover_rate=0.8, 
                                task_name='FlatTerrainTask')

    ga.setup_evaluation_pool()
    best_individual, history = ga.run_ga()

    import json
    import time
    with open(f'test_ga_results_{time.strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump({
            'best_individual': best_individual.chromosome,
            'best_fitness': best_individual.fitness,
            'history': history,
        }, f, indent=2)

    