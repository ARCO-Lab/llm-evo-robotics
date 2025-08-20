import numpy as np

import torch
import os
import pickle
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass

class RobotGenotype:

    num_links: int  = 4

    link_lengths: List[float] = None

    lr: float = 1e-4
    alpha: float = 0.2
    tau: float = 0.005
    gamma: float = 0.99
    batch_size: int = 64
    buffer_capacity: int = 100000

    warmup_steps: int = 1000
    target_entropy_factor: float = 0.01

    def __post_init__(self):
        if self.link_lengths is None:
            self.link_lengths = [80.0] * self.num_links
        assert len(self.link_lengths) == self.num_links

@dataclass

class RobotPhenotype:

    avg_reward: float = 0.0
    success_rate: float = 0.0
    min_distance_to_goal: float = float('inf')

    total_reach: float  = 0.0
    complexity_score: float = 0.0

    trajectory_smoothness: float = 0.0
    collision_frequency: float = 0.0
    exploration_converage: float = 0.0

    action_variance:  float = 0.0

    learning_efficiency: float = 0.0

@dataclass

class Individual:

    genotype: RobotGenotype
    phenotype: RobotPhenotype
    fitness: float = 0.0
    generation: int = 0
    parent_id: Optional[int] = None
    individual_id: str = ""

    def __post_init__(self):
        if not self.individual_id:
            self.individual_id = f"gen({self.generation})_{int(time.time() * 1000) % 100000}"

class FeatureExtractor:

    def __init__(self):
        pass

    def extract_from_training_data(self, training_metrics: Dict, robot_config: Dict) -> RobotPhenotype:

        phenotype = RobotPhenotype()

        phenotype.avg_reward = training_metrics.get('avg_reward', 0.0)
        phenotype.success_rate = training_metrics.get('success_rate', 0.0)
        phenotype.min_distance_to_goal = training_metrics.get('min_distance', float('inf'))

        link_lengths = robot_config.get('link_lengths', [80.0] * 4)
        phenotype.total_reach = sum(link_lengths)

        phenotype.complexity_score = len(link_lengths) + np.var(link_lengths) / 100.0

        phenotype.trajectory_smoothness = training_metrics.get('trajectory_smoothness', 0.0)
        phenotype.collision_frequency = training_metrics.get('collision_rate', 0.0)
        phenotype.exploration_coverage = training_metrics.get('exploration_area',0.0)

        phenotype.action_variance = training_metrics.get('action_variance',0.0)
        phenotype.leanring_efficiency = training_metrics.get('learning_rate',0.0)

        return phenotype
    
    def discretize_features(self, phenotype: RobotPhenotype) -> Tuple[int, ...]:

        reward_bin = min(9, max(0, int(phenotype.avg_reward + 100) / 20 ))

        complexity_bin = min(4, max(0, int(phenotype.complexity_score / 2)))

        smoothness_bin = min(4, max(0, int(phenotype.trajectory_smoothness * 5)))

        collision_bin = min(4, max(0, int(phenotype.collision_frequency * 5)))

        reach_bin = min(5, max(0, int((phenotype.total_reach - 200) / 50)))

        return (reward_bin, complexity_bin, smoothness_bin, collision_bin, reach_bin)
    

class MAPElitesArchive:

    def __init__(self, feature_dimensions: Tuple[int, ...] = (10,5,5,5,6), 
                 
                 save_dir: str = 'map_elites_archive'):
        
        self.feature_dimensions = feature_dimensions
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok = True)

        self.archive: Dict[Tuple[int, ...], Individual] = {}

        self.generation = 0
        self.total_evaluations = 0
        self.improvement_count = 0

        print(f"MAP-Elites 存档初始化")
        print(f"网络维度： {feature_dimensions}")
        print(f"总单元数： {np.prod(feature_dimensions)}")

    def add_individual(self, individual: Individual) -> bool:

        extractor = FeatureExtractor()

        coords = extractor.discretize_features(individual.phenotype)

        self.total_evaluations  += 1

        if coords not in self.archive or individual.fitness > self.archive[coords].fitness:
            self.improvement_count += 1
            print(f"个体添加到位置{coords}, 适应度： {individual.fitness: .2f}")
            return True
        return False
    
    def get_random_elite(self) -> Optional[Individual]:
        if not self.archive:
            return None
        return max(self.archive.values(), key=lambda x: x.fitness)
    
    def get_best_individual(self) -> Optional[Individual]:

        if not self.archive:
            return None
        return max(self.archive.values(), key=lambda x: x.fitness)
    
    def get_statistics(self) -> Dict:

        if not self.archive:
            return {'size':0, 'coverage': 'best_fitness': -float('inf')}
        fitnesses = [ind.fitness for ind in self.archive.values()]
        coverage = len(self.archive/ np.prod(self.feature_dimensions))

        return {
            'size': len(self.archive),
            'coverage': coverage,
            'best_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'total_evaluations': self.total_evaluations,
            'improvement_rate': self.improvement_count / max(1, self.total_evaluations)
        }
    
    def save_archive(self, filename: Optional[str] = None):

        if filename is None:
            filename = f"archive_gen_{self.generation}.pkl"

        filepath = os.path.join(self.save_dir, filename)
        archive_data = {
            'archive': self.archive,
            'feature_dimensions': self.feature_dimensions,
            'generation': self.generation,
            'total_evaluations': self.total_evaluations,
            'improvement_count': self.improvement_count
        }
        with open(filepath, 'wb') as f:
            pickle.dump(archive_data, f)

        print(f"存档已保存： {filepath}")

class RobotMulator:

    def __init__(self, mutation_rates: Dict[str, float] = None):

        self.mutation_rates = mutation_rates or {
            'link_lengths': 0.1,
            'num_links': 0.05,
            'sac_params': 0.3,
            'training_params': 0.2
        }

    def mutate(self, parent: RobotGenotype) -> RobotGenotype:

        mutant = RobotGenotype(
            num_links = parent.num_links,
            link_lengths = parent.link_lengths.copy(),
            lr = parent.lr,
            alpha=parent.alpha,
            tau=parent.tau,
            gamma=parent.gamma,
            batch_size=parent.batch_size,
            buffer_capacity=parent.buffer_capacity,
            warmup_steps=parent.warmup_steps,
            target_entropy_factor = parent.target_entropy_factor
        )

        if np.random.random() < self.mutation_rates['link_lengths']:
            idx = np.random.randint(0, len(mutant.link_lengths))
            mutant.link_lengths[idx] *= np.random.uniform(0.8, 1.2)
            mutant.link_lengths[idx] = max(20, min(120, mutant.link_lengths[idx]))
        
        if np.random.random() < self.mutation_rates['num_links']:

            if np.random.random() < 0.5 and mutant.num_links > 2:
                mutant.num_links -= 1
                mutant.link_lengths = mutant.link_lengths[:-1]
            else:
                mutant.num_links += 1
                new_length = np.random.uniform(40, 80)
                mutant.link_lengths.append(new_length)
        #TODO: 训练参数变异
        # if np.random.random() < self.mutation_rates['sac_params']:
        #             # 学习率变异
        #             mutant.lr *= np.random.lognormal(0, 0.3)
        #             mutant.lr = np.clip(mutant.lr, 1e-6, 1e-3)
                    
        #             # Alpha变异
        #             mutant.alpha += np.random.normal(0, 0.1)
        #             mutant.alpha = np.clip(mutant.alpha, 0.01, 2.0)
                    
        #             # Tau变异
        #             mutant.tau *= np.random.uniform(0.8, 1.2)
        #             mutant.tau = np.clip(mutant.tau, 0.001, 0.01)
                
        # # 🎯 训练参数变异
        # if np.random.random() < self.mutation_rates['training_params']:
        #     # Batch size变异 (在2的幂中选择)
        #     batch_sizes = [32, 64, 128, 256]
        #     mutant.batch_size = np.random.choice(batch_sizes)
            
        #     # Warmup steps变异
        #     mutant.warmup_steps = int(mutant.warmup_steps * np.random.uniform(0.5, 2.0))
        #     mutant.warmup_steps = max(100, min(5000, mutant.warmup_steps))
        
        # return mutant      

        def random_genotype(self) -> RobotGenotype:

            num_links = np.random.randint(3, 7)
            link_lengths = [np.random.uniform(40, 100) for _ in range(num_links)]

            return RobotGenotype(
                num_links=num_links,
                link_lengths=link_lengths,
                lr=10 ** np.random.uniform(-6, -3),  # 1e-6 到 1e-3
                alpha=np.random.uniform(0.1, 1.0),
                tau=np.random.uniform(0.001, 0.01),
                gamma=np.random.uniform(0.95, 0.999),
                batch_size=np.random.choice([32, 64, 128, 256]),
                buffer_capacity=np.random.choice([5000, 10000, 20000]),
                warmup_steps=np.random.randint(500, 3000),
                target_entropy_factor=np.random.uniform(0.5, 1.2)

            )