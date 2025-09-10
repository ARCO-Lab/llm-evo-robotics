"""
Fixed Genetic Algorithm Fitness Evaluation System
Uses the verified EnvironmentPathAnalyzer from test_shortest_path_visualization.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from typing import Dict, List, Tuple
import math
from dataclasses import dataclass

# 直接从 test_shortest_path_visualization.py 复制已验证的类
@dataclass
class Point:
    x: float
    y: float
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class EnvironmentPathAnalyzer:
    """Environment path analyzer with fixed collision detection - VERIFIED"""
    
    def __init__(self, start_point: List[float], goal_point: List[float], 
                 obstacles: List[Dict], safety_buffer: float = 15.0):
        self.start_point = start_point
        self.goal_point = goal_point
        self.obstacles = obstacles
        self.safety_buffer = safety_buffer
        
        # Calculate environment shortest path immediately
        self.shortest_path_result = self._compute_environment_shortest_path()
    
    def _compute_environment_shortest_path(self):
        """Calculate environment shortest path - pure geometric problem"""
        start = Point(self.start_point[0], self.start_point[1])
        goal = Point(self.goal_point[0], self.goal_point[1])
        
        # 1. Calculate direct distance
        direct_distance = start.distance_to(goal)
        
        # 2. Check if direct path is collision-free (FIXED)
        if self._is_path_collision_free(start, goal):
            return {
                'path': [start, goal],
                'path_length': direct_distance,
                'method': 'direct',
                'key_points': [start, goal]
            }
        
        # 3. Calculate detour path
        return self._find_detour_path(start, goal)
    
    def _is_path_collision_free(self, start: Point, end: Point, num_samples: int = 50) -> bool:
        """FIXED: Check if path is collision-free using dense sampling"""
        # Sample multiple points along the path
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_x = start.x + t * (end.x - start.x)
            sample_y = start.y + t * (end.y - start.y)
            sample_point = Point(sample_x, sample_y)
            
            # Check if this sample point is too close to any obstacle
            for obstacle in self.obstacles:
                if obstacle.get('shape') == 'segment':
                    seg_start = Point(obstacle['points'][0][0], obstacle['points'][0][1])
                    seg_end = Point(obstacle['points'][1][0], obstacle['points'][1][1])
                    
                    distance = self._point_to_segment_distance(sample_point, seg_start, seg_end)
                    if distance < self.safety_buffer:
                        return False
        
        return True
    
    def _point_to_segment_distance(self, point: Point, seg_start: Point, seg_end: Point) -> float:
        """Calculate shortest distance from point to line segment"""
        seg_vec = np.array([seg_end.x - seg_start.x, seg_end.y - seg_start.y])
        point_vec = np.array([point.x - seg_start.x, point.y - seg_start.y])
        
        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq == 0:
            return np.linalg.norm(point_vec)
        
        t = max(0, min(1, np.dot(point_vec, seg_vec) / seg_len_sq))
        projection = seg_start.x + t * seg_vec[0], seg_start.y + t * seg_vec[1]
        
        return math.sqrt((point.x - projection[0])**2 + (point.y - projection[1])**2)
    
    def _find_detour_path(self, start: Point, goal: Point):
        """Calculate detour path"""
        # Analyze obstacle boundaries
        obstacle_bounds = self._get_obstacle_bounds()
        
        # Generate key navigation points with more options
        key_points = [start, goal]
        buffer = self.safety_buffer * 2  # Use larger buffer for navigation points
        
        # Upper detour points
        upper_left = Point(obstacle_bounds['min_x'] - buffer, obstacle_bounds['min_y'] - buffer)
        upper_right = Point(obstacle_bounds['max_x'] + buffer, obstacle_bounds['min_y'] - buffer)
        
        # Lower detour points
        lower_left = Point(obstacle_bounds['min_x'] - buffer, obstacle_bounds['max_y'] + buffer)
        lower_right = Point(obstacle_bounds['max_x'] + buffer, obstacle_bounds['max_y'] + buffer)
        
        # Side points for going around
        left_middle = Point(obstacle_bounds['min_x'] - buffer, 
                           (obstacle_bounds['min_y'] + obstacle_bounds['max_y']) / 2)
        right_middle = Point(obstacle_bounds['max_x'] + buffer, 
                            (obstacle_bounds['min_y'] + obstacle_bounds['max_y']) / 2)
        
        # Add more navigation options
        key_points.extend([
            upper_left, upper_right, lower_left, lower_right, 
            left_middle, right_middle
        ])
        
        # Use Dijkstra algorithm to find shortest collision-free path
        return self._dijkstra_shortest_path(key_points, start, goal)
    
    def _get_obstacle_bounds(self):
        """Get obstacle boundaries"""
        all_points = []
        for obstacle in self.obstacles:
            if obstacle.get('shape') == 'segment':
                all_points.extend(obstacle['points'])
        
        if not all_points:
            return {'min_x': 0, 'max_x': 800, 'min_y': 0, 'max_y': 600}
        
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        
        return {
            'min_x': min(xs),
            'max_x': max(xs),
            'min_y': min(ys),
            'max_y': max(ys)
        }
    
    def _dijkstra_shortest_path(self, key_points: List[Point], start: Point, goal: Point):
        """Calculate shortest path using Dijkstra algorithm"""
        # Build graph - only connect points if path between them is collision-free
        n = len(key_points)
        graph = [[float('inf')] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if path between key_points[i] and key_points[j] is collision-free
                    if self._is_path_collision_free(key_points[i], key_points[j]):
                        distance = key_points[i].distance_to(key_points[j])
                        graph[i][j] = distance
        
        # Find start and goal indices
        start_idx = 0
        goal_idx = 1
        
        # Dijkstra algorithm
        distances = [float('inf')] * n
        distances[start_idx] = 0
        previous = [-1] * n
        visited = [False] * n
        
        for _ in range(n):
            # Find unvisited node with minimum distance
            u = -1
            for v in range(n):
                if not visited[v] and (u == -1 or distances[v] < distances[u]):
                    u = v
            
            if u == -1 or distances[u] == float('inf'):
                break
                
            visited[u] = True
            
            # Update neighbors
            for v in range(n):
                if not visited[v] and graph[u][v] != float('inf'):
                    new_dist = distances[u] + graph[u][v]
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        previous[v] = u
        
        # Reconstruct path
        if distances[goal_idx] == float('inf'):
            # Return direct line for visualization (even though it's blocked)
            return {
                'path': [start, goal],
                'path_length': start.distance_to(goal),
                'method': 'direct_blocked',
                'key_points': [start, goal]
            }
        
        path = []
        current = goal_idx
        while current != -1:
            path.append(key_points[current])
            current = previous[current]
        path.reverse()
        
        return {
            'path': path,
            'path_length': distances[goal_idx],
            'method': 'dijkstra',
            'key_points': key_points
        }
    
    def get_direct_distance(self):
        """Get direct distance"""
        start = Point(self.start_point[0], self.start_point[1])
        goal = Point(self.goal_point[0], self.goal_point[1])
        return start.distance_to(goal)
    
    def get_shortest_path_length(self):
        """Get shortest path length"""
        return self.shortest_path_result['path_length']
    
    def get_path_complexity(self):
        """Get path complexity"""
        return self.get_shortest_path_length() / self.get_direct_distance()

class RobotCapabilityAnalyzer:
    """Robot capability analyzer"""
    
    def __init__(self, environment_analyzer: EnvironmentPathAnalyzer, safety_factor: float = 0.85):
        self.env_analyzer = environment_analyzer
        self.safety_factor = safety_factor
        self.required_path_length = environment_analyzer.get_shortest_path_length()
        self.direct_distance = environment_analyzer.get_direct_distance()
    
    def analyze_robot(self, link_lengths: List[float]) -> Dict:
        """Analyze robot capability relative to environment requirements"""
        total_length = sum(link_lengths)
        max_reach = total_length * self.safety_factor
        
        # Capability classification based on ACTUAL requirements
        if max_reach < self.direct_distance:
            capability = "insufficient_for_direct"
            reachable = False
            confidence = max_reach / self.direct_distance
        elif max_reach < self.required_path_length:
            capability = "insufficient_for_path" 
            reachable = False
            confidence = max_reach / self.required_path_length
        else:
            capability = "sufficient"
            reachable = True
            confidence = min(1.0, max_reach / self.required_path_length)
        
        return {
            'total_length': total_length,
            'max_reach': max_reach,
            'required_direct': self.direct_distance,
            'required_path': self.required_path_length,
            'reachable': reachable,
            'capability': capability,
            'confidence': confidence,
            'reach_ratio': max_reach / self.required_path_length,
            'length_gap': max(0, self.required_path_length - max_reach),
        }

class GeneticFitnessEvaluator:
    """遗传算法的分层Fitness评估器 - 使用验证过的路径分析"""
    
    def __init__(self, 
                 target_point: List[float] = [600, 550],
                 start_point: List[float] = [480, 620],
                 obstacles: List[Dict] = None):
        """
        Args:
            target_point: 目标点坐标
            start_point: 起始点坐标  
            obstacles: 障碍物列表
        """
        self.target_point = target_point
        self.start_point = start_point
        self.obstacles = obstacles or self._get_default_obstacles()
        
        # 1. 先分析环境 (与机器人无关)
        self.env_analyzer = EnvironmentPathAnalyzer(
            start_point, target_point, self.obstacles
        )
        
        # 2. 再分析机器人能力
        self.robot_analyzer = RobotCapabilityAnalyzer(self.env_analyzer)
        
        # 为了向后兼容，保留这个属性
        self.direct_distance = self.env_analyzer.get_direct_distance()
        
        print(f"🎯 Fitness评估器初始化 (使用验证过的路径分析)")
        print(f"   起始点: {start_point}")
        print(f"   目标点: {target_point}")
        print(f"   直线距离: {self.direct_distance:.1f}px")
        print(f"   最短路径: {self.env_analyzer.get_shortest_path_length():.1f}px")
        print(f"   路径复杂度: {self.env_analyzer.get_path_complexity():.2f}")
    
    def evaluate_fitness(self, link_lengths: List[float], 
                        training_performance: Dict = None) -> Dict:
        """
        评估机器人的综合fitness - 使用修复后的逻辑
        
        Args:
            link_lengths: 机器人链节长度列表
            training_performance: 训练性能数据 (可选)
        
        Returns:
            包含fitness分数和详细分析的字典
        """
        # 分析机器人能力
        robot_analysis = self.robot_analyzer.analyze_robot(link_lengths)
        
        # 基于能力等级的动态fitness策略
        if robot_analysis['capability'] == 'insufficient_for_direct':
            return self._evaluate_insufficient_for_direct(link_lengths, robot_analysis, training_performance)
        elif robot_analysis['capability'] == 'insufficient_for_path':
            return self._evaluate_insufficient_for_path(link_lengths, robot_analysis, training_performance)
        else:
            return self._evaluate_sufficient_robot(link_lengths, robot_analysis, training_performance)
    
    def _evaluate_insufficient_for_direct(self, link_lengths: List[float], 
                                        robot_analysis: Dict, training_performance: Dict = None) -> Dict:
        """评估连直线距离都达不到的机器人"""
        total_length = robot_analysis['total_length']
        
        print(f"📏 评估直线不足机器人: 总长{total_length:.1f} vs 直线{robot_analysis['required_direct']:.1f}")
        
        # 基础分数：基于接近直线距离的程度
        base_score = robot_analysis['confidence'] * 0.2  # 最高0.2分
        
        # 🎯 结构合理性奖励
        structure_bonus = self._evaluate_link_structure(link_lengths) * 0.1
        
        # 🎯 长度增长潜力奖励
        growth_potential = min(0.1, (total_length - 50) / max(1, robot_analysis['required_direct'] - 50))
        
        # 🎯 训练努力奖励
        training_bonus = 0.0
        if training_performance:
            max_distance = training_performance.get('max_distance_covered', 0)
            training_bonus = min(0.05, max_distance / robot_analysis['required_direct'] * 0.1)
        
        total_fitness = base_score + structure_bonus + growth_potential + training_bonus
        
        return {
            'fitness': total_fitness,
            'category': 'insufficient_length',
            'base_score': base_score,
            'structure_bonus': structure_bonus,
            'growth_potential': growth_potential,
            'training_bonus': training_bonus,
            'reachable': False,
            'confidence': robot_analysis['confidence'],
            'strategy': 'encourage_length_growth',
            'reason': f'长度不足，鼓励增长: {total_length:.1f}/{robot_analysis["required_direct"]:.1f}'
        }
    
    def _evaluate_insufficient_for_path(self, link_lengths: List[float], 
                                      robot_analysis: Dict, training_performance: Dict = None) -> Dict:
        """评估能达到直线但达不到绕行路径的机器人"""
        total_length = robot_analysis['total_length']
        
        print(f"⚖️ 评估路径不足机器人: 总长{total_length:.1f}")
        
        # 基础分数：比直线不足的要高
        base_score = 0.2 + robot_analysis['confidence'] * 0.3  # 0.2-0.5分
        
        # 🎯 结构合理性
        structure_bonus = self._evaluate_link_structure(link_lengths) * 0.1
        
        # 🎯 训练表现奖励 (有限，因为物理上无法成功)
        training_bonus = 0.0
        if training_performance:
            progress = training_performance.get('max_distance_covered', 0)
            near_success = training_performance.get('near_success_rate', 0)
            training_bonus = min(0.1, progress / robot_analysis['required_path'] * 0.05 + near_success * 0.05)
        
        total_fitness = base_score + structure_bonus + training_bonus
        
        return {
            'fitness': total_fitness,
            'category': 'marginal_length', 
            'base_score': base_score,
            'structure_bonus': structure_bonus,
            'training_bonus': training_bonus,
            'reachable': False,
            'confidence': robot_analysis['confidence'],
            'strategy': 'optimize_reachability',
            'reason': f'能达直线但达不到路径: {total_length:.1f}/{robot_analysis["required_path"]:.1f}'
        }
    
    def _evaluate_sufficient_robot(self, link_lengths: List[float], 
                                 robot_analysis: Dict, training_performance: Dict = None) -> Dict:
        """评估长度充足的机器人"""
        total_length = robot_analysis['total_length']
        
        print(f"✅ 评估长度充足机器人: 总长{total_length:.1f}")
        
        # 高基础分数：因为物理上可以成功
        base_score = 0.6
        
        # 🎯 主要看训练表现
        training_score = 0.0
        if training_performance:
            success_rate = training_performance.get('success_rate', 0)
            avg_reward = training_performance.get('avg_reward', 0)
            efficiency = training_performance.get('efficiency', 0)
            
            training_score = (
                success_rate * 0.25 +                    # 成功率权重最高
                min(0.1, avg_reward / 100) +            # 奖励缩放
                efficiency * 0.05                        # 效率奖励
            )
        
        # 🎯 结构优化奖励 (避免过度冗余)
        structure_score = self._evaluate_link_structure(link_lengths) * 0.05
        
        # 🎯 长度惩罚 (避免无意义的过长)
        optimal_length = robot_analysis['required_path'] * 1.2  # 20% 缓冲
        if total_length > optimal_length:
            length_penalty = (total_length - optimal_length) / optimal_length * 0.1
        else:
            length_penalty = 0.0
        
        total_fitness = base_score + training_score + structure_score - length_penalty
        
        return {
            'fitness': total_fitness,
            'category': 'sufficient_length',
            'base_score': base_score,
            'training_score': training_score,
            'structure_score': structure_score,
            'length_penalty': length_penalty,
            'reachable': True,
            'confidence': robot_analysis['confidence'],
            'strategy': 'optimize_performance',
            'reason': f'长度充足，优化训练表现: 成功率{training_performance.get("success_rate", 0):.2f}' if training_performance else '长度充足，等待训练数据'
        }
    
    def _evaluate_link_structure(self, link_lengths: List[float]) -> float:
        """评估链节结构的合理性"""
        if len(link_lengths) < 2:
            return 0.0
        
        # 🎯 长度均匀性 (避免极端不均匀)
        mean_length = np.mean(link_lengths)
        uniformity = 1.0 - np.std(link_lengths) / (mean_length + 1e-8)
        uniformity = max(0, min(1, uniformity))
        
        # 🎯 链节数量合理性
        num_links = len(link_lengths)
        if num_links < 2:
            count_score = 0.0
        elif num_links <= 4:  # 理想范围
            count_score = 1.0
        elif num_links <= 6:  # 可接受
            count_score = 0.7
        else:  # 过多
            count_score = 0.3
        
        # 🎯 最小长度检查 (避免过短的无效链节)
        min_useful_length = 20.0
        useful_links = sum(1 for length in link_lengths if length >= min_useful_length)
        useful_ratio = useful_links / len(link_lengths)
        
        structure_score = (uniformity * 0.4 + count_score * 0.4 + useful_ratio * 0.2)
        return structure_score
    
    def _get_default_obstacles(self) -> List[Dict]:
        """获取锯齿形障碍物配置 (来自reacher_with_zigzag_obstacles.yaml)"""
        return [
            # 上方锯齿障碍物
            {'shape': 'segment', 'points': [[500, 487], [550, 537]]},
            {'shape': 'segment', 'points': [[550, 537], [600, 487]]},
            {'shape': 'segment', 'points': [[600, 487], [650, 537]]},
            {'shape': 'segment', 'points': [[650, 537], [700, 487]]},
            # 下方锯齿障碍物  
            {'shape': 'segment', 'points': [[500, 612], [550, 662]]},
            {'shape': 'segment', 'points': [[550, 662], [600, 612]]},
            {'shape': 'segment', 'points': [[600, 612], [650, 662]]},
            {'shape': 'segment', 'points': [[650, 662], [700, 612]]},
        ]
    
    def compare_individuals(self, individual_a: Dict, individual_b: Dict) -> Dict:
        """比较两个个体的fitness"""
        fitness_a = individual_a['fitness_result']['fitness']
        fitness_b = individual_b['fitness_result']['fitness']
        
        return {
            'winner': 'A' if fitness_a > fitness_b else 'B',
            'fitness_diff': abs(fitness_a - fitness_b),
            'category_a': individual_a['fitness_result']['category'],
            'category_b': individual_b['fitness_result']['category'],
            'analysis': self._compare_analysis(individual_a['fitness_result'], individual_b['fitness_result'])
        }
    
    def _compare_analysis(self, result_a: Dict, result_b: Dict) -> str:
        """分析两个fitness结果的差异"""
        if result_a['category'] != result_b['category']:
            return f"不同类别: {result_a['category']} vs {result_b['category']}"
        
        if result_a['category'] == 'insufficient_length':
            return f"长度比较: {result_a.get('confidence', 0):.2f} vs {result_b.get('confidence', 0):.2f}"
        elif result_a['category'] == 'marginal_length':
            return f"可达性比较: {result_a.get('confidence', 0):.2f} vs {result_b.get('confidence', 0):.2f}"
        else:
            return f"训练表现比较: {result_a.get('training_score', 0):.2f} vs {result_b.get('training_score', 0):.2f}"


def test_genetic_fitness_evaluator():
    """测试遗传算法fitness评估器 - 使用验证过的路径分析"""
    print("🧪 测试遗传算法Fitness评估器 (使用验证过的路径分析)\n")
    
    evaluator = GeneticFitnessEvaluator()
    
    # 测试不同长度的机器人
    test_robots = [
        {
            'name': '严重不足机器人',
            'link_lengths': [30, 30, 30],  # 总长90
            'training_performance': {
                'max_distance_covered': 80,
                'success_rate': 0.0,
                'avg_reward': -50
            }
        },
        {
            'name': '勉强够机器人', 
            'link_lengths': [50, 50, 40],  # 总长140
            'training_performance': {
                'success_rate': 0.1,
                'avg_reward': 20,
                'efficiency': 0.3,
                'near_success_rate': 0.3
            }
        },
        {
            'name': '长度充足机器人',
            'link_lengths': [60, 60, 60],  # 总长180
            'training_performance': {
                'success_rate': 0.8,
                'avg_reward': 85,
                'efficiency': 0.7
            }
        },
        {
            'name': '过长机器人',
            'link_lengths': [80, 80, 80, 80],  # 总长320
            'training_performance': {
                'success_rate': 0.6,
                'avg_reward': 70,
                'efficiency': 0.4
            }
        }
    ]
    
    results = []
    
    for robot in test_robots:
        print(f"\n{'='*50}")
        print(f"🤖 评估: {robot['name']}")
        print(f"{'='*50}")
        
        result = evaluator.evaluate_fitness(
            link_lengths=robot['link_lengths'],
            training_performance=robot['training_performance']
        )
        
        results.append({
            'name': robot['name'],
            'link_lengths': robot['link_lengths'],
            'fitness_result': result
        })
        
        print(f"📊 Fitness结果:")
        print(f"   总分: {result['fitness']:.3f}")
        print(f"   类别: {result['category']}")
        print(f"   策略: {result['strategy']}")
        print(f"   原因: {result['reason']}")
        print(f"   可达: {result.get('reachable', 'N/A')}")
        
        # 详细分解
        print(f"📈 分数分解:")
        for key, value in result.items():
            if key.endswith('_score') or key.endswith('_bonus') or key.endswith('_penalty'):
                print(f"   {key}: {value:.3f}")
    
    # 比较分析
    print(f"\n{'='*50}")
    print(f"🏆 排名分析")
    print(f"{'='*50}")
    
    # 按fitness排序
    sorted_results = sorted(results, key=lambda x: x['fitness_result']['fitness'], reverse=True)
    
    for i, result in enumerate(sorted_results):
        print(f"#{i+1}: {result['name']} - Fitness: {result['fitness_result']['fitness']:.3f}")
        print(f"     策略: {result['fitness_result']['strategy']}")
    
    # 对比分析
    print(f"\n📊 对比分析:")
    best = sorted_results[0]
    worst = sorted_results[-1]
    
    comparison = evaluator.compare_individuals(best, worst)
    print(f"最佳 vs 最差: {comparison['analysis']}")
    print(f"Fitness差距: {comparison['fitness_diff']:.3f}")
    
    return evaluator, results


if __name__ == "__main__":
    evaluator, results = test_genetic_fitness_evaluator()