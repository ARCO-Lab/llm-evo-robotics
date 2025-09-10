import numpy as np
from typing import List, Dict, Tuple, Optional
import heapq
from dataclasses import dataclass
import math

@dataclass
class Point:
    x: float
    y: float
    
    def __iter__(self):
        return iter([self.x, self.y])
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class ShortestPathReachabilityEstimator:
    """基于最短路径算法的可达性估计器"""
    
    def __init__(self, safety_buffer: float = 15.0):
        """
        Args:
            safety_buffer: 与障碍物的安全距离(像素)
        """
        self.safety_buffer = safety_buffer
        
    def estimate_reachability(self, 
                            start_point: List[float], 
                            goal_point: List[float], 
                            link_lengths: List[float],
                            obstacles: List[Dict],
                            grid_resolution: int = 10) -> Dict:
        """
        基于最短路径的可达性估计
        
        Args:
            start_point: 起始点 [x, y]
            goal_point: 目标点 [x, y]
            link_lengths: 链节长度
            obstacles: 障碍物列表
            grid_resolution: 路径搜索的网格分辨率
        """
        start = Point(start_point[0], start_point[1])
        goal = Point(goal_point[0], goal_point[1])
        
        # 1. 基础长度检查
        direct_distance = start.distance_to(goal)
        max_reach = sum(link_lengths) * 0.85  # 85%安全系数
        
        if max_reach < direct_distance:
            return {
                'reachable': False,
                'confidence': 0.0,
                'reason': f'总长度不足: 需要{direct_distance:.1f}, 最大{max_reach:.1f}',
                'shortest_path_length': float('inf'),
                'direct_distance': direct_distance,
                'path_efficiency': 0.0,  # 🔧 添加缺失的字段
                'path_points': [],       # 🔧 添加缺失的字段
                'analysis_details': {    # 🔧 添加缺失的字段
                    'reachable': False,
                    'confidence': 0.0,
                    'reason': '总长度不足'
                }
            }
        
        # 2. 计算最短无碰撞路径
        shortest_path_result = self._find_shortest_collision_free_path(
            start, goal, obstacles, grid_resolution
        )
        
        # 3. 分析路径可达性
        path_analysis = self._analyze_path_reachability(
            shortest_path_result, link_lengths, obstacles
        )
        
        return {
            'reachable': path_analysis['reachable'],
            'confidence': path_analysis['confidence'],
            'reason': path_analysis['reason'],
            'shortest_path_length': shortest_path_result['path_length'],
            'direct_distance': direct_distance,
            'path_efficiency': direct_distance / shortest_path_result['path_length'] if shortest_path_result['path_length'] > 0 else 0,
            'path_points': shortest_path_result['path'],
            'analysis_details': path_analysis
        }
    
    def _find_shortest_collision_free_path(self, start: Point, goal: Point, 
                                         obstacles: List[Dict], grid_resolution: int) -> Dict:
        """使用A*算法找到最短无碰撞路径"""
        
        # 🎯 针对你的"W"形障碍物，我们可以用更智能的方法
        # 而不是暴力网格搜索
        
        # 1. 检查直线路径
        if self._is_path_collision_free(start, goal, obstacles):
            return {
                'path': [start, goal],
                'path_length': start.distance_to(goal),
                'method': 'direct'
            }
        
        # 2. 分析关键点（障碍物的拐点）
        key_points = self._extract_key_navigation_points(obstacles, start, goal)
        
        # 3. 使用Dijkstra算法在关键点间找最短路径
        shortest_path = self._dijkstra_on_key_points(start, goal, key_points, obstacles)
        
        return shortest_path
    
    def _extract_key_navigation_points(self, obstacles: List[Dict], 
                                     start: Point, goal: Point) -> List[Point]:
        """提取关键导航点"""
        key_points = []
        
        # 添加起点和终点
        key_points.extend([start, goal])
        
        # 🎯 针对"W"形障碍物的特殊处理
        obstacle_points = []
        for obstacle in obstacles:
            if obstacle.get('shape') == 'segment':
                points = obstacle['points']
                obstacle_points.extend([Point(p[0], p[1]) for p in points])
        
        if not obstacle_points:
            return key_points
        
        # 分析障碍物布局
        obstacle_bounds = self._get_obstacle_bounds_from_points(obstacle_points)
        
        # 添加绕行关键点
        buffer = self.safety_buffer
        
        # 上方绕行点
        upper_left = Point(obstacle_bounds['min_x'] - buffer, obstacle_bounds['min_y'] - buffer)
        upper_right = Point(obstacle_bounds['max_x'] + buffer, obstacle_bounds['min_y'] - buffer)
        
        # 下方绕行点  
        lower_left = Point(obstacle_bounds['min_x'] - buffer, obstacle_bounds['max_y'] + buffer)
        lower_right = Point(obstacle_bounds['max_x'] + buffer, obstacle_bounds['max_y'] + buffer)
        
        # 🎯 中间通道点（如果存在）
        channel_points = self._find_channel_navigation_points(obstacles)
        
        key_points.extend([upper_left, upper_right, lower_left, lower_right])
        key_points.extend(channel_points)
        
        # 过滤掉与障碍物碰撞的点
        valid_points = [p for p in key_points if not self._point_in_obstacles(p, obstacles)]
        
        return valid_points
    
    def _find_channel_navigation_points(self, obstacles: List[Dict]) -> List[Point]:
        """找到通道中的导航点"""
        # 分析上下障碍物组
        upper_obstacles = []
        lower_obstacles = []
        
        for obstacle in obstacles:
            if obstacle.get('shape') == 'segment':
                points = obstacle['points']
                avg_y = (points[0][1] + points[1][1]) / 2
                if avg_y < 550:  # 根据你的配置调整
                    upper_obstacles.append(obstacle)
                else:
                    lower_obstacles.append(obstacle)
        
        if not (upper_obstacles and lower_obstacles):
            return []
        
        # 计算通道边界
        upper_max_y = max(max(obs['points'][0][1], obs['points'][1][1]) for obs in upper_obstacles)
        lower_min_y = min(min(obs['points'][0][1], obs['points'][1][1]) for obs in lower_obstacles)
        
        gap_width = lower_min_y - upper_max_y
        
        # 如果通道足够宽，添加导航点
        if gap_width > self.safety_buffer * 2:
            channel_y = (upper_max_y + lower_min_y) / 2
            
            # 通道入口和出口点
            left_x = min(min(obs['points'][0][0], obs['points'][1][0]) for obs in upper_obstacles + lower_obstacles)
            right_x = max(max(obs['points'][0][0], obs['points'][1][0]) for obs in upper_obstacles + lower_obstacles)
            
            return [
                Point(left_x - self.safety_buffer, channel_y),
                Point(left_x + 50, channel_y),  # 通道内部点
                Point(right_x - 50, channel_y), # 通道内部点
                Point(right_x + self.safety_buffer, channel_y)
            ]
        
        return []
    
    def _dijkstra_on_key_points(self, start: Point, goal: Point, 
                               key_points: List[Point], obstacles: List[Dict]) -> Dict:
        """在关键点间使用Dijkstra算法"""
        
        # 构建图：计算所有关键点间的距离
        nodes = key_points
        graph = {}
        
        for i, node_a in enumerate(nodes):
            graph[i] = {}
            for j, node_b in enumerate(nodes):
                if i != j:
                    # 检查两点间是否可以直线连接
                    if self._is_path_collision_free(node_a, node_b, obstacles):
                        graph[i][j] = node_a.distance_to(node_b)
        
        # 找到起点和终点的索引
        start_idx = 0  # 起点总是第一个
        goal_idx = 1   # 终点总是第二个
        
        # Dijkstra算法
        distances = {i: float('inf') for i in range(len(nodes))}
        distances[start_idx] = 0
        previous = {}
        unvisited = set(range(len(nodes)))
        
        while unvisited:
            # 找到距离最小的未访问节点
            current = min(unvisited, key=lambda x: distances[x])
            
            if current == goal_idx:
                break
                
            unvisited.remove(current)
            
            # 更新邻居距离
            for neighbor, weight in graph.get(current, {}).items():
                if neighbor in unvisited:
                    new_distance = distances[current] + weight
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current
        
        # 重构路径
        if goal_idx not in previous and goal_idx != start_idx:
            return {
                'path': [],
                'path_length': float('inf'),
                'method': 'no_path_found'
            }
        
        # 回溯路径
        path = []
        current = goal_idx
        while current is not None:
            path.append(nodes[current])
            current = previous.get(current)
        
        path.reverse()
        
        return {
            'path': path,
            'path_length': distances[goal_idx],
            'method': 'dijkstra'
        }
    
    def _analyze_path_reachability(self, path_result: Dict, 
                                 link_lengths: List[float], obstacles: List[Dict]) -> Dict:
        """分析路径的可达性"""
        
        if path_result['path_length'] == float('inf'):
            return {
                'reachable': False,
                'confidence': 0.0,
                'reason': '无可用路径'
            }
        
        max_reach = sum(link_lengths) * 0.85  # 85%安全系数
        path_length = path_result['path_length']
        
        # 基础长度检查
        if path_length > max_reach:
            return {
                'reachable': False,
                'confidence': 0.0,
                'reason': f'路径太长: 需要{path_length:.1f}, 最大{max_reach:.1f}'
            }
        
        # 计算置信度
        length_ratio = path_length / max_reach
        confidence = 1.0 - length_ratio  # 路径越短置信度越高
        
        # 路径复杂度分析
        path_complexity = self._analyze_path_complexity(path_result['path'])
        complexity_penalty = path_complexity * 0.3
        
        final_confidence = max(0.1, confidence - complexity_penalty)
        
        return {
            'reachable': final_confidence > 0.3,  # 30%以上认为可达
            'confidence': final_confidence,
            'reason': f'最短路径长度{path_length:.1f}, 复杂度{path_complexity:.2f}',
            'path_complexity': path_complexity,
            'length_ratio': length_ratio
        }
    
    def _analyze_path_complexity(self, path: List[Point]) -> float:
        """分析路径复杂度（转弯次数和角度）"""
        if len(path) < 3:
            return 0.0
        
        total_turn_angle = 0.0
        turn_count = 0
        
        for i in range(1, len(path) - 1):
            # 计算转弯角度
            v1 = np.array([path[i].x - path[i-1].x, path[i].y - path[i-1].y])
            v2 = np.array([path[i+1].x - path[i].x, path[i+1].y - path[i].y])
            
            # 计算夹角
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            
            if angle > np.pi / 6:  # 大于30度才算转弯
                total_turn_angle += angle
                turn_count += 1
        
        # 归一化复杂度 (0-1)
        complexity = min(1.0, (turn_count * 0.2 + total_turn_angle / np.pi) / 2)
        return complexity
    
    # 辅助方法
    def _is_path_collision_free(self, start: Point, end: Point, obstacles: List[Dict]) -> bool:
        """检查两点间直线路径是否无碰撞"""
        for obstacle in obstacles:
            if obstacle.get('shape') == 'segment':
                if self._line_intersects_segment(start, end, obstacle['points'], self.safety_buffer):
                    return False
        return True
    
    def _line_intersects_segment(self, line_start: Point, line_end: Point, 
                               segment_points: List[List[float]], buffer: float) -> bool:
        """检查线段是否与障碍物线段相交（考虑安全缓冲）"""
        # 简化实现：检查线段到线段的最短距离
        seg_start = Point(segment_points[0][0], segment_points[0][1])
        seg_end = Point(segment_points[1][0], segment_points[1][1])
        
        distance = self._segment_to_segment_distance(line_start, line_end, seg_start, seg_end)
        return distance < buffer
    
    def _segment_to_segment_distance(self, seg1_start: Point, seg1_end: Point, 
                                   seg2_start: Point, seg2_end: Point) -> float:
        """计算两线段间最短距离"""
        # 实现省略，使用几何算法计算
        # 这里返回一个简化的点到线段距离
        return min(
            self._point_to_segment_distance(seg1_start, seg2_start, seg2_end),
            self._point_to_segment_distance(seg1_end, seg2_start, seg2_end),
            self._point_to_segment_distance(seg2_start, seg1_start, seg1_end),
            self._point_to_segment_distance(seg2_end, seg1_start, seg1_end)
        )
    
    def _point_to_segment_distance(self, point: Point, seg_start: Point, seg_end: Point) -> float:
        """计算点到线段的最短距离"""
        # 向量计算
        seg_vec = np.array([seg_end.x - seg_start.x, seg_end.y - seg_start.y])
        point_vec = np.array([point.x - seg_start.x, point.y - seg_start.y])
        
        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq == 0:
            return np.linalg.norm(point_vec)
        
        t = max(0, min(1, np.dot(point_vec, seg_vec) / seg_len_sq))
        projection = seg_start.x + t * seg_vec[0], seg_start.y + t * seg_vec[1]
        
        return math.sqrt((point.x - projection[0])**2 + (point.y - projection[1])**2)
    
    def _get_obstacle_bounds_from_points(self, points: List[Point]) -> Dict:
        """从点列表获取边界"""
        if not points:
            return {}
        
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        
        return {
            'min_x': min(xs),
            'max_x': max(xs),
            'min_y': min(ys),
            'max_y': max(ys)
        }
    
    def _point_in_obstacles(self, point: Point, obstacles: List[Dict]) -> bool:
        """检查点是否在障碍物内"""
        for obstacle in obstacles:
            if obstacle.get('shape') == 'segment':
                # 检查点是否太靠近线段
                seg_start = Point(obstacle['points'][0][0], obstacle['points'][0][1])
                seg_end = Point(obstacle['points'][1][0], obstacle['points'][1][1])
                distance = self._point_to_segment_distance(point, seg_start, seg_end)
                if distance < self.safety_buffer:
                    return True
        return False

# 使用示例
def test_shortest_path_reachability():
    """测试基于最短路径的可达性估计"""
    
    # 从配置文件加载障碍物
    obstacles = [
        {'shape': 'segment', 'points': [[500, 487], [550, 537]]},
        {'shape': 'segment', 'points': [[550, 537], [600, 487]]},
        {'shape': 'segment', 'points': [[600, 487], [650, 537]]},
        {'shape': 'segment', 'points': [[650, 537], [700, 487]]},
        {'shape': 'segment', 'points': [[500, 612], [550, 662]]},
        {'shape': 'segment', 'points': [[550, 662], [600, 612]]},
        {'shape': 'segment', 'points': [[600, 612], [650, 662]]},
        {'shape': 'segment', 'points': [[650, 662], [700, 612]]},
    ]
    
    estimator = ShortestPathReachabilityEstimator(safety_buffer=15.0)
    
    # 测试不同长度的机器人
    test_cases = [
        {'name': '短臂', 'lengths': [40, 40, 40]},    # 总长120
        {'name': '中臂', 'lengths': [60, 60, 60]},    # 总长180  
        {'name': '长臂', 'lengths': [80, 80, 80]},    # 总长240
    ]
    
    for case in test_cases:
        result = estimator.estimate_reachability(
            start_point=[480, 620],
            goal_point=[600, 550],
            link_lengths=case['lengths'],
            obstacles=obstacles
        )
        
        print(f"\n{case['name']}机器人 (总长{sum(case['lengths'])}):")
        print(f"  可达性: {result['reachable']}")
        print(f"  置信度: {result['confidence']:.2f}")
        print(f"  直线距离: {result['direct_distance']:.1f}")
        print(f"  最短路径: {result['shortest_path_length']:.1f}")
        print(f"  路径效率: {result['path_efficiency']:.2f}")
        print(f"  原因: {result['reason']}")

if __name__ == "__main__":
    test_shortest_path_reachability()