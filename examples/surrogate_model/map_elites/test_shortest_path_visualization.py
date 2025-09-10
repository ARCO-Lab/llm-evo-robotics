"""
Fixed Path Visualization Test
Corrected collision detection algorithm
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../2d_reacher/envs/'))

import numpy as np
import pygame
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class EnvironmentPathAnalyzer:
    """Environment path analyzer with fixed collision detection"""
    
    def __init__(self, start_point: List[float], goal_point: List[float], 
                 obstacles: List[Dict], safety_buffer: float = 15.0):
        self.start_point = start_point
        self.goal_point = goal_point
        self.obstacles = obstacles
        self.safety_buffer = safety_buffer
        
        # Calculate environment shortest path immediately
        self.shortest_path_result = self._compute_environment_shortest_path()
        
        print("Environment Path Analysis Complete:")
        print(f"   Direct distance: {self.get_direct_distance():.1f}px")
        print(f"   Shortest path: {self.get_shortest_path_length():.1f}px")
        print(f"   Path complexity: {self.get_path_complexity():.2f}")
        print(f"   Path method: {self.shortest_path_result['method']}")
        
        # Debug: Check if direct path has collision
        start = Point(self.start_point[0], self.start_point[1])
        goal = Point(self.goal_point[0], self.goal_point[1])
        is_direct_clear = self._is_path_collision_free(start, goal)
        print(f"   Direct path collision-free: {is_direct_clear}")
    
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
        print("   Direct path blocked, calculating detour...")
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
        
        print(f"   Building graph with {n} key points...")
        connections = 0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if path between key_points[i] and key_points[j] is collision-free
                    if self._is_path_collision_free(key_points[i], key_points[j]):
                        distance = key_points[i].distance_to(key_points[j])
                        graph[i][j] = distance
                        connections += 1
        
        print(f"   Found {connections} valid connections")
        
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
            print("   WARNING: No collision-free path found!")
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
        
        print(f"   Found detour path with {len(path)} waypoints")
        
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
    
    def get_path_points(self):
        """Get path points"""
        return self.shortest_path_result['path']
    
    def get_key_points(self):
        """Get key navigation points"""
        return self.shortest_path_result.get('key_points', [])

class PathVisualizationTester:
    """Fixed path visualization tester"""
    
    def __init__(self, width: int = 1000, height: int = 800):
        """Initialize visualization environment"""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Fixed Path Visualization")
        
        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Color definitions
        self.colors = {
            'background': (240, 240, 240),
            'obstacle': (100, 100, 100),
            'start': (0, 255, 0),
            'goal': (255, 0, 0),
            'direct_path': (255, 100, 100),
            'shortest_path': (0, 100, 255),
            'key_points': (255, 165, 0),
            'text': (0, 0, 0),
            'grid': (200, 200, 200),
            'blocked_path': (255, 0, 0)
        }
        
        # Configure start/goal points and obstacles
        self.obstacles = self._load_reacher2d_obstacles()
        self.start_point = [580, 720]  # Test with the problematic start point
        self.goal_point = [600, 550]
        
        # Analyze environment (robot-independent)
        self.env_analyzer = EnvironmentPathAnalyzer(
            self.start_point, self.goal_point, self.obstacles
        )
        
        print("Fixed Path Visualization Tester initialized")
    
    def _load_reacher2d_obstacles(self) -> List[Dict]:
        """Load zigzag obstacles from reacher2d_env"""
        return [
            # Upper zigzag obstacles
            {'shape': 'segment', 'points': [[500, 487], [550, 537]]},
            {'shape': 'segment', 'points': [[550, 537], [600, 487]]},
            {'shape': 'segment', 'points': [[600, 487], [650, 537]]},
            {'shape': 'segment', 'points': [[650, 537], [700, 487]]},
            # Lower zigzag obstacles  
            {'shape': 'segment', 'points': [[500, 612], [550, 662]]},
            {'shape': 'segment', 'points': [[550, 662], [600, 612]]},
            {'shape': 'segment', 'points': [[600, 612], [650, 662]]},
            {'shape': 'segment', 'points': [[650, 662], [700, 612]]},
        ]
    
    def visualize_paths(self):
        """Visualize both direct and shortest paths"""
        clock = pygame.time.Clock()
        show_key_points = True
        show_samples = False
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_k:
                        show_key_points = not show_key_points
                        print(f"Show key points: {show_key_points}")
                    elif event.key == pygame.K_s:
                        show_samples = not show_samples
                        print(f"Show collision samples: {show_samples}")
                    elif event.key == pygame.K_q:
                        running = False
            
            # Clear screen
            self.screen.fill(self.colors['background'])
            
            # Draw grid
            self._draw_grid()
            
            # Draw obstacles
            self._draw_obstacles()
            
            # Draw start and goal points
            self._draw_start_goal()
            
            # Draw direct path
            self._draw_direct_path()
            
            # Draw collision samples if enabled
            if show_samples:
                self._draw_collision_samples()
            
            # Draw shortest path
            self._draw_shortest_path(show_key_points)
            
            # Draw info panel
            self._draw_info_panel()
            
            # Draw controls
            self._draw_controls()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
    
    def _draw_grid(self):
        """Draw grid"""
        for x in range(0, self.width, 50):
            pygame.draw.line(self.screen, self.colors['grid'], (x, 0), (x, self.height), 1)
        for y in range(0, self.height, 50):
            pygame.draw.line(self.screen, self.colors['grid'], (0, y), (self.width, y), 1)
    
    def _draw_obstacles(self):
        """Draw obstacles"""
        for obstacle in self.obstacles:
            if obstacle.get('shape') == 'segment':
                points = obstacle['points']
                start_pos = (int(points[0][0]), int(points[0][1]))
                end_pos = (int(points[1][0]), int(points[1][1]))
                
                # Obstacle segments
                pygame.draw.line(self.screen, self.colors['obstacle'], start_pos, end_pos, 8)
                
                # Safety buffer
                buffer_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.line(buffer_surface, (*self.colors['obstacle'], 50), start_pos, end_pos, 30)
                self.screen.blit(buffer_surface, (0, 0))
    
    def _draw_start_goal(self):
        """Draw start and goal points"""
        # Start point (green)
        start_pos = (int(self.start_point[0]), int(self.start_point[1]))
        pygame.draw.circle(self.screen, self.colors['start'], start_pos, 15)
        pygame.draw.circle(self.screen, (255, 255, 255), start_pos, 12)
        pygame.draw.circle(self.screen, self.colors['start'], start_pos, 8)
        
        # Goal point (red)
        goal_pos = (int(self.goal_point[0]), int(self.goal_point[1]))
        pygame.draw.circle(self.screen, self.colors['goal'], goal_pos, 15)
        pygame.draw.circle(self.screen, (255, 255, 255), goal_pos, 12)
        pygame.draw.circle(self.screen, self.colors['goal'], goal_pos, 8)
    
    def _draw_direct_path(self):
        """Draw direct path (red dashed line)"""
        start_pos = (int(self.start_point[0]), int(self.start_point[1]))
        goal_pos = (int(self.goal_point[0]), int(self.goal_point[1]))
        
        # Check if direct path is blocked
        start = Point(self.start_point[0], self.start_point[1])
        goal = Point(self.goal_point[0], self.goal_point[1])
        is_blocked = not self.env_analyzer._is_path_collision_free(start, goal)
        
        color = self.colors['blocked_path'] if is_blocked else self.colors['direct_path']
        self._draw_dashed_line(start_pos, goal_pos, color, 3)
    
    def _draw_collision_samples(self):
        """Draw collision detection sample points"""
        start = Point(self.start_point[0], self.start_point[1])
        goal = Point(self.goal_point[0], self.goal_point[1])
        
        num_samples = 20
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_x = start.x + t * (goal.x - start.x)
            sample_y = start.y + t * (goal.y - start.y)
            sample_point = Point(sample_x, sample_y)
            
            # Check if this sample is in collision
            in_collision = False
            for obstacle in self.obstacles:
                if obstacle.get('shape') == 'segment':
                    seg_start = Point(obstacle['points'][0][0], obstacle['points'][0][1])
                    seg_end = Point(obstacle['points'][1][0], obstacle['points'][1][1])
                    
                    distance = self.env_analyzer._point_to_segment_distance(sample_point, seg_start, seg_end)
                    if distance < self.env_analyzer.safety_buffer:
                        in_collision = True
                        break
            
            # Draw sample point
            color = (255, 0, 0) if in_collision else (0, 255, 0)
            pos = (int(sample_x), int(sample_y))
            pygame.draw.circle(self.screen, color, pos, 3)
    
    def _draw_shortest_path(self, show_key_points: bool):
        """Draw shortest obstacle-free path"""
        path_points = self.env_analyzer.get_path_points()
        method = self.env_analyzer.shortest_path_result['method']
        
        if len(path_points) >= 2:
            # Convert to pygame coordinates
            pygame_points = [(int(p.x), int(p.y)) for p in path_points]
            
            # Choose color based on method
            if method == 'direct_blocked':
                color = self.colors['blocked_path']
                width = 4
            else:
                color = self.colors['shortest_path']
                width = 6
            
            # Draw path
            for i in range(len(pygame_points) - 1):
                start_pos = pygame_points[i]
                end_pos = pygame_points[i + 1]
                pygame.draw.line(self.screen, color, start_pos, end_pos, width)
            
            # Draw key points if enabled and it's a detour path
            if show_key_points and method != 'direct' and method != 'direct_blocked':
                for point in pygame_points[1:-1]:  # Exclude start and goal
                    pygame.draw.circle(self.screen, self.colors['key_points'], point, 8)
                    pygame.draw.circle(self.screen, (255, 255, 255), point, 5)
    
    def _draw_info_panel(self):
        """Draw information panel"""
        panel_x = self.width - 300
        panel_y = 10
        
        # Background
        panel_rect = pygame.Rect(panel_x - 10, panel_y - 5, 290, 180)
        pygame.draw.rect(self.screen, (255, 255, 255), panel_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), panel_rect, 2)
        
        y_offset = 10
        
        # Check if direct path is blocked
        start = Point(self.start_point[0], self.start_point[1])
        goal = Point(self.goal_point[0], self.goal_point[1])
        is_direct_blocked = not self.env_analyzer._is_path_collision_free(start, goal)
        
        # Path information
        info_lines = [
            "Path Analysis:",
            f"Start: ({self.start_point[0]}, {self.start_point[1]})",
            f"Goal: ({self.goal_point[0]}, {self.goal_point[1]})",
            f"Direct distance: {self.env_analyzer.get_direct_distance():.1f}px",
            f"Direct blocked: {'YES' if is_direct_blocked else 'NO'}",
            f"Shortest path: {self.env_analyzer.get_shortest_path_length():.1f}px",
            f"Path complexity: {self.env_analyzer.get_path_complexity():.2f}",
            f"Method: {self.env_analyzer.shortest_path_result['method']}",
            "",
            "Legend:",
            "Red dashed = Direct path",
            "Blue solid = Shortest path",
            "Red solid = Blocked path"
        ]
        
        for line in info_lines:
            if line:
                text_surface = self.small_font.render(line, True, self.colors['text'])
                self.screen.blit(text_surface, (panel_x, panel_y + y_offset))
            y_offset += 14
    
    def _draw_controls(self):
        """Draw control instructions"""
        controls = [
            "Controls:",
            "K - Toggle key points",
            "S - Toggle collision samples",
            "Q - Quit"
        ]
        
        y_start = self.height - 100
        for i, control in enumerate(controls):
            color = self.colors['text'] if i == 0 else (100, 100, 100)
            text_surface = self.small_font.render(control, True, color)
            self.screen.blit(text_surface, (10, y_start + i * 20))
    
    def _draw_dashed_line(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int], 
                         color: Tuple[int, int, int], width: int, dash_length: int = 10):
        """Draw dashed line"""
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dashes = int(distance / dash_length)
        
        for i in range(0, dashes, 2):
            start_ratio = i / dashes
            end_ratio = min((i + 1) / dashes, 1.0)
            
            dash_start = (
                int(x1 + (x2 - x1) * start_ratio),
                int(y1 + (y2 - y1) * start_ratio)
            )
            dash_end = (
                int(x1 + (x2 - x1) * end_ratio),
                int(y1 + (y2 - y1) * end_ratio)
            )
            
            pygame.draw.line(self.screen, color, dash_start, dash_end, width)

def main():
    """Main test function"""
    print("Starting FIXED Path Visualization Test")
    print("=" * 50)
    
    # Create visualization tester
    tester = PathVisualizationTester()
    
    print("\nVisualization ready:")
    print("Red dashed line = Direct path (blocked)")
    print("Blue solid line = Shortest detour path")
    print("Press S to see collision detection samples")
    
    # Start visualization
    tester.visualize_paths()
    
    print("\nVisualization test completed")

if __name__ == "__main__":
    main()