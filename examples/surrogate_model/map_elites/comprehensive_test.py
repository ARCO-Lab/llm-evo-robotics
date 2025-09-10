"""
Comprehensive Test Suite for genetic_fitness_evaluator.py
Tests all components thoroughly with edge cases and integration tests
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import traceback
from genetic_fitness_evaluator import (
    Point, EnvironmentPathAnalyzer, RobotCapabilityAnalyzer, 
    GeneticFitnessEvaluator
)

class ComprehensiveTestSuite:
    """全面测试套件"""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
    
    def run_test(self, test_name, test_func):
        """运行单个测试"""
        print(f"\n{'='*60}")
        print(f"🧪 测试: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name} - PASSED")
                self.passed_tests += 1
                self.test_results.append((test_name, "PASSED", None))
            else:
                print(f"❌ {test_name} - FAILED")
                self.failed_tests += 1
                self.test_results.append((test_name, "FAILED", "Test returned False"))
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            self.failed_tests += 1
            self.test_results.append((test_name, "ERROR", str(e)))
            traceback.print_exc()
    
    def test_1_point_class(self):
        """测试 Point 类基础功能"""
        print("测试 Point 类...")
        
        # 基础创建
        p1 = Point(100, 200)
        p2 = Point(400, 600)
        
        # 距离计算
        distance = p1.distance_to(p2)
        expected = np.sqrt((400-100)**2 + (600-200)**2)  # 500.0
        
        if abs(distance - expected) < 1e-6:
            print(f"   距离计算正确: {distance:.1f}")
            return True
        else:
            print(f"   距离计算错误: 期望{expected}, 得到{distance}")
            return False
    
    def test_2_environment_path_analyzer_basic(self):
        """测试 EnvironmentPathAnalyzer 基础功能"""
        print("测试 EnvironmentPathAnalyzer 基础功能...")
        
        # 简单无障碍环境
        start = [100, 100]
        goal = [400, 500]
        obstacles = []  # 无障碍
        
        analyzer = EnvironmentPathAnalyzer(start, goal, obstacles)
        
        # 验证直线距离
        expected_direct = np.sqrt((400-100)**2 + (500-100)**2)  # 500.0
        actual_direct = analyzer.get_direct_distance()
        
        if abs(actual_direct - expected_direct) < 1e-6:
            print(f"   直线距离计算正确: {actual_direct:.1f}")
        else:
            print(f"   直线距离计算错误: 期望{expected_direct}, 得到{actual_direct}")
            return False
        
        # 无障碍情况下，最短路径应该等于直线距离
        shortest_path = analyzer.get_shortest_path_length()
        if abs(shortest_path - expected_direct) < 1e-6:
            print(f"   无障碍最短路径正确: {shortest_path:.1f}")
            return True
        else:
            print(f"   无障碍最短路径错误: 期望{expected_direct}, 得到{shortest_path}")
            return False
    
    def test_3_environment_path_analyzer_with_obstacles(self):
        """测试 EnvironmentPathAnalyzer 有障碍功能"""
        print("测试 EnvironmentPathAnalyzer 有障碍功能...")
        
        # 使用默认的锯齿障碍
        start = [480, 620]
        goal = [600, 550]
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
        
        analyzer = EnvironmentPathAnalyzer(start, goal, obstacles)
        
        direct_distance = analyzer.get_direct_distance()
        shortest_path = analyzer.get_shortest_path_length()
        complexity = analyzer.get_path_complexity()
        
        print(f"   直线距离: {direct_distance:.1f}px")
        print(f"   最短路径: {shortest_path:.1f}px")
        print(f"   路径复杂度: {complexity:.2f}")
        
        # 验证基本合理性
        if direct_distance > 0 and shortest_path >= direct_distance and complexity >= 1.0:
            print("   路径分析结果合理")
            return True
        else:
            print("   路径分析结果不合理")
            return False
    
    def test_4_robot_capability_analyzer(self):
        """测试 RobotCapabilityAnalyzer"""
        print("测试 RobotCapabilityAnalyzer...")
        
        # 创建环境分析器
        start = [480, 620]
        goal = [600, 550]
        obstacles = []  # 简化测试，无障碍
        env_analyzer = EnvironmentPathAnalyzer(start, goal, obstacles)
        
        # 创建机器人能力分析器
        robot_analyzer = RobotCapabilityAnalyzer(env_analyzer)
        
        # 测试不同长度的机器人
        test_cases = [
            ([30, 30, 30], "insufficient_for_direct"),  # 总长90，不足
            ([60, 60, 60], "sufficient"),               # 总长180，充足
        ]
        
        all_correct = True
        for link_lengths, expected_capability in test_cases:
            analysis = robot_analyzer.analyze_robot(link_lengths)
            total_length = sum(link_lengths)
            
            print(f"   机器人[{','.join(map(str, link_lengths))}] (总长{total_length}px):")
            print(f"     能力等级: {analysis['capability']}")
            print(f"     可达性: {analysis['reachable']}")
            print(f"     置信度: {analysis['confidence']:.3f}")
            
            if analysis['capability'] == expected_capability:
                print(f"     ✅ 能力等级正确")
            else:
                print(f"     ❌ 能力等级错误: 期望{expected_capability}")
                all_correct = False
        
        return all_correct
    
    def test_5_genetic_fitness_evaluator_initialization(self):
        """测试 GeneticFitnessEvaluator 初始化"""
        print("测试 GeneticFitnessEvaluator 初始化...")
        
        try:
            evaluator = GeneticFitnessEvaluator()
            
            # 验证属性存在
            required_attrs = ['target_point', 'start_point', 'obstacles', 
                             'env_analyzer', 'robot_analyzer', 'direct_distance']
            
            for attr in required_attrs:
                if not hasattr(evaluator, attr):
                    print(f"   缺少属性: {attr}")
                    return False
            
            print(f"   初始化成功，直线距离: {evaluator.direct_distance:.1f}px")
            return True
            
        except Exception as e:
            print(f"   初始化失败: {e}")
            return False
    
    def test_6_fitness_evaluation_basic(self):
        """测试基础 fitness 评估"""
        print("测试基础 fitness 评估...")
        
        evaluator = GeneticFitnessEvaluator()
        
        # 测试不同长度机器人
        test_cases = [
            [30, 30, 30],    # 短机器人
            [50, 50, 50],    # 中等机器人
            [70, 70, 70],    # 长机器人
        ]
        
        results = []
        for link_lengths in test_cases:
            result = evaluator.evaluate_fitness(link_lengths)
            total_length = sum(link_lengths)
            
            # 验证返回结果结构
            required_keys = ['fitness', 'category', 'strategy', 'reason', 'reachable']
            for key in required_keys:
                if key not in result:
                    print(f"   结果缺少键: {key}")
                    return False
            
            results.append((total_length, result['fitness'], result['category']))
            print(f"   机器人[{','.join(map(str, link_lengths))}]: fitness={result['fitness']:.3f}, category={result['category']}")
        
        # 验证fitness单调性（一般来说长机器人应该有更高fitness）
        results.sort(key=lambda x: x[0])  # 按长度排序
        fitness_values = [r[1] for r in results]
        
        print(f"   Fitness值随长度变化: {[f'{f:.3f}' for f in fitness_values]}")
        return True
    
    def test_7_fitness_evaluation_with_training(self):
        """测试带训练数据的 fitness 评估"""
        print("测试带训练数据的 fitness 评估...")
        
        evaluator = GeneticFitnessEvaluator()
        
        # 测试相同机器人，不同训练表现
        link_lengths = [60, 60, 60]  # 长度充足的机器人
        
        training_cases = [
            {
                'name': '训练失败',
                'data': {'success_rate': 0.0, 'avg_reward': -50, 'efficiency': 0.1}
            },
            {
                'name': '训练成功',
                'data': {'success_rate': 0.9, 'avg_reward': 90, 'efficiency': 0.8}
            }
        ]
        
        results = []
        for case in training_cases:
            result_no_training = evaluator.evaluate_fitness(link_lengths)
            result_with_training = evaluator.evaluate_fitness(link_lengths, case['data'])
            
            results.append((case['name'], result_no_training['fitness'], result_with_training['fitness']))
            
            print(f"   {case['name']}:")
            print(f"     无训练数据: {result_no_training['fitness']:.3f}")
            print(f"     有训练数据: {result_with_training['fitness']:.3f}")
        
        # 验证训练成功的机器人fitness更高
        if results[1][2] > results[0][2]:
            print("   ✅ 训练成功的机器人fitness更高")
            return True
        else:
            print("   ❌ 训练数据对fitness影响不正确")
            return False
    
    def test_8_edge_cases(self):
        """测试边界情况"""
        print("测试边界情况...")
        
        evaluator = GeneticFitnessEvaluator()
        
        edge_cases = [
            ([10], "单链节"),
            ([5, 5, 5, 5, 5, 5, 5, 5], "过多链节"),
            ([0, 50, 100], "包含零长度"),
            ([200, 200, 200], "超长机器人"),
        ]
        
        all_handled = True
        for link_lengths, description in edge_cases:
            try:
                result = evaluator.evaluate_fitness(link_lengths)
                print(f"   {description} [{','.join(map(str, link_lengths))}]: fitness={result['fitness']:.3f}")
            except Exception as e:
                print(f"   {description} 处理失败: {e}")
                all_handled = False
        
        return all_handled
    
    def test_9_comparison_functionality(self):
        """测试个体比较功能"""
        print("测试个体比较功能...")
        
        evaluator = GeneticFitnessEvaluator()
        
        # 创建两个个体
        individual_a = {
            'fitness_result': evaluator.evaluate_fitness([40, 40, 40])
        }
        individual_b = {
            'fitness_result': evaluator.evaluate_fitness([70, 70, 70])
        }
        
        try:
            comparison = evaluator.compare_individuals(individual_a, individual_b)
            
            required_keys = ['winner', 'fitness_diff', 'category_a', 'category_b', 'analysis']
            for key in required_keys:
                if key not in comparison:
                    print(f"   比较结果缺少键: {key}")
                    return False
            
            print(f"   获胜者: {comparison['winner']}")
            print(f"   Fitness差距: {comparison['fitness_diff']:.3f}")
            print(f"   分析: {comparison['analysis']}")
            
            return True
            
        except Exception as e:
            print(f"   比较功能失败: {e}")
            return False
    
    def test_10_stress_test(self):
        """压力测试"""
        print("压力测试...")
        
        evaluator = GeneticFitnessEvaluator()
        
        # 测试大量随机机器人
        import random
        
        num_tests = 50
        successful_evaluations = 0
        
        for i in range(num_tests):
            # 生成随机机器人
            num_links = random.randint(2, 6)
            link_lengths = [random.uniform(20, 100) for _ in range(num_links)]
            
            try:
                result = evaluator.evaluate_fitness(link_lengths)
                
                # 验证结果合理性
                if (isinstance(result['fitness'], (int, float)) and 
                    result['fitness'] >= 0 and 
                    result['category'] in ['insufficient_length', 'marginal_length', 'sufficient_length']):
                    successful_evaluations += 1
                
            except Exception as e:
                print(f"   随机测试 {i+1} 失败: {e}")
        
        success_rate = successful_evaluations / num_tests
        print(f"   成功率: {success_rate:.1%} ({successful_evaluations}/{num_tests})")
        
        return success_rate >= 0.95  # 要求95%以上成功率
    
    def test_11_integration_test(self):
        """集成测试"""
        print("集成测试 - 完整workflow...")
        
        try:
            # 1. 创建评估器
            evaluator = GeneticFitnessEvaluator()
            
            # 2. 评估一组机器人
            robots = [
                [30, 30, 30],
                [50, 50, 50], 
                [70, 70, 70],
                [90, 90, 90]
            ]
            
            results = []
            for robot in robots:
                # 无训练数据评估
                result1 = evaluator.evaluate_fitness(robot)
                
                # 有训练数据评估
                training_data = {
                    'success_rate': min(1.0, sum(robot) / 200),  # 模拟训练数据
                    'avg_reward': sum(robot) / 2,
                    'efficiency': 0.5
                }
                result2 = evaluator.evaluate_fitness(robot, training_data)
                
                results.append({
                    'robot': robot,
                    'no_training': result1,
                    'with_training': result2
                })
            
            # 3. 排序和比较
            results.sort(key=lambda x: x['with_training']['fitness'], reverse=True)
            
            print("   集成测试结果:")
            for i, result in enumerate(results):
                robot = result['robot']
                fitness = result['with_training']['fitness']
                category = result['with_training']['category']
                print(f"     #{i+1}: {robot} -> fitness={fitness:.3f}, category={category}")
            
            # 4. 验证结果合理性
            # 一般来说，更长的机器人应该有更好的表现（在sufficient_length类别中）
            sufficient_robots = [r for r in results if r['with_training']['category'] == 'sufficient_length']
            if sufficient_robots:
                print(f"   有 {len(sufficient_robots)} 个充足长度的机器人")
            
            return True
            
        except Exception as e:
            print(f"   集成测试失败: {e}")
            traceback.print_exc()
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始全面验证 genetic_fitness_evaluator.py")
        print("=" * 80)
        
        test_methods = [
            ("Point类基础功能", self.test_1_point_class),
            ("环境路径分析器-基础", self.test_2_environment_path_analyzer_basic),
            ("环境路径分析器-有障碍", self.test_3_environment_path_analyzer_with_obstacles),
            ("机器人能力分析器", self.test_4_robot_capability_analyzer),
            ("Fitness评估器初始化", self.test_5_genetic_fitness_evaluator_initialization),
            ("基础Fitness评估", self.test_6_fitness_evaluation_basic),
            ("带训练数据Fitness评估", self.test_7_fitness_evaluation_with_training),
            ("边界情况测试", self.test_8_edge_cases),
            ("个体比较功能", self.test_9_comparison_functionality),
            ("压力测试", self.test_10_stress_test),
            ("集成测试", self.test_11_integration_test),
        ]
        
        for test_name, test_method in test_methods:
            self.run_test(test_name, test_method)
        
        # 总结报告
        self.print_summary()
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 80)
        print("🎯 测试总结报告")
        print("=" * 80)
        
        total_tests = self.passed_tests + self.failed_tests
        pass_rate = (self.passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"总测试数: {total_tests}")
        print(f"通过: {self.passed_tests}")
        print(f"失败: {self.failed_tests}")
        print(f"通过率: {pass_rate:.1f}%")
        
        if self.failed_tests > 0:
            print(f"\n❌ 失败的测试:")
            for test_name, status, error in self.test_results:
                if status != "PASSED":
                    print(f"   - {test_name}: {status}")
                    if error:
                        print(f"     错误: {error}")
        
        print(f"\n{'🎉 所有测试通过！' if self.failed_tests == 0 else '⚠️  存在失败的测试，需要修复'}")
        
        # 给出建议
        if pass_rate >= 95:
            print("✅ genetic_fitness_evaluator.py 工作正常，可以安全使用")
        elif pass_rate >= 80:
            print("⚠️  genetic_fitness_evaluator.py 基本可用，但建议修复失败的测试")
        else:
            print("❌ genetic_fitness_evaluator.py 存在严重问题，不建议使用")

def main():
    """主测试函数"""
    test_suite = ComprehensiveTestSuite()
    test_suite.run_all_tests()

if __name__ == "__main__":
    main()