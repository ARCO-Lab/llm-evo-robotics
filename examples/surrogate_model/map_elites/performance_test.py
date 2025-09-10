"""
Performance Test for genetic_fitness_evaluator.py
Tests execution time and memory usage
"""

import time
import psutil
import os
from genetic_fitness_evaluator import GeneticFitnessEvaluator

def performance_test():
    """性能测试"""
    print("🚀 性能测试 genetic_fitness_evaluator.py")
    print("=" * 60)
    
    # 1. 初始化性能测试
    print("\n📊 初始化性能测试...")
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    
    evaluator = GeneticFitnessEvaluator()
    
    init_time = time.time() - start_time
    init_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 - start_memory
    
    print(f"   初始化时间: {init_time:.3f}s")
    print(f"   内存使用: {init_memory:.1f}MB")
    
    # 2. 单次评估性能测试
    print("\n📊 单次评估性能测试...")
    test_robot = [60, 60, 60]
    training_data = {'success_rate': 0.8, 'avg_reward': 85, 'efficiency': 0.7}
    
    # 预热
    for _ in range(5):
        evaluator.evaluate_fitness(test_robot, training_data)
    
    # 正式测试
    num_tests = 100
    start_time = time.time()
    
    for _ in range(num_tests):
        result = evaluator.evaluate_fitness(test_robot, training_data)
    
    total_time = time.time() - start_time
    avg_time = total_time / num_tests
    
    print(f"   {num_tests}次评估总时间: {total_time:.3f}s")
    print(f"   平均单次评估时间: {avg_time*1000:.2f}ms")
    print(f"   评估速度: {num_tests/total_time:.0f} evaluations/s")
    
    # 3. 批量评估性能测试
    print("\n📊 批量评估性能测试...")
    import random
    
    # 生成随机机器人群体
    population_size = 200
    population = []
    for _ in range(population_size):
        num_links = random.randint(2, 5)
        link_lengths = [random.uniform(30, 80) for _ in range(num_links)]
        population.append(link_lengths)
    
    start_time = time.time()
    results = []
    
    for robot in population:
        result = evaluator.evaluate_fitness(robot)
        results.append(result)
    
    batch_time = time.time() - start_time
    
    print(f"   {population_size}个机器人评估时间: {batch_time:.3f}s")
    print(f"   平均每个机器人: {batch_time/population_size*1000:.2f}ms")
    print(f"   批量处理速度: {population_size/batch_time:.0f} robots/s")
    
    # 4. 内存稳定性测试
    print("\n📊 内存稳定性测试...")
    initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    # 大量评估测试内存泄漏
    for i in range(1000):
        robot = [random.uniform(20, 100) for _ in range(3)]
        result = evaluator.evaluate_fitness(robot)
        
        if i % 200 == 0:
            current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            print(f"   第{i+1}次评估后内存: {current_memory:.1f}MB (增长: {memory_growth:.1f}MB)")
    
    final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    total_growth = final_memory - initial_memory
    
    if total_growth < 50:  # 50MB以内认为正常
        print(f"   ✅ 内存使用稳定，总增长: {total_growth:.1f}MB")
    else:
        print(f"   ⚠️  可能存在内存泄漏，总增长: {total_growth:.1f}MB")
    
    # 5. 性能总结
    print(f"\n🎯 性能测试总结:")
    print(f"   初始化时间: {init_time:.3f}s")
    print(f"   单次评估时间: {avg_time*1000:.2f}ms")
    print(f"   批量处理速度: {population_size/batch_time:.0f} robots/s")
    print(f"   内存增长: {total_growth:.1f}MB")
    
    # 性能评级
    if avg_time < 0.01 and population_size/batch_time > 100 and total_growth < 50:
        print("   🏆 性能评级: 优秀")
    elif avg_time < 0.05 and population_size/batch_time > 50 and total_growth < 100:
        print("   ✅ 性能评级: 良好") 
    else:
        print("   ⚠️  性能评级: 需要优化")

if __name__ == "__main__":
    performance_test()