#!/usr/bin/env python3
"""
简化的可视化测试脚本 - 不依赖enhanced_train.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

def test_matplotlib():
    """测试matplotlib是否工作"""
    print("🎨 测试matplotlib...")
    
    # 创建简单图表
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # 生成测试数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    ax.plot(x, y1, label='sin(x)', color='blue')
    ax.plot(x, y2, label='cos(x)', color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('测试图表')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 保存图表
    os.makedirs('./test_output', exist_ok=True)
    test_path = './test_output/matplotlib_test.png'
    plt.savefig(test_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if os.path.exists(test_path):
        print(f"✅ matplotlib测试成功: {test_path}")
        return True
    else:
        print("❌ matplotlib测试失败")
        return False

def test_map_elites_visualizer_basic():
    """测试MAP-Elites可视化器的基本功能"""
    print("🗺️  测试MAP-Elites可视化器...")
    
    try:
        # 直接导入可视化器，不通过其他模块
        sys.path.append(os.path.dirname(__file__))
        from map_elites_visualizer import MAPElitesVisualizer
        
        # 创建可视化器
        viz = MAPElitesVisualizer(output_dir="./test_output/map_elites")
        print("✅ MAP-Elites可视化器创建成功")
        
        # 创建简单的演示数据
        import pickle
        from collections import namedtuple
        
        # 模拟存档结构
        Individual = namedtuple('Individual', ['individual_id', 'genotype', 'phenotype', 'generation', 'parent_id', 'fitness', 'fitness_details'])
        Genotype = namedtuple('Genotype', ['num_links', 'link_lengths', 'lr', 'alpha'])
        Phenotype = namedtuple('Phenotype', ['avg_reward', 'success_rate', 'min_distance'])
        Archive = namedtuple('Archive', ['archive', 'total_evaluations', 'generation'])
        
        # 生成演示个体
        np.random.seed(42)
        individuals = {}
        
        for i in range(20):
            genotype = Genotype(
                num_links=np.random.randint(2, 6),
                link_lengths=np.random.uniform(20, 80, np.random.randint(2, 6)).tolist(),
                lr=np.random.uniform(1e-5, 1e-2),
                alpha=np.random.uniform(0.1, 0.5)
            )
            
            phenotype = Phenotype(
                avg_reward=np.random.uniform(-100, 50),
                success_rate=np.random.uniform(0, 1),
                min_distance=np.random.uniform(10, 200)
            )
            
            fitness = np.random.uniform(0.1, 1.0)
            
            individual = Individual(
                individual_id=f"test_{i}",
                genotype=genotype,
                phenotype=phenotype,
                generation=np.random.randint(0, 5),
                parent_id=f"test_{max(0, i-1)}" if i > 0 else None,
                fitness=fitness,
                fitness_details={
                    'category': np.random.choice(['insufficient_for_direct', 'insufficient_for_path', 'sufficient_length']),
                    'strategy': np.random.choice(['length_optimization', 'hybrid_optimization', 'performance_optimization']),
                    'reason': 'Test reason'
                }
            )
            
            individuals[f"test_{i}"] = individual
        
        # 直接保存数据，不使用pickle
        archive_path = "./test_output/test_archive.pkl"
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        
        # 创建一个简单的存档字典用于pickle
        archive_data = {
            'archive': individuals,
            'total_evaluations': len(individuals),
            'generation': 5
        }
        
        # 保存为字典
        with open(archive_path, 'wb') as f:
            pickle.dump(archive_data, f)
        
        # 创建一个简单的存档对象用于可视化器
        class SimpleArchive:
            def __init__(self, data):
                self.archive = data['archive']
                self.total_evaluations = data['total_evaluations']
                self.generation = data['generation']
        
        # 手动设置存档到可视化器（跳过load_archive方法）
        with open(archive_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        viz.archive = SimpleArchive(loaded_data)
        print(f"✅ 手动加载存档成功: {len(viz.archive.archive)}个个体")
        
        # 测试热力图生成
        heatmap_path = viz.create_fitness_heatmap()
        if heatmap_path and os.path.exists(heatmap_path):
            print(f"✅ 热力图生成成功: {heatmap_path}")
            return True
        else:
            print("❌ 热力图生成失败")
            return False
            
    except Exception as e:
        print(f"❌ MAP-Elites可视化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_loss_visualizer_basic():
    """测试网络loss可视化器的基本功能"""
    print("🧠 测试网络Loss可视化器...")
    
    try:
        from network_loss_visualizer import NetworkLossVisualizer
        
        # 创建可视化器
        viz = NetworkLossVisualizer(output_dir="./test_output/loss_viz")
        print("✅ 网络Loss可视化器创建成功")
        
        # 创建演示训练日志
        import json
        
        log_dir = "./test_output/demo_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成模拟训练数据
        np.random.seed(42)
        n_steps = 500
        
        training_data = []
        for i in range(n_steps):
            # 模拟loss下降
            actor_loss = 5.0 * np.exp(-i/200) + 0.1 + 0.1 * np.random.random()
            critic_loss = 8.0 * np.exp(-i/180) + 0.1 + 0.15 * np.random.random()
            
            metrics = {
                'step': i,
                'actor_loss': max(0.01, actor_loss),
                'critic_loss': max(0.01, critic_loss),
                'total_loss': max(0.02, actor_loss + critic_loss),
                'learning_rate': 3e-4 * (0.999 ** (i // 100)),
                'alpha': 0.2 + 0.1 * np.sin(i / 50),
                'entropy': 1.5 + 0.5 * np.cos(i / 30)
            }
            training_data.append(metrics)
        
        # 保存训练数据
        with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # 测试加载和可视化
        if viz.load_training_logs(log_dir):
            curves_path = viz.create_loss_curves()
            if curves_path and os.path.exists(curves_path):
                print(f"✅ Loss曲线生成成功: {curves_path}")
                return True
            else:
                print("❌ Loss曲线生成失败")
                return False
        else:
            print("❌ 训练日志加载失败")
            return False
            
    except Exception as e:
        print(f"❌ 网络Loss可视化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🧪 简化可视化功能测试")
    print("=" * 50)
    
    results = {}
    
    # 1. 测试matplotlib
    results['matplotlib'] = test_matplotlib()
    
    # 2. 测试MAP-Elites可视化器
    results['map_elites_viz'] = test_map_elites_visualizer_basic()
    
    # 3. 测试网络Loss可视化器
    results['loss_viz'] = test_network_loss_visualizer_basic()
    
    # 总结结果
    print("\n📋 测试总结")
    print("=" * 30)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    print(f"\n✅ 通过: {passed}/{total}")
    
    if all(results.values()):
        print("🎉 所有基础可视化功能正常工作!")
        print("\n💡 接下来可以:")
        print("   1. 修复enhanced_train.py的语法错误")
        print("   2. 运行完整的可视化集成测试")
        print("   3. 启动带可视化的MAP-Elites训练")
        return True
    else:
        print("⚠️  部分功能存在问题，请检查错误信息")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
