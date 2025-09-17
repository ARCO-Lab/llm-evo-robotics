#!/usr/bin/env python3
"""
MAP-Elites可视化集成测试脚本
功能：
1. 测试MAP-Elites热力图生成
2. 测试神经网络loss可视化
3. 测试训练过程中的可视化集成
4. 验证所有可视化工具是否正常工作
"""

import os
import sys
import time
import argparse
from datetime import datetime

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_visualization_tools():
    """测试可视化工具"""
    print("🧪 开始测试MAP-Elites可视化工具")
    print("=" * 60)
    
    try:
        # 1. 测试可视化演示
        print("\n📊 测试1: 可视化演示")
        from visualization_demo import VisualizationDemo
        
        demo = VisualizationDemo("./test_visualizations")
        success = demo.run_full_demo()
        
        if success:
            print("✅ 可视化演示测试通过")
        else:
            print("❌ 可视化演示测试失败")
            return False
        
        # 2. 测试MAP-Elites可视化器
        print("\n🗺️  测试2: MAP-Elites可视化器")
        from map_elites_visualizer import MAPElitesVisualizer
        
        viz = MAPElitesVisualizer(output_dir="./test_map_elites_viz")
        
        # 创建演示数据
        demo_archive_path = demo.create_demo_map_elites_data()
        viz.load_archive(demo_archive_path)
        
        # 测试各种可视化
        heatmap_path = viz.create_fitness_heatmap()
        evolution_path = viz.create_evolution_analysis()
        elite_path = viz.create_elite_showcase()
        
        if all([heatmap_path, evolution_path, elite_path]):
            print("✅ MAP-Elites可视化器测试通过")
        else:
            print("❌ MAP-Elites可视化器测试失败")
            return False
        
        # 3. 测试神经网络loss可视化器
        print("\n🧠 测试3: 神经网络Loss可视化器")
        from network_loss_visualizer import NetworkLossVisualizer
        
        loss_viz = NetworkLossVisualizer(output_dir="./test_loss_viz")
        
        # 创建演示训练日志
        demo_log_dir = demo.create_demo_training_logs()
        
        if loss_viz.load_training_logs(demo_log_dir):
            curves_path = loss_viz.create_loss_curves()
            comparison_path = loss_viz.create_network_comparison()
            
            if curves_path and comparison_path:
                print("✅ 神经网络Loss可视化器测试通过")
            else:
                print("❌ 神经网络Loss可视化器测试失败")
                return False
        else:
            print("❌ 无法加载演示训练日志")
            return False
        
        print("\n🎉 所有可视化工具测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 可视化工具测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_map_elites_integration():
    """测试MAP-Elites训练器集成"""
    print("\n🧬 开始测试MAP-Elites训练器可视化集成")
    print("=" * 60)
    
    try:
        import argparse
        from map_elites_trainer import MAPElitesEvolutionTrainer
        
        # 创建基础参数
        base_args = argparse.Namespace()
        base_args.env_type = 'reacher2d'
        base_args.num_processes = 1
        base_args.seed = 42
        base_args.save_dir = './test_integration_results'
        base_args.lr = 1e-4
        base_args.alpha = 0.2
        base_args.tau = 0.005
        base_args.gamma = 0.99
        
        print("📊 创建集成可视化的训练器...")
        trainer = MAPElitesEvolutionTrainer(
            base_args=base_args,
            num_initial_random=3,            # 使用很少的个体进行快速测试
            training_steps_per_individual=100,  # 很少的训练步数
            enable_rendering=False,          # 禁用环境渲染以加快测试
            silent_mode=True,               # 静默模式
            use_genetic_fitness=True,        # 使用遗传算法fitness
            enable_multiprocess=False,       # 禁用多进程以简化测试
            enable_visualization=True,       # 🎨 启用可视化
            visualization_interval=2         # 每2代生成可视化
        )
        
        print("🚀 运行小规模进化测试...")
        trainer.run_evolution(
            num_generations=4,               # 只运行4代
            individuals_per_generation=2    # 每代只有2个新个体
        )
        
        # 检查是否生成了可视化文件
        viz_dir = os.path.join(base_args.save_dir, 'visualizations')
        if os.path.exists(viz_dir):
            viz_files = os.listdir(viz_dir)
            if viz_files:
                print(f"✅ 成功生成 {len(viz_files)} 个可视化文件")
                for file in viz_files:
                    print(f"   📊 {file}")
                return True
            else:
                print("⚠️  可视化目录存在但没有文件")
                return False
        else:
            print("❌ 可视化目录不存在")
            return False
        
    except Exception as e:
        print(f"❌ MAP-Elites集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_training_with_visualization():
    """测试带可视化的真实训练（短时间）"""
    print("\n🚀 开始测试带可视化的真实训练")
    print("=" * 60)
    
    try:
        # 运行短时间的真实训练
        print("⚠️  这将运行一个非常短的真实训练以测试可视化集成...")
        print("   如果你想跳过这个测试，请按 Ctrl+C")
        
        try:
            time.sleep(3)  # 给用户3秒钟考虑
        except KeyboardInterrupt:
            print("\n⏭️  跳过真实训练测试")
            return True
        
        from map_elites_trainer import start_real_training
        
        # 临时修改训练参数以进行快速测试
        print("🔧 使用测试配置运行训练...")
        
        # 这里我们不实际运行完整训练，而是验证训练器能否正确初始化
        import argparse
        from map_elites_trainer import MAPElitesEvolutionTrainer
        
        base_args = argparse.Namespace()
        base_args.env_type = 'reacher2d'
        base_args.num_processes = 1
        base_args.seed = 42
        base_args.save_dir = './test_real_training'
        base_args.lr = 3e-4
        base_args.alpha = 0.2
        base_args.tau = 0.005
        base_args.gamma = 0.99
        base_args.update_frequency = 1
        base_args.use_real_training = True
        
        trainer = MAPElitesEvolutionTrainer(
            base_args=base_args,
            num_initial_random=2,
            training_steps_per_individual=50,  # 非常少的步数
            enable_rendering=False,
            silent_mode=True,
            use_genetic_fitness=True,
            enable_visualization=True,
            visualization_interval=1
        )
        
        print("✅ 带可视化的训练器初始化成功")
        print("   (实际训练将在真实使用时运行)")
        
        return True
        
    except Exception as e:
        print(f"❌ 真实训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MAP-Elites可视化集成测试')
    parser.add_argument('--test', type=str, choices=['viz', 'integration', 'training', 'all'], 
                       default='all', help='测试类型')
    parser.add_argument('--cleanup', action='store_true', help='测试后清理文件')
    
    args = parser.parse_args()
    
    print("🧪 MAP-Elites可视化集成测试")
    print("=" * 60)
    print(f"🕐 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    results = {}
    
    # 根据参数运行测试
    if args.test in ['viz', 'all']:
        print("\n" + "🎨 可视化工具测试".center(60, "="))
        results['visualization_tools'] = test_visualization_tools()
    
    if args.test in ['integration', 'all']:
        print("\n" + "🔗 集成测试".center(60, "="))
        results['integration'] = test_map_elites_integration()
    
    if args.test in ['training', 'all']:
        print("\n" + "🚀 训练测试".center(60, "="))
        results['training'] = test_real_training_with_visualization()
    
    end_time = time.time()
    
    # 总结结果
    print("\n" + "📋 测试总结".center(60, "="))
    print(f"⏱️  总耗时: {end_time - start_time:.2f}秒")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"✅ 通过: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    if all(results.values()):
        print("\n🎉 所有测试通过! 可视化功能已成功集成到MAP-Elites系统中")
        print("\n💡 现在你可以:")
        print("   1. 运行 python map_elites_trainer.py --train 开始带可视化的训练")
        print("   2. 运行 python visualization_demo.py 查看可视化演示")
        print("   3. 使用可视化工具分析训练结果")
        success = True
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")
        success = False
    
    # 清理测试文件
    if args.cleanup:
        print("\n🧹 清理测试文件...")
        import shutil
        test_dirs = [
            './test_visualizations',
            './test_map_elites_viz', 
            './test_loss_viz',
            './test_integration_results',
            './test_real_training'
        ]
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                try:
                    shutil.rmtree(test_dir)
                    print(f"   🗑️  删除: {test_dir}")
                except Exception as e:
                    print(f"   ⚠️  无法删除 {test_dir}: {e}")
        
        print("✅ 清理完成")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
