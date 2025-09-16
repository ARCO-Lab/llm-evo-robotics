#!/usr/bin/env python3
"""
实验结果查看工具
用于查看和分析实验记录
"""

import os
import json
import csv
import argparse
from datetime import datetime
from typing import List, Dict, Any

# 可选依赖
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

def find_experiment_sessions(base_dir: str = "./experiment_results") -> List[str]:
    """查找所有实验会话"""
    if not os.path.exists(base_dir):
        return []
    
    sessions = []
    for item in os.listdir(base_dir):
        session_path = os.path.join(base_dir, item)
        if os.path.isdir(session_path) and item.startswith('session_'):
            session_id = item.replace('session_', '')
            sessions.append(session_id)
    
    return sorted(sessions, reverse=True)

def load_session_results(session_id: str, base_dir: str = "./experiment_results") -> Dict[str, Any]:
    """加载会话结果"""
    session_dir = os.path.join(base_dir, f"session_{session_id}")
    
    if not os.path.exists(session_dir):
        raise FileNotFoundError(f"会话不存在: {session_id}")
    
    results = {}
    
    # 加载JSON结果
    json_file = os.path.join(session_dir, "results.json")
    success_json_file = os.path.join(session_dir, "successful_results.json")
    
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            results['all_results'] = json.load(f)
    
    if os.path.exists(success_json_file):
        with open(success_json_file, 'r', encoding='utf-8') as f:
            results['successful_results'] = json.load(f)
    
    # 加载CSV结果
    csv_file = os.path.join(session_dir, "results.csv")
    if os.path.exists(csv_file) and HAS_PANDAS:
        results['df_all'] = pd.read_csv(csv_file)
    
    success_csv_file = os.path.join(session_dir, "successful_results.csv")
    if os.path.exists(success_csv_file) and HAS_PANDAS:
        results['df_success'] = pd.read_csv(success_csv_file)
    
    # 加载总结
    summary_file = os.path.join(session_dir, "session_summary.txt")
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            results['summary'] = f.read()
    
    results['session_id'] = session_id
    results['session_dir'] = session_dir
    
    return results

def print_session_summary(results: Dict[str, Any]):
    """打印会话总结"""
    print("=" * 60)
    print(f"🧪 实验会话: {results['session_id']}")
    print("=" * 60)
    
    if 'summary' in results:
        print(results['summary'])
    else:
        # 从数据生成简单总结
        if 'df_all' in results:
            df = results['df_all']
            total = len(df)
            successful = len(df[df['is_successful'] == True])
            
            print(f"总实验数: {total}")
            print(f"成功实验数: {successful}")
            print(f"成功率: {successful/total:.1%}" if total > 0 else "成功率: 0%")
            
            if successful > 0:
                print(f"平均fitness: {df['fitness'].mean():.3f}")
                print(f"最佳fitness: {df['fitness'].max():.3f}")

def analyze_successful_structures(results: Dict[str, Any]):
    """分析成功的机器人结构"""
    # 优先使用JSON数据，如果没有pandas的话
    if 'successful_results' in results and results['successful_results']:
        successful_results = results['successful_results']
        print("\n🏆 成功结构分析")
        print("-" * 40)
        
        # 基本统计
        total_successful = len(successful_results)
        print(f"成功实验总数: {total_successful}")
        
        # 按关节数统计
        joint_counts = {}
        fitness_by_joints = {}
        for result in successful_results:
            num_links = result['robot_structure']['num_links']
            joint_counts[num_links] = joint_counts.get(num_links, 0) + 1
            if num_links not in fitness_by_joints:
                fitness_by_joints[num_links] = []
            fitness_by_joints[num_links].append(result['performance']['fitness'])
        
        print("\n📊 按关节数统计:")
        for num_links in sorted(joint_counts.keys()):
            count = joint_counts[num_links]
            avg_fitness = sum(fitness_by_joints[num_links]) / len(fitness_by_joints[num_links])
            max_fitness = max(fitness_by_joints[num_links])
            print(f"   {num_links}关节: {count}个 (平均fitness: {avg_fitness:.3f}, 最高: {max_fitness:.3f})")
        
        # 最佳结果
        best_result = max(successful_results, key=lambda x: x['performance']['fitness'])
        print(f"\n🥇 最佳结果:")
        print(f"   实验ID: {best_result['experiment_id']}")
        print(f"   关节数: {best_result['robot_structure']['num_links']}")
        print(f"   链长: {best_result['robot_structure']['link_lengths']}")
        print(f"   总长度: {best_result['robot_structure']['total_length']:.1f}")
        print(f"   Fitness: {best_result['performance']['fitness']:.3f}")
        print(f"   成功率: {best_result['performance']['success_rate']:.1%}")
        print(f"   代数: {best_result['generation']}")
        
        return
    
    # 如果有pandas，使用pandas分析
    if 'df_success' not in results or (HAS_PANDAS and results['df_success'].empty):
        print("❌ 没有成功的实验结果")
        return
    
    if not HAS_PANDAS:
        print("⚠️ 需要安装pandas进行详细分析: pip install pandas")
        return
        
    df = results['df_success']
    
    print("\n🏆 成功结构分析")
    print("-" * 40)
    
    # 按关节数统计
    print("📊 按关节数统计:")
    joint_stats = df.groupby('num_links').agg({
        'experiment_id': 'count',
        'fitness': ['mean', 'max'],
        'success_rate': 'mean',
        'total_length': 'mean'
    }).round(3)
    print(joint_stats)
    
    # 最佳结果
    print("\n🥇 最佳结果:")
    best_idx = df['fitness'].idxmax()
    best_result = df.loc[best_idx]
    
    print(f"   实验ID: {best_result['experiment_id']}")
    print(f"   关节数: {best_result['num_links']}")
    print(f"   链长: {best_result['link_lengths']}")
    print(f"   总长度: {best_result['total_length']:.1f}")
    print(f"   Fitness: {best_result['fitness']:.3f}")
    print(f"   成功率: {best_result['success_rate']:.1%}")
    print(f"   训练时间: {best_result['training_time']:.1f}s")
    
    # 参数分析
    print(f"\n⚙️ 成功实验的参数范围:")
    print(f"   学习率: {df['lr'].min():.6f} ~ {df['lr'].max():.6f}")
    print(f"   SAC Alpha: {df['alpha'].min():.3f} ~ {df['alpha'].max():.3f}")
    print(f"   总长度: {df['total_length'].min():.1f} ~ {df['total_length'].max():.1f}")

def plot_results(results: Dict[str, Any], save_plots: bool = True):
    """绘制结果图表"""
    if not HAS_PLOTTING:
        print("⚠️ 需要安装matplotlib和seaborn来生成图表:")
        print("   pip install matplotlib seaborn")
        return
        
    if 'df_all' not in results:
        print("❌ 没有数据可绘制")
        return
    
    df = results['df_all']
    
    # 设置绘图风格
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'实验结果分析 - 会话 {results["session_id"]}', fontsize=16)
    
    # 1. Fitness分布
    axes[0, 0].hist(df['fitness'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(df['fitness'].mean(), color='red', linestyle='--', label=f'平均值: {df["fitness"].mean():.3f}')
    axes[0, 0].set_xlabel('Fitness')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('Fitness分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 成功率 vs 关节数
    joint_success = df.groupby('num_links')['is_successful'].agg(['count', 'sum']).reset_index()
    joint_success['success_rate'] = joint_success['sum'] / joint_success['count']
    
    axes[0, 1].bar(joint_success['num_links'], joint_success['success_rate'], 
                   color='lightgreen', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('关节数')
    axes[0, 1].set_ylabel('成功率')
    axes[0, 1].set_title('不同关节数的成功率')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Fitness vs 总长度
    colors = ['red' if success else 'blue' for success in df['is_successful']]
    axes[1, 0].scatter(df['total_length'], df['fitness'], c=colors, alpha=0.6)
    axes[1, 0].set_xlabel('机器人总长度')
    axes[1, 0].set_ylabel('Fitness')
    axes[1, 0].set_title('Fitness vs 机器人总长度')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='成功'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='失败')]
    axes[1, 0].legend(handles=legend_elements)
    
    # 4. 训练时间分布
    axes[1, 1].boxplot([df[df['is_successful'] == False]['training_time'].dropna(),
                        df[df['is_successful'] == True]['training_time'].dropna()],
                       labels=['失败', '成功'])
    axes[1, 1].set_ylabel('训练时间 (秒)')
    axes[1, 1].set_title('成功/失败实验的训练时间对比')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plot_file = os.path.join(results['session_dir'], 'analysis_plots.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 图表已保存: {plot_file}")
    
    plt.show()

def export_successful_structures(results: Dict[str, Any], format: str = 'json'):
    """导出成功的结构"""
    if 'successful_results' not in results:
        print("❌ 没有成功的实验结果")
        return
    
    successful = results['successful_results']
    
    # 提取结构信息
    structures = []
    for result in successful:
        structure = {
            'experiment_id': result['experiment_id'],
            'fitness': result['performance']['fitness'],
            'num_links': result['robot_structure']['num_links'],
            'link_lengths': result['robot_structure']['link_lengths'],
            'total_length': result['robot_structure']['total_length'],
            'lr': result['training_params']['lr'],
            'alpha': result['training_params']['alpha'],
            'success_rate': result['performance']['success_rate'],
            'generation': result['generation']
        }
        structures.append(structure)
    
    # 排序（按fitness降序）
    structures.sort(key=lambda x: x['fitness'], reverse=True)
    
    # 导出
    if format.lower() == 'json':
        export_file = os.path.join(results['session_dir'], 'successful_structures.json')
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(structures, f, indent=2, ensure_ascii=False)
    elif format.lower() == 'csv':
        export_file = os.path.join(results['session_dir'], 'successful_structures.csv')
        if HAS_PANDAS:
            df = pd.DataFrame(structures)
            df.to_csv(export_file, index=False, encoding='utf-8')
        else:
            # 手动写CSV
            with open(export_file, 'w', newline='', encoding='utf-8') as f:
                if structures:
                    fieldnames = structures[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(structures)
    
    print(f"📁 成功结构已导出: {export_file}")
    return export_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="查看实验结果")
    parser.add_argument('session_id', nargs='?', help='会话ID')
    parser.add_argument('--list', '-l', action='store_true', help='列出所有会话')
    parser.add_argument('--plot', '-p', action='store_true', help='生成图表')
    parser.add_argument('--export', '-e', choices=['json', 'csv'], help='导出成功结构')
    parser.add_argument('--base-dir', default='./experiment_results', help='结果目录')
    
    args = parser.parse_args()
    
    # 列出所有会话
    if args.list:
        sessions = find_experiment_sessions(args.base_dir)
        if not sessions:
            print("❌ 没有找到实验会话")
            return
        
        print("📋 可用的实验会话:")
        for i, session in enumerate(sessions, 1):
            # 格式化时间显示
            try:
                dt = datetime.strptime(session, "%Y%m%d_%H%M%S")
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                time_str = session
            
            print(f"   {i}. {session} ({time_str})")
        return
    
    # 如果没有指定会话ID，使用最新的
    if not args.session_id:
        sessions = find_experiment_sessions(args.base_dir)
        if not sessions:
            print("❌ 没有找到实验会话")
            return
        args.session_id = sessions[0]
        print(f"🔄 使用最新会话: {args.session_id}")
    
    try:
        # 加载结果
        results = load_session_results(args.session_id, args.base_dir)
        
        # 显示总结
        print_session_summary(results)
        
        # 分析成功结构
        analyze_successful_structures(results)
        
        # 生成图表
        if args.plot:
            try:
                plot_results(results)
            except ImportError:
                print("⚠️ 需要安装matplotlib和seaborn来生成图表:")
                print("   pip install matplotlib seaborn")
            except Exception as e:
                print(f"⚠️ 生成图表失败: {e}")
        
        # 导出成功结构
        if args.export:
            export_successful_structures(results, args.export)
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ 加载结果失败: {e}")

if __name__ == "__main__":
    main()
