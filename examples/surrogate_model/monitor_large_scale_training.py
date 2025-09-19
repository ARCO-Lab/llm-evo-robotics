#!/usr/bin/env python3
"""
大规模训练监控器
实时监控30代×10个体×5000步的MAP-Elites训练进度和损失记录
"""

import os
import time
import json
import subprocess
from datetime import datetime, timedelta

def monitor_training_progress():
    """监控训练进度"""
    experiment_name = "large_scale_30gen_10pop"
    loss_log_dir = f"enhanced_multi_network_logs/{experiment_name}_multi_network_loss"
    map_elites_dir = "map_elites_experiments"
    
    start_time = time.time()
    
    print("🔍 大规模MAP-Elites训练监控器")
    print("=" * 60)
    print(f"实验名称: {experiment_name}")
    print(f"配置: 30代 × 10个体/代 × 5000步/个体")
    print(f"预计总训练步数: 1,500,000 步")
    print(f"监控开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            print(f"\n📊 监控报告 #{iteration} - {datetime.now().strftime('%H:%M:%S')}")
            print(f"运行时间: {timedelta(seconds=int(elapsed_time))}")
            
            # 检查训练进程
            try:
                result = subprocess.run(['pgrep', '-f', 'enhanced_multi_network_extractor'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    pids = result.stdout.strip().split('\n')
                    print(f"🟢 训练进程状态: 运行中 ({len(pids)} 个进程)")
                else:
                    print("🔴 训练进程状态: 未运行")
            except:
                print("⚠️ 无法检查训练进程状态")
            
            # 检查损失记录
            if os.path.exists(loss_log_dir):
                print(f"📁 损失记录目录: {loss_log_dir}")
                
                # 统计各网络的损失记录数
                network_stats = {}
                for network in ['ppo', 'attention', 'gnn', 'sac', 'total']:
                    csv_file = os.path.join(loss_log_dir, f"{network}_losses.csv")
                    if os.path.exists(csv_file):
                        try:
                            with open(csv_file, 'r') as f:
                                lines = f.readlines()
                                record_count = len(lines) - 1  # 减去头部
                                network_stats[network] = record_count
                        except:
                            network_stats[network] = 0
                    else:
                        network_stats[network] = 0
                
                print("📊 损失记录统计:")
                for network, count in network_stats.items():
                    print(f"   {network.upper()}: {count} 条记录")
                
                total_records = sum(network_stats.values())
                print(f"   总计: {total_records} 条损失记录")
                
            else:
                print("⚠️ 损失记录目录尚未创建")
            
            # 检查MAP-Elites实验结果
            if os.path.exists(map_elites_dir):
                try:
                    # 统计个体数量
                    individuals = [d for d in os.listdir(map_elites_dir) 
                                 if d.startswith('individual_') and 
                                    os.path.isdir(os.path.join(map_elites_dir, d))]
                    
                    print(f"🧬 MAP-Elites进度:")
                    print(f"   已训练个体数: {len(individuals)}")
                    
                    if len(individuals) > 0:
                        # 估算进度
                        total_expected_individuals = 30 * 10  # 30代 × 10个体
                        progress_percentage = (len(individuals) / total_expected_individuals) * 100
                        print(f"   总体进度: {progress_percentage:.1f}% ({len(individuals)}/{total_expected_individuals})")
                        
                        # 估算剩余时间
                        if len(individuals) > 1 and elapsed_time > 60:
                            avg_time_per_individual = elapsed_time / len(individuals)
                            remaining_individuals = total_expected_individuals - len(individuals)
                            estimated_remaining_time = remaining_individuals * avg_time_per_individual
                            
                            print(f"   平均每个体用时: {timedelta(seconds=int(avg_time_per_individual))}")
                            print(f"   预计剩余时间: {timedelta(seconds=int(estimated_remaining_time))}")
                            print(f"   预计完成时间: {(datetime.now() + timedelta(seconds=estimated_remaining_time)).strftime('%H:%M:%S')}")
                    
                except Exception as e:
                    print(f"⚠️ 无法统计MAP-Elites进度: {e}")
            else:
                print("⚠️ MAP-Elites实验目录尚未创建")
            
            # 检查系统资源
            try:
                # 检查内存使用
                result = subprocess.run(['free', '-h'], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        mem_line = lines[1].split()
                        if len(mem_line) >= 3:
                            total_mem = mem_line[1]
                            used_mem = mem_line[2]
                            print(f"💾 内存使用: {used_mem}/{total_mem}")
            except:
                pass
            
            print("-" * 60)
            
            # 等待下次检查
            time.sleep(30)  # 每30秒检查一次
            
    except KeyboardInterrupt:
        print("\n🛑 监控被中断")
    except Exception as e:
        print(f"\n❌ 监控出错: {e}")

def show_final_summary():
    """显示最终总结"""
    experiment_name = "large_scale_30gen_10pop"
    loss_log_dir = f"enhanced_multi_network_logs/{experiment_name}_multi_network_loss"
    
    print("\n🎉 大规模训练监控总结")
    print("=" * 60)
    
    if os.path.exists(loss_log_dir):
        # 显示最终损失统计
        stats_file = os.path.join(loss_log_dir, "comprehensive_loss_statistics.json")
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                print("📈 最终损失统计:")
                print(f"   总网络数: {stats['experiment_info']['total_networks']}")
                print(f"   总记录数: {stats['experiment_info']['total_records']}")
                
                for network, network_stats in stats['network_stats'].items():
                    print(f"\n📊 {network.upper()} 网络:")
                    print(f"   记录数: {network_stats['total_records']}")
                    
                    for metric, metric_stats in network_stats['metrics'].items():
                        if 'loss' in metric.lower():
                            trend_icon = "📉" if metric_stats['trend'] == 'decreasing' else "📈"
                            print(f"   {metric}: {metric_stats['avg']:.3f} (趋势: {trend_icon})")
                
            except Exception as e:
                print(f"⚠️ 无法读取统计文件: {e}")
        
        print(f"\n📁 完整结果保存在: {loss_log_dir}")
    else:
        print("⚠️ 损失记录目录不存在")

if __name__ == "__main__":
    try:
        monitor_training_progress()
    except KeyboardInterrupt:
        print("\n🛑 监控结束")
    finally:
        show_final_summary()
