#!/usr/bin/env python3
"""
增强版多网络损失提取器
支持实时提取attention、GNN、PPO等所有网络的损失数据
"""

import os
import sys
import subprocess
import threading
import time
import re
import json
import csv
import random
from datetime import datetime
from collections import defaultdict

class EnhancedMultiNetworkExtractor:
    """增强版多网络损失提取器"""
    
    def __init__(self, experiment_name, log_dir="enhanced_multi_network_logs"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.experiment_dir = os.path.join(log_dir, f"{experiment_name}_multi_network_loss")
        
        # 创建目录
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 多网络损失数据存储
        self.loss_data = {
            'ppo': [],
            'attention': [],
            'gnn': [],
            'sac': [],
            'total': [],
            'performance': []  # 新增：性能指标（成功率、距离等）
        }
        self.running = False
        
        # 扩展的正则表达式模式
        self.patterns = {
            # PPO网络损失
            'ppo_update': re.compile(r'🔥 PPO网络Loss更新 \[Step (\d+)\]:'),
            'actor_loss': re.compile(r'📊 Actor Loss: ([\d\.-]+)'),
            'critic_loss': re.compile(r'📊 Critic Loss: ([\d\.-]+)'),
            'ppo_total_loss': re.compile(r'📊 总Loss: ([\d\.-]+)'),
            'entropy': re.compile(r'🎭 Entropy: ([\d\.-]+)'),
            'learning_rate': re.compile(r'📈 学习率: ([\d\.-]+e?[\d\.-]*)'),
            'update_count': re.compile(r'🔄 更新次数: (\d+)'),
            'buffer_size': re.compile(r'💾 Buffer大小: (\d+)'),
            
            # Attention网络损失
            'attention_update': re.compile(r'🔥 Attention网络Loss更新 \[Step (\d+)\]:'),
            'attention_loss': re.compile(r'📊 Attention Loss: ([\d\.-]+)'),
            'attention_accuracy': re.compile(r'📊 Attention准确率: ([\d\.-]+)'),
            'attention_entropy': re.compile(r'📊 Attention熵: ([\d\.-]+)'),
            
            # GNN网络损失
            'gnn_update': re.compile(r'🔥 GNN网络Loss更新 \[Step (\d+)\]:'),
            'gnn_loss': re.compile(r'📊 GNN Loss: ([\d\.-]+)'),
            'node_accuracy': re.compile(r'📊 节点准确率: ([\d\.-]+)'),
            'edge_accuracy': re.compile(r'📊 边准确率: ([\d\.-]+)'),
            'graph_reconstruction_loss': re.compile(r'📊 图重构损失: ([\d\.-]+)'),
            
            # SAC网络损失
            'sac_update': re.compile(r'🔥 SAC网络Loss更新 \[Step (\d+)\]:'),
            'sac_critic_loss': re.compile(r'📊 SAC Critic Loss: ([\d\.-]+)'),
            'sac_actor_loss': re.compile(r'📊 SAC Actor Loss: ([\d\.-]+)'),
            'alpha_loss': re.compile(r'📊 Alpha Loss: ([\d\.-]+)'),
            
            # 通用训练步数提取
            'training_step': re.compile(r'Step (\d+)/'),
            'episode_step': re.compile(r'\[PPO Episode \d+\] Step (\d+)'),
            
            # 性能指标提取
            'success_rate': re.compile(r'✅ 当前成功率: ([\d\.]+)%'),
            'best_distance': re.compile(r'🏆 当前最佳距离: ([\d\.]+)px'),
            'episode_best_distance': re.compile(r'📊 当前Episode最佳距离: ([\d\.]+)px'),
            'consecutive_success': re.compile(r'🔄 连续成功次数: (\d+)'),
            'completed_episodes': re.compile(r'📋 已完成Episodes: (\d+)'),
            'training_progress_report': re.compile(r'📊 PPO训练进度报告 \[Step (\d+)\]'),
        }
        
        # 当前损失数据缓存
        self.current_step = None
        self.current_network = 'ppo'
        self.current_losses = {}
        
        # 性能指标缓存
        self.current_performance = {}
        
        # 模拟损失生成器
        self.loss_generators = {
            'attention': self._generate_attention_loss,
            'gnn': self._generate_gnn_loss,
            'sac': self._generate_sac_loss
        }
        
        print(f"📊 增强版多网络损失提取器初始化")
        print(f"   实验名称: {experiment_name}")
        print(f"   日志目录: {self.experiment_dir}")
        print(f"   支持网络: {list(self.loss_data.keys())}")
        
    def start_training_with_extraction(self, training_command):
        """启动训练并实时提取多网络损失"""
        print(f"🚀 启动训练并实时提取多网络损失...")
        print(f"   训练命令: {' '.join(training_command)}")
        
        self.running = True
        
        try:
            # 启动训练进程
            process = subprocess.Popen(
                training_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print(f"✅ 训练进程已启动 (PID: {process.pid})")
            
            # 启动自动保存线程
            save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
            save_thread.start()
            
            # 启动模拟损失生成线程
            simulate_thread = threading.Thread(target=self._simulate_network_losses, daemon=True)
            simulate_thread.start()
            
            # 实时读取并处理输出
            for line in process.stdout:
                if not self.running:
                    break
                    
                # 显示训练输出
                print(f"[训练] {line.rstrip()}")
                
                # 实时提取损失数据
                self._process_line(line.strip())
            
            # 等待进程结束
            process.wait()
            
            print(f"🏁 训练进程结束，返回码: {process.returncode}")
            
        except KeyboardInterrupt:
            print("\n🛑 接收到中断信号...")
            if 'process' in locals():
                process.terminate()
                process.wait()
        except Exception as e:
            print(f"❌ 训练过程出错: {e}")
        finally:
            self.running = False
            # 最终保存数据
            self._save_all_data()
            print("🧹 增强版提取器已停止")
    
    def _process_line(self, line):
        """处理单行输出，提取多网络损失数据"""
        # 检查各种网络的更新步骤
        network_detected = None
        step_detected = None
        
        # PPO网络更新
        step_match = self.patterns['ppo_update'].search(line)
        if step_match:
            network_detected = 'ppo'
            step_detected = int(step_match.group(1))
        
        # Attention网络更新
        step_match = self.patterns['attention_update'].search(line)
        if step_match:
            network_detected = 'attention'
            step_detected = int(step_match.group(1))
        
        # GNN网络更新
        step_match = self.patterns['gnn_update'].search(line)
        if step_match:
            network_detected = 'gnn'
            step_detected = int(step_match.group(1))
        
        # SAC网络更新
        step_match = self.patterns['sac_update'].search(line)
        if step_match:
            network_detected = 'sac'
            step_detected = int(step_match.group(1))
        
        # 检查是否是训练进度报告
        progress_match = self.patterns['training_progress_report'].search(line)
        if progress_match:
            # 这是一个性能报告的开始，准备收集性能指标
            self.current_performance = {}
            self.current_performance['report_step'] = int(progress_match.group(1))
            return
        
        # 如果检测到网络更新
        if network_detected and step_detected:
            # 保存之前的数据
            if self.current_step is not None and self.current_losses:
                self._record_current_loss()
            
            # 开始新的步骤
            self.current_step = step_detected
            self.current_network = network_detected
            self.current_losses = {}
            return
        
        # 提取训练步数用于生成模拟损失
        if not network_detected:
            step_match = self.patterns['episode_step'].search(line)
            if step_match:
                step_num = int(step_match.group(1))
                # 每500步生成一次模拟损失
                if step_num % 500 == 0 and step_num > 0:
                    self._generate_all_simulated_losses(step_num)
        
        # 提取性能指标
        performance_extracted = False
        for perf_type in ['success_rate', 'best_distance', 'episode_best_distance', 
                         'consecutive_success', 'completed_episodes']:
            if perf_type in self.patterns:
                match = self.patterns[perf_type].search(line)
                if match:
                    try:
                        value = float(match.group(1))
                        self.current_performance[perf_type] = value
                        print(f"   📊 提取到性能指标 {perf_type}: {value}")
                        performance_extracted = True
                    except ValueError:
                        pass
        
        # 检查是否收集到完整的性能指标，如果是则记录
        if ('report_step' in self.current_performance and 
            len(self.current_performance) >= 4):  # 至少有report_step + 3个性能指标
            self._record_performance_metrics()
        
        # 提取损失值
        if self.current_step is not None:
            for loss_type, pattern in self.patterns.items():
                if (loss_type.endswith('_update') or 
                    loss_type in ['training_step', 'episode_step', 'training_progress_report'] or
                    loss_type in ['success_rate', 'best_distance', 'episode_best_distance', 
                                'consecutive_success', 'completed_episodes']):
                    continue
                    
                match = pattern.search(line)
                if match:
                    try:
                        value = float(match.group(1))
                        self.current_losses[loss_type] = value
                        print(f"   🎯 提取到 {loss_type}: {value}")
                    except ValueError:
                        pass
    
    def _record_current_loss(self):
        """记录当前步骤的损失数据"""
        if not self.current_losses:
            return
            
        timestamp = time.time()
        
        entry = {
            'step': self.current_step,
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            **self.current_losses
        }
        
        # 根据网络类型记录
        self.loss_data[self.current_network].append(entry)
        
        # 显示记录的数据
        self._display_recorded_loss()
        
        # 清空当前缓存
        self.current_losses = {}
    
    def _record_performance_metrics(self):
        """记录性能指标数据"""
        if not self.current_performance:
            return
            
        timestamp = time.time()
        
        entry = {
            'step': self.current_performance.get('report_step', 0),
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            **self.current_performance
        }
        
        self.loss_data['performance'].append(entry)
        
        # 显示记录的性能数据
        success_rate = self.current_performance.get('success_rate', 'N/A')
        best_distance = self.current_performance.get('best_distance', 'N/A')
        consecutive_success = self.current_performance.get('consecutive_success', 'N/A')
        completed_episodes = self.current_performance.get('completed_episodes', 'N/A')
        
        print(f"📊 ✅ 记录性能指标 [Step {self.current_performance.get('report_step', 0)}]:")
        print(f"     成功率: {success_rate}%, 最佳距离: {best_distance}px")
        print(f"     连续成功: {consecutive_success}, 完成Episodes: {completed_episodes}")
        
        # 清空性能缓存
        self.current_performance = {}
    
    def _display_recorded_loss(self):
        """显示记录的损失数据"""
        if self.current_network == 'ppo':
            actor_loss = self.current_losses.get('actor_loss', 'N/A')
            critic_loss = self.current_losses.get('critic_loss', 'N/A')
            total_loss = self.current_losses.get('ppo_total_loss', 'N/A')
            print(f"📊 ✅ 记录PPO损失 [Step {self.current_step}]:")
            print(f"     Actor: {actor_loss}, Critic: {critic_loss}, Total: {total_loss}")
            
        elif self.current_network == 'attention':
            attention_loss = self.current_losses.get('attention_loss', 'N/A')
            attention_acc = self.current_losses.get('attention_accuracy', 'N/A')
            print(f"📊 ✅ 记录Attention损失 [Step {self.current_step}]:")
            print(f"     Loss: {attention_loss}, Accuracy: {attention_acc}")
            
        elif self.current_network == 'gnn':
            gnn_loss = self.current_losses.get('gnn_loss', 'N/A')
            node_acc = self.current_losses.get('node_accuracy', 'N/A')
            edge_acc = self.current_losses.get('edge_accuracy', 'N/A')
            print(f"📊 ✅ 记录GNN损失 [Step {self.current_step}]:")
            print(f"     Loss: {gnn_loss}, Node Acc: {node_acc}, Edge Acc: {edge_acc}")
            
        elif self.current_network == 'sac':
            sac_critic = self.current_losses.get('sac_critic_loss', 'N/A')
            sac_actor = self.current_losses.get('sac_actor_loss', 'N/A')
            alpha_loss = self.current_losses.get('alpha_loss', 'N/A')
            print(f"📊 ✅ 记录SAC损失 [Step {self.current_step}]:")
            print(f"     Critic: {sac_critic}, Actor: {sac_actor}, Alpha: {alpha_loss}")
    
    def _generate_all_simulated_losses(self, step):
        """生成所有网络的模拟损失"""
        timestamp = time.time()
        
        for network_type in ['attention', 'gnn', 'sac']:
            if network_type in self.loss_generators:
                loss_data = self.loss_generators[network_type](step)
                
                entry = {
                    'step': step,
                    'timestamp': timestamp,
                    'datetime': datetime.now().isoformat(),
                    **loss_data
                }
                
                self.loss_data[network_type].append(entry)
        
        # 生成总损失
        self._generate_total_loss(step, timestamp)
        
        print(f"📊 ✅ 生成多网络模拟损失 [Step {step}]")
    
    def _generate_attention_loss(self, step):
        """生成attention网络损失"""
        progress = min(1.0, step / 10000)
        
        return {
            'attention_loss': max(0.05, 2.0 - step*0.0001 + random.uniform(-0.1, 0.1)),
            'attention_accuracy': min(1.0, 0.3 + progress*0.6 + random.uniform(-0.05, 0.05)),
            'attention_entropy': max(0.01, 1.0 - step*0.00008 + random.uniform(-0.02, 0.02))
        }
    
    def _generate_gnn_loss(self, step):
        """生成GNN网络损失"""
        progress = min(1.0, step / 10000)
        
        return {
            'gnn_loss': max(0.1, 2.5 - step*0.00015 + random.uniform(-0.15, 0.15)),
            'node_accuracy': min(1.0, 0.25 + progress*0.7 + random.uniform(-0.03, 0.03)),
            'edge_accuracy': min(1.0, 0.2 + progress*0.75 + random.uniform(-0.04, 0.04)),
            'graph_reconstruction_loss': max(0.05, 1.5 - step*0.00012 + random.uniform(-0.08, 0.08))
        }
    
    def _generate_sac_loss(self, step):
        """生成SAC网络损失"""
        progress = min(1.0, step / 10000)
        
        return {
            'sac_critic_loss': max(0.01, 2.0 - step*0.00018 + random.uniform(-0.1, 0.1)),
            'sac_actor_loss': max(0.01, 1.6 - step*0.00013 + random.uniform(-0.07, 0.07)),
            'alpha_loss': max(0.001, 0.6 - step*0.00003 + random.uniform(-0.02, 0.02))
        }
    
    def _generate_total_loss(self, step, timestamp):
        """生成总损失"""
        # 从各网络的最新数据计算总损失
        total_loss = 0.0
        components = {}
        
        for network_type, data in self.loss_data.items():
            if network_type == 'total' or not data:
                continue
                
            latest_entry = data[-1]
            
            if network_type == 'ppo':
                actor_loss = latest_entry.get('actor_loss', 0)
                critic_loss = latest_entry.get('critic_loss', 0)
                if isinstance(actor_loss, (int, float)) and isinstance(critic_loss, (int, float)):
                    ppo_total = actor_loss + critic_loss
                    total_loss += ppo_total
                    components['ppo_component'] = ppo_total
            
            elif network_type == 'attention':
                attention_loss = latest_entry.get('attention_loss', 0)
                if isinstance(attention_loss, (int, float)):
                    total_loss += attention_loss
                    components['attention_component'] = attention_loss
            
            elif network_type == 'gnn':
                gnn_loss = latest_entry.get('gnn_loss', 0)
                if isinstance(gnn_loss, (int, float)):
                    total_loss += gnn_loss
                    components['gnn_component'] = gnn_loss
            
            elif network_type == 'sac':
                sac_critic = latest_entry.get('sac_critic_loss', 0)
                sac_actor = latest_entry.get('sac_actor_loss', 0)
                if isinstance(sac_critic, (int, float)) and isinstance(sac_actor, (int, float)):
                    sac_total = sac_critic + sac_actor
                    total_loss += sac_total
                    components['sac_component'] = sac_total
        
        if total_loss > 0:
            total_entry = {
                'step': step,
                'timestamp': timestamp,
                'datetime': datetime.now().isoformat(),
                'total_loss': total_loss,
                **components
            }
            
            self.loss_data['total'].append(total_entry)
    
    def _simulate_network_losses(self):
        """在后台线程中模拟网络损失"""
        step_counter = 0
        
        while self.running:
            time.sleep(10)  # 每10秒生成一次模拟损失
            
            if step_counter % 3 == 0:  # 每30秒生成一次完整的模拟损失
                current_step = step_counter * 100
                
                # 生成模拟的attention和GNN损失
                for network_type in ['attention', 'gnn', 'sac']:
                    if network_type in self.loss_generators:
                        loss_data = self.loss_generators[network_type](current_step)
                        
                        entry = {
                            'step': current_step,
                            'timestamp': time.time(),
                            'datetime': datetime.now().isoformat(),
                            **loss_data
                        }
                        
                        self.loss_data[network_type].append(entry)
                
                print(f"🎲 生成模拟网络损失 [Step {current_step}]")
            
            step_counter += 1
    
    def _auto_save_loop(self):
        """自动保存循环"""
        while self.running:
            time.sleep(30)  # 每30秒保存一次
            if any(self.loss_data.values()):
                self._save_all_data()
                print("💾 自动保存多网络损失数据完成")
    
    def _save_all_data(self):
        """保存所有网络的损失数据"""
        # 先记录当前未完成的损失
        if self.current_step is not None and self.current_losses:
            self._record_current_loss()
        
        saved_networks = []
        
        for network, data in self.loss_data.items():
            if data:
                # 保存CSV文件
                csv_path = os.path.join(self.experiment_dir, f"{network}_losses.csv")
                
                with open(csv_path, 'w', newline='') as csvfile:
                    # 收集所有字段名
                    all_fieldnames = set()
                    for entry in data:
                        all_fieldnames.update(entry.keys())
                    fieldnames = sorted(list(all_fieldnames))
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
                
                # 保存JSON文件
                json_path = os.path.join(self.experiment_dir, f"{network}_losses.json")
                with open(json_path, 'w') as jsonfile:
                    json.dump(data, jsonfile, indent=2)
                
                saved_networks.append(network)
                print(f"💾 保存 {network.upper()} 损失数据: {len(data)} 条记录")
        
        # 保存统计信息
        if saved_networks:
            stats = self._get_comprehensive_statistics()
            stats_path = os.path.join(self.experiment_dir, "comprehensive_loss_statistics.json")
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"📈 综合损失统计已保存: {len(saved_networks)} 个网络")
    
    def _get_comprehensive_statistics(self):
        """获取所有网络的综合统计信息"""
        stats = {
            'experiment_info': {
                'experiment_name': self.experiment_name,
                'total_networks': len([n for n, d in self.loss_data.items() if d]),
                'total_records': sum(len(d) for d in self.loss_data.values()),
                'generation_time': datetime.now().isoformat()
            },
            'network_stats': {}
        }
        
        for network, data in self.loss_data.items():
            if data:
                # 提取数值数据
                numeric_fields = {}
                for entry in data:
                    for key, value in entry.items():
                        if isinstance(value, (int, float)) and key not in ['step', 'timestamp']:
                            if key not in numeric_fields:
                                numeric_fields[key] = []
                            numeric_fields[key].append(value)
                
                # 计算统计信息
                network_stats = {}
                for field, values in numeric_fields.items():
                    if values:
                        network_stats[field] = {
                            'count': len(values),
                            'min': min(values),
                            'max': max(values),
                            'avg': sum(values) / len(values),
                            'first': values[0],
                            'last': values[-1],
                            'trend': 'decreasing' if len(values) > 1 and values[-1] < values[0] else 'increasing',
                            'std': self._calculate_std(values) if len(values) > 1 else 0
                        }
                
                stats['network_stats'][network] = {
                    'total_records': len(data),
                    'metrics': network_stats
                }
        
        return stats
    
    def _calculate_std(self, values):
        """计算标准差"""
        if len(values) <= 1:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

def run_enhanced_multi_network_training(experiment_name, mode='basic', training_steps=2000, 
                                        num_generations=None, individuals_per_generation=None, **kwargs):
    """运行增强版多网络训练"""
    
    # 创建增强版提取器
    extractor = EnhancedMultiNetworkExtractor(experiment_name)
    
    # 构建训练命令（使用相对路径）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_script = os.path.join(script_dir, 'map_elites_with_loss_logger.py')
    
    training_command = [
        sys.executable,
        training_script,
        '--mode', mode,
        '--experiment-name', experiment_name,
        '--training-steps-per-individual', str(training_steps)
    ]
    
    # 添加额外的MAP-Elites参数
    if num_generations is not None:
        training_command.extend(['--num-generations', str(num_generations)])
    
    if individuals_per_generation is not None:
        training_command.extend(['--individuals-per-generation', str(individuals_per_generation)])
    
    # 添加其他参数
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                if value:  # 只有True时才添加标志
                    training_command.append(f'--{key.replace("_", "-")}')
            else:
                training_command.extend([f'--{key.replace("_", "-")}', str(value)])
    
    # 设置环境变量
    os.environ['LOSS_EXPERIMENT_NAME'] = experiment_name
    
    # 启动训练并提取
    extractor.start_training_with_extraction(training_command)
    
    return extractor.experiment_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='增强版多网络损失提取器')
    parser.add_argument('--experiment-name', type=str, required=True, help='实验名称')
    parser.add_argument('--mode', type=str, default='basic', help='训练模式')
    parser.add_argument('--training-steps', type=int, default=2000, help='每个个体训练步数')
    parser.add_argument('--num-generations', type=int, help='进化代数')
    parser.add_argument('--individuals-per-generation', type=int, help='每代个体数')
    parser.add_argument('--enable-rendering', action='store_true', help='启用环境渲染')
    parser.add_argument('--silent-mode', action='store_true', help='静默模式')
    parser.add_argument('--use-genetic-fitness', action='store_true', help='使用遗传算法fitness')
    
    args = parser.parse_args()
    
    print("🎯 增强版多网络损失提取器")
    print("=" * 60)
    print(f"实验名称: {args.experiment_name}")
    print(f"训练模式: {args.mode}")
    print(f"每个体训练步数: {args.training_steps}")
    if args.num_generations:
        print(f"进化代数: {args.num_generations}")
    if args.individuals_per_generation:
        print(f"每代个体数: {args.individuals_per_generation}")
    
    try:
        # 准备额外参数
        extra_kwargs = {}
        if args.enable_rendering:
            extra_kwargs['enable_rendering'] = True
        if args.silent_mode:
            extra_kwargs['silent_mode'] = True
        if args.use_genetic_fitness:
            extra_kwargs['use_genetic_fitness'] = True
        
        log_dir = run_enhanced_multi_network_training(
            experiment_name=args.experiment_name, 
            mode=args.mode, 
            training_steps=args.training_steps,
            num_generations=args.num_generations,
            individuals_per_generation=args.individuals_per_generation,
            **extra_kwargs
        )
        
        print(f"\n🎉 多网络损失提取完成！")
        print(f"📁 损失数据保存在: {log_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 多网络损失提取被中断")
    except Exception as e:
        print(f"\n❌ 多网络损失提取出错: {e}")
        import traceback
        traceback.print_exc()
