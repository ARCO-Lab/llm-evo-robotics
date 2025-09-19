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
        
        # 只记录真实存在的网络损失数据
        self.loss_data = {
            'ppo': [],           # PPO网络有真实损失输出
            'performance': []    # 性能指标有真实输出（成功率、距离等）
            # 注意：只有在训练输出中真实存在时才会动态添加其他网络
        }
        self.running = False
        
        # 正则表达式模式 - 只匹配真实存在的输出
        self.patterns = {
            # PPO网络损失（真实存在）
            'ppo_update': re.compile(r'🔥 PPO网络Loss更新 \[Step (\d+)\]:'),
            'actor_loss': re.compile(r'📊 Actor Loss: ([\d\.-]+)'),
            'critic_loss': re.compile(r'📊 Critic Loss: ([\d\.-]+)'),
            'ppo_total_loss': re.compile(r'📊 总Loss: ([\d\.-]+)'),
            'entropy': re.compile(r'🎭 Entropy: ([\d\.-]+)'),
            'learning_rate': re.compile(r'📈 学习率: ([\d\.-]+e?[\d\.-]*)'),
            'update_count': re.compile(r'🔄 更新次数: (\d+)'),
            'buffer_size': re.compile(r'💾 Buffer大小: (\d+)'),
            
            # 性能指标（真实存在）
            'success_rate': re.compile(r'✅ 当前成功率: ([\d\.]+)%'),
            'best_distance': re.compile(r'🏆 当前最佳距离: ([\d\.]+)px'),
            'episode_best_distance': re.compile(r'📊 当前Episode最佳距离: ([\d\.]+)px'),
            'consecutive_success': re.compile(r'🔄 连续成功次数: (\d+)'),
            'completed_episodes': re.compile(r'📋 已完成Episodes: (\d+)'),
            'training_progress_report': re.compile(r'📊 PPO训练进度报告 \[Step (\d+)\]'),
            
            # 真实的attention网络损失模式（现在已实现）
            'attention_update': re.compile(r'🔥 Attention网络Loss更新 \[Step (\d+)\]:'),
            'attention_actor_grad_norm': re.compile(r'📊 Actor Attention梯度范数: ([\d\.-]+)'),
            'attention_critic_grad_norm': re.compile(r'📊 Critic Attention梯度范数: ([\d\.-]+)'),
            'attention_total_loss': re.compile(r'📊 Attention总损失: ([\d\.-]+)'),
            'attention_param_mean': re.compile(r'📊 Attention参数均值: ([\d\.-]+)'),
            'attention_param_std': re.compile(r'📊 Attention参数标准差: ([\d\.-]+)'),
            
            # 🆕 关节注意力分布模式
            'most_attended_joint': re.compile(r'🎯 最关注关节: Joint (\d+)'),
            'max_joint_attention': re.compile(r'最关注关节: Joint \d+ \(强度: ([\d\.-]+)\)'),
            'attention_concentration': re.compile(r'📊 注意力集中度: ([\d\.-]+)'),
            'attention_entropy': re.compile(r'📊 注意力熵值: ([\d\.-]+)'),
            'joint_attention_distribution': re.compile(r'🔍 关节注意力分布: (.+)'),
            
            # GNN网络损失模式（如果将来实现）
            'gnn_update': re.compile(r'🔥 GNN网络Loss更新 \[Step (\d+)\]:'),
            'gnn_loss': re.compile(r'📊 GNN Loss: ([\d\.-]+)'),
            'node_accuracy': re.compile(r'📊 节点准确率: ([\d\.-]+)'),
            
            # SAC网络损失模式（如果将来实现）
            'sac_update': re.compile(r'🔥 SAC网络Loss更新 \[Step (\d+)\]:'),
            'sac_critic_loss': re.compile(r'📊 SAC Critic Loss: ([\d\.-]+)'),
            
            # 🆕 个体和代数信息提取
            'individual_evaluation': re.compile(r'🧬 评估个体 (.+)'),
            'generation_info': re.compile(r'第(\d+)代'),
        }
        
        # 当前损失数据缓存
        self.current_step = None
        self.current_network = 'ppo'
        self.current_losses = {}
        
        # 性能指标缓存
        self.current_performance = {}
        
        # 🆕 当前个体和代数信息
        self.current_individual_id = None
        self.current_generation = 0
        self.individual_count = 0
        self.individuals_per_generation = 10  # 默认值，可以从命令行参数获取
        
        print(f"📊 真实数据损失提取器初始化")
        print(f"   实验名称: {experiment_name}")
        print(f"   日志目录: {self.experiment_dir}")
        print(f"   🎯 只记录真实存在的网络损失，绝不生成假数据")
        print(f"   📊 当前支持: PPO网络损失 + Individual Reacher性能指标")
        
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
            
            # 不再启动模拟损失生成线程 - 只记录真实数据
            
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
        """处理单行输出，只提取真实存在的损失数据"""
        
        # 🆕 首先检查个体和代数信息
        individual_match = self.patterns['individual_evaluation'].search(line)
        if individual_match:
            self.current_individual_id = individual_match.group(1).strip()
            self.individual_count += 1
            
            # 根据个体计数推算generation（每10个个体一代）
            self.current_generation = (self.individual_count - 1) // self.individuals_per_generation
            
            print(f"   📋 检测到个体: {self.current_individual_id}")
            print(f"   📊 个体计数: {self.individual_count}, 推算代数: {self.current_generation}")
            return
        
        generation_match = self.patterns['generation_info'].search(line)
        if generation_match:
            self.current_generation = int(generation_match.group(1))
            print(f"   📋 检测到代数: {self.current_generation}")
            return
        
        # 检查PPO网络更新（唯一确认存在的真实网络损失）
        step_match = self.patterns['ppo_update'].search(line)
        if step_match:
            # 保存之前的数据
            if self.current_step is not None and self.current_losses:
                self._record_current_loss()
            
            # 开始新的PPO损失记录
            self.current_step = int(step_match.group(1))
            self.current_network = 'ppo'
            self.current_losses = {}
            return
        
        # 检查其他网络更新（如果训练输出中真实存在）
        for network_type in ['attention', 'gnn', 'sac']:
            update_pattern = f'{network_type}_update'
            if update_pattern in self.patterns:
                step_match = self.patterns[update_pattern].search(line)
                if step_match:
                    # 保存之前的数据
                    if self.current_step is not None and self.current_losses:
                        self._record_current_loss()
                    
                    # 开始新的网络损失记录
                    self.current_step = int(step_match.group(1))
                    self.current_network = network_type
                    self.current_losses = {}
                    
                    # 动态添加网络到数据存储
                    if network_type not in self.loss_data:
                        self.loss_data[network_type] = []
                        print(f"   🎯 检测到真实{network_type.upper()}网络损失，开始记录")
                    return
        
        # 检查训练进度报告
        progress_match = self.patterns['training_progress_report'].search(line)
        if progress_match:
            # 保存之前的性能数据
            if self.current_performance:
                self._record_performance_metrics()
            
            # 开始新的性能报告
            self.current_performance = {'report_step': int(progress_match.group(1))}
            return
        
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
        
        # 提取真实损失值
        if self.current_step is not None:
            # 根据当前网络类型提取对应的损失值
            loss_patterns_to_check = []
            
            if self.current_network == 'ppo':
                loss_patterns_to_check = ['actor_loss', 'critic_loss', 'ppo_total_loss', 'entropy', 
                                        'learning_rate', 'update_count', 'buffer_size']
            elif self.current_network == 'attention':
                loss_patterns_to_check = ['attention_actor_grad_norm', 'attention_critic_grad_norm', 
                                        'attention_total_loss', 'attention_param_mean', 'attention_param_std',
                                        'most_attended_joint', 'max_joint_attention', 'attention_concentration',
                                        'attention_entropy', 'joint_attention_distribution']
            elif self.current_network == 'gnn':
                loss_patterns_to_check = ['gnn_loss', 'node_accuracy']
            elif self.current_network == 'sac':
                loss_patterns_to_check = ['sac_critic_loss', 'sac_actor_loss']
            
            for loss_type in loss_patterns_to_check:
                if loss_type in self.patterns:
                    match = self.patterns[loss_type].search(line)
                    if match:
                        try:
                            # 特殊处理关节分布字符串
                            if loss_type == 'joint_attention_distribution':
                                distribution_str = match.group(1)
                                # 解析关节分布字符串，例如 "J0:1.000, J1:1.000, J2:1.000, J3:1.000, J4:0.000, J5:0.000"
                                joint_values = self._parse_joint_distribution(distribution_str)
                                self.current_losses.update(joint_values)
                                print(f"   🎯 提取真实{self.current_network.upper()} 关节分布: {distribution_str}")
                            else:
                                value = float(match.group(1))
                                self.current_losses[loss_type] = value
                                print(f"   🎯 提取真实{self.current_network.upper()} {loss_type}: {value}")
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
            'generation': self.current_generation,
            'individual_id': self.current_individual_id,
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
            'generation': self.current_generation,
            'individual_id': self.current_individual_id,
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
        """显示记录的真实损失数据"""
        if self.current_network == 'ppo':
            actor_loss = self.current_losses.get('actor_loss', 'N/A')
            critic_loss = self.current_losses.get('critic_loss', 'N/A')
            total_loss = self.current_losses.get('ppo_total_loss', 'N/A')
            print(f"📊 ✅ 记录真实PPO损失 [Step {self.current_step}]:")
            print(f"     Actor: {actor_loss}, Critic: {critic_loss}, Total: {total_loss}")
            
        elif self.current_network == 'attention':
            actor_grad = self.current_losses.get('attention_actor_grad_norm', 'N/A')
            critic_grad = self.current_losses.get('attention_critic_grad_norm', 'N/A')
            total_loss = self.current_losses.get('attention_total_loss', 'N/A')
            most_attended = self.current_losses.get('most_attended_joint', 'N/A')
            concentration = self.current_losses.get('attention_concentration', 'N/A')
            print(f"📊 ✅ 记录真实Attention损失 [Step {self.current_step}]:")
            print(f"     Actor梯度: {actor_grad}, Critic梯度: {critic_grad}, 总损失: {total_loss}")
            print(f"     🎯 最关注关节: Joint {most_attended}, 集中度: {concentration}")
            
        elif self.current_network == 'gnn':
            gnn_loss = self.current_losses.get('gnn_loss', 'N/A')
            node_acc = self.current_losses.get('node_accuracy', 'N/A')
            print(f"📊 ✅ 记录真实GNN损失 [Step {self.current_step}]:")
            print(f"     Loss: {gnn_loss}, Node Acc: {node_acc}")
            
        elif self.current_network == 'sac':
            sac_critic = self.current_losses.get('sac_critic_loss', 'N/A')
            sac_actor = self.current_losses.get('sac_actor_loss', 'N/A')
            print(f"📊 ✅ 记录真实SAC损失 [Step {self.current_step}]:")
            print(f"     Critic: {sac_critic}, Actor: {sac_actor}")
    
    def _parse_joint_distribution(self, distribution_str):
        """解析关节注意力分布字符串"""
        joint_values = {}
        
        try:
            # 解析格式: "J0:1.000, J1:1.000, J2:1.000, J3:1.000, J4:0.000, J5:0.000"
            parts = distribution_str.split(', ')
            for part in parts:
                if ':' in part:
                    joint_name, value_str = part.split(':')
                    joint_id = joint_name.strip()  # 例如 "J0"
                    value = float(value_str.strip())
                    joint_values[f'{joint_id}_attention'] = value
        except Exception as e:
            joint_values['joint_distribution_parse_error'] = str(e)
        
        return joint_values
    
    def _auto_save_loop(self):
        """自动保存循环"""
        while self.running:
            time.sleep(30)  # 每30秒保存一次
            if any(self.loss_data.values()):
                self._save_all_data()
                print("💾 自动保存真实损失数据完成")
    
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
                print(f"💾 保存真实 {network.upper()} 数据: {len(data)} 条记录")
        
        # 保存统计信息
        if saved_networks:
            stats = self._get_comprehensive_statistics()
            stats_path = os.path.join(self.experiment_dir, "comprehensive_loss_statistics.json")
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"📈 真实损失统计已保存: {len(saved_networks)} 个网络")
    
    def _get_comprehensive_statistics(self):
        """获取所有网络的综合统计信息"""
        stats = {
            'experiment_info': {
                'experiment_name': self.experiment_name,
                'data_type': 'real_only',  # 标明这是真实数据
                'total_networks': len([n for n, d in self.loss_data.items() if d]),
                'total_records': sum(len(d) for d in self.loss_data.values()),
                'generation_time': datetime.now().isoformat(),
                'note': 'Only real loss data from training output, no simulated data'
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
    
    # 🆕 设置每代个体数，用于generation计算
    if individuals_per_generation is not None:
        extractor.individuals_per_generation = individuals_per_generation
    
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
        
        print(f"\n🎉 真实损失提取完成！")
        print(f"📁 真实损失数据保存在: {log_dir}")
        print(f"📊 只包含训练输出中真实存在的网络损失，无任何模拟数据")
        
    except KeyboardInterrupt:
        print("\n⚠️ 多网络损失提取被中断")
    except Exception as e:
        print(f"\n❌ 多网络损失提取出错: {e}")
        import traceback
        traceback.print_exc()
