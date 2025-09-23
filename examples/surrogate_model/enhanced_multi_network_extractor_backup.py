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
        
        # 🔧 确保日志保存在正确的位置（surrogate_model目录下）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(script_dir, log_dir)
        self.experiment_dir = os.path.join(self.log_dir, f"{experiment_name}_multi_network_loss")
        
        # 创建目录
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 🆕 Individual成功次数记录（独立文件）
        self.individual_success_files = {}  # individual_id -> 文件路径
        self.current_individual_success = 0  # 当前individual的成功次数
        
        # 只记录真实存在的网络损失数据
        self.loss_data = {
            'sac': [],           # SAC网络有真实损失输出
            'performance': []    # 性能指标有真实输出（成功率、距离等）
            # 注意：只有在训练输出中真实存在时才会动态添加其他网络
        }
        self.running = False
        
        # 正则表达式模式 - 只匹配真实存在的输出
        self.patterns = {
            # SAC网络损失（真实存在）
            'sac_update': re.compile(r'🔥 SAC网络Loss更新 \[Step (\d+)\]:'),
            'actor_loss': re.compile(r'📊 Actor Loss: ([\d\.-]+)'),
            'critic_loss': re.compile(r'📊 Critic Loss: ([\d\.-]+)'),
            'sac_total_loss': re.compile(r'📊 总Loss: ([\d\.-]+)'),
            'alpha_loss': re.compile(r'📊 Alpha Loss: ([\d\.-]+)'),
            'alpha': re.compile(r'📊 Alpha: ([\d\.-]+)'),
            'learning_rate': re.compile(r'📈 学习率: ([\d\.-]+e?[\d\.-]*)'),
            'update_count': re.compile(r'🔄 更新次数: (\d+)'),
            'buffer_size': re.compile(r'💾 Buffer大小: (\d+)'),
            'q1_mean': re.compile(r'📊 Q1均值: ([\d\.-]+)'),
            'q2_mean': re.compile(r'📊 Q2均值: ([\d\.-]+)'),
            'q1_std': re.compile(r'📊 Q1标准差: ([\d\.-]+)'),
            'q2_std': re.compile(r'📊 Q2标准差: ([\d\.-]+)'),
            'entropy_term': re.compile(r'📊 熵项: ([\d\.-]+)'),
            'q_term': re.compile(r'📊 Q值项: ([\d\.-]+)'),
            
            # 性能指标（真实存在）
            'success_rate': re.compile(r'✅ 当前成功率: ([\d\.]+)%'),
            'best_distance': re.compile(r'🏆 当前最佳距离: ([\d\.]+)px'),
            'episode_best_distance': re.compile(r'📊 当前Episode最佳距离: ([\d\.]+)px'),
            'consecutive_success': re.compile(r'🔄 连续成功次数: (\d+)'),
            'completed_episodes': re.compile(r'📋 已完成Episodes: (\d+)'),
            'training_progress_report': re.compile(r'📊 PPO训练进度报告 \[Step (\d+)\]'),
            
            # 🆕 成功事件检测（独立记录）
            'goal_reached': re.compile(r'🎯 \[DEBUG\] 到达目标但需继续维持，继续当前.*episode'),
            'goal_reached_with_distance': re.compile(r'🎯 到达目标! 距离: ([\d\.]+)px，进入下一个episode'),
            
            # 真实的attention网络损失模式（现在已实现）
            'attention_update': re.compile(r'🔥 Attention网络Loss更新 \[Step (\d+)\]:'),
            # 🆕 独立的attention网络loss模式
            'attention_actor_loss': re.compile(r'📊 Actor Attention Loss: ([\d\.-]+)'),
            'attention_critic_main_loss': re.compile(r'📊 Critic Main Attention Loss: ([\d\.-]+)'),
            'attention_critic_value_loss': re.compile(r'📊 Critic Value Attention Loss: ([\d\.-]+)'),
            
            # 梯度范数信息
            'attention_actor_grad_norm': re.compile(r'🔍 Actor Attention梯度范数: ([\d\.-]+)'),
            'attention_critic_main_grad_norm': re.compile(r'🔍 Critic Main Attention梯度范数: ([\d\.-]+)'),
            'attention_critic_value_grad_norm': re.compile(r'🔍 Critic Value Attention梯度范数: ([\d\.-]+)'),
            'attention_total_loss': re.compile(r'📊 Attention总损失: ([\d\.-]+)'),
            'attention_param_mean': re.compile(r'📊 Attention参数均值: ([\d\.-]+)'),
            'attention_param_std': re.compile(r'📊 Attention参数标准差: ([\d\.-]+)'),
            
            # 🆕 分离的Actor和Critic attention网络参数
            'attention_actor_param_mean': re.compile(r'📊 Actor Attention参数: 均值=([\d\.-]+), 标准差=([\d\.-]+)'),
            'attention_critic_param_mean': re.compile(r'📊 Critic Attention参数: 均值=([\d\.-]+), 标准差=([\d\.-]+)'),
            
            # 🆕 关节注意力分布模式
            'most_attended_joint': re.compile(r'🎯 最关注关节: Joint (\d+)'),
            'most_important_joint': re.compile(r'🎯 最重要关节: Joint (\d+)'),
            'max_joint_importance': re.compile(r'最重要关节: Joint \d+ \(重要性: ([\d\.-]+)\)'),
            'importance_concentration': re.compile(r'📊 重要性集中度: ([\d\.-]+)'),
            'importance_entropy': re.compile(r'📊 重要性熵值: ([\d\.-]+)'),
            'robot_num_joints': re.compile(r'🤖 机器人结构: (\d+)关节'),
            'robot_structure_info': re.compile(r'🤖 机器人结构: \d+关节 \((.+?)\)'),
            'joint_usage_ranking': re.compile(r'🏆 关节使用排名: (.+)'),
            
            # 🆕 动态关节数据模式（支持任意关节数）
            'joint_activity': re.compile(r'🔍 关节活跃度: (.+)'),
            'joint_importance': re.compile(r'🎯 关节重要性: (.+)'),  # 🆕 添加关节重要性模式
            'joint_angles': re.compile(r'📐 关节角度幅度: (.+)'),
            'joint_velocities': re.compile(r'⚡ 关节速度幅度: (.+)'),
            'link_lengths': re.compile(r'📏 Link长度: (.+)'),
            
            # GNN网络损失模式（如果将来实现）
            'gnn_update': re.compile(r'🔥 GNN网络Loss更新 \[Step (\d+)\]:'),
            'gnn_loss': re.compile(r'📊 GNN Loss: ([\d\.-]+)'),
            'node_accuracy': re.compile(r'📊 节点准确率: ([\d\.-]+)'),
            
            # PPO网络损失模式（如果将来实现）
            'ppo_update': re.compile(r'🔥 PPO网络Loss更新 \[Step (\d+)\]:'),
            'ppo_critic_loss': re.compile(r'📊 PPO Critic Loss: ([\d\.-]+)'),
            'ppo_actor_loss': re.compile(r'📊 PPO Actor Loss: ([\d\.-]+)'),
            'ppo_total_loss': re.compile(r'📊 PPO总Loss: ([\d\.-]+)'),
            'entropy': re.compile(r'🎭 Entropy: ([\d\.-]+)'),
            
            # 🆕 个体和代数信息提取
            'individual_evaluation': re.compile(r'🧬 评估个体 (.+)'),
            'individual_id_setting': re.compile(r'🆔 设置Individual ID: (.+)'),
            'generation_info': re.compile(r'第(\d+)代'),
        }
        
        # 当前损失数据缓存
        self.current_step = None
        self.current_network = 'sac'
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
        print(f"   📊 当前支持: SAC网络损失 + Individual Reacher性能指标")
        
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
        
        # 🆕 检查成功事件（独立记录到individual专用文件）
        goal_reached_match = self.patterns['goal_reached'].search(line)
        goal_reached_with_distance_match = self.patterns['goal_reached_with_distance'].search(line)
        
        if goal_reached_match or goal_reached_with_distance_match:
            # 如果有距离信息就用，否则设为0
            if goal_reached_with_distance_match:
                distance = float(goal_reached_with_distance_match.group(1))
            else:
                distance = 0.0  # DEBUG格式没有距离信息
                
            self.current_individual_success += 1
            
            print(f"   🎉 检测到成功事件! 距离: {distance}px")
            print(f"   📊 Individual {self.current_individual_id} 成功次数: {self.current_individual_success}")
            
            # 记录到individual专用文件
            self._record_individual_success(distance)
            return
        
        # 🆕 检查Individual ID设置
        individual_id_match = self.patterns['individual_id_setting'].search(line)
        if individual_id_match:
            new_individual_id = individual_id_match.group(1).strip()
            
            # 如果是新的individual，重置成功计数
            if self.current_individual_id != new_individual_id:
                self.current_individual_success = 0
                print(f"   🔄 切换到新Individual: {new_individual_id}")
            
            self.current_individual_id = new_individual_id
            print(f"   🆔 检测到Individual ID设置: {self.current_individual_id}")
            return
        
        generation_match = self.patterns['generation_info'].search(line)
        if generation_match:
            self.current_generation = int(generation_match.group(1))
            print(f"   📋 检测到代数: {self.current_generation}")
            return
        
        # 检查SAC网络更新（唯一确认存在的真实网络损失）
        step_match = self.patterns['sac_update'].search(line)
        if step_match:
            # 保存之前的数据
            if self.current_step is not None and self.current_losses:
                self._record_current_loss()
            
            # 开始新的SAC损失记录
            self.current_step = int(step_match.group(1))
            self.current_network = 'sac'
            self.current_losses = {}
            return
        
        # 检查其他网络更新（如果训练输出中真实存在）
        for network_type in ['attention', 'gnn', 'ppo']:
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
            
            if self.current_network == 'sac':
                loss_patterns_to_check = ['actor_loss', 'critic_loss', 'sac_total_loss', 'alpha_loss', 'alpha',
                                        'learning_rate', 'update_count', 'buffer_size', 'q1_mean', 'q2_mean',
                                        'q1_std', 'q2_std', 'entropy_term', 'q_term']
            elif self.current_network == 'ppo':
                loss_patterns_to_check = ['ppo_actor_loss', 'ppo_critic_loss', 'ppo_total_loss', 'entropy', 
                                        'learning_rate', 'update_count', 'buffer_size']
            elif self.current_network == 'attention':
                loss_patterns_to_check = [
                    # 🆕 独立的attention网络loss
                    'attention_actor_loss', 'attention_critic_main_loss', 'attention_critic_value_loss',
                    'attention_total_loss',
                    # 梯度范数
                    'attention_actor_grad_norm', 'attention_critic_main_grad_norm', 'attention_critic_value_grad_norm',
                    # 参数统计
                    'attention_param_mean', 'attention_param_std',
                    'attention_actor_param_mean', 'attention_actor_param_std',
                    'attention_critic_param_mean', 'attention_critic_param_std',
                    # 关节分析
                    'most_important_joint', 'max_joint_importance', 'importance_concentration',
                    'importance_entropy', 'robot_num_joints', 'robot_structure_info',
                    'joint_usage_ranking', 'joint_activity', 'joint_importance', 'joint_angles', 
                    'joint_velocities', 'link_lengths'
                ]
            elif self.current_network == 'gnn':
                loss_patterns_to_check = ['gnn_loss', 'node_accuracy']
            
            for loss_type in loss_patterns_to_check:
                if loss_type in self.patterns:
                    match = self.patterns[loss_type].search(line)
                    if match:
                        try:
                            # 特殊处理关节分布字符串
                            if loss_type in ['joint_activity', 'joint_importance', 'joint_angles', 'joint_velocities', 'link_lengths', 'joint_usage_ranking']:
                                distribution_str = match.group(1)
                                # 解析关节分布字符串，例如 "J0:1.000, J1:1.000, J2:1.000"
                                joint_values = self._parse_joint_distribution(distribution_str, loss_type)
                                self.current_losses.update(joint_values)
                                print(f"   🎯 提取真实{self.current_network.upper()} {loss_type}: {distribution_str[:50]}...")
                            elif loss_type in ['robot_structure_info']:
                                # 字符串类型数据
                                value_str = match.group(1)
                                self.current_losses[loss_type] = value_str
                                print(f"   🎯 提取真实{self.current_network.upper()} {loss_type}: {value_str}")
                            elif loss_type in ['attention_actor_param_mean', 'attention_critic_param_mean']:
                                # 特殊处理：提取均值和标准差
                                mean_value = float(match.group(1))
                                std_value = float(match.group(2))
                                if loss_type == 'attention_actor_param_mean':
                                    self.current_losses['attention_actor_param_mean'] = mean_value
                                    self.current_losses['attention_actor_param_std'] = std_value
                                    print(f"   🎯 提取真实{self.current_network.upper()} Actor参数: 均值={mean_value:.6f}, 标准差={std_value:.6f}")
                                else:
                                    self.current_losses['attention_critic_param_mean'] = mean_value
                                    self.current_losses['attention_critic_param_std'] = std_value
                                    print(f"   🎯 提取真实{self.current_network.upper()} Critic参数: 均值={mean_value:.6f}, 标准差={std_value:.6f}")
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
    
    def _record_individual_success(self, distance):
        """记录individual的成功事件到专用文件"""
        if not self.current_individual_id:
            return
            
        # 为individual创建专用成功记录文件
        if self.current_individual_id not in self.individual_success_files:
            # 🔧 Individual成功记录也保存在同一个实验目录下
            success_file = os.path.join(self.experiment_dir, f"individual_{self.current_individual_id}_success.csv")
            self.individual_success_files[self.current_individual_id] = success_file
            
            # 创建文件并写入表头
            with open(success_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'timestamp', 'datetime', 'distance_to_goal', 'success_count', 'individual_id', 'generation'])
            
            print(f"   📁 为Individual {self.current_individual_id} 创建成功记录文件: {success_file}")
        
        # 记录成功事件
        success_file = self.individual_success_files[self.current_individual_id]
        timestamp = time.time()
        
        with open(success_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_step if self.current_step is not None else 0,
                timestamp,
                datetime.now().isoformat(),
                distance,
                self.current_individual_success,
                self.current_individual_id,
                self.current_generation
            ])
        
        print(f"   💾 成功事件已记录到Individual专用文件")
    
    def _display_recorded_loss(self):
        """显示记录的真实损失数据"""
        if self.current_network == 'sac':
            actor_loss = self.current_losses.get('actor_loss', 'N/A')
            critic_loss = self.current_losses.get('critic_loss', 'N/A')
            alpha_loss = self.current_losses.get('alpha_loss', 'N/A')
            alpha = self.current_losses.get('alpha', 'N/A')
            q1_mean = self.current_losses.get('q1_mean', 'N/A')
            q2_mean = self.current_losses.get('q2_mean', 'N/A')
            print(f"📊 ✅ 记录真实SAC损失 [Step {self.current_step}]:")
            print(f"     Actor: {actor_loss}, Critic: {critic_loss}, Alpha: {alpha_loss}")
            print(f"     Alpha值: {alpha}, Q1均值: {q1_mean}, Q2均值: {q2_mean}")
            
        elif self.current_network == 'ppo':
            actor_loss = self.current_losses.get('ppo_actor_loss', 'N/A')
            critic_loss = self.current_losses.get('ppo_critic_loss', 'N/A')
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
            
    
    def _parse_joint_distribution(self, distribution_str, data_type):
        """解析关节分布字符串 - 支持多种数据类型"""
        joint_values = {}
        
        try:
            # 解析格式: "J0:1.000, J1:1.000, J2:1.000" 或 "L0:80.0px, L1:70.0px"
            parts = distribution_str.split(', ')
            for part in parts:
                if ':' in part:
                    joint_name, value_str = part.split(':')
                    joint_id = joint_name.strip()  # 例如 "J0" 或 "L0"
                    
                    # 清理数值字符串（移除px等单位）
                    clean_value_str = value_str.strip().replace('px', '').replace('°', '')
                    value = float(clean_value_str)
                    
                    # 根据数据类型设置字段名
                    if data_type == 'joint_activity':
                        joint_values[f'{joint_id}_activity'] = value
                    elif data_type == 'joint_importance':
                        joint_values[f'{joint_id}_importance'] = value
                    elif data_type == 'joint_angles':
                        joint_values[f'{joint_id}_angle_magnitude'] = value
                    elif data_type == 'joint_velocities':
                        joint_values[f'{joint_id}_velocity_magnitude'] = value
                    elif data_type == 'link_lengths':
                        joint_values[f'{joint_id}_length'] = value
                    elif data_type == 'joint_usage_ranking':
                        joint_values[f'{joint_id}_usage_rank'] = value
                    else:
                        joint_values[f'{joint_id}_attention'] = value
                        
        except Exception as e:
            joint_values[f'{data_type}_parse_error'] = str(e)
        
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
                    # 🆕 对于attention网络，强制包含20个关节的统一格式
                    if network == 'attention':
                        # 强制包含所有20个关节和link的字段
                        base_fieldnames = set()
                        for entry in data:
                            base_fieldnames.update(entry.keys())
                        
                        # 🆕 强制包含独立的attention loss字段
                        base_fieldnames.add('attention_actor_loss')
                        base_fieldnames.add('attention_critic_main_loss')
                        base_fieldnames.add('attention_critic_value_loss')
                        base_fieldnames.add('attention_critic_main_grad_norm')
                        
                        # 添加20个关节的所有字段
                        for i in range(20):
                            base_fieldnames.add(f'joint_{i}_activity')
                            base_fieldnames.add(f'joint_{i}_importance') 
                            base_fieldnames.add(f'joint_{i}_angle_magnitude')
                            base_fieldnames.add(f'joint_{i}_velocity_magnitude')
                            base_fieldnames.add(f'link_{i}_length')
                        
                        fieldnames = sorted(list(base_fieldnames))
                        
                        # 确保所有数据条目都包含独立attention loss字段和20个关节的字段
                        for entry in data:
                            # 🆕 确保独立attention loss字段存在（不存在的填0.0）
                            if 'attention_actor_loss' not in entry:
                                entry['attention_actor_loss'] = 0.0
                            if 'attention_critic_main_loss' not in entry:
                                entry['attention_critic_main_loss'] = 0.0
                            if 'attention_critic_value_loss' not in entry:
                                entry['attention_critic_value_loss'] = 0.0
                            if 'attention_critic_main_grad_norm' not in entry:
                                entry['attention_critic_main_grad_norm'] = 0.0
                            
                            # 确保20个关节的字段存在（不存在的填-1）
                            for i in range(20):
                                if f'joint_{i}_activity' not in entry:
                                    entry[f'joint_{i}_activity'] = -1.0
                                if f'joint_{i}_importance' not in entry:
                                    entry[f'joint_{i}_importance'] = -1.0
                                if f'joint_{i}_angle_magnitude' not in entry:
                                    entry[f'joint_{i}_angle_magnitude'] = -1.0
                                if f'joint_{i}_velocity_magnitude' not in entry:
                                    entry[f'joint_{i}_velocity_magnitude'] = -1.0
                                if f'link_{i}_length' not in entry:
                                    entry[f'link_{i}_length'] = -1.0
                    else:
                        # 其他网络使用原有的动态字段收集
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
