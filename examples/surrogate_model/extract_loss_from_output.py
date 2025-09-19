#!/usr/bin/env python3
"""
从训练输出中提取损失数据
实时解析训练输出并记录损失数据
"""

import re
import os
import csv
import json
import time
from datetime import datetime
from collections import defaultdict

class LossExtractor:
    """从训练输出中提取损失数据"""
    
    def __init__(self, experiment_name, log_dir="extracted_loss_logs"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.experiment_dir = os.path.join(log_dir, f"{experiment_name}_extracted_loss")
        
        # 创建目录
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 损失数据存储
        self.loss_data = defaultdict(list)
        
        # 正则表达式模式
        self.patterns = {
            'ppo_update': re.compile(r'🔥 PPO网络Loss更新 \[Step (\d+)\]:'),
            'actor_loss': re.compile(r'📊 Actor Loss: ([\d\.-]+)'),
            'critic_loss': re.compile(r'📊 Critic Loss: ([\d\.-]+)'),
            'total_loss': re.compile(r'📊 总Loss: ([\d\.-]+)'),
            'entropy': re.compile(r'🎭 Entropy: ([\d\.-]+)'),
            'learning_rate': re.compile(r'📈 学习率: ([\d\.-]+e?[\d\.-]*)'),
            'update_count': re.compile(r'🔄 更新次数: (\d+)'),
            'buffer_size': re.compile(r'💾 Buffer大小: (\d+)')
        }
        
        print(f"📊 损失提取器初始化")
        print(f"   实验名称: {experiment_name}")
        print(f"   日志目录: {self.experiment_dir}")
        
    def extract_from_line(self, line, step_context=None):
        """从单行文本中提取损失数据"""
        # 检查是否是PPO更新行
        step_match = self.patterns['ppo_update'].search(line)
        if step_match:
            return int(step_match.group(1))
        
        # 如果有步数上下文，提取损失值
        if step_context is not None:
            losses = {}
            
            for loss_type, pattern in self.patterns.items():
                if loss_type == 'ppo_update':
                    continue
                    
                match = pattern.search(line)
                if match:
                    try:
                        value = float(match.group(1))
                        losses[loss_type] = value
                    except ValueError:
                        pass
            
            if losses:
                self.record_loss(step_context, losses)
                
        return None
        
    def record_loss(self, step, losses):
        """记录损失数据"""
        timestamp = time.time()
        
        entry = {
            'step': step,
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            **losses
        }
        
        self.loss_data['ppo'].append(entry)
        actor_loss = losses.get('actor_loss', 'N/A')
        critic_loss = losses.get('critic_loss', 'N/A')
        actor_str = f"{actor_loss:.3f}" if isinstance(actor_loss, (int, float)) else str(actor_loss)
        critic_str = f"{critic_loss:.3f}" if isinstance(critic_loss, (int, float)) else str(critic_loss)
        print(f"📊 记录PPO损失 [Step {step}]: Actor={actor_str}, Critic={critic_str}")
        
    def save_data(self):
        """保存损失数据"""
        if not self.loss_data:
            print("⚠️ 没有损失数据可保存")
            return
            
        for network, data in self.loss_data.items():
            if data:
                # 保存CSV文件
                csv_path = os.path.join(self.experiment_dir, f"{network}_losses.csv")
                
                with open(csv_path, 'w', newline='') as csvfile:
                    if data:
                        # 收集所有可能的字段名
                        all_fieldnames = set()
                        for entry in data:
                            all_fieldnames.update(entry.keys())
                        fieldnames = sorted(list(all_fieldnames))
                        
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(data)
                
                print(f"💾 保存 {network} 损失数据: {len(data)} 条记录 -> {csv_path}")
                
                # 保存JSON文件
                json_path = os.path.join(self.experiment_dir, f"{network}_losses.json")
                with open(json_path, 'w') as jsonfile:
                    json.dump(data, jsonfile, indent=2)
        
        # 保存统计信息
        stats = self.get_statistics()
        stats_path = os.path.join(self.experiment_dir, "loss_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📈 损失统计信息已保存: {stats_path}")
        
    def get_statistics(self):
        """获取损失统计信息"""
        stats = {}
        
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
                            'trend': 'decreasing' if values[-1] < values[0] else 'increasing'
                        }
                
                stats[network] = network_stats
        
        return stats

def extract_from_file(log_file_path, experiment_name):
    """从日志文件中提取损失数据"""
    extractor = LossExtractor(experiment_name)
    
    current_step = None
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # 检查是否是新的步数
                step = extractor.extract_from_line(line)
                if step is not None:
                    current_step = step
                
                # 提取损失数据
                extractor.extract_from_line(line, current_step)
        
        # 保存数据
        extractor.save_data()
        
        print("✅ 损失提取完成")
        
    except Exception as e:
        print(f"❌ 提取失败: {e}")

if __name__ == "__main__":
    # 测试提取器
    print("🧪 测试损失提取器")
    
    # 模拟训练输出
    test_lines = [
        "🔥 PPO网络Loss更新 [Step 1063]:",
        "   📊 Actor Loss: 0.357086",
        "   📊 Critic Loss: 19.341845", 
        "   📊 总Loss: 19.698931",
        "   🎭 Entropy: -0.405246",
        "   📈 学习率: 8.24e-06",
        "🔥 PPO网络Loss更新 [Step 1127]:",
        "   📊 Actor Loss: 0.367207",
        "   📊 Critic Loss: 35.931519",
        "   📊 总Loss: 36.298727",
        "   🎭 Entropy: -0.405081"
    ]
    
    extractor = LossExtractor("test_extract")
    
    current_step = None
    for line in test_lines:
        step = extractor.extract_from_line(line)
        if step is not None:
            current_step = step
        extractor.extract_from_line(line, current_step)
    
    extractor.save_data()
    print("✅ 测试完成")
