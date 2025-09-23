#!/usr/bin/env python3
"""
Individual成功次数记录器
每个individual进程单独记录自己的成功次数，使用CSV或JSON格式
"""

import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import threading

@dataclass
class IndividualSuccessRecord:
    """Individual成功记录数据结构"""
    individual_id: str
    timestamp: str
    episode_number: int
    success: bool
    fitness: float
    distance: float
    episode_reward: float
    episode_steps: int
    maintain_progress: float
    notes: str = ""

class IndividualSuccessLogger:
    """Individual成功次数记录器 - 每个进程独立使用"""
    
    def __init__(self, individual_id: str, log_dir: str = "individual_success_logs", 
                 format_type: str = "csv"):
        """
        初始化Individual成功记录器
        
        Args:
            individual_id: Individual ID
            log_dir: 日志保存目录
            format_type: 记录格式 ("csv" 或 "json")
        """
        self.individual_id = individual_id
        self.log_dir = log_dir
        self.format_type = format_type.lower()
        self.process_id = os.getpid()
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"individual_{individual_id}_pid_{self.process_id}_{timestamp}"
        
        if self.format_type == "csv":
            self.log_file = os.path.join(log_dir, f"{base_filename}.csv")
            self._init_csv_file()
        else:
            self.log_file = os.path.join(log_dir, f"{base_filename}.json")
            self._init_json_file()
        
        # 统计计数器
        self.total_episodes = 0
        self.successful_episodes = 0
        self.records_cache = []
        
        # 线程锁（防止并发写入）
        self._lock = threading.Lock()
        
        print(f"🗂️ Individual成功记录器已初始化")
        print(f"   Individual ID: {individual_id}")
        print(f"   进程ID: {self.process_id}")
        print(f"   记录文件: {self.log_file}")
        print(f"   记录格式: {self.format_type.upper()}")
    
    def _init_csv_file(self):
        """初始化CSV文件"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow([
                    'individual_id', 'timestamp', 'episode_number', 'success',
                    'fitness', 'distance', 'episode_reward', 'episode_steps',
                    'maintain_progress', 'notes'
                ])
    
    def _init_json_file(self):
        """初始化JSON文件"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2, ensure_ascii=False)
    
    def log_episode(self, episode_number: int, success: bool, fitness: float = 0.0,
                   distance: float = 0.0, episode_reward: float = 0.0, 
                   episode_steps: int = 0, maintain_progress: float = 0.0,
                   notes: str = "") -> None:
        """
        记录一个episode的结果
        
        Args:
            episode_number: Episode编号
            success: 是否成功
            fitness: 适应度分数
            distance: 最终距离
            episode_reward: Episode总奖励
            episode_steps: Episode步数
            maintain_progress: 维持进度(0-1)
            notes: 备注信息
        """
        with self._lock:
            # 创建记录
            record = IndividualSuccessRecord(
                individual_id=self.individual_id,
                timestamp=datetime.now().isoformat(),
                episode_number=episode_number,
                success=success,
                fitness=fitness,
                distance=distance,
                episode_reward=episode_reward,
                episode_steps=episode_steps,
                maintain_progress=maintain_progress,
                notes=notes
            )
            
            # 更新统计
            self.total_episodes += 1
            if success:
                self.successful_episodes += 1
            
            # 保存记录
            self._save_record(record)
            
            # 缓存记录
            self.records_cache.append(record)
            
            # 输出日志
            status = "✅" if success else "❌"
            success_rate = self.successful_episodes / self.total_episodes
            print(f"{status} Individual {self.individual_id} Episode {episode_number}: "
                  f"成功={success}, 适应度={fitness:.3f}, 距离={distance:.1f}px")
            print(f"   总Episodes: {self.total_episodes}, 成功: {self.successful_episodes} "
                  f"({success_rate:.1%})")
    
    def _save_record(self, record: IndividualSuccessRecord):
        """保存记录到文件"""
        try:
            if self.format_type == "csv":
                self._save_to_csv(record)
            else:
                self._save_to_json(record)
        except Exception as e:
            print(f"❌ 保存记录失败: {e}")
    
    def _save_to_csv(self, record: IndividualSuccessRecord):
        """保存到CSV文件"""
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                record.individual_id, record.timestamp, record.episode_number,
                record.success, record.fitness, record.distance, record.episode_reward,
                record.episode_steps, record.maintain_progress, record.notes
            ])
    
    def _save_to_json(self, record: IndividualSuccessRecord):
        """保存到JSON文件"""
        # 读取现有数据
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        
        # 添加新记录
        data.append(asdict(record))
        
        # 写回文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_success_rate(self) -> float:
        """获取当前成功率"""
        if self.total_episodes == 0:
            return 0.0
        return self.successful_episodes / self.total_episodes
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'individual_id': self.individual_id,
            'process_id': self.process_id,
            'total_episodes': self.total_episodes,
            'successful_episodes': self.successful_episodes,
            'success_rate': self.get_success_rate(),
            'log_file': self.log_file,
            'format_type': self.format_type
        }
    
    def print_summary(self):
        """打印统计摘要"""
        stats = self.get_statistics()
        print(f"\n📊 Individual {self.individual_id} 成功统计摘要:")
        print(f"   进程ID: {stats['process_id']}")
        print(f"   总Episodes: {stats['total_episodes']}")
        print(f"   成功Episodes: {stats['successful_episodes']}")
        print(f"   成功率: {stats['success_rate']:.1%}")
        print(f"   记录文件: {stats['log_file']}")
        print(f"   记录格式: {stats['format_type'].upper()}")
    
    def close(self):
        """关闭记录器，输出最终统计"""
        self.print_summary()
        print(f"✅ Individual {self.individual_id} 成功记录器已关闭")


# 🧪 测试函数
def test_individual_success_logger():
    """测试Individual成功记录器"""
    print("🧪 测试Individual成功记录器...")
    
    # 创建记录器
    logger = IndividualSuccessLogger(
        individual_id="test_001",
        log_dir="./test_individual_logs",
        format_type="csv"
    )
    
    # 模拟记录一些episodes
    import random
    for i in range(10):
        success = random.random() > 0.3  # 70%成功率
        fitness = random.uniform(0.5, 1.0) if success else random.uniform(0.0, 0.5)
        distance = random.uniform(10, 50) if success else random.uniform(50, 200)
        
        logger.log_episode(
            episode_number=i+1,
            success=success,
            fitness=fitness,
            distance=distance,
            episode_reward=random.uniform(-100, 500),
            episode_steps=random.randint(100, 500),
            maintain_progress=random.uniform(0.0, 1.0),
            notes=f"测试episode {i+1}"
        )
    
    # 输出统计
    logger.close()
    
    print("✅ 测试完成")


if __name__ == "__main__":
    test_individual_success_logger()
