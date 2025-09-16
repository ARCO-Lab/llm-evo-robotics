#!/usr/bin/env python3
"""
实验成功记录系统
记录成功的机器人结构、训练参数和性能指标
"""

import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class RobotStructure:
    """机器人结构信息"""
    num_links: int
    link_lengths: List[float]
    total_length: float
    
    def __post_init__(self):
        if self.total_length is None:
            self.total_length = sum(self.link_lengths)

@dataclass
class TrainingParameters:
    """训练参数"""
    lr: float
    alpha: float  # SAC alpha
    training_steps: int
    buffer_capacity: int
    batch_size: int
    
@dataclass
class PerformanceMetrics:
    """性能指标"""
    fitness: float
    success_rate: float
    avg_reward: float
    training_time: float
    episodes_completed: int
    final_distance_to_target: float
    path_efficiency: Optional[float] = None
    
@dataclass
class ExperimentResult:
    """完整的实验结果"""
    experiment_id: str
    timestamp: str
    robot_structure: RobotStructure
    training_params: TrainingParameters
    performance: PerformanceMetrics
    generation: int
    parent_id: Optional[str]
    success_threshold: float
    is_successful: bool
    notes: str = ""

class SuccessLogger:
    """实验成功记录器"""
    
    def __init__(self, base_dir: str = "./experiment_results", success_threshold: float = 0.7):
        """
        初始化成功记录器
        
        Args:
            base_dir: 结果保存目录
            success_threshold: 成功判定的fitness阈值
        """
        self.base_dir = base_dir
        self.success_threshold = success_threshold
        
        # 创建保存目录
        os.makedirs(base_dir, exist_ok=True)
        
        # 生成实验会话ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # 文件路径
        self.json_file = os.path.join(self.session_dir, "results.json")
        self.csv_file = os.path.join(self.session_dir, "results.csv")
        self.success_json = os.path.join(self.session_dir, "successful_results.json")
        self.success_csv = os.path.join(self.session_dir, "successful_results.csv")
        self.summary_file = os.path.join(self.session_dir, "session_summary.txt")
        
        # 初始化文件
        self._initialize_files()
        
        # 统计计数器
        self.total_experiments = 0
        self.successful_experiments = 0
        self.results_cache = []
        
        print(f"🗂️ 实验记录器已初始化")
        print(f"   会话ID: {self.session_id}")
        print(f"   保存目录: {self.session_dir}")
        print(f"   成功阈值: {self.success_threshold}")
    
    def _initialize_files(self):
        """初始化记录文件"""
        # CSV表头
        csv_headers = [
            'experiment_id', 'timestamp', 'generation', 'parent_id',
            'num_links', 'link_lengths', 'total_length',
            'lr', 'alpha', 'training_steps', 'buffer_capacity', 'batch_size',
            'fitness', 'success_rate', 'avg_reward', 'training_time', 
            'episodes_completed', 'final_distance_to_target', 'path_efficiency',
            'success_threshold', 'is_successful', 'notes'
        ]
        
        # 创建CSV文件
        for csv_path in [self.csv_file, self.success_csv]:
            if not os.path.exists(csv_path):
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_headers)
        
        # 创建JSON文件
        for json_path in [self.json_file, self.success_json]:
            if not os.path.exists(json_path):
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=2, ensure_ascii=False)
    
    def log_result(self, 
                   individual_id: str,
                   robot_structure: RobotStructure,
                   training_params: TrainingParameters,
                   performance: PerformanceMetrics,
                   generation: int = 0,
                   parent_id: Optional[str] = None,
                   notes: str = "") -> bool:
        """
        记录实验结果
        
        Args:
            individual_id: 个体ID
            robot_structure: 机器人结构
            training_params: 训练参数
            performance: 性能指标
            generation: 代数
            parent_id: 父代ID
            notes: 备注
            
        Returns:
            bool: 是否为成功实验
        """
        # 判断是否成功
        is_successful = performance.fitness >= self.success_threshold
        
        # 创建实验结果
        result = ExperimentResult(
            experiment_id=individual_id,
            timestamp=datetime.now().isoformat(),
            robot_structure=robot_structure,
            training_params=training_params,
            performance=performance,
            generation=generation,
            parent_id=parent_id,
            success_threshold=self.success_threshold,
            is_successful=is_successful,
            notes=notes
        )
        
        # 更新统计
        self.total_experiments += 1
        if is_successful:
            self.successful_experiments += 1
        
        # 保存结果
        self._save_result(result, is_successful)
        
        # 缓存结果
        self.results_cache.append(result)
        
        # 输出日志
        status = "✅ 成功" if is_successful else "❌ 失败"
        print(f"{status} 实验记录: {individual_id}")
        print(f"   机器人: {robot_structure.num_links}关节, 总长度: {robot_structure.total_length:.1f}")
        print(f"   性能: fitness={performance.fitness:.3f}, 成功率={performance.success_rate:.1%}")
        print(f"   总实验: {self.total_experiments}, 成功: {self.successful_experiments} ({self.successful_experiments/self.total_experiments:.1%})")
        
        return is_successful
    
    def _save_result(self, result: ExperimentResult, is_successful: bool):
        """保存结果到文件"""
        # 转换为字典
        result_dict = asdict(result)
        
        # 保存到JSON - 所有结果
        self._append_to_json(self.json_file, result_dict)
        
        # 保存到CSV - 所有结果
        self._append_to_csv(self.csv_file, result_dict)
        
        # 如果成功，也保存到成功结果文件
        if is_successful:
            self._append_to_json(self.success_json, result_dict)
            self._append_to_csv(self.success_csv, result_dict)
    
    def _append_to_json(self, filepath: str, result_dict: Dict):
        """追加结果到JSON文件"""
        try:
            # 读取现有数据
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 添加新结果
            data.append(result_dict)
            
            # 写回文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"⚠️ 保存JSON失败: {e}")
    
    def _append_to_csv(self, filepath: str, result_dict: Dict):
        """追加结果到CSV文件"""
        try:
            with open(filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 展开嵌套字典
                row = [
                    result_dict['experiment_id'],
                    result_dict['timestamp'],
                    result_dict['generation'],
                    result_dict['parent_id'],
                    result_dict['robot_structure']['num_links'],
                    str(result_dict['robot_structure']['link_lengths']),
                    result_dict['robot_structure']['total_length'],
                    result_dict['training_params']['lr'],
                    result_dict['training_params']['alpha'],
                    result_dict['training_params']['training_steps'],
                    result_dict['training_params']['buffer_capacity'],
                    result_dict['training_params']['batch_size'],
                    result_dict['performance']['fitness'],
                    result_dict['performance']['success_rate'],
                    result_dict['performance']['avg_reward'],
                    result_dict['performance']['training_time'],
                    result_dict['performance']['episodes_completed'],
                    result_dict['performance']['final_distance_to_target'],
                    result_dict['performance'].get('path_efficiency', ''),
                    result_dict['success_threshold'],
                    result_dict['is_successful'],
                    result_dict['notes']
                ]
                
                writer.writerow(row)
                
        except Exception as e:
            print(f"⚠️ 保存CSV失败: {e}")
    
    def get_success_statistics(self) -> Dict[str, Any]:
        """获取成功统计信息"""
        if self.total_experiments == 0:
            return {
                'total_experiments': 0,
                'successful_experiments': 0,
                'success_rate': 0.0,
                'avg_fitness': 0.0,
                'best_fitness': 0.0,
                'successful_structures': []
            }
        
        successful_results = [r for r in self.results_cache if r.is_successful]
        all_fitness = [r.performance.fitness for r in self.results_cache]
        
        stats = {
            'session_id': self.session_id,
            'total_experiments': self.total_experiments,
            'successful_experiments': self.successful_experiments,
            'success_rate': self.successful_experiments / self.total_experiments,
            'avg_fitness': np.mean(all_fitness) if all_fitness else 0.0,
            'best_fitness': max(all_fitness) if all_fitness else 0.0,
            'successful_structures': []
        }
        
        # 成功结构统计
        for result in successful_results:
            structure_info = {
                'experiment_id': result.experiment_id,
                'num_links': result.robot_structure.num_links,
                'link_lengths': result.robot_structure.link_lengths,
                'total_length': result.robot_structure.total_length,
                'fitness': result.performance.fitness,
                'success_rate': result.performance.success_rate,
                'generation': result.generation
            }
            stats['successful_structures'].append(structure_info)
        
        return stats
    
    def generate_summary(self):
        """生成实验总结"""
        stats = self.get_success_statistics()
        
        summary = f"""
🧪 实验会话总结
==========================================
会话ID: {self.session_id}
开始时间: {self.session_id[:8]} {self.session_id[9:11]}:{self.session_id[11:13]}:{self.session_id[13:15]}
成功阈值: {self.success_threshold}

📊 总体统计
------------------------------------------
总实验数: {stats['total_experiments']}
成功实验数: {stats['successful_experiments']}
成功率: {stats['success_rate']:.1%}
平均fitness: {stats['avg_fitness']:.3f}
最佳fitness: {stats['best_fitness']:.3f}

🏆 成功结构列表
------------------------------------------
"""
        
        if stats['successful_structures']:
            for i, structure in enumerate(stats['successful_structures'], 1):
                summary += f"""
{i}. 实验ID: {structure['experiment_id']}
   关节数: {structure['num_links']}
   链长: {structure['link_lengths']}
   总长度: {structure['total_length']:.1f}
   Fitness: {structure['fitness']:.3f}
   成功率: {structure['success_rate']:.1%}
   代数: {structure['generation']}
"""
        else:
            summary += "\n❌ 本次实验中没有成功的结构\n"
        
        summary += f"""
📁 文件位置
------------------------------------------
结果目录: {self.session_dir}
所有结果: results.json, results.csv
成功结果: successful_results.json, successful_results.csv
本总结: session_summary.txt

🔍 查看结果
------------------------------------------
python examples/surrogate_model/map_elites/view_results.py {self.session_id}
"""
        
        # 保存总结
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(summary)
        return summary
    
    def close(self):
        """关闭记录器，生成最终总结"""
        print(f"\n🏁 实验会话结束: {self.session_id}")
        self.generate_summary()
        print(f"📊 最终统计: {self.successful_experiments}/{self.total_experiments} 成功 ({self.successful_experiments/self.total_experiments:.1%} 成功率)")

# 便捷函数
def create_robot_structure(num_links: int, link_lengths: List[float]) -> RobotStructure:
    """创建机器人结构"""
    return RobotStructure(
        num_links=num_links,
        link_lengths=link_lengths,
        total_length=sum(link_lengths)
    )

def create_training_params(lr: float, alpha: float, training_steps: int = 500, 
                         buffer_capacity: int = 10000, batch_size: int = 64) -> TrainingParameters:
    """创建训练参数"""
    return TrainingParameters(
        lr=lr,
        alpha=alpha,
        training_steps=training_steps,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size
    )

def create_performance_metrics(fitness: float, success_rate: float, avg_reward: float,
                             training_time: float, episodes_completed: int,
                             final_distance_to_target: float, path_efficiency: Optional[float] = None) -> PerformanceMetrics:
    """创建性能指标"""
    return PerformanceMetrics(
        fitness=fitness,
        success_rate=success_rate,
        avg_reward=avg_reward,
        training_time=training_time,
        episodes_completed=episodes_completed,
        final_distance_to_target=final_distance_to_target,
        path_efficiency=path_efficiency
    )

# 测试函数
def test_success_logger():
    """测试成功记录器"""
    logger = SuccessLogger(success_threshold=0.6)
    
    # 模拟一些实验结果
    test_results = [
        {
            'id': 'test_001',
            'structure': create_robot_structure(3, [60.0, 50.0, 40.0]),
            'params': create_training_params(0.001, 0.2),
            'performance': create_performance_metrics(0.75, 0.8, 15.2, 45.6, 20, 2.3),
            'notes': '成功的3关节机器人'
        },
        {
            'id': 'test_002', 
            'structure': create_robot_structure(4, [70.0, 60.0, 50.0, 40.0]),
            'params': create_training_params(0.0005, 0.15),
            'performance': create_performance_metrics(0.45, 0.4, 8.7, 52.3, 15, 5.8),
            'notes': '失败的4关节机器人'
        },
        {
            'id': 'test_003',
            'structure': create_robot_structure(5, [80.0, 70.0, 60.0, 50.0, 40.0]),
            'params': create_training_params(0.002, 0.25),
            'performance': create_performance_metrics(0.82, 0.9, 22.1, 38.9, 25, 1.2),
            'notes': '非常成功的5关节机器人'
        }
    ]
    
    # 记录结果
    for result in test_results:
        logger.log_result(
            result['id'],
            result['structure'],
            result['params'],
            result['performance'],
            generation=1,
            notes=result['notes']
        )
    
    # 生成总结
    logger.close()

if __name__ == "__main__":
    test_success_logger()
