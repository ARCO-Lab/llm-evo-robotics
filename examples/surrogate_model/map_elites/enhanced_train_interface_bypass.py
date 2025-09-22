"""
MAP-Elites训练接口 - 绕过enhanced_train.py导入问题的版本
"""
import os
import sys
import argparse
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
import json
import pickle
import subprocess
import tempfile
from collections import defaultdict

# 添加路径以便导入其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..', '..', '2d_reacher', 'envs'))

class MAPElitesTrainingInterface:
    """
    MAP-Elites训练接口 - 绕过enhanced_train.py的版本
    
    这个版本通过subprocess直接调用enhanced_train.py脚本，避免导入问题
    """
    
    def __init__(self, enable_rendering=False, verbose=True):
        self.enable_rendering = enable_rendering
        self.verbose = verbose
        self.enhanced_train_path = os.path.join(
            os.path.dirname(__file__), '..', 'enhanced_train_backup.py'
        )
        
        if not os.path.exists(self.enhanced_train_path):
            raise FileNotFoundError(f"找不到enhanced_train.py: {self.enhanced_train_path}")
    
    def run_training(self, args) -> Dict[str, Any]:
        """
        运行训练 - 通过subprocess调用enhanced_train.py
        """
        if self.verbose:
            print(f"🚀 [MAP-Elites Interface] 开始训练个体...")
        
        # 准备命令行参数
        cmd = [
            'python', self.enhanced_train_path,
            '--env-name', 'reacher2d',
            '--algo', 'sac',
            '--seed', str(getattr(args, 'seed', 42)),
            '--num-processes', '1',  # MAP-Elites使用单进程
            '--lr', str(getattr(args, 'lr', 3e-4)),
            '--gamma', str(getattr(args, 'gamma', 0.99)),
            '--alpha', str(getattr(args, 'alpha', 0.1)),
            '--batch-size', str(getattr(args, 'batch_size', 64)),
            '--buffer-capacity', str(getattr(args, 'buffer_capacity', 10000)),
            '--warmup-steps', str(getattr(args, 'warmup_steps', 1000)),
            '--num-env-steps', str(getattr(args, 'total_steps', 10000)),
            '--save-dir', getattr(args, 'save_dir', './temp_training'),
            '--log-interval', '100',
            '--save-interval', '1000',
            '--render-interval', '500' if self.enable_rendering else '999999',
        ]
        
        # 添加机器人配置
        if hasattr(args, 'num_joints'):
            cmd.extend(['--num-joints', str(args.num_joints)])
        if hasattr(args, 'link_lengths'):
            cmd.extend(['--link-lengths'] + [str(x) for x in args.link_lengths])
        
        # 禁用渲染（如果需要）
        if not self.enable_rendering:
            cmd.append('--no-render')
        
        # 添加CUDA控制
        cmd.append('--no-cuda')  # MAP-Elites默认使用CPU
        
        if self.verbose:
            print(f"🔧 [训练命令] {' '.join(cmd)}")
        
        try:
            # 创建临时目录用于输出
            with tempfile.TemporaryDirectory() as temp_dir:
                # 更新保存目录到临时目录
                save_dir_idx = cmd.index('--save-dir') + 1
                cmd[save_dir_idx] = temp_dir
                
                # 运行训练
                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5分钟超时
                    cwd=os.path.dirname(self.enhanced_train_path)
                )
                end_time = time.time()
                
                # 解析结果
                training_results = self._parse_training_output(
                    result.stdout, result.stderr, temp_dir, end_time - start_time
                )
                
                if result.returncode != 0:
                    print(f"⚠️ [训练警告] 进程退出码: {result.returncode}")
                    if self.verbose:
                        print(f"stderr: {result.stderr[:500]}...")
                
                return training_results
                
        except subprocess.TimeoutExpired:
            print("⏱️ [训练超时] 训练超时，返回默认结果")
            return self._get_default_results()
        except Exception as e:
            print(f"❌ [训练错误] {e}")
            return self._get_default_results()
    
    def _parse_training_output(self, stdout: str, stderr: str, save_dir: str, duration: float) -> Dict[str, Any]:
        """
        解析训练输出，提取关键指标
        """
        results = {
            'avg_reward': -500.0,  # 默认较低奖励
            'success_rate': 0.0,
            'min_distance': 500.0,
            'episode_count': 0,
            'total_steps': 0,
            'training_duration': duration,
            'final_loss': 10.0,
            'learning_progress': 0.0,
            'training_stability': 0.1,
            'convergence_speed': 0.1
        }
        
        try:
            # 从stdout中提取信息
            lines = stdout.split('\n')
            for line in lines:
                line = line.strip()
                
                # 提取奖励信息
                if 'avg_reward' in line.lower() or '平均奖励' in line:
                    try:
                        # 寻找数字
                        import re
                        numbers = re.findall(r'-?\d+\.?\d*', line)
                        if numbers:
                            reward = float(numbers[-1])  # 取最后一个数字
                            results['avg_reward'] = max(results['avg_reward'], reward)
                    except:
                        pass
                
                # 提取成功率
                if 'success' in line.lower() or '成功率' in line:
                    try:
                        import re
                        numbers = re.findall(r'\d+\.?\d*', line)
                        if numbers:
                            success_rate = float(numbers[-1])
                            if success_rate <= 1.0:  # 确保是百分比形式
                                results['success_rate'] = success_rate
                            elif success_rate <= 100.0:  # 可能是百分比
                                results['success_rate'] = success_rate / 100.0
                    except:
                        pass
                
                # 提取距离信息
                if 'distance' in line.lower() or '距离' in line:
                    try:
                        import re
                        numbers = re.findall(r'\d+\.?\d*', line)
                        if numbers:
                            distance = float(numbers[-1])
                            results['min_distance'] = min(results['min_distance'], distance)
                    except:
                        pass
                
                # 提取episode数量
                if 'episode' in line.lower():
                    try:
                        import re
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            episode_num = int(numbers[0])
                            results['episode_count'] = max(results['episode_count'], episode_num)
                    except:
                        pass
            
            # 尝试从保存的日志文件中读取更多信息
            try:
                log_files = []
                if os.path.exists(save_dir):
                    for root, dirs, files in os.walk(save_dir):
                        for file in files:
                            if file.endswith('.json') or file.endswith('.txt'):
                                log_files.append(os.path.join(root, file))
                
                for log_file in log_files:
                    try:
                        if log_file.endswith('.json'):
                            with open(log_file, 'r') as f:
                                log_data = json.load(f)
                                if isinstance(log_data, list) and log_data:
                                    # 取最后几个记录的平均值
                                    recent_data = log_data[-5:]
                                    if 'reward' in str(recent_data):
                                        rewards = [entry.get('reward', entry.get('avg_reward', -500)) 
                                                 for entry in recent_data if isinstance(entry, dict)]
                                        if rewards:
                                            results['avg_reward'] = np.mean(rewards)
                    except:
                        continue
            except:
                pass
            
            # 计算衍生指标
            if results['avg_reward'] > -100:
                results['learning_progress'] = min(1.0, (results['avg_reward'] + 500) / 600)
            
            if results['episode_count'] > 0:
                results['training_stability'] = min(1.0, results['episode_count'] / 50.0)
                results['convergence_speed'] = min(1.0, results['episode_count'] / 100.0)
            
            # 根据训练时长调整指标
            if duration < 30:  # 训练时间太短
                results['training_stability'] *= 0.5
                results['convergence_speed'] *= 0.5
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ [解析警告] 解析训练输出时出错: {e}")
        
        return results
    
    def _get_default_results(self) -> Dict[str, Any]:
        """
        返回默认的训练结果（用于错误情况）
        """
        return {
            'avg_reward': -500.0,
            'success_rate': 0.0,
            'min_distance': 500.0,
            'episode_count': 0,
            'total_steps': 0,
            'training_duration': 0.0,
            'final_loss': 10.0,
            'learning_progress': 0.0,
            'training_stability': 0.1,
            'convergence_speed': 0.1
        }

def test_interface():
    """
    测试训练接口
    """
    print("🧪 测试MAP-Elites训练接口...")
    
    # 创建测试参数
    test_args = argparse.Namespace()
    test_args.seed = 42
    test_args.lr = 3e-4
    test_args.gamma = 0.99
    test_args.alpha = 0.1
    test_args.batch_size = 64
    test_args.buffer_capacity = 1000
    test_args.warmup_steps = 100
    test_args.total_steps = 1000
    test_args.num_joints = 3
    test_args.link_lengths = [60.0, 40.0, 30.0]
    test_args.save_dir = './test_bypass_interface'
    
    # 创建接口并运行测试
    interface = MAPElitesTrainingInterface(enable_rendering=False, verbose=True)
    results = interface.run_training(test_args)
    
    print("✅ 测试完成！结果:")
    for key, value in results.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_interface()
