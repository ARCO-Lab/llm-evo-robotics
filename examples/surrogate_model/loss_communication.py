#!/usr/bin/env python3
"""
损失通信模块
用于在训练进程和损失记录器之间进行实时通信
"""

import os
import json
import time
import tempfile
import fcntl
from typing import Dict, Any
from pathlib import Path

class LossCommunicator:
    """损失通信器 - 用于进程间损失数据传递"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.comm_dir = Path(tempfile.gettempdir()) / f"loss_comm_{experiment_name}"
        self.comm_dir.mkdir(exist_ok=True)
        
        # 通信文件路径
        self.loss_file = self.comm_dir / "losses.jsonl"
        self.status_file = self.comm_dir / "status.json"
        
        # 初始化状态文件
        self._write_status("initialized")
        
    def send_loss(self, network: str, step: int, losses: Dict[str, float], timestamp: float = None):
        """发送损失数据到记录器"""
        if timestamp is None:
            timestamp = time.time()
            
        loss_data = {
            'network': network,
            'step': step,
            'timestamp': timestamp,
            'losses': losses
        }
        
        # 原子写入损失数据
        try:
            with open(self.loss_file, 'a') as f:
                # 文件锁确保原子写入
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(loss_data) + '\n')
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            print(f"⚠️ 发送损失数据失败: {e}")
    
    def receive_losses(self):
        """接收损失数据（损失记录器调用）"""
        if not self.loss_file.exists():
            return []
            
        losses = []
        try:
            with open(self.loss_file, 'r') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            loss_data = json.loads(line)
                            losses.append(loss_data)
                        except json.JSONDecodeError:
                            continue
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                            
            # 读取后清空文件
            if losses:
                self.loss_file.write_text("")
                
        except Exception as e:
            print(f"⚠️ 接收损失数据失败: {e}")
            
        return losses
    
    def _write_status(self, status: str):
        """写入状态信息"""
        status_data = {
            'status': status,
            'timestamp': time.time(),
            'experiment': self.experiment_name
        }
        
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f)
        except Exception as e:
            print(f"⚠️ 写入状态失败: {e}")
    
    def cleanup(self):
        """清理通信文件"""
        try:
            if self.loss_file.exists():
                self.loss_file.unlink()
            if self.status_file.exists():
                self.status_file.unlink()
            if self.comm_dir.exists():
                self.comm_dir.rmdir()
        except Exception as e:
            print(f"⚠️ 清理通信文件失败: {e}")


class RealTimeLossCollector:
    """实时损失收集器 - 在损失记录器进程中运行"""
    
    def __init__(self, experiment_name: str, loss_logger_interface):
        self.experiment_name = experiment_name
        self.communicator = LossCommunicator(experiment_name)
        self.loss_logger = loss_logger_interface
        self.running = False
        
    def start_collecting(self):
        """开始收集损失数据"""
        self.running = True
        print(f"🔄 开始实时收集损失数据: {self.experiment_name}")
        
        while self.running:
            try:
                # 接收新的损失数据
                new_losses = self.communicator.receive_losses()
                
                # 记录到损失记录器
                for loss_data in new_losses:
                    network = loss_data['network']
                    step = loss_data['step']
                    timestamp = loss_data['timestamp']
                    losses = loss_data['losses']
                    
                    # 调用损失记录器
                    from loss_logger_interface import log_network_loss
                    log_network_loss(network, step, losses, timestamp)
                
                if new_losses:
                    print(f"📊 收集到 {len(new_losses)} 条损失数据")
                
                # 短暂休眠避免过度占用CPU
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ 收集损失数据时出错: {e}")
                time.sleep(1)
        
        print("🛑 停止收集损失数据")
        self.communicator.cleanup()
    
    def stop_collecting(self):
        """停止收集"""
        self.running = False


# 便捷函数
def send_training_loss(experiment_name: str, network: str, step: int, losses: Dict[str, float]):
    """发送训练损失的便捷函数"""
    communicator = LossCommunicator(experiment_name)
    communicator.send_loss(network, step, losses)


if __name__ == "__main__":
    # 测试通信
    experiment = "test_comm"
    
    # 模拟发送端
    comm = LossCommunicator(experiment)
    
    # 发送一些测试数据
    for i in range(10):
        comm.send_loss('ppo', i, {'actor_loss': 1.0 - i*0.1, 'critic_loss': 0.8 - i*0.05})
        time.sleep(0.1)
    
    # 模拟接收端
    received = comm.receive_losses()
    print(f"接收到 {len(received)} 条数据")
    for data in received:
        print(f"  {data}")
    
    comm.cleanup()
    print("✅ 通信测试完成")
