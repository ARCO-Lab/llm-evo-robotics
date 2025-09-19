#!/usr/bin/env python3
"""
损失记录器接口 - 简化版
用于在MAP-Elites训练代码中快速集成网络损失记录功能
"""

import os
import sys
import time
from network_loss_logger import init_network_loss_logger, log_network_loss, cleanup_network_loss_logger, get_network_loss_logger

class LossLoggerInterface:
    """损失记录器接口 - 简化使用"""
    
    _instance = None
    
    def __init__(self, experiment_name=None, log_dir="network_loss_logs", 
                 networks=['attention', 'ppo', 'gnn'], update_interval=10.0, auto_start=True):
        """
        初始化损失记录器接口（单例模式）
        
        Args:
            experiment_name: 实验名称
            log_dir: 日志目录
            networks: 监控的网络列表
            update_interval: 图表更新间隔
            auto_start: 是否自动启动
        """
        if LossLoggerInterface._instance is not None:
            print("⚠️  LossLoggerInterface已初始化，使用现有实例")
            # 复制现有实例的属性
            existing = LossLoggerInterface._instance
            self.logger = existing.logger
            self.experiment_name = existing.experiment_name
            self.log_dir = existing.log_dir
            self.networks = existing.networks
            self.update_interval = existing.update_interval
            return
            
        LossLoggerInterface._instance = self
        
        self.logger = None
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.networks = networks
        self.update_interval = update_interval
        
        if auto_start:
            self.start()
        
        print(f"🎯 LossLoggerInterface已初始化")
    
    def start(self):
        """启动损失记录器"""
        if self.logger is not None:
            print("⚠️  损失记录器已启动")
            return
            
        self.logger = init_network_loss_logger(
            experiment_name=self.experiment_name,
            log_dir=self.log_dir,
            networks=self.networks,
            update_interval=self.update_interval
        )
        
        print(f"✅ 损失记录器已启动 - 实验: {self.logger.experiment_name}")
        return self.logger
    
    def stop(self):
        """停止损失记录器"""
        if self.logger is not None:
            cleanup_network_loss_logger()
            self.logger = None
            print("🛑 损失记录器已停止")
    
    @classmethod
    def get_instance(cls, **kwargs):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance
    
    def log_attention_loss(self, step, attention_loss_dict, timestamp=None):
        """记录attention网络损失"""
        log_network_loss('attention', step, attention_loss_dict, timestamp)
    
    def log_ppo_loss(self, step, ppo_loss_dict, timestamp=None):
        """记录PPO网络损失"""
        log_network_loss('ppo', step, ppo_loss_dict, timestamp)
    
    def log_gnn_loss(self, step, gnn_loss_dict, timestamp=None):
        """记录GNN网络损失"""
        log_network_loss('gnn', step, gnn_loss_dict, timestamp)
    
    def log_custom_loss(self, network_name, step, loss_dict, timestamp=None):
        """记录自定义网络损失"""
        log_network_loss(network_name, step, loss_dict, timestamp)
    
    def get_log_dir(self):
        """获取日志目录"""
        if self.logger:
            return self.logger.experiment_dir
        return None
    
    def is_alive(self):
        """检查记录器是否还在运行"""
        if self.logger:
            return self.logger.is_alive()
        return False


# 便捷函数 - 全局接口
def start_loss_logging(experiment_name=None, **kwargs):
    """启动损失记录的便捷函数"""
    interface = LossLoggerInterface.get_instance(experiment_name=experiment_name, **kwargs)
    return interface

def stop_loss_logging():
    """停止损失记录的便捷函数"""
    interface = LossLoggerInterface.get_instance(auto_start=False)
    interface.stop()

def log_attention_loss(step, loss_dict, timestamp=None):
    """记录attention损失的便捷函数"""
    log_network_loss('attention', step, loss_dict, timestamp)

def log_ppo_loss(step, loss_dict, timestamp=None):
    """记录PPO损失的便捷函数"""
    log_network_loss('ppo', step, loss_dict, timestamp)

def log_gnn_loss(step, loss_dict, timestamp=None):
    """记录GNN损失的便捷函数"""
    log_network_loss('gnn', step, loss_dict, timestamp)

def log_custom_network_loss(network_name, step, loss_dict, timestamp=None):
    """记录自定义网络损失的便捷函数"""
    log_network_loss(network_name, step, loss_dict, timestamp)

def get_loss_log_directory():
    """获取损失日志目录的便捷函数"""
    interface = LossLoggerInterface.get_instance(auto_start=False)
    return interface.get_log_dir()

def is_loss_logger_alive():
    """检查损失记录器是否还在运行"""
    interface = LossLoggerInterface.get_instance(auto_start=False)
    return interface.is_alive()


# 装饰器 - 自动记录函数的损失
def auto_log_loss(network_name, step_param='step'):
    """
    装饰器：自动记录函数返回的损失值
    
    Args:
        network_name: 网络名称
        step_param: 步数参数名
    
    Usage:
        @auto_log_loss('ppo', 'training_step')
        def train_ppo(training_step, ...):
            # 训练逻辑
            return {'actor_loss': 0.5, 'critic_loss': 0.3}
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # 尝试获取步数
            step = None
            if step_param in kwargs:
                step = kwargs[step_param]
            else:
                # 尝试从位置参数获取（假设是第一个参数）
                if args:
                    step = args[0]
            
            # 如果返回值是字典，记录为损失
            if isinstance(result, dict) and step is not None:
                log_network_loss(network_name, step, result)
            
            return result
        return wrapper
    return decorator


# 上下文管理器
class LossLoggingContext:
    """损失记录上下文管理器"""
    
    def __init__(self, experiment_name=None, **kwargs):
        self.experiment_name = experiment_name
        self.kwargs = kwargs
        self.interface = None
    
    def __enter__(self):
        self.interface = start_loss_logging(self.experiment_name, **self.kwargs)
        return self.interface
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        stop_loss_logging()


# 使用示例
if __name__ == "__main__":
    print("🧪 测试损失记录器接口")
    
    # 方法1: 直接使用便捷函数
    print("\n=== 方法1: 便捷函数 ===")
    logger_interface = start_loss_logging(
        experiment_name="test_interface",
        networks=['attention', 'ppo', 'gnn', 'custom']
    )
    
    for step in range(50):
        log_attention_loss(step, {'attention_loss': 2.0 - step*0.01})
        log_ppo_loss(step, {'actor_loss': 1.5 - step*0.008, 'critic_loss': 1.2 - step*0.006})
        log_gnn_loss(step, {'gnn_loss': 2.5 - step*0.012})
        log_custom_network_loss('custom', step, {'custom_loss': 1.0 - step*0.005})
        
        if step % 10 == 0:
            print(f"Step {step} - 日志目录: {get_loss_log_directory()}")
        
        time.sleep(0.05)
    
    stop_loss_logging()
    
    # 方法2: 使用上下文管理器
    print("\n=== 方法2: 上下文管理器 ===")
    with LossLoggingContext(experiment_name="test_context") as logger:
        for step in range(30):
            log_attention_loss(step, {'attention_loss': 1.8 - step*0.01})
            if step % 10 == 0:
                print(f"Context Step {step}")
            time.sleep(0.05)
    
    # 方法3: 使用装饰器
    print("\n=== 方法3: 装饰器 ===")
    start_loss_logging(experiment_name="test_decorator")
    
    @auto_log_loss('ppo')
    def mock_ppo_training(step):
        # 模拟PPO训练
        return {
            'actor_loss': max(0.01, 1.5 - step*0.01),
            'critic_loss': max(0.01, 1.2 - step*0.008)
        }
    
    for step in range(20):
        mock_ppo_training(step)
        if step % 5 == 0:
            print(f"Decorator Step {step}")
        time.sleep(0.05)
    
    stop_loss_logging()
    
    print("✅ 测试完成")
